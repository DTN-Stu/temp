`timescale 1ns/1ps
// =============================================================================
// Module: softmax_requantizer
// -----------------------------------------------------------------------------
// Exp-free probability generator for the FP4 attention path.
//
// Consumes one block of BLOCK_SIZE attention scores straight out of the matmul
// accumulator and produces the P operand for the following P*V matmul:
//
//   code_out      BLOCK_SIZE x E2M1 magnitude codes (probabilities, sign = 0)
//   p_blk_out     one Log8 S4.3 block scale
//   csum_out      sum of M_int = 2*|level|, this block's contribution to l
//   l_contrib_out S_blk * csum in Q16
//   resc_*        factor 2^(m_old - m_new) for the online rescale of O and l
//
// -----------------------------------------------------------------------------
// WHY THERE IS NO exp
//
//   softmax :  p_t = E_t / l,   E_t = 2^(s_t - m),   l = sum_u E_u
//   FP4 scale:  S = max(p) / 6 = E_bmax / (6 l)
//
//   The quantiser encodes p_t / S, and p_t and S carry the same 1/l:
//
//       p_t / S = (E_t / l) / (E_bmax / (6 l)) = 6 * E_t / E_bmax
//               = 6 * 2^-(s_bmax - s_t) = 6 * 2^-d_t
//
//   l and m cancel, so the code depends only on d_t, a plain integer
//   subtraction. Rounding 6*2^-d_t to the nearest E2M1 level therefore becomes
//   a comparison of d_t against seven constants: the arithmetic midpoints of
//   E2M1 pushed through d = log2(6 / beta).
//
//   The unnormalised convention S = E_bmax / 6 is used because l is unknown
//   until the whole row has been consumed. l is accumulated from the quantised
//   codes and folded into the final output scale by the consumer.
//
// -----------------------------------------------------------------------------
// TIMING (one block per cycle, back to back)
//
//   latency 1 : m_out / m_valid / row_start_out / resc_*
//       The running max is fed back to the next block, so its loop must close
//       in one cycle. The 16-input max tree is therefore combinational on the
//       input side.
//
//   latency 2 : code_out / p_blk_out / csum_out / l_contrib_out / block_empty
//       One more register stage for subtract + threshold + CSD scale.
//
//   If the max tree cannot close timing, register half of it and add one cycle
//   to BOTH paths; the block rate must then drop accordingly.
//
// -----------------------------------------------------------------------------
// INVARIANTS
//   * p_blk_out <= -21 always, because s_bmax - m <= 0 by construction.
//   * a masked key gets code 0 and contributes nothing to csum or the max tree.
//   * on row_start_out the consumer must CLEAR (not scale) O and l.
//   * BLOCK_SIZE is fixed at 16: the max tree is a 4-level 8-4-2-1 reduction.
// =============================================================================
module softmax_requantizer #(
    parameter int BLOCK_SIZE  = 16,  // fixed at 16 (see INVARIANTS)
    parameter int SCORE_FRAC  = 16,  // scores are Q16 two's complement
    parameter int NUM_QUERIES = 1,
    parameter int QW = (NUM_QUERIES > 1) ? $clog2(NUM_QUERIES) : 1
) (
    input  logic                        clk,
    input  logic                        rst_n,

    // ---------------------------------------------------------- score block
    input  logic                        in_valid,
    input  logic signed [BLOCK_SIZE-1:0][31:0] score_in,
    input  logic [BLOCK_SIZE-1:0]       key_mask,     // 1 = key visible
    input  logic                        row_start,    // first block of a row
    input  logic                        clear,        // reset ALL query maxima
    input  logic [QW-1:0]               query_idx,    // query state select

    // ------------------------------------------------------- P operand (lat 2)
    output logic                        out_valid,
    output logic [4*BLOCK_SIZE-1:0]     code_out,     // per key: {1'b0, mag[2:0]}
    output logic signed [7:0]           p_blk_out,    // Log8 S4.3 block scale
    output logic [7:0]                  csum_out,     // sum of M_int = 2*|level|
    output logic signed [31:0]          l_contrib_out,// S_blk * csum in Q16
    output logic                        block_empty,  // no visible key

    // ------------------------------------- online max / rescale (latency 1)
    output logic                        m_valid,
    output logic signed [31:0]          m_out,        // updated running max
    output logic                        row_start_out,
    output logic                        resc_valid,   // running max increased
    output logic signed [7:0]           resc_q,       // octave part of 2^(dm)
    output logic        [2:0]           resc_r        // CSD residual index
);

    // =========================================================================
    // 1. Constants
    // =========================================================================
    // Seven E2M1 arithmetic-midpoint boundaries, expressed as the distance
    // d = log2(6 / beta) in Q16.  A key with d <= TH[k] gets the k-th code.
    //
    //   beta  6.0/4.0  4.0/3.0  3.0/2.0  2.0/1.5  1.5/1.0  1.0/0.5  0.5/0.0
    //   d     0.2630   0.7776   1.2630   1.7776   2.2630   3.0000   4.5850
    //   Q16   17238    50961    82774    116497   148310   196608   300480
    localparam logic [32:0] TH_6P0 = 33'd17238;
    localparam logic [32:0] TH_4P0 = 33'd50961;
    localparam logic [32:0] TH_3P0 = 33'd82774;
    localparam logic [32:0] TH_2P0 = 33'd116497;
    localparam logic [32:0] TH_1P5 = 33'd148310;
    localparam logic [32:0] TH_1P0 = 33'd196608;
    localparam logic [32:0] TH_0P5 = 33'd300480;

    // 8*log2(6) = 20.68, rounded to 21.  The constant is identical for every
    // block, so it cancels exactly in the consumer's O/l normalisation; only
    // the fractional part of p_blk matters.
    localparam logic signed [32:0] P_OFFSET = -33'sd21;

    // Rounding term for p = 8*(s_bmax - m)/2^SCORE_FRAC, i.e. half an LSB of
    // the shift below.  Rounding halves the scale error versus truncation
    // (4.4% -> 2.2% RMS) and can only make the scale larger, which is the safe
    // direction (no clipping).
    localparam logic signed [32:0] P_RND = 33'sd1 << (SCORE_FRAC - 4);

    // CSD approximation of 32 * 2^(r/8), identical to logscale_frac_scaler.
    localparam logic [5:0] CSD_Q5 [0:7] = '{6'd32, 6'd35, 6'd38, 6'd41,
                                           6'd46, 6'd49, 6'd54, 6'd59};

    // Most negative signed value: used to make masked keys lose the max tree.
    localparam logic signed [31:0] S_MIN = 32'sh8000_0000;

    // =========================================================================
    // 2. Helper functions
    // =========================================================================
    // Nearest E2M1 magnitude for a normalised value 6*2^-d.  Because the value
    // decreases monotonically with d, this is a pure threshold function.
    function automatic logic [2:0] mag_of(input logic [32:0] d);
        if      (d <= TH_6P0) mag_of = 3'b111;   // 6.0
        else if (d <= TH_4P0) mag_of = 3'b110;   // 4.0
        else if (d <= TH_3P0) mag_of = 3'b101;   // 3.0
        else if (d <= TH_2P0) mag_of = 3'b100;   // 2.0
        else if (d <= TH_1P5) mag_of = 3'b011;   // 1.5
        else if (d <= TH_1P0) mag_of = 3'b010;   // 1.0
        else if (d <= TH_0P5) mag_of = 3'b001;   // 0.5
        else                  mag_of = 3'b000;   // 0.0
    endfunction

    // M_int = 2 * |level|, the same encoding fp4_mult_lut uses.  Summing M_int
    // instead of the level keeps the row sum integral; the factor 2 is a
    // constant and cancels in the consumer's O/l.
    function automatic logic [3:0] mint_of(input logic [2:0] mag);
        case (mag)
            3'd0:    mint_of = 4'd0;    // 0.0
            3'd1:    mint_of = 4'd1;    // 0.5
            3'd2:    mint_of = 4'd2;    // 1.0
            3'd3:    mint_of = 4'd3;    // 1.5
            3'd4:    mint_of = 4'd4;    // 2.0
            3'd5:    mint_of = 4'd6;    // 3.0
            3'd6:    mint_of = 4'd8;    // 4.0
            default: mint_of = 4'd12;   // 6.0
        endcase
    endfunction

    // =========================================================================
    // 3. Stage 0 (combinational): causal mask + 16-input max tree
    // =========================================================================
    logic signed [BLOCK_SIZE-1:0][31:0] score_msk;
    logic               any_valid;

    int sm_t_msk;
    int sm_t_s1;
    int sm_t_code;
    int sm_t_out;
    int sm_i;
    int sm_q;

    always_comb begin
        for (sm_t_msk = 0; sm_t_msk < BLOCK_SIZE; sm_t_msk++)
            score_msk[sm_t_msk] = key_mask[sm_t_msk] ? score_in[sm_t_msk] : S_MIN;
        any_valid = |key_mask;
    end

    logic signed [7:0][31:0] tree_l1;
    logic signed [3:0][31:0] tree_l2;
    logic signed [1:0][31:0] tree_l3;
    logic signed [31:0] blk_max;

    always_comb begin
        for (sm_i = 0; sm_i < 8; sm_i++)
            tree_l1[sm_i] = ($signed(score_msk[2*sm_i]) > $signed(score_msk[2*sm_i+1])) ? score_msk[2*sm_i] : score_msk[2*sm_i+1];
        for (sm_i = 0; sm_i < 4; sm_i++)
            tree_l2[sm_i] = ($signed(tree_l1[2*sm_i]) > $signed(tree_l1[2*sm_i+1])) ? tree_l1[2*sm_i] : tree_l1[2*sm_i+1];
        for (sm_i = 0; sm_i < 2; sm_i++)
            tree_l3[sm_i] = ($signed(tree_l2[2*sm_i]) > $signed(tree_l2[2*sm_i+1])) ? tree_l2[2*sm_i] : tree_l2[2*sm_i+1];
        blk_max = ($signed(tree_l3[0]) > $signed(tree_l3[1])) ? tree_l3[0] : tree_l3[1];
    end

    // =========================================================================
    // 4. Running max and rescale factor (one-cycle loop)
    // =========================================================================
    logic signed [NUM_QUERIES-1:0][31:0] m_q;
    logic signed [31:0] m_curr;
    logic signed [31:0] m_next;
    logic               resc_req;

    assign m_curr = m_q[query_idx];

    always_comb begin
        if (!any_valid) begin
            m_next   = m_curr;                    // empty block: hold
            resc_req = 1'b0;
        end else if (row_start) begin
            m_next   = blk_max;                   // new row: reload
            resc_req = 1'b0;
        end else begin
            m_next   = (blk_max > m_curr) ? blk_max : m_curr;
            resc_req = (blk_max > m_curr);
        end
    end

    // c = 2^(m_old - m_new) with m_old - m_new <= 0, split into an octave part
    // and a 3-bit CSD residual so the consumer reuses logscale_frac_scaler.
    logic signed [32:0] resc_delta;               // <= 0
    logic signed [32:0] resc_oct;
    logic signed [7:0]  resc_q_d;
    logic        [2:0]  resc_r_d;

    assign resc_delta = {m_curr[31], m_curr} - {m_next[31], m_next};
    assign resc_oct   = resc_delta >>> SCORE_FRAC;

    always_comb begin
        if      (resc_oct < -33'sd128) resc_q_d = -8'sd128;
        else if (resc_oct >  33'sd127) resc_q_d =  8'sd127;
        else                           resc_q_d =  resc_oct[7:0];
        resc_r_d = resc_delta[SCORE_FRAC-1 -: 3];
    end

    // =========================================================================
    // 5. Stage 1 registers
    // =========================================================================
    logic signed [BLOCK_SIZE-1:0][31:0] s1_score;
    logic [BLOCK_SIZE-1:0] s1_mask;
    logic signed [31:0]    s1_blk_max;
    logic signed [31:0]    s1_m;          // m_next of this block
    logic                  s1_valid;
    logic                  s1_row_start;
    logic                  s1_any;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (sm_q = 0; sm_q < NUM_QUERIES; sm_q++) m_q[sm_q] <= S_MIN;
            s1_valid     <= 1'b0;
            s1_row_start <= 1'b0;
            s1_any       <= 1'b0;
            s1_mask      <= '0;
            s1_blk_max   <= S_MIN;
            s1_m         <= S_MIN;
            for (sm_t_s1 = 0; sm_t_s1 < BLOCK_SIZE; sm_t_s1++) s1_score[sm_t_s1] <= 32'sd0;
        end else begin
            // `clear` resets every query's running max at once, which is what
            // lets the drain run one block per cycle.  Clearing to S_MIN makes
            // the very first block of each query take max(S_MIN, blk_max) =
            // blk_max on the normal path, so no per-block seed pulse is needed.
            if (clear) begin
                for (sm_q = 0; sm_q < NUM_QUERIES; sm_q++) m_q[sm_q] <= S_MIN;
            end else if (in_valid) begin
                m_q[query_idx] <= m_next;
            end
            s1_valid     <= in_valid;
            s1_row_start <= in_valid & row_start;
            s1_any       <= in_valid & any_valid;
            s1_mask      <= in_valid ? key_mask : '0;
            s1_blk_max   <= blk_max;
            s1_m         <= m_next;
            for (sm_t_s1 = 0; sm_t_s1 < BLOCK_SIZE; sm_t_s1++) s1_score[sm_t_s1] <= score_msk[sm_t_s1];
        end
    end

    // =========================================================================
    // 6. Stage 2 (combinational): scale, codes, code sum, l contribution
    // =========================================================================
    // 6a. Block scale:  p = 8*log2(S_blk) = 8*(s_bmax - m)/2^SCORE_FRAC - 8*log2(6)
    logic signed [32:0] blk_max_minus_m;
    logic signed [32:0] p_wide;
    logic signed [7:0]  p_blk;

    assign blk_max_minus_m = {s1_blk_max[31], s1_blk_max} - {s1_m[31], s1_m};

    always_comb begin
        p_wide = ((blk_max_minus_m + P_RND) >>> (SCORE_FRAC - 3)) + P_OFFSET;
        if      (p_wide < -33'sd128) p_blk = -8'sd128;
        else if (p_wide >  33'sd127) p_blk =  8'sd127;
        else                         p_blk =  p_wide[7:0];
    end

    // 6b. Per-key distance, magnitude, code and code sum
    logic [BLOCK_SIZE-1:0][32:0] score_dist;
    logic [BLOCK_SIZE-1:0][2:0]  mag;
    logic [BLOCK_SIZE-1:0][3:0]  code_d;
    logic [7:0]  csum_d;

    always_comb begin
        csum_d = 8'd0;
        for (sm_t_code = 0; sm_t_code < BLOCK_SIZE; sm_t_code++) begin
            score_dist[sm_t_code]   = {s1_blk_max[31], s1_blk_max} - {s1_score[sm_t_code][31], s1_score[sm_t_code]};
            mag[sm_t_code]    = mag_of(score_dist[sm_t_code]);
            code_d[sm_t_code] = s1_mask[sm_t_code] ? {1'b0, mag[sm_t_code]} : 4'b0000;
            csum_d    = csum_d + (s1_mask[sm_t_code] ? {4'd0, mint_of(mag[sm_t_code])} : 8'd0);
        end
    end

    // 6c. l contribution: S_blk * sum(level) * 2^SCORE_FRAC
    //     csum_d is 2*sum(level) (the M_int encoding used by fp4_mult_lut),
    //     so the final shift is 6 instead of 5.  This removes the extra
    //     factor 2 that would otherwise make O/l half of the true softmax
    //     average.  S_blk = 2^(p/8) = 2^q * CSD[r]/32 with q = p>>>3, r = p&7.
    logic signed [7:0]  q_s;
    logic        [2:0]  r_s;
    logic        [14:0] prod;
    logic        [4:0]  sh;
    logic signed [31:0] l_contrib;

    always_comb begin
        q_s  = p_blk >>> 3;                     // [-16, 15]
        r_s  = p_blk[2:0];                      // p mod 8, always non-negative
        prod = csum_d * CSD_Q5[r_s];            // <= 192*59
        sh   = q_s[4:0] + 5'(SCORE_FRAC);       // q + SCORE_FRAC (0..13 in range)
        if (sh > 5'd15) sh = 5'd15;             // defensive: unreachable while p<=-21
        if (sh >= 5'd6) l_contrib = 32'(prod) << (sh - 5'd6);
        else            l_contrib = 32'(prod) >> (5'd6 - sh);
    end

    // =========================================================================
    // 7. Output registers
    // =========================================================================
    // latency 1: must be available before the next block starts
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_valid       <= 1'b0;
            m_out         <= S_MIN;
            row_start_out <= 1'b0;
            resc_valid    <= 1'b0;
            resc_q        <= 8'sd0;
            resc_r        <= 3'd0;
        end else begin
            m_valid       <= in_valid & any_valid;
            m_out         <= m_next;
            row_start_out <= in_valid & row_start;
            resc_valid    <= in_valid & any_valid & resc_req;
            resc_q        <= resc_q_d;
            resc_r        <= resc_r_d;
        end
    end

    // latency 2: the P operand
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid     <= 1'b0;
            code_out      <= '0;
            p_blk_out     <= 8'sd0;
            csum_out      <= 8'd0;
            l_contrib_out <= 32'sd0;
            block_empty   <= 1'b0;
        end else begin
            out_valid   <= s1_valid;
            block_empty <= s1_valid & ~s1_any;
            if (s1_valid) begin
                for (sm_t_out = 0; sm_t_out < BLOCK_SIZE; sm_t_out++)
                    code_out[4*sm_t_out +: 4] <= code_d[sm_t_out];
                p_blk_out     <= p_blk;
                csum_out      <= csum_d;
                l_contrib_out <= l_contrib;
            end
        end
    end

endmodule
