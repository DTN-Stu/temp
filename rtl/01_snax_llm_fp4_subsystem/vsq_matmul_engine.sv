`timescale 1ns/1ps
`include "pe_defines.svh"

// =============================================================================
// vsq_matmul_engine — 16-Lane LogScale-Lazy MatMul Engine
//   Front-end: FP4 dot-product → log-scale split (q, r) → CSD frac scaler →
//              bidirectional octave shifter → 32-bit saturating RMW add
//   Accumulator: 1W2R latch array (write commits next cycle per two-phase
//              protocol; read is combinational; Port B = rolling slot drain)
//
// REQUIRED FILELIST (compiled TOGETHER):
//     fp4_mult_lut.sv  compressor_4to2.sv  dot_product_16.sv  latch_array.sv
//     vsq_matmul_engine.sv  + testbench
//   Do NOT co-compile with vsq_feba_core_top*.sv or vsq_pe_*.sv: they define
//   helper modules with the same names (logscale_lazy_drain_norm, etc.).
//
// INTERFACE CONVENTION (deliberately different from legacy vsq_feba_core_top):
//   - act_broadcast : ONE shared activation packet (64b FP4 + 8b exp) → all lanes
//   - wt_in[0:15]   : PER-LANE weight packets (16 output channels)
//   (legacy top was the opposite: act_in[16] + wt_broadcast)
//
// PIPELINE (2 stages — timing-relaxed, mirrors vsq_feba_core_top_latch_2r1w):
//   - Stage 1 (cycle T)    : dot → (q, r) → CSD scaler → shifter (feed-forward)
//   - Stage 1→2 registers  : lane_valid_q1 / lane_acc_idx_q1 / aligned_prod_q1
//   - Stage 2 (cycle T+1)  : same-cycle RMW (raddr1 == waddr == idx_q1) + write
//     command; commits per the latch_array two-phase protocol.
//   - Drain : request at Td → drain word registered (Td+1) → FP16 normalize
//     (combinational) → output registered (Td+2); core_vec_valid pulses at Td+2.
//     Rolling-drain overlap = drain request aligned with the accumulate's
//     STAGE-2 cycle (one cycle after its operand input), matching the legacy
//     golden-model semantics.
// =============================================================================

// =============================================================================
// Module: logscale_lazy_drain_norm (Operand-Isolated Lazy FP16 Drain Normalizer)
// Description: Converts a drained 32-bit static fixed-point integer (Q(32-FRAC).FRAC) into IEEE 754 FP16
//              using a 2-level Hierarchical 32-bit LZC and a Single Left Shift.
//              Placed ONLY on the Collector Drain path (outside the RMW loop!).
// =============================================================================
module logscale_lazy_drain_norm #(
    parameter int FRAC_BITS = 16
)(
    input  logic signed [31:0] norm_in,  // Operand-isolated 32-bit fixed-point integer
    output logic        [15:0] fp16_out  // Normalized IEEE 754 FP16 output
);

    logic        fp_sign;
    logic [31:0] abs_y;

    assign fp_sign = norm_in[31];
    assign abs_y   = fp_sign ? unsigned'(-norm_in) : unsigned'(norm_in);

    // Hierarchical 32-bit LZC (4 x 8-bit chunks)
    logic [3:0][2:0] lzc_sub;
    logic [3:0] sub_has_one;

    genvar g;
    generate
        for (g = 0; g < 4; g++) begin : gen_lzc_32
            wire [7:0] chunk = abs_y[g*8 +: 8];
            assign sub_has_one[g] = |chunk;
            always_comb begin
                unique casez (chunk)
                    8'b1???????: lzc_sub[g] = 3'd0;
                    8'b01??????: lzc_sub[g] = 3'd1;
                    8'b001?????: lzc_sub[g] = 3'd2;
                    8'b0001????: lzc_sub[g] = 3'd3;
                    8'b00001???: lzc_sub[g] = 3'd4;
                    8'b000001??: lzc_sub[g] = 3'd5;
                    8'b0000001?: lzc_sub[g] = 3'd6;
                    8'b00000001: lzc_sub[g] = 3'd7;
                    default:     lzc_sub[g] = 3'd7;
                endcase
            end
        end
    endgenerate

    logic [5:0] lzc;
    always_comb begin
        unique casez (sub_has_one)
            4'b1???: lzc = {3'd0, lzc_sub[3]};
            4'b01??: lzc = 6'd8  + {3'd0, lzc_sub[2]};
            4'b001?: lzc = 6'd16 + {3'd0, lzc_sub[1]};
            4'b0001: lzc = 6'd24 + {3'd0, lzc_sub[0]};
            default: lzc = 6'd32;
        endcase
    end

    logic [4:0]        msb_pos;
    logic signed [7:0] raw_exp;

    assign msb_pos = (lzc >= 6'd32) ? 5'd0 : (5'd31 - lzc[4:0]);
    assign raw_exp = $signed({3'b000, msb_pos}) - 8'(signed'(FRAC_BITS)) + 8'sd15;

    // Single-Shift Fixed-Wire Extraction
    logic [31:0] norm_shifted;
    logic [9:0]  mant_bits;
    logic        guard_bit, round_bit, sticky_bit;

    assign norm_shifted = abs_y << lzc[4:0];
    assign mant_bits    = norm_shifted[30:21];
    assign guard_bit    = norm_shifted[20];
    assign round_bit    = norm_shifted[19];
    assign sticky_bit   = |norm_shifted[18:0];

    logic        round_up;
    logic [10:0] rounded_mant;
    logic signed [7:0] final_exp;

    assign round_up     = guard_bit & (round_bit | sticky_bit | mant_bits[0]);
    assign rounded_mant = {1'b0, mant_bits} + {10'd0, round_up};
    assign final_exp    = rounded_mant[10] ? (raw_exp + 8'sd1) : raw_exp;

    always_comb begin
        if (abs_y == 32'd0 || final_exp <= 8'sd0) begin
            fp16_out = 16'h0000;
        end else if (final_exp >= 8'sd31) begin
            fp16_out = {fp_sign, 5'b11110, 10'h3FF};
        end else begin
            fp16_out = {fp_sign, final_exp[4:0], rounded_mant[9:0]};
        end
    end

endmodule


// =============================================================================
// Module: scale_adder — log-scale exponent combine (8-bit signed, eighths)
//   res = scale_a + scale_w;  q = floor(res/8) (octave part, signed slice);
//                             r = res[2:0]      (= (a+w) mod 8 residual)
// =============================================================================
module scale_adder(
    input  logic signed [7:0]  scale_a,
    input  logic signed [7:0]  scale_w,
    output logic signed [5:0]  q,
    output logic        [2:0]  r
);

    wire signed [8:0] res = scale_a + scale_w;
    assign q = res[8:3];
    assign r = res[2:0];

endmodule


// =============================================================================
// Module: logscale_frac_scaler — P × m_r / 32 via 3-term CSD (Q2 × Q5 = Q7)
//   m_r approximates 2^(r/8) as a 3-term signed power-of-2 decomposition:
//       r: 0    1     2     3     4     5     6     7
//       m: 32   35    38    41    46    49    54    59
//   Zero-negator arithmetic: 1's-complement terms + 2 absorbed carry-ins
//   (carry-vector LSB for sub_t1, final CPA carry-in for sub_t2).
// =============================================================================
module logscale_frac_scaler (
    input  logic signed [12:0] prod_in,      // Q2 dot-product (13-bit signed)
    input  logic        [2:0]  r,            // fractional exponent residual (0..7)
    output logic signed [18:0] scaled_prod   // Q7 = prod_in × m_r
);

    // Sign-extend 13-bit Q2 input to 19 bits
    logic signed [18:0] p_ext;
    assign p_ext = {{6{prod_in[12]}}, prod_in};

    // Term 0: 2-to-1 MUX (r >= 6 selects 64*P, else 32*P)
    logic [18:0] t0;
    assign t0 = (r[2] & r[1]) ? (p_ext <<< 6) : (p_ext <<< 5);

    // Term 1 & Term 2 magnitudes + 1's complement inversion flags
    logic [18:0] t1_mag, t2_mag;
    logic        sub_t1, sub_t2;

    always_comb begin
        unique case (r)
            3'd0: begin t1_mag = 19'd0;         sub_t1 = 1'b0; t2_mag = 19'd0;         sub_t2 = 1'b0; end // 32P
            3'd1: begin t1_mag = (p_ext <<< 1); sub_t1 = 1'b0; t2_mag = p_ext;         sub_t2 = 1'b0; end // 32P + 2P + P = 35P
            3'd2: begin t1_mag = (p_ext <<< 2); sub_t1 = 1'b0; t2_mag = (p_ext <<< 1); sub_t2 = 1'b0; end // 32P + 4P + 2P = 38P
            3'd3: begin t1_mag = (p_ext <<< 3); sub_t1 = 1'b0; t2_mag = p_ext;         sub_t2 = 1'b0; end // 32P + 8P + P = 41P
            3'd4: begin t1_mag = (p_ext <<< 4); sub_t1 = 1'b0; t2_mag = (p_ext <<< 1); sub_t2 = 1'b1; end // 32P + 16P - 2P = 46P
            3'd5: begin t1_mag = (p_ext <<< 4); sub_t1 = 1'b0; t2_mag = p_ext;         sub_t2 = 1'b0; end // 32P + 16P + P = 49P
            3'd6: begin t1_mag = (p_ext <<< 3); sub_t1 = 1'b1; t2_mag = (p_ext <<< 1); sub_t2 = 1'b1; end // 64P - 8P - 2P = 54P
            3'd7: begin t1_mag = (p_ext <<< 2); sub_t1 = 1'b1; t2_mag = p_ext;         sub_t2 = 1'b1; end // 64P - 4P - P = 59P
        endcase
    end

    logic [18:0] t1, t2;
    assign t1 = sub_t1 ? (~t1_mag) : t1_mag;
    assign t2 = sub_t2 ? (~t2_mag) : t2_mag;

    // Single 19-bit 3:2 Carry-Save Adder (1 Full-Adder delay, zero carry ripple)
    logic [18:0] csa_sum, csa_cout;
    assign csa_sum  = t0 ^ t1 ^ t2;
    assign csa_cout = (t0 & t1) | (t1 & t2) | (t0 & t2);

    // Inject sub_t1 into bit 0 of the left-shifted carry vector
    logic [18:0] csa_carry_shifted;
    assign csa_carry_shifted = {csa_cout[17:0], sub_t1};

    // Final 19-bit CPA with sub_t2 as carry-in
    assign scaled_prod = $signed(csa_sum + csa_carry_shifted + 19'(sub_t2));

endmodule


// =============================================================================
// Module: bidirectional_shifter — Q7 (19b) × 2^net_shift, net_shift = q + (FRAC-7)
//   net_shift ≥ 0 : left shift, 34-bit intermediate, 32-bit signed saturation
//   net_shift < 0 : narrow (19-bit) arithmetic right shift + sign-extend
//   Bit-exact with the golden model (gen_vsq_patterns.py lazy_stage1_fe):
//   same zero / right / >+15-saturate / else-wide branch order.
// =============================================================================
module bidirectional_shifter #(
    parameter int FRAC_BITS = 16
)(
    input  logic signed [18:0] scaled_prod,      // Q7 product (from CSD scaler)
    input  logic signed [5:0]  q,                // octave exponent floor((a+w)/8)
    output logic signed [31:0] aligned_prod_32   // Q(32-FRAC_BITS).FRAC_BITS aligned product (saturated)
);

    localparam int BASE_SHIFT = FRAC_BITS - 7;   // Q7 → Q18 alignment (+11 for 18)

    logic signed [6:0] net_shift;
    assign net_shift = $signed({q[5], q}) + 7'(signed'(BASE_SHIFT));

    logic [3:0]         lshift_amt;
    logic [4:0]         rshift_amt;
    logic signed [33:0] lshifted_wide;
    logic signed [18:0] rshifted_narrow;

    // Magnitude-checked saturation helpers for net_shift in [16, 25] — constant-
    // threshold form (reviewer proposal; bit-exact with the previous form):
    //   W = scaled_prod << (net_shift-16)  =>  aligned = W * 2^16 must fit int32:
    //     positive overflow : W >= +2^15  (any bit of W[32:15] set, W positive)
    //     negative overflow : W <  -2^15  (upper bits not all ones, W negative)
    //   (replaces the 2^(31-net_shift) comparator + its 34-bit negation)
    logic signed [33:0] wide16;
    logic               sat_pos_hi;
    logic               sat_neg_hi;

    assign lshift_amt      = net_shift[3:0];
    assign rshift_amt      = (-net_shift > 7'sd19) ? 5'd19 : unsigned'(-net_shift[4:0]);
    assign lshifted_wide   = 34'(scaled_prod) <<< lshift_amt;
    assign rshifted_narrow = scaled_prod >>> rshift_amt;

    assign wide16     = 34'(scaled_prod) <<< net_shift[3:0];   // scaled*2^(net-16), net 16..25
    assign sat_pos_hi = ~wide16[33] & (|wide16[32:15]);
    assign sat_neg_hi =  wide16[33] & ~(&wide16[32:15]);

    always_comb begin
        if (scaled_prod == 19'sd0) begin
            aligned_prod_32 = 32'sd0;
        end else if (net_shift < 7'sd0) begin
            aligned_prod_32 = {{13{rshifted_narrow[18]}}, rshifted_narrow};
        end else if (net_shift > 7'sd15) begin
            // Bit-exact magnitude-checked saturation (matches golden lazy_stage1_fe):
            //   * net >= 26 : |scaled_prod| >= 32 always overflows -> saturate
            //   * net 16..25: |W| >= 2^15 overflows (constant-threshold reduce);
            //     +2^31 saturates, -2^31 is representable and passes through
            if (net_shift >= 7'sd26) begin
                aligned_prod_32 = scaled_prod[18] ? 32'sh8000_0000 : 32'sh7FFF_FFFF;
            end else if (sat_pos_hi) begin
                aligned_prod_32 = 32'sh7FFF_FFFF;
            end else if (sat_neg_hi) begin
                aligned_prod_32 = 32'sh8000_0000;
            end else begin
                aligned_prod_32 = {wide16[15:0], 16'd0};
            end
        end else begin
            if (lshifted_wide > 34'sd2147483647) begin
                aligned_prod_32 = 32'sh7FFF_FFFF;
            end else if (lshifted_wide < -34'sd2147483648) begin
                aligned_prod_32 = 32'sh8000_0000;
            end else begin
                aligned_prod_32 = lshifted_wide[31:0];
            end
        end
    end

endmodule


// =============================================================================
// Module: accum_adder — 32-bit saturating signed adder (sat_add32)
// =============================================================================
module accum_adder (

    input logic signed [31:0]  a, 
    input logic signed [31:0]  b,   
    output logic signed [31:0] sum

); 

    function automatic logic signed [31:0] sat_add32(
        input logic signed [31:0] a,
        input logic signed [31:0] b
    );
        logic signed [32:0] s;
        begin
            s = $signed({a[31], a}) + $signed({b[31], b});
            if (s > 33'sd2147483647)       sat_add32 = 32'sh7FFF_FFFF;
            else if (s < -33'sd2147483648) sat_add32 = 32'sh8000_0000;
            else                           sat_add32 = s[31:0];
        end
    endfunction

    assign sum = sat_add32(a, b); 

endmodule


// =============================================================================
// Module: vsq_matmul_engine
// =============================================================================
module vsq_matmul_engine #(
    parameter int NUM_LANES_P = NUM_LANES,        
    parameter int ACC_DEPTH_P = ACCUMULATOR_DEPTH,   
    parameter int PKT_W       = PACKET_WIDTH,      // 72 bits (64b FP4 + 8b Exp)
    parameter int FRAC_BITS   = 16                
)
(
    input logic                            clk, 
    input logic                            reset_n, 

    input  logic [NUM_LANES_P-1:0]         lane_valid_in,
    input  logic [$clog2(ACC_DEPTH_P)-1:0] lane_acc_idx,             // Current accumulation slot
    input  logic [PKT_W-1:0]               act_broadcast,            // Shared Act vector broadcast
    input  logic [NUM_LANES_P-1:0][PKT_W-1:0] wt_in,  // Per-lane Weight vectors

    // Drain & Tile Control (Supports Simultaneous Compute & Rolling Drain)
    input  logic                           tile_drain_req,
    input  logic [$clog2(ACC_DEPTH_P)-1:0] tile_drain_acc_idx,       // Slot to drain across all lanes
    input  logic                           tile_clear_on_drain,

    // Online Rescale Interface (for FlashAttention O-accumulator)
    input  logic                           acc_rescale_req,
    input  logic signed [7:0]              resc_q,
    input  logic [2:0]                     resc_r,

    // Direct Raw 32-bit Score Output (for Softmax Requantizer Bypass)
    output logic [NUM_LANES_P*32-1:0]      raw_drained_word,
    output logic                           raw_drained_valid,

    // Final FP16 Vector Scaled Output (256-bit SIMD / 16 x FP16)
    output logic [NUM_LANES_P-1:0][15:0]  core_vec_out,
    output logic [NUM_LANES_P*16-1:0]      core_vec_packed,
    output logic                           core_vec_valid
);

    localparam int TOTAL_W = NUM_LANES_P * 32;   // 512 bits per slot
    localparam int SLOT_AW = $clog2(ACC_DEPTH_P);

    // -------------------------------------------------------------------------
    // STAGE 1 (cycle T): per-lane front-end
    //   dot → (q, r) split → CSD frac scaler → bidirectional shifter
    // -------------------------------------------------------------------------
    int mm_l_s1a;
    int mm_l_s1b;
    int mm_l_out;

    logic signed [NUM_LANES_P-1:0][15:0] dot_out_array;
    logic signed [NUM_LANES_P-1:0][5:0]  q_array;
    logic        [NUM_LANES_P-1:0][2:0]  r_array;
    logic signed [NUM_LANES_P-1:0][18:0] scaled_prod;
    logic signed [NUM_LANES_P-1:0][31:0] fe_aligned_prod;

    // -------------------------------------------------------------------------
    // Stage 1A -> Stage 1B pipeline registers (Dot-product & Scale adder output)
    // -------------------------------------------------------------------------
    logic [NUM_LANES_P-1:0]              lane_valid_s1a;
    logic [SLOT_AW-1:0]                  lane_acc_idx_s1a;
    logic                                acc_rescale_req_s1a;
    logic signed [7:0]                   resc_q_s1a;
    logic        [2:0]                   resc_r_s1a;
    logic signed [NUM_LANES_P-1:0][15:0] dot_out_q1;
    logic signed [NUM_LANES_P-1:0][5:0]  q_array_q1;
    logic        [NUM_LANES_P-1:0][2:0]  r_array_q1;

    always_ff @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            lane_valid_s1a      <= '0;
            lane_acc_idx_s1a    <= '0;
            acc_rescale_req_s1a <= 1'b0;
            resc_q_s1a          <= 8'sd0;
            resc_r_s1a          <= 3'd0;
            for (mm_l_s1a = 0; mm_l_s1a < NUM_LANES_P; mm_l_s1a++) begin
                dot_out_q1[mm_l_s1a] <= 16'sd0;
                q_array_q1[mm_l_s1a] <= 6'sd0;
                r_array_q1[mm_l_s1a] <= 3'd0;
            end
        end else begin
            lane_valid_s1a      <= lane_valid_in;
            lane_acc_idx_s1a    <= lane_acc_idx;
            acc_rescale_req_s1a <= acc_rescale_req;
            resc_q_s1a          <= resc_q;
            resc_r_s1a          <= resc_r;
            for (mm_l_s1a = 0; mm_l_s1a < NUM_LANES_P; mm_l_s1a++) begin
                if (lane_valid_in[mm_l_s1a]) begin
                    dot_out_q1[mm_l_s1a] <= dot_out_array[mm_l_s1a];
                    q_array_q1[mm_l_s1a] <= q_array[mm_l_s1a];
                    r_array_q1[mm_l_s1a] <= r_array[mm_l_s1a];
                end
            end
        end
    end

    // -------------------------------------------------------------------------
    // Stage 1B -> Stage 2 pipeline registers (Aligned product & RMW setup)
    // -------------------------------------------------------------------------
    logic [NUM_LANES_P-1:0] lane_valid_q1;
    logic [SLOT_AW-1:0]     lane_acc_idx_q1;
    logic signed [NUM_LANES_P-1:0][31:0] aligned_prod_q1;

    // Online-rescale sideband travels with the P block through Stage 1 so
    // Stage 2 rescales the O accumulator in the same cycle it adds the P
    // product.  The core drives these directly from the P-FIFO head.
    logic                   acc_rescale_req_q1;
    logic signed [7:0]      resc_q_q1;
    logic        [2:0]      resc_r_q1;

    always_ff @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            lane_valid_q1      <= '0;
            lane_acc_idx_q1    <= '0;
            acc_rescale_req_q1 <= 1'b0;
            resc_q_q1          <= 8'sd0;
            resc_r_q1          <= 3'd0;
            for (mm_l_s1b = 0; mm_l_s1b < NUM_LANES_P; mm_l_s1b++) begin
                aligned_prod_q1[mm_l_s1b] <= 32'sd0;
            end
        end else begin
            lane_valid_q1      <= lane_valid_s1a;
            lane_acc_idx_q1    <= lane_acc_idx_s1a;
            acc_rescale_req_q1 <= acc_rescale_req_s1a;
            resc_q_q1          <= resc_q_s1a;
            resc_r_q1          <= resc_r_s1a;
            for (mm_l_s1b = 0; mm_l_s1b < NUM_LANES_P; mm_l_s1b++) begin
                if (lane_valid_s1a[mm_l_s1b]) begin
                    aligned_prod_q1[mm_l_s1b] <= fe_aligned_prod[mm_l_s1b];
                end
            end
        end
    end

    wire any_lane_valid_q1 = |lane_valid_q1;

    // -------------------------------------------------------------------------
    // STAGE 2 (cycle T+1): 32-bit saturating RMW add + latch-array write command
    // -------------------------------------------------------------------------
    logic signed [NUM_LANES_P-1:0][31:0] old_res;
    logic signed [NUM_LANES_P-1:0][31:0] new_res;

    logic [TOTAL_W-1:0] wdata, rdata1, rdata2;

    // Virtual clear mask (`slot_cleared` semantics, identical to legacy top):
    //   SET on reset / drain-with-clear; CLEARED when Stage 2 commits.
    logic [ACC_DEPTH_P-1:0] slot_cleared;

    genvar i;
    generate 
        for (i = 0; i < NUM_LANES_P; i++) begin: GEN_MAC_LANE
            dot_product_16 u_dot_product_16(
                .act_vec        (act_broadcast[63:0]),
                .wt_vec         (wt_in[i][63:0]),
                .dot_out        (dot_out_array[i])
            );

            scale_adder u_scale_adder(
                .scale_a    (signed'(act_broadcast[71:64])),
                .scale_w    (signed'(wt_in[i][71:64])),
                .q          (q_array[i]),
                .r          (r_array[i])
            );  

            logscale_frac_scaler u_logscale_frac_scaler(
                .prod_in            (signed'(dot_out_q1[i][12:0])),
                .r                  (r_array_q1[i]),
                .scaled_prod        (scaled_prod[i])
            ); 

            bidirectional_shifter #(.FRAC_BITS(FRAC_BITS))
            u_bidirectional_shifter (
                .scaled_prod        (scaled_prod[i]),
                .q                  (q_array_q1[i]),
                .aligned_prod_32    (fe_aligned_prod[i])
            );

            // Stage-2 RMW: old accumulator (masked if slot is 'cleared')
            assign old_res[i] = slot_cleared[lane_acc_idx_q1] ? 32'sd0
                                                       : signed'(rdata1[i*32 +: 32]);

            logic signed [31:0] rescaled_old_res;
            logic signed [31:0] effective_old_res;

            vsq_rescaler_32b u_rescaler (
                .val_in     (old_res[i]),
                .resc_q     (resc_q_q1),
                .resc_r     (resc_r_q1),
                .scaled_out (rescaled_old_res)
            );

            assign effective_old_res = acc_rescale_req_q1 ? rescaled_old_res : old_res[i];

            accum_adder u_accum_adder(
                .a         (aligned_prod_q1[i]),  
                .b         (effective_old_res),  
                .sum       (new_res[i]) 
            );

            // Per-lane gating: invalid lanes write back their (masked) effective old value
            assign wdata[i*32 +: 32] = lane_valid_q1[i] ? new_res[i] : effective_old_res;
        end 
    endgenerate

    // -------------------------------------------------------------------------
    // 1-Write 2-Read Latch Array RMW Accumulator
    //   Port A: same-cycle RMW read (raddr1 == waddr)
    //   Port B: rolling slot drain read
    // -------------------------------------------------------------------------
    latch_array #(
        .DEPTH (ACC_DEPTH_P),
        .DW    (TOTAL_W)
    ) u_latch_array (
        .clk                (clk),    
        .reset_n            (reset_n),

        .waddr              (lane_acc_idx_q1),  
        .wdata              (wdata),  
        .wen                (any_lane_valid_q1 | acc_rescale_req_q1),
        .ren1               (any_lane_valid_q1 | acc_rescale_req_q1),  // Port-A read gated to accumulate or rescale cycles
        .ren2               (tile_drain_req),     // Port-B read gated to drain windows
        .raddr1             (lane_acc_idx_q1), 
        .raddr2             (tile_drain_acc_idx), 
        .rdata1             (rdata1), 
        .rdata2             (rdata2)  
    );

    // -------------------------------------------------------------------------
    // Virtual clear mask
    // -------------------------------------------------------------------------
    always_ff @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            slot_cleared <= '1;
        end else begin
            if (any_lane_valid_q1) begin
                slot_cleared[lane_acc_idx_q1] <= 1'b0;
            end
            if (tile_drain_req && tile_clear_on_drain) begin
                slot_cleared[tile_drain_acc_idx] <= 1'b1;   // drain+clear wins
            end
        end
    end

    // -------------------------------------------------------------------------
    // Drain path: Port-B read + same-slot bypass + zero-mask
    // -------------------------------------------------------------------------
    logic [TOTAL_W-1:0] drain_word_comb;

    always_comb begin
        if (any_lane_valid_q1 && (tile_drain_acc_idx == lane_acc_idx_q1)) begin
            drain_word_comb = wdata;        // slot finishing Stage 2 -> bypass
        end else if (slot_cleared[tile_drain_acc_idx]) begin
            drain_word_comb = '0;
        end else begin
            drain_word_comb = rdata2;
        end
    end

    // Operand-isolated drain register (only updates on drain requests)
    logic               drain_req_q1;
    logic [TOTAL_W-1:0] drained_int32_word_q1;

    always_ff @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            drain_req_q1          <= 1'b0;
            drained_int32_word_q1 <= '0;
        end else begin
            drain_req_q1 <= tile_drain_req;
            if (tile_drain_req) begin
                drained_int32_word_q1 <= drain_word_comb;
            end
        end
    end

    // 16× lazy FP16 normalizer (switching activity only during drain cycles)
    logic [NUM_LANES_P-1:0][15:0] drained_fp16;
    logic [NUM_LANES_P*16-1:0] drained_fp16_packed;

    generate
        for (i = 0; i < NUM_LANES_P; i++) begin : GEN_DRAIN_NORM
            logscale_lazy_drain_norm #(
                .FRAC_BITS (FRAC_BITS)
            ) u_drain_norm (
                .norm_in  (signed'(drained_int32_word_q1[i*32 +: 32])),
                .fp16_out (drained_fp16[i])
            );
            assign drained_fp16_packed[i*16 +: 16] = drained_fp16[i];
        end
    endgenerate

    // -------------------------------------------------------------------------
    // Final output registers (drain latency: request Td → outputs at Td+2)
    // -------------------------------------------------------------------------
    assign raw_drained_word  = drained_int32_word_q1;
    assign raw_drained_valid = drain_req_q1;

    always_ff @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            core_vec_valid  <= 1'b0;
            core_vec_packed <= '0;
            for (mm_l_out = 0; mm_l_out < NUM_LANES_P; mm_l_out++) begin
                core_vec_out[mm_l_out] <= 16'h0000;
            end
        end else begin
            core_vec_valid <= drain_req_q1;
            if (drain_req_q1) begin
                core_vec_packed <= drained_fp16_packed;
                for (mm_l_out = 0; mm_l_out < NUM_LANES_P; mm_l_out++) begin
                    core_vec_out[mm_l_out] <= drained_fp16[mm_l_out];
                end
            end
        end
    end

endmodule
