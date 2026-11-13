// =============================================================================
// Module: snax_fused_llm_core.sv
// Description: Core tính toán Fused LLM Engine tích hợp:
//              - vsq_matmul_engine: 16 Lanes x 16-element FP4 DotProduct
//              - softmax_requantizer: Exp-free FP4 Requantization
//              - p_fifo: Transient Ring Buffer cho ma trận P
//              - l_accumulator: Online Sum và sinh 1/l
//              - final_normalizer: Chuẩn hóa O với 1/l và xuất IEEE 754 FP16
//              - slq_encoder: Log Quantizer (FP16 -> 72-bit FP4)
// =============================================================================
`timescale 1ns/1ps
`include "pe_defines.svh"

module snax_fused_llm_core #(
    parameter int NUM_LANES_P   = NUM_LANES,         // 16
    parameter int ACC_DEPTH_P   = ACCUMULATOR_DEPTH, // 64 / 128
    parameter int PKT_W         = PACKET_WIDTH,      // 72 bits
    parameter int FRAC_BITS     = 16,
    parameter int P_FIFO_DEPTH  = 64,                // P blocks; 64*16 = 1024 keys
    parameter int NUM_QUERIES   = 1,
    // 0: legacy TBs that never backpressure the FP16 output
    // 1: top-level integration with shared log-quantizer (honours ready)
    parameter bit USE_OUTPUT_READY = 1'b0
)(
    input  logic                            clk,
    input  logic                            rst_n,

    // -------------------------------------------------------------------------
    // Control & Configuration Interface (từ Shell/CSR Manager)
    // -------------------------------------------------------------------------
    input  logic [1:0]                      cfg_mode_i,       // 00: GEMM (W*A), 01: Attn Q*K, 10: Attn P*V
    input  logic                            cfg_row_start_i,   // Bắt đầu row mới trong Softmax
    input  logic [NUM_LANES_P-1:0]          cfg_key_mask_i,    // Causal mask 16-bit
    input  logic                            cfg_enable_slq_i,  // 1: Bật re-quantizer FP16->FP4 cho layer sau

    // Drain & Accumulator Control
    input  logic                            tile_drain_req_i,
    input  logic [$clog2(ACC_DEPTH_P)-1:0]  tile_drain_idx_i,
    input  logic                            tile_clear_on_drain_i,

    // Accumulation Control
    input  logic [NUM_LANES_P-1:0]          lane_valid_i,
    input  logic [$clog2(ACC_DEPTH_P)-1:0]  lane_acc_idx_i,

    // -------------------------------------------------------------------------
    // Input Data Streams
    // -------------------------------------------------------------------------
    // Stream Activation: từ Activation Streamer ngoài HOẶC loopback từ p_fifo
    input  logic [PKT_W-1:0]                ext_act_broadcast_i,
    input  logic                            ext_act_valid_i,

    // Stream Weights / Keys / Values: 16 Lanes song song (16 x 72-bit = 1152 bits)
    input  logic [NUM_LANES_P-1:0][PKT_W-1:0] wt_in_i,
    input  logic                            wt_valid_i,

    // -------------------------------------------------------------------------
    // Output Interfaces
    // -------------------------------------------------------------------------
    // Ngõ ra FP16 chuẩn hóa (16 x 16-bit = 256 bits)
    output logic [NUM_LANES_P*16-1:0]       out_vec_fp16_packed_o,
    output logic [NUM_LANES_P-1:0][15:0]    out_vec_fp16_lanes_o,
    output logic                            out_vec_fp16_valid_o,
    input  logic                            out_vec_fp16_ready_i,
    // Flow control for the accumulator drain: high while the output FIFO can
    // still accept the vector that a new drain would produce.
    output logic                            out_fifo_ready_o,

    // Deprecated: FP4 requantization moved to snax_logq_shared at top level.
    // Kept as tied-off ports so the legacy core TBs still elaborate.
    output logic [PKT_W-1:0]                out_vec_fp4_packet_o,
    output logic                            out_vec_fp4_valid_o,

    // Trạng thái nội bộ
    output logic                            p_fifo_full_o,
    output logic                            p_fifo_empty_o,
    output logic                            p_fifo_almost_full_o,
    output logic                            l_ready_o
);

    localparam int SLOT_AW  = $clog2(ACC_DEPTH_P);
    localparam int QW       = (NUM_QUERIES > 1) ? $clog2(NUM_QUERIES) : 1;
    localparam int P_SB_W   = 12 + QW;         // query + resc_valid + q + r
    localparam int P_FIFO_W = PKT_W + P_SB_W;

    // ln(2) in Q16.16.  Multiplying the QK score by this before the base-2
    // softmax makes 2^(s' - m') equal to exp(s - m), i.e. converts the
    // exp-free base-2 softmax to the standard base-e softmax.
    localparam logic signed [31:0] LN2_Q16 = 32'sd45426;

    function automatic logic signed [31:0] sat32(input logic signed [63:0] v);
        if (v > 64'sd2147483647)
            return 32'sh7FFF_FFFF;
        else if (v < -64'sd2147483648)
            return 32'sh8000_0000;
        else
            return v[31:0];
    endfunction

    // =========================================================================
    // 1. ACTIVATION MUX, P-FIFO SIDEBAND, ROW-START / RESCALE ALIGNMENT
    // =========================================================================
    // Mode 10 (P*V) reads the P packet from the FIFO.  The FIFO also carries
    // the per-block rescale factor, so O is rescaled at consume time (not at
    // softmax production time).
    logic [PKT_W-1:0]     act_broadcast_selected;
    logic [P_FIFO_W-1:0]  p_fifo_rdata;
    logic [PKT_W-1:0]     p_fifo_packet;
    logic                 p_fifo_resc_valid;
    logic signed [7:0]    p_fifo_resc_q;
    logic        [2:0]    p_fifo_resc_r;
    logic [QW-1:0]        p_fifo_query;
    logic                 p_fifo_empty, p_fifo_full;
    logic                 p_fifo_pop;
    logic [$clog2(P_FIFO_DEPTH):0] p_fifo_count;
    logic                 p_fifo_almost_full;

    assign p_fifo_full_o  = p_fifo_full;
    assign p_fifo_empty_o = p_fifo_empty;

    // Softmax has a two-cycle pipeline.  Reserve two slots so a score that
    // was accepted while the FIFO had room cannot have its P block dropped
    // two cycles later.
    assign p_fifo_almost_full = (p_fifo_count >= (P_FIFO_DEPTH - 2));
    assign p_fifo_almost_full_o = p_fifo_almost_full;

    // Sideband row_start: qualified by tile_drain_req_i and aligned with raw_drained_valid (latency 1).
    // This removes the need for edge-detection toggle bubbles, enabling zero-bubble 1-cycle-per-slot draining.
    logic drain_is_row_start_q;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) drain_is_row_start_q <= 1'b0;
        else if (tile_drain_req_i) drain_is_row_start_q <= cfg_row_start_i && (cfg_mode_i == 2'b01);
        else drain_is_row_start_q <= 1'b0;
    end

    logic row_start_d1, row_start_d2, row_start_d3;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            row_start_d1 <= 1'b0;
            row_start_d2 <= 1'b0;
            row_start_d3 <= 1'b0;
        end else begin
            row_start_d1 <= drain_is_row_start_q;
            row_start_d2 <= row_start_d1;
            row_start_d3 <= row_start_d2;
        end
    end

    // The drain slot carries the query index (S slots 8..15 or O slots 0..7).
    // Register it at the drain request; the raw score valid arrives next cycle
    // and feeds softmax/l with the same query.
    logic [QW-1:0] drain_query_idx;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) drain_query_idx <= '0;
        else if (tile_drain_req_i) drain_query_idx <= tile_drain_idx_i[QW-1:0];
    end

    logic [QW-1:0] sm_query_idx;
    logic [QW-1:0] sm_query_idx_d1, sm_query_idx_d2, sm_query_idx_d3;
    assign sm_query_idx = drain_query_idx;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sm_query_idx_d1 <= '0;
            sm_query_idx_d2 <= '0;
            sm_query_idx_d3 <= '0;
        end else begin
            sm_query_idx_d1 <= sm_query_idx;
            sm_query_idx_d2 <= sm_query_idx_d1;
            sm_query_idx_d3 <= sm_query_idx_d2;
        end
    end

    // A P block is consumed only when all 16 lanes are valid and the FIFO
    // is non-empty.  In Mode 10 the activation comes from the FIFO, so the
    // external activation valid is not a consumption condition.
    wire compute_fire = &lane_valid_i;

    logic [NUM_LANES_P-1:0] matmul_lane_valid;
    
    assign matmul_lane_valid = (cfg_mode_i == 2'b10)
        ? (lane_valid_i & {NUM_LANES_P{!p_fifo_empty}})
        : lane_valid_i;

    assign p_fifo_packet     = p_fifo_rdata[PKT_W-1:0];
    assign p_fifo_resc_r     = p_fifo_rdata[PKT_W +: 3];
    assign p_fifo_resc_q     = signed'(p_fifo_rdata[PKT_W+3 +: 8]);
    assign p_fifo_resc_valid = p_fifo_rdata[PKT_W+11];
    assign p_fifo_query      = p_fifo_rdata[PKT_W+12 +: QW];

    always_comb begin
        if (cfg_mode_i == 2'b10) begin
            act_broadcast_selected = p_fifo_empty ? {PKT_W{1'b0}} : p_fifo_packet;
        end else begin
            act_broadcast_selected = ext_act_broadcast_i;
        end
    end

    // =========================================================================
    // 2. KHỐI MATMUL ENGINE (vsq_matmul_engine)
    // =========================================================================
    logic                            engine_tile_drain;
    logic [SLOT_AW-1:0]              engine_drain_idx;
    logic                            engine_clear_on_drain;

    logic                            acc_rescale_req;
    logic signed [7:0]               resc_q;
    logic        [2:0]               resc_r;

    logic [NUM_LANES_P*32-1:0]       raw_drained_word;
    logic                            raw_drained_valid;
    logic [NUM_LANES_P*16-1:0]       gemm_vec_packed;
    logic [NUM_LANES_P-1:0][15:0]    gemm_vec_out;
    logic                            gemm_vec_valid;

    assign engine_tile_drain     = tile_drain_req_i;
    assign engine_drain_idx      = tile_drain_idx_i;
    assign engine_clear_on_drain = tile_clear_on_drain_i;

    vsq_matmul_engine #(
        .NUM_LANES_P (NUM_LANES_P),
        .ACC_DEPTH_P (ACC_DEPTH_P),
        .PKT_W       (PKT_W),
        .FRAC_BITS   (FRAC_BITS)
    ) u_matmul_engine (
        .clk                 (clk),
        .reset_n             (rst_n),
        .lane_valid_in       (matmul_lane_valid),
        .lane_acc_idx        (lane_acc_idx_i),
        .act_broadcast       (act_broadcast_selected),
        .wt_in               (wt_in_i),
        .tile_drain_req      (engine_tile_drain),
        .tile_drain_acc_idx  (engine_drain_idx),
        .tile_clear_on_drain (engine_clear_on_drain),
        .acc_rescale_req     (acc_rescale_req),
        .resc_q              (resc_q),
        .resc_r              (resc_r),
        .raw_drained_word    (raw_drained_word),
        .raw_drained_valid   (raw_drained_valid),
        .core_vec_out        (gemm_vec_out),
        .core_vec_packed     (gemm_vec_packed),
        .core_vec_valid      (gemm_vec_valid)
    );

    // Bóc tách raw 32-bit drained scores thành mảng
    logic signed [NUM_LANES_P-1:0][31:0] score_array;
    genvar l;
    generate
        for (l = 0; l < NUM_LANES_P; l++) begin : GEN_SCORE_ARRAY
            assign score_array[l] = signed'(raw_drained_word[l*32 +: 32]);
        end
    endgenerate

    // Convert the exp-free base-2 score to the standard base-e score:
    //   s' = s * ln(2)
    // so softmax_requantizer's 2^(s'-m') becomes exp(s-m).
    int core_i;
    logic signed [63:0] score_scaled_prod;

    logic signed [NUM_LANES_P-1:0][31:0] score_for_softmax;
    always_comb begin
        for (core_i = 0; core_i < NUM_LANES_P; core_i++) begin
            score_scaled_prod = $signed({{32{score_array[core_i][31]}}, score_array[core_i]}) *
                                $signed({{32{LN2_Q16[31]}}, LN2_Q16});
            score_for_softmax[core_i] = sat32(score_scaled_prod >>> 16);
        end
    end

    // =========================================================================
    // 3. KHỐI SOFTMAX REQUANTIZER (Exp-free Softmax)
    // =========================================================================
    logic                         sm_out_valid;
    logic [4*NUM_LANES_P-1:0]     sm_code_out;
    logic signed [7:0]            sm_p_blk_out;
    logic signed [31:0]           sm_l_contrib_out;
    logic                         sm_resc_valid;
    logic signed [7:0]            sm_resc_q;
    logic        [2:0]            sm_resc_r;
    logic                         sm_m_valid;
    logic signed [31:0]           sm_m_out;

    // Softmax accepts a drained score only when the P FIFO can take the
    // resulting block.  The shell must also gate tile_drain_req_i with
    // p_fifo_almost_full_o so the raw score is not lost.
    //
    // The drain region matters as well: while the sequencer drains the previous
    // group's O accumulator during the next group's QK, cfg_mode is still 2'b01
    // but those words are output vectors, not scores.  Only S-slot drains
    // (index >= ACC_DEPTH/2) may enter softmax.
    logic drain_is_o_q;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) drain_is_o_q <= 1'b0;
        else if (tile_drain_req_i) drain_is_o_q <= ~tile_drain_idx_i[SLOT_AW-1];
    end

    wire sm_in_valid = (cfg_mode_i == 2'b01) && raw_drained_valid &&
                       !p_fifo_almost_full && !drain_is_o_q;

    // Pipeline registers to isolate LN2 multiplier from softmax_requantizer
    logic signed [NUM_LANES_P-1:0][31:0] score_for_softmax_q;
    logic                                sm_in_valid_q;
    logic [NUM_LANES_P-1:0]              cfg_key_mask_q;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sm_in_valid_q       <= 1'b0;
            cfg_key_mask_q      <= '0;
            score_for_softmax_q <= '0;
        end else begin
            sm_in_valid_q  <= sm_in_valid;
            cfg_key_mask_q <= cfg_key_mask_i;
            if (sm_in_valid) begin
                score_for_softmax_q <= score_for_softmax;
            end
        end
    end

    softmax_requantizer #(
        .BLOCK_SIZE  (NUM_LANES_P),
        .SCORE_FRAC  (FRAC_BITS),
        .NUM_QUERIES (NUM_QUERIES)
    ) u_softmax_requantizer (
        .clk           (clk),
        .rst_n         (rst_n),
        .in_valid      (sm_in_valid_q),
        .score_in      (score_for_softmax_q),
        .key_mask      (cfg_key_mask_q),
        .row_start     (row_start_d1),
        .clear         (1'b0),
        .query_idx     (sm_query_idx_d1),
        .out_valid     (sm_out_valid),
        .code_out      (sm_code_out),
        .p_blk_out     (sm_p_blk_out),
        .csum_out      (),
        .l_contrib_out (sm_l_contrib_out),
        .block_empty   (),
        .m_valid       (sm_m_valid),
        .m_out         (sm_m_out),
        .row_start_out (),
        .resc_valid    (sm_resc_valid),
        .resc_q        (sm_resc_q),
        .resc_r        (sm_resc_r)
    );

    // Delay the rescale sideband by one cycle to align with sm_out_valid
    // (softmax latency 1 -> 2).  This is the factor carried by the P block.
    logic                 sm_resc_valid_d1;
    logic signed [7:0]    sm_resc_q_d1;
    logic        [2:0]    sm_resc_r_d1;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sm_resc_valid_d1 <= 1'b0;
            sm_resc_q_d1     <= 8'sd0;
            sm_resc_r_d1     <= 3'd0;
        end else begin
            sm_resc_valid_d1 <= sm_resc_valid;
            sm_resc_q_d1     <= sm_resc_q;
            sm_resc_r_d1     <= sm_resc_r;
        end
    end

    // O is rescaled when the corresponding P block is consumed, using the
    // sideband stored with that P block.  The matmul registers this sideband
    // through Stage 1 so Stage 2 applies it with the P product.
    assign acc_rescale_req = (cfg_mode_i == 2'b10) &&
                             compute_fire && !p_fifo_empty &&
                             p_fifo_resc_valid;
    assign resc_q          = p_fifo_resc_q;
    assign resc_r          = p_fifo_resc_r;

    // =========================================================================
    // 4. TRANSIENT P_FIFO (Đệm vòng chứa gói xác suất P 72-bit)
    // =========================================================================
    // P packet (72b) + rescale sideband (12b): {valid, q[7:0], r[2:0]}.
    wire [P_FIFO_W-1:0] p_packet_wdata = {
        sm_query_idx_d3,
        sm_resc_valid_d1,
        sm_resc_q_d1,
        sm_resc_r_d1,
        sm_p_blk_out,
        sm_code_out
    };

    // Consume a P block only on a real compute beat, never on an empty FIFO.
    assign p_fifo_pop = (cfg_mode_i == 2'b10) &&
                        compute_fire && !p_fifo_empty;

    p_fifo #(
        .DATA_WIDTH (P_FIFO_W),
        .DEPTH      (P_FIFO_DEPTH)
    ) u_p_fifo (
        .clk     (clk),
        .rst_n   (rst_n),
        .clear   (drain_is_row_start_q && (sm_query_idx == '0)),
        .push    (sm_out_valid),
        .wdata   (p_packet_wdata),
        .full    (p_fifo_full),
        .pop     (p_fifo_pop),
        .rdata   (p_fifo_rdata),
        .empty   (p_fifo_empty),
        .count   (p_fifo_count)
    );

    // =========================================================================
    // 5. BỘ TÍCH LŨY MẪU SỐ VÀ SINH NGHỊCH ĐẢO 1/L (l_accumulator)
    // =========================================================================
    logic [15:0] inv_l_mant;
    logic [4:0]  inv_l_exp;
    logic        l_accum_ready;

    assign l_ready_o = l_accum_ready;

    logic [NUM_QUERIES-1:0][15:0] inv_l_mant_q;
    logic [NUM_QUERIES-1:0][4:0]  inv_l_exp_q;

    l_accumulator #(
        .SCORE_FRAC  (FRAC_BITS),
        .NUM_QUERIES (NUM_QUERIES)
    ) u_l_accumulator (
        .clk            (clk),
        .rst_n          (rst_n),
        .query_idx      (sm_query_idx_d3),
        .row_start      (row_start_d3),
        .clear          (1'b0),
        .l_valid        (sm_out_valid),
        .resc_valid     (sm_resc_valid_d1),
        .resc_q         (sm_resc_q_d1),
        .resc_r         (sm_resc_r_d1),
        .l_contrib_in   (sm_l_contrib_out),
        .l_total_out    (),
        .l_ready        (l_accum_ready),
        .inv_l_mant     (inv_l_mant),
        .inv_l_exp      (inv_l_exp),
        .inv_l_mant_q   (inv_l_mant_q),
        .inv_l_exp_q    (inv_l_exp_q)
    );

    // =========================================================================
    // 6. KHỐI CHUẨN HÓA ĐẦU RA (final_normalizer)
    // =========================================================================
    logic                      norm_drain_req;
    logic                      norm_bypass_inv_l;
    logic                      norm_vec_valid;
    logic [NUM_LANES_P-1:0][15:0] norm_vec_lanes;
    logic [NUM_LANES_P*16-1:0] norm_vec_packed;

    // Ở mode P*V (10), cần nhân với 1/l (bypass = 0). Ở mode GEMM chuẩn (00), bỏ qua 1/l (bypass = 1).
    //
    // The normalizer is qualified by the DRAIN REGION, not by the operating
    // mode.  An O-slot drain (index 0..ACC_DEPTH/2-1) is always normalized:
    // that is what lets the sequencer keep draining the previous group's O
    // accumulator while the next group's QK fills the S slots, which happens in
    // attention mode.  An S-slot drain feeds softmax and must not come here.
    assign norm_drain_req    = raw_drained_valid &&
                               ((cfg_mode_i == 2'b10) || drain_is_o_q);
    assign norm_bypass_inv_l = (cfg_mode_i == 2'b00);

    // The O-drain slot selects which query's reciprocal is used.
    logic [QW-1:0] norm_query;
    logic [15:0]   norm_inv_l_mant;
    logic [4:0]    norm_inv_l_exp;
    assign norm_query      = drain_query_idx;
    assign norm_inv_l_mant = inv_l_mant_q[norm_query];
    assign norm_inv_l_exp  = inv_l_exp_q[norm_query];

    final_normalizer #(
        .NUM_LANES_P (NUM_LANES_P),
        .FRAC_BITS   (FRAC_BITS)
    ) u_final_normalizer (
        .clk          (clk),
        .rst_n        (rst_n),
        .drain_req    (norm_drain_req),
        .bypass_inv_l (norm_bypass_inv_l),
        .accum_in     (score_array),
        .inv_l_mant   (norm_inv_l_mant),
        .inv_l_exp    (norm_inv_l_exp),
        .vec_valid    (norm_vec_valid),
        .vec_out      (norm_vec_lanes),
        .vec_packed   (norm_vec_packed)
    );

    logic [NUM_LANES_P*16-1:0] src_packed;
    logic                      src_valid;

    always_comb begin
        if (cfg_mode_i == 2'b00) begin
            src_packed = gemm_vec_packed;
            src_valid  = gemm_vec_valid;
        end else begin
            src_packed = norm_vec_packed;
            src_valid  = norm_vec_valid;
        end
    end

    // Output hold register: the FP16 vector is held until the downstream
    // consumer accepts it.
    //
    // NOTE: an 8-entry FIFO was tried here to let the sequencer drain the
    // accumulator one slot per cycle.  It produced a wrong vector for the last
    // slot of the last group, so it was reverted; the drain is paced by the
    // consumer again.  out_fifo_ready_o is kept as an always-ready stub.
    logic          out_hold_q;
    logic [NUM_LANES_P*16-1:0] out_hold_packed_q;

    wire out_ready = USE_OUTPUT_READY ? out_vec_fp16_ready_i : 1'b1;

    assign out_vec_fp16_packed_o = out_hold_q ? out_hold_packed_q : src_packed;
    assign out_vec_fp16_valid_o  = out_hold_q | src_valid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_hold_q        <= 1'b0;
            out_hold_packed_q <= '0;
        end else if (out_vec_fp16_valid_o && !out_ready) begin
            out_hold_q        <= 1'b1;
            out_hold_packed_q <= out_vec_fp16_packed_o;
        end else if (out_vec_fp16_valid_o && out_ready) begin
            out_hold_q        <= 1'b0;
        end
    end

    assign out_fifo_ready_o = 1'b1;

    assign out_vec_fp4_packet_o = '0;
    assign out_vec_fp4_valid_o  = 1'b0;

    generate
        for (genvar k = 0; k < NUM_LANES_P; k++) begin : GEN_FP16_LANES
            assign out_vec_fp16_lanes_o[k] = out_vec_fp16_packed_o[k*16 +: 16];
        end
    endgenerate

endmodule
