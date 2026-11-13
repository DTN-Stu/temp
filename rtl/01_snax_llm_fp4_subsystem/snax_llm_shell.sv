// =============================================================================
// Module: snax_llm_shell.sv
// Description: SNAX Accelerator Shell Wrapper cho Fused FP4 LLM Engine
//              - Tích hợp CSR Manager điều khiển chế độ Prefill / Decode / GEMM
//              - Tích hợp bộ chốt giữ Activation Freeze cho Phase Decode (Streaming Weight)
//              - Phối hợp bắt tay Ready/Valid với 16 kênh Weight Streamer
// =============================================================================
`timescale 1ns/1ps
`include "pe_defines.svh"

module snax_llm_shell #(
    parameter int NUM_LANES_P = NUM_LANES,        // 16
    parameter int ACC_DEPTH_P = ACCUMULATOR_DEPTH, // 64
    parameter int PKT_W       = PACKET_WIDTH       // 72 bits
)(
    input  logic                                  clk,
    input  logic                                  rst_n,

    // -------------------------------------------------------------------------
    // 1. Giao tiếp SNAX CSR Manager (RW & RO Registers)
    // -------------------------------------------------------------------------
    input  logic                                  csr_we_i,
    input  logic [7:0]                            csr_addr_i,
    input  logic [31:0]                           csr_wdata_i,
    output logic [31:0]                           csr_rdata_o,

    // -------------------------------------------------------------------------
    // 2. Activation Stream (72-bit Packet từ Activation Streamer)
    // -------------------------------------------------------------------------
    input  logic [PKT_W-1:0]                      act_stream_data_i,
    input  logic                                  act_stream_valid_i,
    output logic                                  act_stream_ready_o,

    // -------------------------------------------------------------------------
    // 3. Weight Streams (16 Lanes song song: 16 x 72-bit = 1152 bits)
    // -------------------------------------------------------------------------
    input  logic [NUM_LANES_P-1:0][PKT_W-1:0]      wt_stream_data_i,
    input  logic [NUM_LANES_P-1:0]                wt_stream_valid_i,
    output logic [NUM_LANES_P-1:0]                wt_stream_ready_o,

    // -------------------------------------------------------------------------
    // 4. Output Streams (Kết quả tính toán đưa ra Output Streamer)
    // -------------------------------------------------------------------------
    // Kênh FP16 (256-bit SIMD)
    output logic [NUM_LANES_P*16-1:0]             out_fp16_data_o,
    output logic                                  out_fp16_valid_o,
    input  logic                                  out_fp16_ready_i,

    // Kênh FP4 Requantized (72-bit Packet)
    output logic [PKT_W-1:0]                      out_fp4_data_o,
    output logic                                  out_fp4_valid_o,
    input  logic                                  out_fp4_ready_i,

    // Trạng thái ra hệ thống
    output logic                                  engine_busy_o,
    output logic                                  engine_done_irq_o
);

    // =========================================================================
    // PHẦN 1: THANH GHI CẤU HÌNH CSR (snax_llm_csr)
    // =========================================================================
    logic [1:0]             csr_cfg_mode_q;        // 00: GEMM, 01: Attn QK, 10: Attn PV
    logic                   csr_cfg_row_start_q;
    logic [NUM_LANES_P-1:0] csr_cfg_key_mask_q;
    logic                   csr_cfg_enable_slq_q;
    logic                   csr_start_pulse;
    logic [31:0]            csr_tile_k_q;
    logic [31:0]            csr_tile_n_q;

    logic                   is_busy_q, is_done_q;
    logic [31:0]            perf_cycles_q;

    // Core status
    logic                   core_p_fifo_full;
    logic                   core_p_fifo_almost_full;
    logic                   core_p_fifo_empty;
    logic                   core_l_ready;

    assign engine_busy_o     = is_busy_q;
    assign engine_done_irq_o = is_done_q;

    // Ghi CSR từ CPU
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            csr_cfg_mode_q       <= '0;
            csr_cfg_row_start_q  <= 1'b0;
            csr_cfg_key_mask_q   <= '1; // Mặc định mở toàn bộ mask
            csr_cfg_enable_slq_q <= 1'b0;
            csr_start_pulse      <= 1'b0;
            csr_tile_k_q         <= '0;
            csr_tile_n_q         <= '0;
        end else begin
            csr_start_pulse <= 1'b0;
            // row_start is a one-cycle level; the core converts it to a pulse.
            csr_cfg_row_start_q <= 1'b0;
            if (csr_we_i) begin
                case (csr_addr_i)
                    8'h00: csr_start_pulse      <= csr_wdata_i[0];
                    8'h04: csr_cfg_mode_q       <= csr_wdata_i[1:0];
                    8'h08: csr_cfg_row_start_q  <= csr_wdata_i[0];
                    8'h0C: csr_cfg_key_mask_q   <= csr_wdata_i[NUM_LANES_P-1:0];
                    8'h10: csr_cfg_enable_slq_q <= csr_wdata_i[0];
                    8'h14: csr_tile_k_q         <= csr_wdata_i;
                    8'h18: csr_tile_n_q         <= csr_wdata_i;
                    default: ;
                endcase
            end
        end
    end

    // Đọc CSR
    always_comb begin
        case (csr_addr_i)
            8'h00: csr_rdata_o = {30'd0, is_done_q, is_busy_q};
            8'h04: csr_rdata_o = {30'd0, csr_cfg_mode_q};
            8'h14: csr_rdata_o = csr_tile_k_q;
            8'h1C: csr_rdata_o = perf_cycles_q;
            8'h20: csr_rdata_o = {28'd0, core_p_fifo_full,
                                  core_p_fifo_almost_full,
                                  core_p_fifo_empty, core_l_ready};
            default: csr_rdata_o = '0;
        endcase
    end

    // =========================================================================
    // PHẦN 2: BỘ CHỐT ACTIVATION CHO PHASE DECODE (FREEZE & BROADCAST)
    // =========================================================================
    // Trong Phase Decode (sinh từng token): activation Q chỉ cần nạp 1 lần,
    // sau đó chốt lại để broadcast liên tục cho 16 lanes nhân với dòng Weights
    logic [PKT_W-1:0] act_latched_q;
    logic             act_is_latched_q;
    logic [PKT_W-1:0] effective_act_data;
    logic             effective_act_valid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            act_latched_q    <= '0;
            act_is_latched_q <= 1'b0;
        end else begin
            if (csr_start_pulse) begin
                act_is_latched_q <= 1'b0;
            end else if (act_stream_valid_i && act_stream_ready_o) begin
                act_latched_q    <= act_stream_data_i;
                act_is_latched_q <= 1'b1;
            end
        end
    end

    // Nếu ở chế độ Decode: dùng giá trị chốt nếu đã nạp
    always_comb begin
        if (csr_cfg_mode_q == 2'b10) begin
            effective_act_data  = act_latched_q;
            effective_act_valid = act_is_latched_q;
        end else begin
            effective_act_data  = act_stream_data_i;
            effective_act_valid = act_stream_valid_i;
        end
    end

    // act_stream_ready_o is defined after compute_act_valid (see Part 3).

    // =========================================================================
    // PHẦN 3: FSM ĐIỀU KHIỂN & BỘ ĐẾM HIỆU NĂNG
    // =========================================================================
    logic [$clog2(ACC_DEPTH_P)-1:0] current_acc_slot_q;
    logic [31:0]                    k_step_cnt_q;
    logic [31:0]                    n_tile_cnt_q;
    logic                           tile_drain_req;
    logic [$clog2(ACC_DEPTH_P)-1:0] tile_drain_idx;
    logic                           tile_clear_on_drain;

    wire all_wt_valid = &wt_stream_valid_i;

    // In Mode 10 (P*V) the activation operand comes from the P FIFO, not from
    // the external Q stream.  In Mode 01, stop QK at the almost-full point so
    // the two-cycle softmax pipeline cannot overrun the FIFO; the host can
    // switch to P*V to drain it.
    wire drain_wait = (csr_cfg_mode_q == 2'b01) && core_p_fifo_almost_full;
    wire compute_act_valid = (csr_cfg_mode_q == 2'b10)
        ? !core_p_fifo_empty
        : (effective_act_valid && !drain_wait);

    wire compute_fire = is_busy_q && all_wt_valid && compute_act_valid;

    // Toàn bộ 16 kênh weight sẵn sàng nhận dữ liệu tiếp theo
    assign wt_stream_ready_o = {NUM_LANES_P{is_busy_q && compute_act_valid}};

    // Stop the activation stream with the compute.  In Mode 10 this is a
    // one-shot Q latch; in GEMM/QK it follows compute_act_valid.
    assign act_stream_ready_o = is_busy_q &&
        ((csr_cfg_mode_q == 2'b10) ? !act_is_latched_q : compute_act_valid);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            is_busy_q           <= 1'b0;
            is_done_q           <= 1'b0;
            perf_cycles_q       <= '0;
            current_acc_slot_q  <= '0;
            k_step_cnt_q        <= '0;
            n_tile_cnt_q        <= '0;
            tile_drain_req      <= 1'b0;
            tile_drain_idx      <= '0;
            tile_clear_on_drain <= 1'b0;
        end else begin
            tile_drain_req <= 1'b0;

            if (csr_start_pulse && !is_busy_q) begin
                is_busy_q          <= 1'b1;
                is_done_q          <= 1'b0;
                perf_cycles_q      <= '0;
                current_acc_slot_q <= '0;
                k_step_cnt_q       <= '0;
                n_tile_cnt_q       <= '0;
            end else if (is_busy_q) begin
                perf_cycles_q <= perf_cycles_q + 1'b1;

                if (compute_fire) begin
                    // Tăng bước tích lũy theo chiều K
                    if (k_step_cnt_q < csr_tile_k_q - 1'b1) begin
                        k_step_cnt_q <= k_step_cnt_q + 1'b1;
                    end else if (!core_p_fifo_almost_full) begin
                        k_step_cnt_q <= '0;
                        // Đã tích lũy xong 1 tile theo chiều K -> Yêu cầu xả (Drain)
                        tile_drain_req      <= 1'b1;
                        tile_drain_idx      <= current_acc_slot_q;
                        tile_clear_on_drain <= 1'b1;

                        // Chuyển sang slot tích lũy tiếp theo trong Latch Array
                        if (n_tile_cnt_q < csr_tile_n_q - 1'b1) begin
                            n_tile_cnt_q       <= n_tile_cnt_q + 1'b1;
                            current_acc_slot_q <= current_acc_slot_q + 1'b1;
                        end else begin
                            // Hoàn thành toàn bộ ma trận!
                            is_busy_q <= 1'b0;
                            is_done_q <= 1'b1;
                        end
                    end
                end
            end
        end
    end

    // =========================================================================
    // PHẦN 4: KHỞI TẠO CORE TÍNH TOÁN (snax_fused_llm_core)
    // =========================================================================
    snax_fused_llm_core #(
        .NUM_LANES_P (NUM_LANES_P),
        .ACC_DEPTH_P (ACC_DEPTH_P),
        .PKT_W       (PKT_W)
    ) u_core (
        .clk                   (clk),
        .rst_n                 (rst_n),
        .cfg_mode_i            (csr_cfg_mode_q),
        .cfg_row_start_i       (csr_cfg_row_start_q),
        .cfg_key_mask_i        (csr_cfg_key_mask_q),
        .cfg_enable_slq_i      (csr_cfg_enable_slq_q),
        .tile_drain_req_i      (tile_drain_req),
        .tile_drain_idx_i      (tile_drain_idx),
        .tile_clear_on_drain_i (tile_clear_on_drain),
        .lane_valid_i          ({NUM_LANES_P{compute_fire}}),
        .lane_acc_idx_i        (current_acc_slot_q),
        .ext_act_broadcast_i   (effective_act_data),
        .ext_act_valid_i       (effective_act_valid),
        .wt_in_i               (wt_stream_data_i),
        .wt_valid_i            (all_wt_valid),
        .out_vec_fp16_packed_o (out_fp16_data_o),
        .out_vec_fp16_lanes_o  (),
        .out_vec_fp16_valid_o  (out_fp16_valid_o),
        .out_vec_fp16_ready_i  (1'b1),
        .out_vec_fp4_packet_o  (out_fp4_data_o),
        .out_vec_fp4_valid_o   (out_fp4_valid_o),
        .p_fifo_full_o         (core_p_fifo_full),
        .p_fifo_empty_o        (core_p_fifo_empty),
        .p_fifo_almost_full_o  (core_p_fifo_almost_full),
        .l_ready_o             (core_l_ready)
    );

endmodule
