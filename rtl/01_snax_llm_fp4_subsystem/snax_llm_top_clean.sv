// =============================================================================
// Module: snax_llm_top_clean.sv
// Description: Kiến trúc Clean TOP Subsystem theo chuẩn Block Diagram:
//
//   [DRAM/DDR]
//       ▲
//       │ (AXI / Memory Bus)
//   ┌───┴───┐
//   │  DMA  │ ──► Master Port 0 (Nạp / Xả TCDM)
//   └───────┘
//
//   ┌───────┐
//   │ RV32I │ ──► Master Port 1 (RoPE, RMSNorm, SwiGLU tại chỗ trên TCDM)
//   └───────┘     + CSR Bus cấu hình Streamers & Core
//
//   [INTERLEAVED CROSSBAR (21 Masters x 32 Banks)]:
//     - Master 0      : DMA (DRAM <-> TCDM)
//     - Master 1      : RV32I CPU Memory Port
//     - Master 2      : Activation / Q Streamer (Broadcast 1 lane)
//     - Masters 3..18 : Weight / KV Streamer (16 Lanes song song)
//     - Master 19     : Write Streamer (Hỗ trợ cả FP4 72b và FP16 256b)
//     - Master 20     : Read Streamer (Đọc vector FP16 256b từ TCDM - Pipelined)
//
//   [LUỒNG DỮ LIỆU SẠCH 1 CHIỀU]:
//     Core FP16 Out ────────┬──► Write Streamer (Ghi thẳng FP16 ra cho CPU RoPE/Norm)
//                           │
//                           ├──► Mux Port 0 ──┐
//                           │                 │
//     Core FP16 Out ──► Transpose Buffer      │
//                       (Xoay trục 16x16)     ├──► [LOG-QUANTIZER] ──► [Write Streamer]
//                                   │         │       (72b FP4)         (Ghi xuống TCDM)
//                                   └──► Mux 1│
//                                             │
//     Read Streamer (FP16 từ TCDM) ────► Mux 2┘
// =============================================================================
`timescale 1ns/1ps

module snax_llm_top_clean #(
    parameter int unsigned DATA_WIDTH  = 72,
    parameter int unsigned STRB_WIDTH  = 9,
    parameter int unsigned ADDR_WIDTH  = 32,
    parameter int unsigned NUM_BANKS   = 32,
    parameter int unsigned BANK_DEPTH  = 1024,
    parameter int unsigned NUM_LANES   = 16,
    parameter int unsigned ACC_DEPTH   = 16
)(
    input  logic                    clk,
    input  logic                    rst_n,

    // =========================================================================
    // 1. CỔNG CPU RV32I ĐỘC LẬP (CSR Bus & TCDM Master Port)
    // =========================================================================
    // SNAX CSR request / response (RV32I điều phối hệ thống)
    input  logic                    cpu_csr_req_valid_i,
    output logic                    cpu_csr_req_ready_o,
    input  logic                    cpu_csr_req_write_i,
    input  logic [11:0]             cpu_csr_req_addr_i,
    input  logic [31:0]             cpu_csr_req_data_i,
    output logic                    cpu_csr_rsp_valid_o,
    output logic [31:0]             cpu_csr_rsp_data_o,
    input  logic                    cpu_csr_rsp_ready_i,

    // Cổng bộ nhớ TCDM riêng của CPU (Master Port 1: chạy RoPE, RMSNorm, SiLU)
    input  logic                    cpu_mem_req_valid_i,
    output logic                    cpu_mem_req_ready_o,
    input  logic [ADDR_WIDTH-1:0]   cpu_mem_req_addr_i,
    input  logic                    cpu_mem_req_write_i,
    input  logic [DATA_WIDTH-1:0]   cpu_mem_req_wdata_i,
    input  logic [STRB_WIDTH-1:0]   cpu_mem_req_strb_i,
    output logic                    cpu_mem_rsp_valid_o,
    output logic [DATA_WIDTH-1:0]   cpu_mem_rsp_rdata_o,

    // =========================================================================
    // 2. CỔNG DMA ĐỘC LẬP (Master Port 0: Nạp Trọng số & Xả Output với DRAM)
    // =========================================================================
    input  logic                    dma_mem_req_valid_i,
    output logic                    dma_mem_req_ready_o,
    input  logic [ADDR_WIDTH-1:0]   dma_mem_req_addr_i,
    input  logic                    dma_mem_req_write_i,
    input  logic [DATA_WIDTH-1:0]   dma_mem_req_wdata_i,
    input  logic [STRB_WIDTH-1:0]   dma_mem_req_strb_i,
    output logic                    dma_mem_rsp_valid_o,
    output logic [DATA_WIDTH-1:0]   dma_mem_rsp_rdata_o,

    // DMA descriptor handshake (sequencer ring-buffer prefetch, master 0 owner)
    output logic                    dma_desc_valid_o,
    input  logic                    dma_desc_ready_i,
    output logic [1:0]              dma_desc_dir_o,
    output logic [ADDR_WIDTH-1:0]   dma_desc_dram_addr_o,
    output logic [ADDR_WIDTH-1:0]   dma_desc_tcdm_addr_o,
    output logic [31:0]             dma_desc_size_o,
    input  logic                    dma_desc_done_i,

    // Trạng thái hệ thống
    output logic                    subsystem_busy_o,
    output logic                    subsystem_done_irq_o
);

    localparam int NUM_MASTERS = 21;

    // -------------------------------------------------------------------------
    // Tín hiệu nội bộ Interconnect (21 Masters -> 32 Banks)
    // -------------------------------------------------------------------------
    logic [NUM_MASTERS-1:0]          m_req_valid;
    logic [NUM_MASTERS-1:0]          m_req_ready;
    logic [NUM_MASTERS-1:0][31:0]    m_req_addr;
    logic [NUM_MASTERS-1:0]          m_req_write;
    logic [NUM_MASTERS-1:0][71:0]    m_req_wdata;
    logic [NUM_MASTERS-1:0][8:0]     m_req_strb;
    logic [NUM_MASTERS-1:0]          m_rsp_valid;
    logic [NUM_MASTERS-1:0][71:0]    m_rsp_rdata;

    logic [NUM_BANKS-1:0]            b_req;
    logic [NUM_BANKS-1:0]            b_write;
    logic [NUM_BANKS-1:0][9:0]       b_addr;
    logic [NUM_BANKS-1:0][71:0]      b_wdata;
    logic [NUM_BANKS-1:0][8:0]       b_strb;
    logic [NUM_BANKS-1:0][71:0]      b_rdata;
    logic [NUM_BANKS-1:0]            b_rvalid;

    // -------------------------------------------------------------------------
    // Tín hiệu điều khiển Streamers & Core (Từ CSR hoặc Sequencer)
    // -------------------------------------------------------------------------
    logic        csr_start_pulse;
    logic        csr_src_sel;

    // Activation Streamer Signals
    logic        act_start, act_busy;
    logic [31:0] act_base, act_tokens, act_token_stride, act_k_blocks, act_k_stride;
    logic        act_m_req_valid, act_m_req_write, act_m_req_ready;
    logic [31:0] act_m_req_addr;
    logic [71:0] act_m_req_wdata;
    logic [8:0]  act_m_req_strb;
    logic [71:0] act_m_rsp_rdata;
    logic        act_m_rsp_valid;
    logic [71:0] act_stream_data;
    logic        act_stream_valid, act_stream_ready;

    // Weight Streamer Signals (16 Channels)
    logic        wt_start, wt_busy;
    logic [31:0] wt_base, wt_repeat, wt_k_blocks, wt_k_stride, wt_num_tiles, wt_tile_step;
    logic [15:0] wt_m_req_valid, wt_m_req_write, wt_m_req_ready;
    logic [15:0][31:0] wt_m_req_addr;
    logic [15:0][71:0] wt_m_req_wdata;
    logic [15:0][8:0]  wt_m_req_strb;
    logic [15:0][71:0] wt_m_rsp_rdata;
    logic [15:0]       wt_m_rsp_valid;
    logic [15:0][71:0] wt_stream_data;
    logic [15:0]       wt_stream_valid, wt_stream_ready;

    // Output Write Streamer Signals
    logic        wo_start, wo_busy, wo_done;
    logic        wo_mode_fp16; // 1: Ghi FP16 ra cho CPU, 0: Ghi FP4 nén
    logic        wo_m_req_valid, wo_m_req_write, wo_m_req_ready;
    logic [31:0] wo_m_req_addr;
    logic [71:0] wo_m_req_wdata;
    logic [8:0]  wo_m_req_strb;
    logic        wo_fp4_ready, wo_fp16_ready;
    logic [71:0] wo_fp4_data;
    logic        wo_fp4_valid;

    // TCDM FP16 Read Streamer Signals (Pipelined)
    logic        rd_start, rd_busy, rd_done;
    logic [31:0] rd_base, rd_num_vectors, rd_line_stride;
    logic        rd_m_req_valid, rd_m_req_write, rd_m_req_ready;
    logic [31:0] rd_m_req_addr;
    logic [71:0] rd_m_req_wdata;
    logic [8:0]  rd_m_req_strb;
    logic [71:0] rd_m_rsp_rdata;
    logic        rd_m_rsp_valid;
    logic [255:0] rd_fp16_data;
    logic         rd_fp16_valid, rd_fp16_ready;

    // Core Interface Signals
    logic [1:0]  cfg_mode;
    logic        cfg_row_start;
    logic [15:0] cfg_key_mask;
    logic        cfg_enable_slq;
    logic        tile_drain_req;
    logic [3:0]  tile_drain_idx;
    logic        tile_clear_on_drain;
    logic [15:0] lane_valid;
    logic [3:0]  lane_acc_idx;
    logic        p_fifo_full, p_fifo_empty, p_fifo_almost_full, l_ready;

    logic [255:0] core_fp16_packed;
    logic         core_fp16_valid;
    logic         core_fp16_ready;
    logic         core_out_fifo_ready;

    // -------------------------------------------------------------------------
    // CSR / sequencer configuration
    // -------------------------------------------------------------------------
    logic        csr_act_ready, csr_wt_ready;
    logic        csr_act_start;
    logic [31:0] csr_act_base, csr_act_k_blocks, csr_act_k_stride;
    logic [31:0] csr_act_tokens, csr_act_token_stride;
    logic        csr_wt_start;
    logic [31:0] csr_wt_base, csr_wt_repeat, csr_wt_k_blocks, csr_wt_k_stride;
    logic [31:0] csr_wt_num_tiles, csr_wt_tile_step;
    logic        csr_wo_start, csr_wo_mode_fp16;
    logic [31:0] csr_wo_base, csr_wo_num_vectors, csr_wo_line_stride;
    logic [1:0]  csr_core_mode;
    logic        csr_core_row_start;
    logic [15:0] csr_core_key_mask;
    logic        csr_core_enable_slq;
    logic        csr_core_tile_drain_req;
    logic [3:0]  csr_core_tile_drain_idx;
    logic        csr_core_tile_clear;
    logic [15:0] csr_core_lane_valid;
    logic [3:0]  csr_core_lane_acc_idx;
    logic        csr_v_start;
    logic [31:0] csr_v_base, csr_v_num_batches;
    logic        csr_rd_start;
    logic [31:0] csr_rd_base, csr_rd_num_vectors, csr_rd_line_stride;

    // Sequencer descriptor from CSR
    logic [1:0]  csr_seq_op_mode;
    logic        csr_seq_is_causal, csr_seq_enable_dma;
    logic [7:0]  csr_seq_attn_h;
    logic [15:0] csr_seq_attn_q_tiles, csr_seq_attn_dv_tiles;
    logic [15:0] csr_seq_attn_kv_tiles, csr_seq_attn_k_blocks;
    logic [15:0] csr_seq_gemm_n_tiles, csr_seq_gemm_m_tiles, csr_seq_gemm_k_blocks;
    logic [3:0]  csr_seq_attn_m_q;
    logic [31:0] csr_seq_tcdm_ring_base, csr_seq_tcdm_q_base;
    logic [31:0] csr_seq_tcdm_out_base, csr_seq_dram_kv_base, csr_seq_stride_tile;

    logic        seq_act_start, seq_act_ready;
    logic [31:0] seq_act_base, seq_act_k_blocks, seq_act_k_stride;
    logic [31:0] seq_act_tokens, seq_act_token_stride;
    logic        seq_wt_start;
    logic [31:0] seq_wt_base, seq_wt_repeat, seq_wt_k_blocks, seq_wt_k_stride;
    logic [31:0] seq_wt_num_tiles, seq_wt_tile_step;
    logic [15:0] seq_wt_ready;
    logic [1:0]  seq_core_mode;
    logic        seq_core_row_start;
    logic [15:0] seq_core_key_mask;
    logic        seq_core_enable_slq;
    logic        seq_core_tile_drain_req;
    logic [3:0]  seq_core_tile_drain_idx;
    logic        seq_core_tile_clear;
    logic [15:0] seq_core_lane_valid;
    logic [3:0]  seq_core_lane_acc_idx;
    logic        seq_wo_start;
    logic [1:0]  seq_wo_mode;
    logic [31:0] seq_wo_base, seq_wo_num_vectors, seq_wo_line_stride;
    logic [31:0] seq_wo_num_batches, seq_wo_batch_stride;
    logic        seq_v_start, seq_rd_start;
    logic        seq_busy, seq_done;
    logic        csr_cfg_ready;

    // -------------------------------------------------------------------------
    // V transpose controller / log-quantizer arbitration
    // -------------------------------------------------------------------------
    logic        v_active, v_busy, v_done, v_col_issue, v_row_ready;
    logic [31:0] v_base, v_num_batches;
    logic [255:0] quant_fp16_in;
    logic         quant_in_valid, quant_accept;
    logic [1:0]   cfg_quant_src_sel;

    // -------------------------------------------------------------------------
    // Write-out descriptor selected between sequencer and CSR
    // -------------------------------------------------------------------------
    logic [1:0]  wo_cfg_mode;
    logic [31:0] wo_cfg_base, wo_cfg_nvec, wo_cfg_line, wo_cfg_nb, wo_cfg_bs;

    // =========================================================================
    // 3. TẦNG BỘ NHỚ TCDM & BỘ CHUYỂN MẠCH CROSSBAR 21 MASTERS
    // =========================================================================
    // Master 0: DMA Port (DRAM <-> TCDM)
    assign m_req_valid[0]       = dma_mem_req_valid_i;
    assign m_req_addr[0]        = dma_mem_req_addr_i;
    assign m_req_write[0]       = dma_mem_req_write_i;
    assign m_req_wdata[0]       = dma_mem_req_wdata_i;
    assign m_req_strb[0]        = dma_mem_req_strb_i;
    assign dma_mem_req_ready_o  = m_req_ready[0];
    assign dma_mem_rsp_valid_o  = m_rsp_valid[0];
    assign dma_mem_rsp_rdata_o  = m_rsp_rdata[0];

    // Master 1: CPU RV32I Port (TCDM L1 Scratchpad Access cho RoPE/Norm/SiLU)
    assign m_req_valid[1]       = cpu_mem_req_valid_i;
    assign m_req_addr[1]        = cpu_mem_req_addr_i;
    assign m_req_write[1]       = cpu_mem_req_write_i;
    assign m_req_wdata[1]       = cpu_mem_req_wdata_i;
    assign m_req_strb[1]        = cpu_mem_req_strb_i;
    assign cpu_mem_req_ready_o  = m_req_ready[1];
    assign cpu_mem_rsp_valid_o  = m_rsp_valid[1];
    assign cpu_mem_rsp_rdata_o  = m_rsp_rdata[1];

    // Master 2: Activation / Q Streamer
    assign m_req_valid[2]       = act_m_req_valid;
    assign m_req_addr[2]        = act_m_req_addr;
    assign m_req_write[2]       = act_m_req_write;
    assign m_req_wdata[2]       = act_m_req_wdata;
    assign m_req_strb[2]        = act_m_req_strb;
    assign act_m_req_ready      = m_req_ready[2];
    assign act_m_rsp_valid      = m_rsp_valid[2];
    assign act_m_rsp_rdata      = m_rsp_rdata[2];

    // Masters 3..18: 16-Channel Weight / KV Streamer
    genvar gw;
    generate
        for (gw = 0; gw < 16; gw++) begin : GEN_WT_MASTERS
            assign m_req_valid[3+gw]  = wt_m_req_valid[gw];
            assign m_req_addr[3+gw]   = wt_m_req_addr[gw];
            assign m_req_write[3+gw]  = wt_m_req_write[gw];
            assign m_req_wdata[3+gw]  = wt_m_req_wdata[gw];
            assign m_req_strb[3+gw]   = wt_m_req_strb[gw];
            assign wt_m_req_ready[gw] = m_req_ready[3+gw];
            assign wt_m_rsp_valid[gw] = m_rsp_valid[3+gw];
            assign wt_m_rsp_rdata[gw] = m_rsp_rdata[3+gw];
        end
    endgenerate

    // Master 19: Output Write Streamer
    assign m_req_valid[19]      = wo_m_req_valid;
    assign m_req_addr[19]       = wo_m_req_addr;
    assign m_req_write[19]      = wo_m_req_write;
    assign m_req_wdata[19]      = wo_m_req_wdata;
    assign m_req_strb[19]       = wo_m_req_strb;
    assign wo_m_req_ready       = m_req_ready[19];

    // Master 20: TCDM FP16 Read Streamer
    assign m_req_valid[20]      = rd_m_req_valid;
    assign m_req_addr[20]       = rd_m_req_addr;
    assign m_req_write[20]      = rd_m_req_write;
    assign m_req_wdata[20]      = rd_m_req_wdata;
    assign m_req_strb[20]       = rd_m_req_strb;
    assign rd_m_req_ready       = m_req_ready[20];
    assign rd_m_rsp_valid       = m_rsp_valid[20];
    assign rd_m_rsp_rdata       = m_rsp_rdata[20];

    // Crossbar Interconnect
    interleaved_interconnect_72b #(
        .NUM_MASTERS   (NUM_MASTERS),
        .NUM_BANKS     (NUM_BANKS),
        .DATA_WIDTH    (DATA_WIDTH),
        .ADDR_WIDTH    (ADDR_WIDTH),
        .BANK_DEPTH    (BANK_DEPTH)
    ) u_interconnect (
        .clk           (clk),
        .rst_n         (rst_n),
        .m_req_valid_i (m_req_valid),
        .m_req_addr_i  (m_req_addr),
        .m_req_write_i (m_req_write),
        .m_req_wdata_i (m_req_wdata),
        .m_req_strb_i  (m_req_strb),
        .m_req_ready_o (m_req_ready),
        .m_rsp_valid_o (m_rsp_valid),
        .m_rsp_rdata_o (m_rsp_rdata),
        .b_req_o       (b_req),
        .b_write_o     (b_write),
        .b_addr_o      (b_addr),
        .b_wdata_o     (b_wdata),
        .b_strb_o      (b_strb),
        .b_rdata_i     (b_rdata),
        .b_rvalid_i    (b_rvalid)
    );

    // Multi-bank TCDM (32 Banks)
    multi_bank_tcdm_72b #(
        .NUM_BANKS     (NUM_BANKS),
        .DATA_WIDTH    (DATA_WIDTH),
        .STRB_WIDTH    (STRB_WIDTH),
        .BANK_DEPTH    (BANK_DEPTH)
    ) u_tcdm (
        .clk           (clk),
        .rst_n         (rst_n),
        .bank_req_i    (b_req),
        .bank_write_i  (b_write),
        .bank_addr_i   (b_addr),
        .bank_wdata_i  (b_wdata),
        .bank_strb_i   (b_strb),
        .bank_rdata_o  (b_rdata),
        .bank_rvalid_o (b_rvalid)
    );

    // =========================================================================
    // 3b. CONTROL PLANE: CSR FILE + HARDWARE SEQUENCER
    // =========================================================================
    snax_flash_csr_snax u_csr (
        .clk                      (clk),
        .rst_n                    (rst_n),
        .csr_req_valid_i          (cpu_csr_req_valid_i),
        .csr_req_ready_o          (cpu_csr_req_ready_o),
        .csr_req_write_i          (cpu_csr_req_write_i),
        .csr_req_addr_i           (cpu_csr_req_addr_i),
        .csr_req_data_i           (cpu_csr_req_data_i),
        .csr_rsp_valid_o          (cpu_csr_rsp_valid_o),
        .csr_rsp_data_o           (cpu_csr_rsp_data_o),
        .csr_rsp_ready_i          (cpu_csr_rsp_ready_i),
        .cfg_ready_i              (csr_cfg_ready),
        .seq_busy                 (seq_busy),
        .seq_done                 (seq_done),
        .act_busy                 (act_busy),
        .wt_busy                  (wt_busy),
        .wo_busy                  (wo_busy),
        .v_busy                   (v_busy),
        .v_done                   (v_done),
        .rd_busy                  (rd_busy),
        .rd_done                  (rd_done),
        .p_fifo_full              (p_fifo_full),
        .p_fifo_empty             (p_fifo_empty),
        .p_fifo_almost_full       (p_fifo_almost_full),
        .l_ready                  (l_ready),
        .core_fp16_valid          (core_fp16_valid),
        .start_pulse              (csr_start_pulse),
        .src_sel                  (csr_src_sel),
        .act_ready_force          (csr_act_ready),
        .wt_ready_force           (csr_wt_ready),
        .act_start                (csr_act_start),
        .act_base                 (csr_act_base),
        .act_k_blocks             (csr_act_k_blocks),
        .act_k_stride             (csr_act_k_stride),
        .act_tokens               (csr_act_tokens),
        .act_token_stride         (csr_act_token_stride),
        .wt_start                 (csr_wt_start),
        .wt_base                  (csr_wt_base),
        .wt_repeat                (csr_wt_repeat),
        .wt_k_blocks              (csr_wt_k_blocks),
        .wt_k_stride              (csr_wt_k_stride),
        .wt_num_tiles             (csr_wt_num_tiles),
        .wt_tile_step             (csr_wt_tile_step),
        .wo_start                 (csr_wo_start),
        .wo_mode_fp16             (csr_wo_mode_fp16),
        .wo_base                  (csr_wo_base),
        .wo_num_vectors           (csr_wo_num_vectors),
        .wo_line_stride           (csr_wo_line_stride),
        .core_mode                (csr_core_mode),
        .core_row_start           (csr_core_row_start),
        .core_key_mask            (csr_core_key_mask),
        .core_enable_slq          (csr_core_enable_slq),
        .core_tile_drain_req      (csr_core_tile_drain_req),
        .core_tile_drain_idx      (csr_core_tile_drain_idx),
        .core_tile_clear_on_drain (csr_core_tile_clear),
        .core_lane_valid          (csr_core_lane_valid),
        .core_lane_acc_idx        (csr_core_lane_acc_idx),
        .v_start                  (csr_v_start),
        .v_base                   (csr_v_base),
        .v_num_batches            (csr_v_num_batches),
        .rd_start                 (csr_rd_start),
        .rd_base                  (csr_rd_base),
        .rd_num_vectors           (csr_rd_num_vectors),
        .rd_line_stride           (csr_rd_line_stride),
        .quant_src_sel            (cfg_quant_src_sel),

        .seq_op_mode              (csr_seq_op_mode),
        .seq_is_causal            (csr_seq_is_causal),
        .seq_enable_dma           (csr_seq_enable_dma),
        .seq_attn_h               (csr_seq_attn_h),
        .seq_attn_q_tiles         (csr_seq_attn_q_tiles),
        .seq_attn_m_q             (csr_seq_attn_m_q),
        .seq_attn_dv_tiles        (csr_seq_attn_dv_tiles),
        .seq_attn_kv_tiles        (csr_seq_attn_kv_tiles),
        .seq_attn_k_blocks        (csr_seq_attn_k_blocks),
        .seq_gemm_n_tiles         (csr_seq_gemm_n_tiles),
        .seq_gemm_m_tiles         (csr_seq_gemm_m_tiles),
        .seq_gemm_k_blocks        (csr_seq_gemm_k_blocks),
        .seq_tcdm_ring_base       (csr_seq_tcdm_ring_base),
        .seq_tcdm_q_base          (csr_seq_tcdm_q_base),
        .seq_tcdm_out_base        (csr_seq_tcdm_out_base),
        .seq_dram_kv_base         (csr_seq_dram_kv_base),
        .seq_stride_tile          (csr_seq_stride_tile)
    );

    snax_llm_sequencer #(
        .ADDR_WIDTH           (ADDR_WIDTH),
        .NUM_LANES            (NUM_LANES),
        .NUM_RING_SLOTS       (4),
        .SLOT_SIZE_BYTES      (16 * 1024)
    ) u_seq (
        .clk                   (clk),
        .rst_n                 (rst_n),
        .start_i               (csr_start_pulse && !csr_src_sel && !v_busy),
        .busy_o                (seq_busy),
        .done_o                (seq_done),

        // Runtime Configuration (CPU-programmable via CSR)
        .cfg_op_mode_i         (csr_seq_op_mode),
        .cfg_is_causal_i       (csr_seq_is_causal),
        .cfg_enable_dma_i      (csr_seq_enable_dma),
        .cfg_attn_h_i          (csr_seq_attn_h),
        .cfg_attn_q_tiles_i    (csr_seq_attn_q_tiles),
        .cfg_attn_m_q_i        (csr_seq_attn_m_q),
        .cfg_attn_dv_tiles_i   (csr_seq_attn_dv_tiles),
        .cfg_attn_kv_tiles_i   (csr_seq_attn_kv_tiles),
        .cfg_attn_k_blocks_i   (csr_seq_attn_k_blocks),

        .cfg_gemm_n_tiles_i    (csr_seq_gemm_n_tiles),
        .cfg_gemm_m_tiles_i    (csr_seq_gemm_m_tiles),
        .cfg_gemm_k_blocks_i   (csr_seq_gemm_k_blocks),

        .cfg_tcdm_ring_base_i  (csr_seq_tcdm_ring_base),
        .cfg_tcdm_q_base_i     (csr_seq_tcdm_q_base),
        .cfg_tcdm_out_base_i   (csr_seq_tcdm_out_base),
        .cfg_dram_kv_base_i    (csr_seq_dram_kv_base),
        .cfg_stride_tile_i     (csr_seq_stride_tile),

        .cfg_v_base_i          (csr_v_base),
        .cfg_v_num_batches_i   (csr_v_num_batches),
        .cfg_v_out_base_i      (csr_v_base),
        .cfg_rd_base_i         (csr_rd_base),
        .cfg_rd_num_vectors_i  (csr_rd_num_vectors),
        .cfg_rd_line_stride_i  (csr_rd_line_stride),
        .cfg_rd_out_base_i     (csr_wo_base),

        // DMA descriptor handshake
        .dma_req_valid_o       (dma_desc_valid_o),
        .dma_req_ready_i       (dma_desc_ready_i),
        .dma_req_dir_o         (dma_desc_dir_o),
        .dma_req_dram_addr_o   (dma_desc_dram_addr_o),
        .dma_req_tcdm_addr_o   (dma_desc_tcdm_addr_o),
        .dma_req_size_bytes_o  (dma_desc_size_o),
        .dma_done_i            (dma_desc_done_i),

        // Streamers
        .act_start_o           (seq_act_start),
        .act_base_o            (seq_act_base),
        .act_k_blocks_o        (seq_act_k_blocks),
        .act_k_stride_o        (seq_act_k_stride),
        .act_tokens_o          (seq_act_tokens),
        .act_token_stride_o    (seq_act_token_stride),
        .act_busy_i            (act_busy),
        .act_valid_i           (act_stream_valid),
        .act_ready_o           (seq_act_ready),

        .wt_start_o            (seq_wt_start),
        .wt_base_o             (seq_wt_base),
        .wt_repeat_o           (seq_wt_repeat),
        .wt_k_blocks_o         (seq_wt_k_blocks),
        .wt_k_stride_o         (seq_wt_k_stride),
        .wt_num_tiles_o        (seq_wt_num_tiles),
        .wt_tile_step_o        (seq_wt_tile_step),
        .wt_busy_i             (wt_busy),
        .wt_valid_i            (wt_stream_valid),
        .wt_ready_o            (seq_wt_ready),

        .wo_start_o            (seq_wo_start),
        .wo_mode_o             (seq_wo_mode),
        .wo_base_o             (seq_wo_base),
        .wo_num_vectors_o      (seq_wo_num_vectors),
        .wo_line_stride_o      (seq_wo_line_stride),
        .wo_num_batches_o      (seq_wo_num_batches),
        .wo_batch_stride_o     (seq_wo_batch_stride),
        .wo_busy_i             (wo_busy),
        .wo_done_i             (wo_done),

        .v_start_o             (seq_v_start),
        .v_busy_i              (v_busy),
        .v_done_i              (v_done),
        .rd_start_o            (seq_rd_start),
        .rd_busy_i             (rd_busy),
        .rd_done_i             (rd_done),

        .cfg_mode_o            (seq_core_mode),
        .cfg_row_start_o       (seq_core_row_start),
        .cfg_key_mask_o        (seq_core_key_mask),
        .cfg_enable_slq_o      (seq_core_enable_slq),
        .tile_drain_req_o      (seq_core_tile_drain_req),
        .tile_drain_idx_o      (seq_core_tile_drain_idx),
        .tile_clear_on_drain_o (seq_core_tile_clear),
        .lane_valid_o          (seq_core_lane_valid),
        .lane_acc_idx_o        (seq_core_lane_acc_idx),
        .p_fifo_empty_i        (p_fifo_empty),
        .p_fifo_almost_full_i  (p_fifo_almost_full),
        .core_fp16_valid_i     (core_fp16_valid),
        .core_out_fifo_ready_i (core_fp16_ready)
    );

    // -------------------------------------------------------------------------
    // Control source select: CSR (direct) vs sequencer (autonomous)
    // -------------------------------------------------------------------------
    assign act_start          = csr_src_sel ? csr_act_start : seq_act_start;
    assign act_base           = csr_src_sel ? csr_act_base : seq_act_base;
    assign act_k_blocks       = csr_src_sel ? csr_act_k_blocks : seq_act_k_blocks;
    assign act_k_stride       = csr_src_sel ? csr_act_k_stride : seq_act_k_stride;
    assign act_tokens         = csr_src_sel ? csr_act_tokens : seq_act_tokens;
    assign act_token_stride   = csr_src_sel ? csr_act_token_stride : seq_act_token_stride;
    assign act_stream_ready   = csr_src_sel ? csr_act_ready : seq_act_ready;

    assign wt_start           = csr_src_sel ? csr_wt_start : seq_wt_start;
    assign wt_base            = csr_src_sel ? csr_wt_base : seq_wt_base;
    assign wt_repeat          = csr_src_sel ? csr_wt_repeat : seq_wt_repeat;
    assign wt_k_blocks        = csr_src_sel ? csr_wt_k_blocks : seq_wt_k_blocks;
    assign wt_k_stride        = csr_src_sel ? csr_wt_k_stride : seq_wt_k_stride;
    assign wt_num_tiles       = csr_src_sel ? csr_wt_num_tiles : seq_wt_num_tiles;
    assign wt_tile_step       = csr_src_sel ? csr_wt_tile_step : seq_wt_tile_step;
    assign wt_stream_ready    = csr_src_sel ? {16{csr_wt_ready}} : seq_wt_ready;

    assign wo_start           = csr_src_sel ? csr_wo_start : seq_wo_start;
    assign wo_mode_fp16       = csr_src_sel ? csr_wo_mode_fp16 : (seq_wo_mode == 2'd1);

    // The descriptor always comes from one owner: the sequencer in autonomous
    // mode, the CSR file in direct mode.
    assign wo_cfg_mode = csr_src_sel ? (csr_wo_mode_fp16 ? 2'd1 : 2'd0) : seq_wo_mode;
    assign wo_cfg_base = csr_src_sel ? csr_wo_base        : seq_wo_base;
    assign wo_cfg_nvec = csr_src_sel ? csr_wo_num_vectors: seq_wo_num_vectors;
    assign wo_cfg_line = csr_src_sel ? csr_wo_line_stride: seq_wo_line_stride;
    assign wo_cfg_nb   = csr_src_sel ? 32'd1              : seq_wo_num_batches;
    assign wo_cfg_bs   = csr_src_sel ? 32'd0              : seq_wo_batch_stride;

    assign v_start            = csr_src_sel ? csr_v_start : seq_v_start;
    assign rd_start           = csr_src_sel ? csr_rd_start: seq_rd_start;

    assign cfg_mode           = csr_src_sel ? csr_core_mode : seq_core_mode;
    assign cfg_row_start      = csr_src_sel ? csr_core_row_start : seq_core_row_start;
    assign cfg_key_mask       = csr_src_sel ? csr_core_key_mask : seq_core_key_mask;
    assign cfg_enable_slq     = csr_src_sel ? csr_core_enable_slq : seq_core_enable_slq;
    assign tile_drain_req     = csr_src_sel ? csr_core_tile_drain_req : seq_core_tile_drain_req;
    assign tile_drain_idx     = csr_src_sel ? csr_core_tile_drain_idx : seq_core_tile_drain_idx;
    assign tile_clear_on_drain= csr_src_sel ? csr_core_tile_clear : seq_core_tile_clear;
    assign lane_valid         = csr_src_sel ? csr_core_lane_valid : seq_core_lane_valid;
    assign lane_acc_idx       = csr_src_sel ? csr_core_lane_acc_idx : seq_core_lane_acc_idx;

    assign csr_cfg_ready = !seq_busy && !act_busy && !wt_busy &&
                           !wo_busy && !v_busy && !rd_busy &&
                           !wo_fp4_valid;

    // =========================================================================
    // 4. TRANSPOSE BUFFER & LOG-QUANTIZER (LUỒNG CHUẨN 1 CHIỀU)
    // =========================================================================
    // Transpose Buffer: Nhận 16 hàng FP16 từ Core -> Đọc theo cột
    logic [15:0][15:0] trans_col_data;
    logic [255:0]      trans_fp16_packed;
    logic              trans_wr_en;
    logic [3:0]        trans_wr_row;
    logic [3:0]        trans_rd_col;

    assign trans_fp16_packed = trans_col_data;

    snax_transpose_buffer #(
        .TOKENS     (16),
        .CHANNELS   (16),
        .DATA_WIDTH (16)
    ) u_transpose_buffer (
        .clk      (clk),
        .rst_n    (rst_n),
        .wr_en    (trans_wr_en),
        .wr_row   (trans_wr_row),
        .wr_data  (core_fp16_packed),
        .rd_col   (trans_rd_col),
        .rd_data  (trans_col_data)
    );

    // -------------------------------------------------------------------------
    // V transpose controller: gather 16 core rows -> emit 16 columns
    // -------------------------------------------------------------------------
    assign v_active = v_start | v_busy;

    snax_transpose_ctrl #(
        .TOKENS (16)
    ) u_transpose_ctrl (
        .clk          (clk),
        .rst_n        (rst_n),
        .start_i      (v_start),
        .num_batches_i(v_num_batches),
        .busy_o       (v_busy),
        .done_o       (v_done),
        .row_valid_i  (core_fp16_valid && core_fp16_ready),
        .row_ready_o  (v_row_ready),
        .wr_en_o      (trans_wr_en),
        .wr_row_o     (trans_wr_row),
        .col_valid_o  (v_col_issue),
        .col_accept_i (quant_accept),
        .pipe_idle_i  (quant_accept),
        .rd_col_o     (trans_rd_col)
    );

    // -------------------------------------------------------------------------
    // MUX 3-to-1 CHỌN NGUỒN FP16 VÀO LOG-QUANTIZER
    // cfg_quant_src_sel: 2'b00: Core FP16 (GEMM/Attn Row)
    //                    2'b01: Transpose FP16 (V Column)
    //                    2'b10: Read Stream FP16 (TCDM Readback)
    //
    // The CSR selection is authoritative, except that an in-flight V batch or
    // readback burst overrides it: those sources are not buffered and would be
    // corrupted if the mux switched away mid-batch.
    // -------------------------------------------------------------------------
    // `rd_busy` already covers the whole readback burst: the read stream stays
    // busy until its last vector has been accepted, which is exactly the cycle
    // the mux must still select the readback source.
    wire src_v_sel    = v_active || (cfg_quant_src_sel == 2'b01);
    wire src_rd_sel   = !v_active && (rd_busy || (cfg_quant_src_sel == 2'b10));
    wire src_core_sel = !src_v_sel && !src_rd_sel;

    // Module Log-Quantizer chuẩn (Thuần túy chuyển đổi FP16 -> FP4 72b)
    // The quantizer exposes in_ready/out_ready, so the mux gates each source
    // with that signal and no external buffering is needed.
    always_comb begin
        quant_fp16_in  = core_fp16_packed;
        quant_in_valid = 1'b0;

        if (src_v_sel) begin
            quant_fp16_in  = trans_fp16_packed;
            quant_in_valid = v_col_issue;
        end else if (src_rd_sel) begin
            quant_fp16_in  = rd_fp16_data;
            quant_in_valid = rd_fp16_valid && quant_accept;
        end else if (src_core_sel) begin
            quant_fp16_in  = core_fp16_packed;
            quant_in_valid = core_fp16_valid && quant_accept && !wo_mode_fp16;
        end
    end

    log_quantizer #(
        .BLOCK_SIZE   (16),
        .PACKET_WIDTH (72)
    ) u_log_quantizer (
        .clk        (clk),
        .reset_n    (rst_n),
        .fp16_in    (quant_fp16_in),
        .in_valid   (quant_in_valid),
        .in_ready   (quant_accept),
        .packed_vec (wo_fp4_data),
        .out_valid  (wo_fp4_valid),
        .out_ready  (wo_fp4_ready)
    );

    // -------------------------------------------------------------------------
    // Readback / V / core-FP16 handshakes
    // -------------------------------------------------------------------------
    assign v_base        = csr_v_base;
    assign v_num_batches = csr_v_num_batches;
    assign rd_base        = csr_rd_base;
    assign rd_num_vectors = csr_rd_num_vectors;
    assign rd_line_stride = csr_rd_line_stride;

    assign rd_fp16_ready   = src_rd_sel && quant_accept;

    // The core output feeds exactly one consumer at a time.  `core_fp16_ready`
    // must follow the same selection as the quantizer mux, otherwise a word
    // accepted here would never be taken by any downstream block.
    assign core_fp16_ready = v_active ? v_row_ready : wo_mode_fp16 ? wo_fp16_ready : src_core_sel ? quant_accept : 1'b0;

    // =========================================================================
    // 5. CÁC MODULE STREAMERS & HARDWARE CORE
    // =========================================================================
    // A. Activation Streamer (Vòng trong là Token, vòng ngoài là K)
    snax_streamer_act_bcast #(
        .DATA_WIDTH (72), .ADDR_WIDTH (32), .STRB_WIDTH (9)
    ) u_act_streamer (
        .clk                (clk),
        .rst_n              (rst_n),
        .csr_start_i        (act_start),
        .busy_o             (act_busy),
        .done_irq_o         (),
        .cfg_base_addr_i    (act_base),
        .cfg_tokens_i       (act_tokens),
        .cfg_token_stride_i (act_token_stride),
        .cfg_k_blocks_i     (act_k_blocks),
        .cfg_k_stride_i     (act_k_stride),
        .m_tcdm_req_valid_o (act_m_req_valid),
        .m_tcdm_req_write_o (act_m_req_write),
        .m_tcdm_req_addr_o  (act_m_req_addr),
        .m_tcdm_req_wdata_o (act_m_req_wdata),
        .m_tcdm_req_strb_o  (act_m_req_strb),
        .m_tcdm_req_ready_i (act_m_req_ready),
        .m_tcdm_rsp_rdata_i (act_m_rsp_rdata),
        .m_tcdm_rsp_valid_i (act_m_rsp_valid),
        .act_stream_data_o  (act_stream_data),
        .act_stream_valid_o (act_stream_valid),
        .act_stream_ready_i (act_stream_ready)
    );

    // B. Weight Streamer (16 Kênh song song)
    snax_streamer_wt_16ch #(
        .DATA_WIDTH (72), .ADDR_WIDTH (32), .NUM_LANES (16), .STRB_WIDTH (9)
    ) u_wt_streamer (
        .clk                 (clk),
        .rst_n               (rst_n),
        .csr_start_i         (wt_start),
        .busy_o              (wt_busy),
        .done_irq_o          (),
        .cfg_base_addr_i     (wt_base),
        .cfg_repeat_tokens_i (wt_repeat),
        .cfg_k_blocks_i      (wt_k_blocks),
        .cfg_k_stride_i      (wt_k_stride),
        .cfg_num_tiles_i     (wt_num_tiles),
        .cfg_tile_step_i     (wt_tile_step),
        .m_tcdm_req_valid_o  (wt_m_req_valid),
        .m_tcdm_req_write_o  (wt_m_req_write),
        .m_tcdm_req_addr_o   (wt_m_req_addr),
        .m_tcdm_req_wdata_o  (wt_m_req_wdata),
        .m_tcdm_req_strb_o   (wt_m_req_strb),
        .m_tcdm_req_ready_i  (wt_m_req_ready),
        .m_tcdm_rsp_rdata_i  (wt_m_rsp_rdata),
        .m_tcdm_rsp_valid_i  (wt_m_rsp_valid),
        .wt_stream_data_o    (wt_stream_data),
        .wt_stream_valid_o   (wt_stream_valid),
        .wt_stream_ready_i   (wt_stream_ready)
    );

    // C. Write Streamer (HỖ TRỢ ĐỒNG THỜI CẢ FP4 NÉN VÀ FP16 THÔ CHO CPU)
    //
    // cfg_mode_i: 0 = FP4 linear, 1 = FP16 linear, 2 = V FP4 batch.
    // The job type is decided by whoever starts the engine:
    //   wo_start          -> CSR / sequencer job (FP4 or FP16)
    //   v_start           -> V batch job
    //   rd_start          -> readback job (FP4 linear, one packet per vector)
    // -------------------------------------------------------------------------
    // Write-out descriptor
    //
    // The sequencer owns the complete descriptor.  The top-level only selects
    // between the sequencer (autonomous mode) and the CSR file (direct mode);
    // it never derives mode/address/count from the running engine.
    // -------------------------------------------------------------------------
    wire wo_job_start = wo_start | v_start | rd_start;

    wire [1:0]  wo_mode_sel = wo_cfg_mode;
    wire [31:0] wo_base_sel = wo_cfg_base;
    wire [31:0] wo_nvec_sel = wo_cfg_nvec;
    wire [31:0] wo_line_sel = wo_cfg_line;
    wire [31:0] wo_nb_sel   = wo_cfg_nb;
    wire [31:0] wo_bs_sel   = wo_cfg_bs;

    snax_streamer_write_out #(
        .DATA_WIDTH (72), .ADDR_WIDTH (32), .NUM_LANES (16), .STRB_WIDTH (9)
    ) u_write_out (
        .clk                   (clk),
        .rst_n                 (rst_n),
        .csr_start_i           (wo_job_start),
        .busy_o                (wo_busy),
        .done_irq_o            (wo_done),
        .cfg_mode_i            (wo_mode_sel),
        .cfg_base_addr_i       (wo_base_sel),
        .cfg_num_vectors_i     (wo_nvec_sel),
        .cfg_line_stride_i     (wo_line_sel),
        .cfg_num_batches_i     (wo_nb_sel),
        .cfg_batch_stride_i    (wo_bs_sel),

        // LUỒNG FP16 (Nối trực tiếp từ Core ra ngoài TCDM cho CPU RoPE/RMSNorm)
        .core_out_fp16_data_i  (core_fp16_packed),
        .core_out_fp16_valid_i (core_fp16_valid && wo_mode_fp16 && !v_active),
        .core_out_fp16_ready_o (wo_fp16_ready),

        // LUỒNG FP4 (Từ Log-Quantizer ghi xuống TCDM)
        .core_out_fp4_data_i   (wo_fp4_data),
        .core_out_fp4_valid_i  (wo_fp4_valid),
        .core_out_fp4_ready_o  (wo_fp4_ready),

        // TCDM Master Port 19
        .m_tcdm_req_valid_o    (wo_m_req_valid),
        .m_tcdm_req_write_o    (wo_m_req_write),
        .m_tcdm_req_addr_o     (wo_m_req_addr),
        .m_tcdm_req_wdata_o    (wo_m_req_wdata),
        .m_tcdm_req_strb_o     (wo_m_req_strb),
        .m_tcdm_req_ready_i    (wo_m_req_ready)
    );

    // D. Pipelined TCDM FP16 Read Streamer (Master 20 - Tối ưu 1 request/chu kỳ)
    snax_stream_read_fp16 #(
        .DATA_WIDTH (72), .ADDR_WIDTH (32), .STRB_WIDTH (9)
    ) u_read_fp16 (
        .clk                (clk),
        .rst_n              (rst_n),
        .csr_start_i        (rd_start),
        .busy_o             (rd_busy),
        .done_irq_o         (rd_done),
        .cfg_base_addr_i    (rd_base),
        .cfg_num_vectors_i  (rd_num_vectors),
        .cfg_line_stride_i  (rd_line_stride),
        .m_tcdm_req_valid_o (rd_m_req_valid),
        .m_tcdm_req_write_o (rd_m_req_write),
        .m_tcdm_req_addr_o  (rd_m_req_addr),
        .m_tcdm_req_wdata_o (rd_m_req_wdata),
        .m_tcdm_req_strb_o  (rd_m_req_strb),
        .m_tcdm_req_ready_i (rd_m_req_ready),
        .m_tcdm_rsp_rdata_i (rd_m_rsp_rdata),
        .m_tcdm_rsp_valid_i (rd_m_rsp_valid),
        .fp16_data_o        (rd_fp16_data),
        .fp16_valid_o       (rd_fp16_valid),
        .fp16_ready_i       (rd_fp16_ready)
    );

    // E. Hardware Core (SNAX Fused LLM Core)
    snax_fused_llm_core #(
        .NUM_LANES_P      (16),
        .ACC_DEPTH_P      (ACC_DEPTH),
        .NUM_QUERIES      (8),
        .PKT_W            (72),
        .FRAC_BITS        (16),
        .P_FIFO_DEPTH     (16),
        .USE_OUTPUT_READY (1'b1)
    ) u_core (
        .clk                   (clk),
        .rst_n                 (rst_n),
        .cfg_mode_i            (cfg_mode),
        .cfg_row_start_i       (cfg_row_start),
        .cfg_key_mask_i        (cfg_key_mask),
        .cfg_enable_slq_i      (cfg_enable_slq),
        .tile_drain_req_i      (tile_drain_req),
        .tile_drain_idx_i      (tile_drain_idx),
        .tile_clear_on_drain_i (tile_clear_on_drain),
        .lane_valid_i          (lane_valid),
        .lane_acc_idx_i        (lane_acc_idx),
        .ext_act_broadcast_i   (act_stream_data),
        .ext_act_valid_i       (act_stream_valid),
        .wt_in_i               (wt_stream_data),
        .wt_valid_i            (&wt_stream_valid),
        .out_vec_fp16_packed_o (core_fp16_packed),
        .out_vec_fp16_lanes_o  (),
        .out_vec_fp16_valid_o  (core_fp16_valid),
        .out_vec_fp16_ready_i  (core_fp16_ready),
        .out_fifo_ready_o      (core_out_fifo_ready),
        .out_vec_fp4_packet_o  (),
        .out_vec_fp4_valid_o   (),
        .p_fifo_full_o         (p_fifo_full),
        .p_fifo_empty_o        (p_fifo_empty),
        .p_fifo_almost_full_o  (p_fifo_almost_full),
        .l_ready_o             (l_ready)
    );

    // -------------------------------------------------------------------------
    // System status
    // -------------------------------------------------------------------------
    // `p_fifo_almost_full` is exposed by the core so a future scheduler can stop
    // issuing drains before the P FIFO overflows; no consumer needs it yet.
    assign subsystem_busy_o     = csr_src_sel ? (act_busy | wt_busy | wo_busy | v_busy | rd_busy | wo_fp4_valid) : (seq_busy | v_busy | wo_busy | rd_busy | wo_fp4_valid);
    assign subsystem_done_irq_o = (seq_done | v_done | rd_done | wo_done) && !wo_busy && !wo_fp4_valid;

endmodule
