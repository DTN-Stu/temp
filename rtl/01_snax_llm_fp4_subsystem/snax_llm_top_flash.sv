// =============================================================================
// Module: snax_llm_top_flash
// Description: Integrated SNAX FP4 FlashAttention subsystem top.
//
//   host CSR + host DMA
//        |
//   20-master interleaved xbar (0 host, 1 act, 2..17 wt, 18 out, 19 v-quant)
//        |
//   32-bank x 1024 x 72b TCDM
//        |
//   + snax_streamer_act_bcast
//   + snax_streamer_wt_16ch
//   + snax_streamer_write_out (linear FP4/FP16)
//   + snax_v_quant_path (transpose buffer -> log quantizer -> FP4)
//   + skid_buffer for FP4 backpressure
//   + snax_fused_llm_core (8 queries, 8:8 slots)
//   + snax_flash_seq (hardware h->qi->dvi->kj sequencer)
//   + snax_flash_csr (full CSR file, sequencer/CSR source select)
//
// Host DMA master 0 provides TCDM preload/readback.
// Fixed data map (same as tb_snax_core_flash_system):
//   Q  bank16, word = h*32 + q*2 + kb
//   K  bank0..15, word = h*4 + kj*2 + kb
//   V  bank0..15, word = 16 + h*4 + dvi*2 + kj
//   OUT bank17, word = 0..
// =============================================================================
`timescale 1ns/1ps

module snax_llm_top_flash #(
    parameter int unsigned DATA_WIDTH  = 72,
    parameter int unsigned STRB_WIDTH  = 9,
    parameter int unsigned ADDR_WIDTH  = 32,
    parameter int unsigned NUM_BANKS   = 32,
    parameter int unsigned BANK_DEPTH  = 1024,
    parameter int unsigned NUM_LANES   = 16,
    parameter int unsigned ACC_DEPTH   = 16
)(
    input  logic clk,
    input  logic rst_n,

    // Host CSR
    input  logic        host_csr_we_i,
    input  logic [11:0] host_csr_addr_i,
    input  logic [31:0] host_csr_wdata_i,
    output logic [31:0] host_csr_rdata_o,

    // Host DMA / TCDM port (master 0)
    input  logic                    host_mem_req_valid_i,
    output logic                    host_mem_req_ready_o,
    input  logic [ADDR_WIDTH-1:0]   host_mem_req_addr_i,
    input  logic                    host_mem_req_write_i,
    input  logic [DATA_WIDTH-1:0]   host_mem_req_wdata_i,
    input  logic [STRB_WIDTH-1:0]   host_mem_req_strb_i,
    output logic                    host_mem_rsp_valid_o,
    output logic [DATA_WIDTH-1:0]   host_mem_rsp_rdata_o,

    // Status / debug
    output logic                    subsystem_busy_o,
    output logic                    subsystem_done_irq_o,
    output logic [NUM_LANES*16-1:0] dbg_fp16_packed_o,
    output logic                    dbg_fp16_valid_o
);

    localparam int NUM_MASTERS = 20;

    // -------------------------------------------------------------------------
    // CSR / sequencer source selection
    // -------------------------------------------------------------------------
    logic        csr_start_pulse;
    logic        csr_src_sel;
    logic        csr_act_ready;
    logic        csr_wt_ready;

    logic        csr_act_start;
    logic [31:0] csr_act_base;
    logic [31:0] csr_act_k_blocks;
    logic [31:0] csr_act_k_stride;
    logic [31:0] csr_act_tokens;
    logic [31:0] csr_act_token_stride;

    logic        csr_wt_start;
    logic [31:0] csr_wt_base;
    logic [31:0] csr_wt_repeat;
    logic [31:0] csr_wt_k_blocks;
    logic [31:0] csr_wt_k_stride;
    logic [31:0] csr_wt_num_tiles;
    logic [31:0] csr_wt_tile_step;

    logic        csr_wo_start;
    logic        csr_wo_mode_fp16;
    logic [31:0] csr_wo_base;
    logic [31:0] csr_wo_num_vectors;
    logic [31:0] csr_wo_line_stride;

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
    logic [31:0] csr_v_base;
    logic [31:0] csr_v_num_batches;
    logic        csr_rd_start;
    logic [31:0] csr_rd_base;
    logic [31:0] csr_rd_num_vectors;
    logic [31:0] csr_rd_line_stride;
    logic [1:0]  csr_quant_src_sel;

    logic        seq_act_start;
    logic [31:0] seq_act_base;
    logic [31:0] seq_act_k_blocks;
    logic [31:0] seq_act_k_stride;
    logic [31:0] seq_act_tokens;
    logic [31:0] seq_act_token_stride;
    logic        seq_act_ready;

    logic        seq_wt_start;
    logic [31:0] seq_wt_base;
    logic [31:0] seq_wt_repeat;
    logic [31:0] seq_wt_k_blocks;
    logic [31:0] seq_wt_k_stride;
    logic [31:0] seq_wt_num_tiles;
    logic [31:0] seq_wt_tile_step;
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
    logic [31:0] seq_wo_base;
    logic [31:0] seq_wo_num_vectors;
    logic [31:0] seq_wo_line_stride;

    logic        wo_mode_fp16;
    logic        wo_fp16_valid;

    logic        v_start, v_busy, v_done, v_fp16_ready;
    logic [31:0] v_base, v_num_batches;
    logic        v_m_req_valid, v_m_req_write, v_m_req_ready;
    logic [31:0] v_m_req_addr;
    logic [71:0] v_m_req_wdata;
    logic [8:0]  v_m_req_strb;

    // -------------------------------------------------------------------------
    // Interconnect
    // -------------------------------------------------------------------------
    logic [NUM_MASTERS-1:0]          m_req_valid;
    logic [NUM_MASTERS-1:0]          m_req_ready;
    logic [NUM_MASTERS-1:0][31:0]    m_req_addr;
    logic [NUM_MASTERS-1:0]          m_req_write;
    logic [NUM_MASTERS-1:0][71:0]    m_req_wdata;
    logic [NUM_MASTERS-1:0][8:0]     m_req_strb;
    logic [NUM_MASTERS-1:0]          m_rsp_valid;
    logic [NUM_MASTERS-1:0][71:0]    m_rsp_rdata;

    logic [NUM_BANKS-1:0]         b_req;
    logic [NUM_BANKS-1:0]         b_write;
    logic [NUM_BANKS-1:0][9:0]    b_addr;
    logic [NUM_BANKS-1:0][71:0]   b_wdata;
    logic [NUM_BANKS-1:0][8:0]    b_strb;
    logic [NUM_BANKS-1:0][71:0]   b_rdata;
    logic [NUM_BANKS-1:0]         b_rvalid;

    // -------------------------------------------------------------------------
    // Streamers
    // -------------------------------------------------------------------------
    logic act_start, act_busy;
    logic [31:0] act_base, act_k_blocks, act_k_stride, act_tokens, act_token_stride;
    logic act_m_req_valid, act_m_req_write, act_m_req_ready;
    logic [31:0] act_m_req_addr;
    logic [71:0] act_m_req_wdata;
    logic [8:0]  act_m_req_strb;
    logic [71:0] act_m_rsp_rdata;
    logic        act_m_rsp_valid;
    logic [71:0] act_stream_data;
    logic        act_stream_valid, act_stream_ready;

    logic wt_start, wt_busy;
    logic [31:0] wt_base, wt_repeat, wt_k_blocks, wt_k_stride, wt_num_tiles, wt_tile_step;
    logic [15:0] wt_m_req_valid, wt_m_req_write, wt_m_req_ready;
    logic [15:0][31:0] wt_m_req_addr;
    logic [15:0][71:0] wt_m_req_wdata;
    logic [15:0][8:0]  wt_m_req_strb;
    logic [15:0][71:0] wt_m_rsp_rdata;
    logic [15:0]       wt_m_rsp_valid;
    logic [15:0][71:0] wt_stream_data;
    logic [15:0]       wt_stream_valid, wt_stream_ready;

    logic wo_start, wo_busy;
    logic [31:0] wo_base, wo_num_vectors, wo_line_stride;
    logic wo_m_req_valid, wo_m_req_write, wo_m_req_ready;
    logic [31:0] wo_m_req_addr;
    logic [71:0] wo_m_req_wdata;
    logic [8:0]  wo_m_req_strb;

    logic [255:0] core_fp16_packed;
    logic         core_fp16_valid;
    logic         core_fp16_ready;
    logic [71:0]  core_fp4_packet;
    logic         core_fp4_valid;
    logic wo_fp4_ready;
    logic [71:0]  wo_fp4_data;
    logic         wo_fp4_valid;
    logic         v_active;
    logic         v_packet_ready;
    logic [1:0]   wo_cfg_mode;
    logic [31:0]  wo_cfg_base, wo_cfg_num_vectors, wo_cfg_line_stride;
    logic [31:0]  wo_cfg_num_batches, wo_cfg_batch_stride;

    // Shared log-quantizer
    logic [71:0]  logq_core_pkt, logq_core_pkt_data;
    logic         logq_core_pkt_valid, logq_core_pkt_ready;
    logic [71:0]  logq_v_pkt_data;
    logic         logq_v_pkt_valid, logq_v_pkt_ready;
    logic [71:0]  logq_rd_pkt_data;
    logic         logq_rd_pkt_valid, logq_rd_pkt_ready;
    logic [255:0] logq_core_fp16_data;
    logic         logq_core_fp16_valid, logq_core_fp16_ready;
    logic [255:0] logq_v_col_data;
    logic         logq_v_col_valid, logq_v_col_ready;
    logic [255:0] logq_rd_fp16_data;
    logic         logq_rd_fp16_valid, logq_rd_fp16_ready;
    logic [1:0]   logq_active_src;

    // V path packet handshake
    logic [71:0]  v_col_pkt_data;
    logic         v_col_pkt_valid, v_col_pkt_ready;
    logic [71:0]  v_wo_pkt_data;
    logic         v_wo_pkt_valid, v_wo_pkt_ready;
    logic [255:0] v_core_fp16_data;
    logic         v_core_fp16_valid, v_core_fp16_ready;

    // TCDM readback stream
    logic         rd_start, rd_busy, rd_done;
    logic [31:0]  rd_base, rd_num_vectors, rd_line_stride;
    logic         rd_m_req_valid, rd_m_req_write, rd_m_req_ready;
    logic [31:0]  rd_m_req_addr;
    logic [71:0]  rd_m_req_wdata;
    logic [8:0]   rd_m_req_strb;
    logic [71:0]  rd_m_rsp_rdata;
    logic         rd_m_rsp_valid;

    logic        fp4_skid_push, fp4_skid_pop;
    logic [71:0] fp4_skid_din, fp4_skid_dout;
    logic        fp4_skid_ready, fp4_skid_valid;
    logic [4:0]  fp4_skid_level;

    // -------------------------------------------------------------------------
    // Core control from sequencer
    // -------------------------------------------------------------------------
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
    logic        seq_busy, seq_done;

    snax_flash_csr u_csr (
        .clk                    (clk),
        .rst_n                  (rst_n),
        .csr_we                 (host_csr_we_i),
        .csr_addr               (host_csr_addr_i),
        .csr_wdata              (host_csr_wdata_i),
        .csr_rdata              (host_csr_rdata_o),
        .seq_busy               (seq_busy),
        .seq_done               (seq_done),
        .act_busy               (act_busy),
        .wt_busy                (wt_busy),
        .wo_busy                (wo_busy),
        .v_busy                 (v_busy),
        .v_done                 (v_done),
        .rd_busy                (rd_busy),
        .rd_done                (rd_done),
        .p_fifo_full            (p_fifo_full),
        .p_fifo_empty           (p_fifo_empty),
        .l_ready                (l_ready),
        .core_fp16_valid        (core_fp16_valid),
        .start_pulse            (csr_start_pulse),
        .src_sel                (csr_src_sel),
        .act_ready_force        (csr_act_ready),
        .wt_ready_force         (csr_wt_ready),
        .act_start              (csr_act_start),
        .act_base               (csr_act_base),
        .act_k_blocks           (csr_act_k_blocks),
        .act_k_stride           (csr_act_k_stride),
        .act_tokens             (csr_act_tokens),
        .act_token_stride       (csr_act_token_stride),
        .wt_start               (csr_wt_start),
        .wt_base                (csr_wt_base),
        .wt_repeat              (csr_wt_repeat),
        .wt_k_blocks            (csr_wt_k_blocks),
        .wt_k_stride            (csr_wt_k_stride),
        .wt_num_tiles           (csr_wt_num_tiles),
        .wt_tile_step           (csr_wt_tile_step),
        .wo_start               (csr_wo_start),
        .wo_mode_fp16           (csr_wo_mode_fp16),
        .wo_base                (csr_wo_base),
        .wo_num_vectors         (csr_wo_num_vectors),
        .wo_line_stride         (csr_wo_line_stride),
        .core_mode              (csr_core_mode),
        .core_row_start         (csr_core_row_start),
        .core_key_mask          (csr_core_key_mask),
        .core_enable_slq        (csr_core_enable_slq),
        .core_tile_drain_req    (csr_core_tile_drain_req),
        .core_tile_drain_idx    (csr_core_tile_drain_idx),
        .core_tile_clear_on_drain (csr_core_tile_clear),
        .core_lane_valid        (csr_core_lane_valid),
        .core_lane_acc_idx      (csr_core_lane_acc_idx),
        .v_start                (csr_v_start),
        .v_base                 (csr_v_base),
        .v_num_batches          (csr_v_num_batches),
        .rd_start               (csr_rd_start),
        .rd_base                (csr_rd_base),
        .rd_num_vectors         (csr_rd_num_vectors),
        .rd_line_stride         (csr_rd_line_stride),
        .quant_src_sel          (csr_quant_src_sel)
    );

    assign act_start            = csr_src_sel ? csr_act_start : seq_act_start;
    assign act_base             = csr_src_sel ? csr_act_base : seq_act_base;
    assign act_k_blocks         = csr_src_sel ? csr_act_k_blocks : seq_act_k_blocks;
    assign act_k_stride         = csr_src_sel ? csr_act_k_stride : seq_act_k_stride;
    assign act_tokens           = csr_src_sel ? csr_act_tokens : seq_act_tokens;
    assign act_token_stride     = csr_src_sel ? csr_act_token_stride : seq_act_token_stride;
    assign act_stream_ready     = csr_src_sel ? csr_act_ready : seq_act_ready;

    assign wt_start             = csr_src_sel ? csr_wt_start : seq_wt_start;
    assign wt_base              = csr_src_sel ? csr_wt_base : seq_wt_base;
    assign wt_repeat            = csr_src_sel ? csr_wt_repeat : seq_wt_repeat;
    assign wt_k_blocks          = csr_src_sel ? csr_wt_k_blocks : seq_wt_k_blocks;
    assign wt_k_stride          = csr_src_sel ? csr_wt_k_stride : seq_wt_k_stride;
    assign wt_num_tiles         = csr_src_sel ? csr_wt_num_tiles : seq_wt_num_tiles;
    assign wt_tile_step         = csr_src_sel ? csr_wt_tile_step : seq_wt_tile_step;
    assign wt_stream_ready      = csr_src_sel ? {16{csr_wt_ready}} : seq_wt_ready;

    assign wo_start             = csr_src_sel ? csr_wo_start : seq_wo_start;
    assign wo_mode_fp16         = csr_src_sel ? csr_wo_mode_fp16 : 1'b0;
    assign wo_base              = csr_src_sel ? csr_wo_base : seq_wo_base;
    assign wo_num_vectors       = csr_src_sel ? csr_wo_num_vectors : seq_wo_num_vectors;
    assign wo_line_stride       = csr_src_sel ? csr_wo_line_stride : seq_wo_line_stride;

    assign cfg_mode             = csr_src_sel ? csr_core_mode : seq_core_mode;
    assign cfg_row_start        = csr_src_sel ? csr_core_row_start : seq_core_row_start;
    assign cfg_key_mask         = csr_src_sel ? csr_core_key_mask : seq_core_key_mask;
    assign cfg_enable_slq       = csr_src_sel ? csr_core_enable_slq : seq_core_enable_slq;
    assign tile_drain_req       = csr_src_sel ? csr_core_tile_drain_req : seq_core_tile_drain_req;
    assign tile_drain_idx       = csr_src_sel ? csr_core_tile_drain_idx : seq_core_tile_drain_idx;
    assign tile_clear_on_drain  = csr_src_sel ? csr_core_tile_clear : seq_core_tile_clear;
    assign lane_valid           = csr_src_sel ? csr_core_lane_valid : seq_core_lane_valid;
    assign lane_acc_idx         = csr_src_sel ? csr_core_lane_acc_idx : seq_core_lane_acc_idx;

    assign v_start              = csr_v_start && !seq_busy && !wo_busy &&
                                  !fp4_skid_valid && !v_busy;
    assign v_base               = csr_v_base;
    assign v_num_batches        = csr_v_num_batches;
    assign rd_start             = csr_rd_start && !wo_busy && !fp4_skid_valid &&
                                  !v_busy && !rd_busy;
    assign rd_base              = csr_rd_base;
    assign rd_num_vectors       = csr_rd_num_vectors;
    assign rd_line_stride       = csr_rd_line_stride;

    assign v_active             = v_start | v_busy;

    // FP16 consumers: core output goes to the shared quantizer row port, V
    // transpose, or the readback path depending on the active source.
    assign logq_core_fp16_data  = core_fp16_packed;
    assign logq_core_fp16_valid = core_fp16_valid &&
                                  !v_active && !rd_busy &&
                                  (csr_quant_src_sel != 2'b01) &&
                                  (csr_quant_src_sel != 2'b10);
    assign v_core_fp16_data     = core_fp16_packed;
    assign v_core_fp16_valid    = core_fp16_valid && v_active;

    assign core_fp16_ready      = v_active      ? v_core_fp16_ready :
                                  rd_busy       ? logq_rd_fp16_ready :
                                                  logq_core_fp16_ready;

    // Row packet path: shared quantizer -> FP4 skid -> write_out
    assign core_fp4_packet      = logq_core_pkt_data;
    assign core_fp4_valid       = logq_core_pkt_valid;
    assign logq_core_pkt_ready  = fp4_skid_ready;

    // V packet path: shared quantizer -> V path -> write_out V mode
    assign v_col_pkt_data       = logq_v_pkt_data;
    assign v_col_pkt_valid      = logq_v_pkt_valid;
    assign logq_v_pkt_ready     = v_col_pkt_ready;

    // Write-out source select: V packet > readback packet > row skid
    wire wo_v_sel  = v_active;
    wire wo_rd_sel = !wo_v_sel && (rd_busy || logq_rd_pkt_valid);

    assign wo_cfg_mode          = wo_v_sel ? 2'd2 : 2'd0;
    assign wo_cfg_base          = wo_v_sel ? v_base : wo_base;
    assign wo_cfg_num_vectors   = wo_v_sel ? 32'd16 : wo_num_vectors;
    assign wo_cfg_line_stride   = wo_v_sel ? 32'd16 : wo_line_stride;
    assign wo_cfg_num_batches   = wo_v_sel ? v_num_batches : 32'd1;
    assign wo_cfg_batch_stride  = wo_v_sel ? 32'h0000_0200 : 32'd0;

    assign v_wo_pkt_ready       = wo_v_sel  ? wo_fp4_ready : 1'b0;
    assign logq_rd_pkt_ready    = wo_rd_sel ? wo_fp4_ready : 1'b0;
    assign fp4_skid_pop         = (!wo_v_sel && !wo_rd_sel) ? wo_fp4_ready : 1'b0;

    assign wo_fp4_data          = wo_v_sel  ? v_wo_pkt_data :
                                  wo_rd_sel ? logq_rd_pkt_data : fp4_skid_dout;
    assign wo_fp4_valid         = wo_v_sel  ? v_wo_pkt_valid :
                                  wo_rd_sel ? logq_rd_pkt_valid : fp4_skid_valid;

    assign subsystem_busy_o     = csr_src_sel
                                ? (act_busy | wt_busy | wo_busy | v_busy |
                                   rd_busy | fp4_skid_valid)
                                : (seq_busy | v_busy | wo_busy |
                                   rd_busy | fp4_skid_valid);
    assign subsystem_done_irq_o = (seq_done | v_done | rd_done) &&
                                  !wo_busy && !fp4_skid_valid;
    assign dbg_fp16_packed_o    = core_fp16_packed;
    assign dbg_fp16_valid_o     = core_fp16_valid;

    // -------------------------------------------------------------------------
    // Host master 0
    // -------------------------------------------------------------------------
    assign m_req_valid[0]  = host_mem_req_valid_i;
    assign m_req_addr[0]   = host_mem_req_addr_i;
    assign m_req_write[0]  = host_mem_req_write_i;
    assign m_req_wdata[0]  = host_mem_req_wdata_i;
    assign m_req_strb[0]   = host_mem_req_strb_i;
    assign host_mem_req_ready_o = m_req_ready[0];
    assign host_mem_rsp_valid_o = m_rsp_valid[0];
    assign host_mem_rsp_rdata_o = m_rsp_rdata[0];

    // Act master 1
    assign m_req_valid[1]  = act_m_req_valid;
    assign m_req_addr[1]   = act_m_req_addr;
    assign m_req_write[1]  = act_m_req_write;
    assign m_req_wdata[1]  = act_m_req_wdata;
    assign m_req_strb[1]   = act_m_req_strb;
    assign act_m_req_ready = m_req_ready[1];
    assign act_m_rsp_valid = m_rsp_valid[1];
    assign act_m_rsp_rdata = m_rsp_rdata[1];

    // Wt masters 2..17
    genvar gw;
    generate
        for (gw = 0; gw < 16; gw++) begin : GEN_WT
            assign m_req_valid[2+gw]  = wt_m_req_valid[gw];
            assign m_req_addr[2+gw]   = wt_m_req_addr[gw];
            assign m_req_write[2+gw]  = wt_m_req_write[gw];
            assign m_req_wdata[2+gw]  = wt_m_req_wdata[gw];
            assign m_req_strb[2+gw]   = wt_m_req_strb[gw];
            assign wt_m_req_ready[gw] = m_req_ready[2+gw];
            assign wt_m_rsp_valid[gw] = m_rsp_valid[2+gw];
            assign wt_m_rsp_rdata[gw] = m_rsp_rdata[2+gw];
        end
    endgenerate

    // Output master 18
    assign m_req_valid[18] = wo_m_req_valid;
    assign m_req_addr[18]  = wo_m_req_addr;
    assign m_req_write[18] = wo_m_req_write;
    assign m_req_wdata[18] = wo_m_req_wdata;
    assign m_req_strb[18]  = wo_m_req_strb;
    assign wo_m_req_ready  = m_req_ready[18];

    // V path is a packet source for write_out V mode.
    assign m_req_valid[19] = rd_m_req_valid;
    assign m_req_addr[19]  = rd_m_req_addr;
    assign m_req_write[19] = rd_m_req_write;
    assign m_req_wdata[19] = rd_m_req_wdata;
    assign m_req_strb[19]  = rd_m_req_strb;
    assign rd_m_req_ready  = m_req_ready[19];
    assign rd_m_rsp_valid  = m_rsp_valid[19];
    assign rd_m_rsp_rdata  = m_rsp_rdata[19];

    // -------------------------------------------------------------------------
    // Streamers
    // -------------------------------------------------------------------------
    snax_streamer_act_bcast #(
        .DATA_WIDTH (72), .ADDR_WIDTH (32), .STRB_WIDTH (9)
    ) u_act_streamer (
        .clk                     (clk),
        .rst_n                   (rst_n),
        .csr_start_i             (act_start),
        .busy_o                  (act_busy),
        .done_irq_o              (),
        .cfg_base_addr_i         (act_base),
        .cfg_k_blocks_i          (act_k_blocks),
        .cfg_k_stride_i          (act_k_stride),
        .cfg_tokens_i            (act_tokens),
        .cfg_token_stride_i      (act_token_stride),
        .m_tcdm_req_valid_o      (act_m_req_valid),
        .m_tcdm_req_write_o      (act_m_req_write),
        .m_tcdm_req_addr_o       (act_m_req_addr),
        .m_tcdm_req_wdata_o      (act_m_req_wdata),
        .m_tcdm_req_strb_o       (act_m_req_strb),
        .m_tcdm_req_ready_i      (act_m_req_ready),
        .m_tcdm_rsp_rdata_i      (act_m_rsp_rdata),
        .m_tcdm_rsp_valid_i      (act_m_rsp_valid),
        .act_stream_data_o       (act_stream_data),
        .act_stream_valid_o      (act_stream_valid),
        .act_stream_ready_i      (act_stream_ready)
    );

    snax_streamer_wt_16ch #(
        .DATA_WIDTH (72), .ADDR_WIDTH (32), .NUM_LANES (16), .STRB_WIDTH (9)
    ) u_wt_streamer (
        .clk                   (clk),
        .rst_n                 (rst_n),
        .csr_start_i           (wt_start),
        .busy_o                (wt_busy),
        .done_irq_o            (),
        .cfg_base_addr_i       (wt_base),
        .cfg_repeat_tokens_i   (wt_repeat),
        .cfg_k_blocks_i        (wt_k_blocks),
        .cfg_k_stride_i        (wt_k_stride),
        .cfg_num_tiles_i       (wt_num_tiles),
        .cfg_tile_step_i       (wt_tile_step),
        .m_tcdm_req_valid_o    (wt_m_req_valid),
        .m_tcdm_req_write_o    (wt_m_req_write),
        .m_tcdm_req_addr_o     (wt_m_req_addr),
        .m_tcdm_req_wdata_o    (wt_m_req_wdata),
        .m_tcdm_req_strb_o     (wt_m_req_strb),
        .m_tcdm_req_ready_i    (wt_m_req_ready),
        .m_tcdm_rsp_rdata_i    (wt_m_rsp_rdata),
        .m_tcdm_rsp_valid_i    (wt_m_rsp_valid),
        .wt_stream_data_o      (wt_stream_data),
        .wt_stream_valid_o     (wt_stream_valid),
        .wt_stream_ready_i     (wt_stream_ready)
    );

    snax_streamer_write_out #(
        .DATA_WIDTH (72), .ADDR_WIDTH (32), .NUM_LANES (16), .STRB_WIDTH (9)
    ) u_write_out (
        .clk                    (clk),
        .rst_n                  (rst_n),
        .csr_start_i            (wo_start | v_start),
        .busy_o                 (wo_busy),
        .done_irq_o             (),
        .cfg_mode_i             (wo_cfg_mode),
        .cfg_base_addr_i        (wo_cfg_base),
        .cfg_num_vectors_i      (wo_cfg_num_vectors),
        .cfg_line_stride_i      (wo_cfg_line_stride),
        .cfg_num_batches_i      (wo_cfg_num_batches),
        .cfg_batch_stride_i     (wo_cfg_batch_stride),
        .core_out_fp16_data_i   (core_fp16_packed),
        .core_out_fp16_valid_i  (wo_fp16_valid),
        .core_out_fp16_ready_o  (),
        .core_out_fp4_data_i    (wo_fp4_data),
        .core_out_fp4_valid_i   (wo_fp4_valid),
        .core_out_fp4_ready_o   (wo_fp4_ready),
        .m_tcdm_req_valid_o     (wo_m_req_valid),
        .m_tcdm_req_write_o     (wo_m_req_write),
        .m_tcdm_req_addr_o      (wo_m_req_addr),
        .m_tcdm_req_wdata_o     (wo_m_req_wdata),
        .m_tcdm_req_strb_o      (wo_m_req_strb),
        .m_tcdm_req_ready_i     (wo_m_req_ready)
    );

    skid_buffer #(.DATA_WIDTH (72), .DEPTH (16)) u_fp4_skid (
        .clk      (clk),
        .rst_n    (rst_n),
        .push     (core_fp4_valid),
        .data_in  (core_fp4_packet),
        .pop      (fp4_skid_pop),
        .data_out (fp4_skid_dout),
        .ready    (fp4_skid_ready),
        .valid    (fp4_skid_valid),
        .level    (fp4_skid_level)
    );

    // -------------------------------------------------------------------------
    // V transpose + log-quantizer: packet source for write_out V mode
    // -------------------------------------------------------------------------
    snax_v_quant_path #(
        .NUM_LANES  (16),
        .DATA_WIDTH (72),
        .ADDR_WIDTH (32),
        .STRB_WIDTH (9)
    ) u_v_quant_path (
        .clk                    (clk),
        .rst_n                  (rst_n),
        .csr_start_i            (v_start),
        .busy_o                 (v_busy),
        .done_irq_o             (v_done),
        .cfg_base_addr_i        (v_base),
        .cfg_num_batches_i      (v_num_batches),
        .core_out_fp16_data_i   (v_core_fp16_data),
        .core_out_fp16_valid_i  (v_core_fp16_valid),
        .core_out_fp16_ready_o  (v_core_fp16_ready),
        .col_data_o             (logq_v_col_data),
        .col_valid_o            (logq_v_col_valid),
        .col_ready_i            (logq_v_col_ready),
        .pkt_data_i             (v_col_pkt_data),
        .pkt_valid_i            (v_col_pkt_valid),
        .pkt_ready_o            (v_col_pkt_ready),
        .wo_pkt_data_o          (v_wo_pkt_data),
        .wo_pkt_valid_o         (v_wo_pkt_valid),
        .wo_pkt_ready_i         (v_wo_pkt_ready)
    );

    // -------------------------------------------------------------------------
    // Shared log-quantizer: core row / V column / TCDM readback
    // -------------------------------------------------------------------------
    snax_logq_shared #(
        .NUM_LANES  (16),
        .PKT_W      (72),
        .FIFO_DEPTH (16)
    ) u_logq_shared (
        .clk             (clk),
        .rst_n           (rst_n),
        .core_fp16_data  (logq_core_fp16_data),
        .core_fp16_valid (logq_core_fp16_valid),
        .core_fp16_ready (logq_core_fp16_ready),
        .v_col_data      (logq_v_col_data),
        .v_col_valid     (logq_v_col_valid),
        .v_col_ready     (logq_v_col_ready),
        .rd_fp16_data    (logq_rd_fp16_data),
        .rd_fp16_valid   (logq_rd_fp16_valid),
        .rd_fp16_ready   (logq_rd_fp16_ready),
        .core_pkt_data   (logq_core_pkt_data),
        .core_pkt_valid  (logq_core_pkt_valid),
        .core_pkt_ready  (logq_core_pkt_ready),
        .v_pkt_data      (logq_v_pkt_data),
        .v_pkt_valid     (logq_v_pkt_valid),
        .v_pkt_ready     (logq_v_pkt_ready),
        .rd_pkt_data     (logq_rd_pkt_data),
        .rd_pkt_valid    (logq_rd_pkt_valid),
        .rd_pkt_ready    (logq_rd_pkt_ready),
        .active_src_o    (logq_active_src)
    );

    // -------------------------------------------------------------------------
    // TCDM FP16 readback stream (master 19)
    // -------------------------------------------------------------------------
    snax_stream_read_fp16 #(
        .DATA_WIDTH (72), .ADDR_WIDTH (32), .STRB_WIDTH (9)
    ) u_read_fp16 (
        .clk                   (clk),
        .rst_n                 (rst_n),
        .csr_start_i           (rd_start),
        .busy_o                (rd_busy),
        .done_irq_o            (rd_done),
        .cfg_base_addr_i       (rd_base),
        .cfg_num_vectors_i     (rd_num_vectors),
        .cfg_line_stride_i     (rd_line_stride),
        .m_tcdm_req_valid_o    (rd_m_req_valid),
        .m_tcdm_req_write_o    (rd_m_req_write),
        .m_tcdm_req_addr_o     (rd_m_req_addr),
        .m_tcdm_req_wdata_o    (rd_m_req_wdata),
        .m_tcdm_req_strb_o     (rd_m_req_strb),
        .m_tcdm_req_ready_i    (rd_m_req_ready),
        .m_tcdm_rsp_rdata_i    (rd_m_rsp_rdata),
        .m_tcdm_rsp_valid_i    (rd_m_rsp_valid),
        .fp16_data_o           (logq_rd_fp16_data),
        .fp16_valid_o          (logq_rd_fp16_valid),
        .fp16_ready_i          (logq_rd_fp16_ready)
    );

    // -------------------------------------------------------------------------
    // Core
    // -------------------------------------------------------------------------
    snax_fused_llm_core #(
        .NUM_LANES_P       (16),
        .ACC_DEPTH_P       (ACC_DEPTH),
        .NUM_QUERIES       (8),
        .PKT_W             (72),
        .FRAC_BITS         (16),
        .P_FIFO_DEPTH      (16),
        .USE_OUTPUT_READY  (1'b1)
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
        .out_vec_fp4_packet_o  (),
        .out_vec_fp4_valid_o   (),
        .p_fifo_full_o         (p_fifo_full),
        .p_fifo_empty_o        (p_fifo_empty),
        .p_fifo_almost_full_o  (p_fifo_almost_full),
        .l_ready_o             (l_ready)
    );

    // -------------------------------------------------------------------------
    // Sequencer
    // -------------------------------------------------------------------------
    snax_flash_seq #(
        .H (2), .N_Q (16), .M_Q (8), .D_K (32), .D_V (32), .KV_TILES (2)
    ) u_seq (
        .clk                (clk),
        .rst_n              (rst_n),
        .start_i            (csr_start_pulse && !csr_src_sel && !v_busy),
        .busy_o             (seq_busy),
        .done_o             (seq_done),
        .act_start_o        (seq_act_start),
        .act_base_o         (seq_act_base),
        .act_k_blocks_o     (seq_act_k_blocks),
        .act_k_stride_o     (seq_act_k_stride),
        .act_tokens_o       (seq_act_tokens),
        .act_token_stride_o (seq_act_token_stride),
        .act_busy_i         (act_busy),
        .act_valid_i        (act_stream_valid),
        .act_ready_o        (seq_act_ready),
        .wt_start_o         (seq_wt_start),
        .wt_base_o          (seq_wt_base),
        .wt_repeat_o        (seq_wt_repeat),
        .wt_k_blocks_o      (seq_wt_k_blocks),
        .wt_k_stride_o      (seq_wt_k_stride),
        .wt_num_tiles_o     (seq_wt_num_tiles),
        .wt_tile_step_o     (seq_wt_tile_step),
        .wt_busy_i          (wt_busy),
        .wt_valid_i         (wt_stream_valid),
        .wt_ready_o         (seq_wt_ready),
        .cfg_mode_o         (seq_core_mode),
        .cfg_row_start_o    (seq_core_row_start),
        .cfg_key_mask_o     (seq_core_key_mask),
        .cfg_enable_slq_o   (seq_core_enable_slq),
        .tile_drain_req_o   (seq_core_tile_drain_req),
        .tile_drain_idx_o   (seq_core_tile_drain_idx),
        .tile_clear_on_drain_o (seq_core_tile_clear),
        .lane_valid_o       (seq_core_lane_valid),
        .lane_acc_idx_o     (seq_core_lane_acc_idx),
        .p_fifo_empty_i     (p_fifo_empty),
        .core_fp16_valid_i  (core_fp16_valid),
        .wo_start_o         (seq_wo_start),
        .wo_base_o          (seq_wo_base),
        .wo_num_vectors_o   (seq_wo_num_vectors),
        .wo_line_stride_o   (seq_wo_line_stride)
    );

    // -------------------------------------------------------------------------
    // TCDM + interconnect
    // -------------------------------------------------------------------------
    multi_bank_tcdm_72b #(
        .NUM_BANKS  (NUM_BANKS),
        .DATA_WIDTH (DATA_WIDTH),
        .STRB_WIDTH (STRB_WIDTH),
        .BANK_DEPTH (BANK_DEPTH)
    ) u_tcdm (
        .clk          (clk),
        .rst_n        (rst_n),
        .bank_req_i   (b_req),
        .bank_write_i (b_write),
        .bank_addr_i  (b_addr),
        .bank_wdata_i (b_wdata),
        .bank_strb_i  (b_strb),
        .bank_rdata_o (b_rdata),
        .bank_rvalid_o(b_rvalid)
    );

    interleaved_interconnect_72b #(
        .NUM_MASTERS (NUM_MASTERS),
        .NUM_BANKS   (NUM_BANKS),
        .DATA_WIDTH  (DATA_WIDTH),
        .ADDR_WIDTH  (ADDR_WIDTH),
        .BANK_DEPTH  (BANK_DEPTH)
    ) u_xbar (
        .clk           (clk),
        .rst_n         (rst_n),
        .m_req_valid_i (m_req_valid),
        .m_req_ready_o (m_req_ready),
        .m_req_addr_i  (m_req_addr),
        .m_req_write_i (m_req_write),
        .m_req_wdata_i (m_req_wdata),
        .m_req_strb_i  (m_req_strb),
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

endmodule
