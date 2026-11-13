// =============================================================================
// tb_snax_core_flash_system.sv
//
// System-level integration smoke test:
//   TCDM (32 banks x 1024 x 72b)
//     + 19-master interleaved interconnect
//         master 0      : host preload / readback
//         master 1      : activation/Q broadcast streamer
//         masters 2..17 : 16-channel K/V weight streamer
//         master 18     : output write streamer
//     + snax_streamer_act_bcast
//     + snax_streamer_wt_16ch
//     + snax_streamer_write_out
//     + snax_fused_llm_core (8 queries, 8:8 slots)
//     + skid_buffer for FP4 output -> write streamer
//
// It runs the same loop nest as tb_snax_core_flash_tiling:
//   for h, qi, dvi, kj:  QK -> softmax -> P*V ; final drain -> 1/l -> FP16
// but Q/K/V now come from TCDM through the stream engines.
//
// Dimensions: H=2, N_Q=16, N_KV=32, D_K=32, D_V=32, M_Q=8.
// =============================================================================
`timescale 1ns/1ps
`include "pe_defines.svh"

module tb_snax_core_flash_system;

    localparam int NUM_LANES_P = 16;
    localparam int ACC_DEPTH_P = 16;
    localparam int NUM_QUERIES = 8;
    localparam int PKT_W       = 72;
    localparam int FRAC_BITS   = 16;
    localparam int CLK_PERIOD  = 10;

    localparam int H        = 2;
    localparam int N_Q      = 16;
    localparam int N_KV     = 32;
    localparam int D_K      = 32;
    localparam int D_V      = 32;
    localparam int M_Q      = 8;
    localparam int K_BLOCKS = D_K / 16; // 2
    localparam int Q_TILES  = N_Q / M_Q; // 2
    localparam int DV_TILES = D_V / 16;  // 2
    localparam int KV_TILES = N_KV / 16; // 2

    // TCDM address: word [18:9], bank [8:4], byte offset [3:0].
    // 16-byte slot per bank word; 512-byte stride to move one word.
    localparam int WORD_STRIDE = 32'h200;

    logic clk, rst_n;

    // -------------------------------------------------------------------------
    // Host master 0
    // -------------------------------------------------------------------------
    logic        host_req_valid;
    logic [31:0] host_req_addr;
    logic        host_req_write;
    logic [71:0] host_req_wdata;
    logic [8:0]  host_req_strb;
    logic        host_req_ready;
    logic        host_rsp_valid;
    logic [71:0] host_rsp_rdata;

    // -------------------------------------------------------------------------
    // Interconnect master arrays
    // -------------------------------------------------------------------------
    localparam int NUM_MASTERS = 19;
    logic [NUM_MASTERS-1:0]          m_req_valid;
    logic [NUM_MASTERS-1:0]          m_req_ready;
    logic [NUM_MASTERS-1:0][31:0]    m_req_addr;
    logic [NUM_MASTERS-1:0]          m_req_write;
    logic [NUM_MASTERS-1:0][71:0]    m_req_wdata;
    logic [NUM_MASTERS-1:0][8:0]     m_req_strb;
    logic [NUM_MASTERS-1:0]          m_rsp_valid;
    logic [NUM_MASTERS-1:0][71:0]    m_rsp_rdata;

    // -------------------------------------------------------------------------
    // TCDM bank ports
    // -------------------------------------------------------------------------
    logic [31:0]          b_req;
    logic [31:0]          b_write;
    logic [31:0][9:0]     b_addr;
    logic [31:0][71:0]    b_wdata;
    logic [31:0][8:0]     b_strb;
    logic [31:0][71:0]    b_rdata;
    logic [31:0]          b_rvalid;

    // -------------------------------------------------------------------------
    // Activation streamer
    // -------------------------------------------------------------------------
    logic        act_start, act_busy, act_done;
    logic        act_freeze;
    logic [31:0] act_freeze_cycles;
    logic [31:0] act_base, act_k_blocks, act_k_stride, act_tokens, act_token_stride;

    logic        act_m_req_valid, act_m_req_write, act_m_req_ready;
    logic [31:0] act_m_req_addr;
    logic [71:0] act_m_req_wdata;
    logic [8:0]  act_m_req_strb;
    logic [71:0] act_m_rsp_rdata;
    logic        act_m_rsp_valid;

    logic [71:0] act_stream_data;
    logic        act_stream_valid;
    logic        act_stream_ready;

    // -------------------------------------------------------------------------
    // 16-channel weight streamer
    // -------------------------------------------------------------------------
    logic        wt_start, wt_busy, wt_done;
    logic [31:0] wt_base, wt_repeat, wt_k_blocks, wt_k_stride, wt_num_tiles, wt_tile_step;

    logic [15:0]       wt_m_req_valid, wt_m_req_write, wt_m_req_ready;
    logic [15:0][31:0] wt_m_req_addr;
    logic [15:0][71:0] wt_m_req_wdata;
    logic [15:0][8:0]  wt_m_req_strb;
    logic [15:0][71:0] wt_m_rsp_rdata;
    logic [15:0]       wt_m_rsp_valid;

    logic [15:0][71:0] wt_stream_data;
    logic [15:0]       wt_stream_valid;
    logic [15:0]       wt_stream_ready;

    // -------------------------------------------------------------------------
    // Output write streamer
    // -------------------------------------------------------------------------
    logic        wo_start, wo_busy, wo_done;
    logic        wo_transpose;
    logic [31:0] wo_base, wo_num_vectors, wo_line_stride;
    logic        wo_m_req_valid, wo_m_req_write, wo_m_req_ready;
    logic [31:0] wo_m_req_addr;
    logic [71:0] wo_m_req_wdata;
    logic [8:0]  wo_m_req_strb;

    logic [255:0] core_fp16_packed;
    logic         core_fp16_valid;
    logic [71:0]  core_fp4_packet;
    logic         core_fp4_valid;
    logic         wo_fp16_ready;
    logic         wo_fp4_ready;
    logic [71:0]  wo_fp4_data;

    // FP4 skid between core and write streamer
    logic        fp4_skid_push, fp4_skid_pop;
    logic [71:0] fp4_skid_din, fp4_skid_dout;
    logic        fp4_skid_ready, fp4_skid_valid;
    logic [4:0]  fp4_skid_level;

    // -------------------------------------------------------------------------
    // Core control
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

    logic [71:0] ext_act_broadcast;
    logic        ext_act_valid;
    logic        wt_valid_all;

    logic        p_fifo_full, p_fifo_empty, p_fifo_almost_full, l_ready;

    // -------------------------------------------------------------------------
    // TB memories / bookkeeping
    // -------------------------------------------------------------------------
    logic [71:0] mem_q [0:H*N_Q*K_BLOCKS - 1];                 // 64
    logic [71:0] mem_k [0:H*KV_TILES*K_BLOCKS*16 - 1];         // 128
    logic [71:0] mem_v [0:H*DV_TILES*KV_TILES*16 - 1];         // 128
    logic [15:0] mem_golden [0:H*N_Q*D_V - 1];                 // 1024

    int h, qi, dvi, kj, kb, m;
    int out_count;
    int out_total, out_base;
    int exp_h, exp_qi, exp_dvi, exp_m;
    int dbg_wait;
    int match_count, mismatch_count;
    int mon_l, mon_q_global, mon_v_col, mon_golden_idx;
    logic [15:0] mon_actual, mon_golden;
    real mon_actual_val, mon_golden_val, mon_diff, mon_abs_diff;
    real max_abs_err, sum_abs_err, sum_sq_err;

    // -------------------------------------------------------------------------
    // DUTs
    // -------------------------------------------------------------------------
    snax_fused_llm_core #(
        .NUM_LANES_P  (NUM_LANES_P),
        .ACC_DEPTH_P  (ACC_DEPTH_P),
        .NUM_QUERIES  (NUM_QUERIES),
        .PKT_W        (PKT_W),
        .FRAC_BITS    (FRAC_BITS),
        .P_FIFO_DEPTH (16)
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
        .ext_act_broadcast_i   (ext_act_broadcast),
        .ext_act_valid_i       (ext_act_valid),
        .wt_in_i               (wt_stream_data),
        .wt_valid_i            (wt_valid_all),
        .out_vec_fp16_packed_o (core_fp16_packed),
        .out_vec_fp16_lanes_o  (),
        .out_vec_fp16_valid_o  (core_fp16_valid),
        .out_vec_fp4_packet_o  (core_fp4_packet),
        .out_vec_fp4_valid_o   (core_fp4_valid),
        .p_fifo_full_o         (p_fifo_full),
        .p_fifo_empty_o        (p_fifo_empty),
        .p_fifo_almost_full_o  (p_fifo_almost_full),
        .l_ready_o             (l_ready)
    );

    snax_streamer_act_bcast #(
        .DATA_WIDTH (72),
        .ADDR_WIDTH (32),
        .STRB_WIDTH (9)
    ) u_act_streamer (
        .clk                     (clk),
        .rst_n                   (rst_n),
        .csr_start_i             (act_start),
        .busy_o                  (act_busy),
        .done_irq_o              (act_done),
        .cfg_freeze_mode_i       (act_freeze),
        .cfg_freeze_cycles_i     (act_freeze_cycles),
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
        .DATA_WIDTH (72),
        .ADDR_WIDTH (32),
        .NUM_LANES  (16),
        .STRB_WIDTH (9)
    ) u_wt_streamer (
        .clk                   (clk),
        .rst_n                 (rst_n),
        .csr_start_i           (wt_start),
        .busy_o                (wt_busy),
        .done_irq_o            (wt_done),
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
        .DATA_WIDTH (72),
        .ADDR_WIDTH (32),
        .NUM_LANES  (16),
        .STRB_WIDTH (9)
    ) u_write_out (
        .clk                    (clk),
        .rst_n                  (rst_n),
        .csr_start_i            (wo_start),
        .busy_o                 (wo_busy),
        .done_irq_o             (wo_done),
        .cfg_mode_transpose_i   (wo_transpose),
        .cfg_base_addr_i        (wo_base),
        .cfg_num_vectors_i      (wo_num_vectors),
        .cfg_line_stride_i      (wo_line_stride),
        .core_out_fp16_data_i   (core_fp16_packed),
        .core_out_fp16_valid_i  (core_fp16_valid),
        .core_out_fp16_ready_o  (wo_fp16_ready),
        .core_out_fp4_data_i    (wo_fp4_data),
        .core_out_fp4_valid_i   (fp4_skid_valid),
        .core_out_fp4_ready_o   (wo_fp4_ready),
        .m_tcdm_req_valid_o     (wo_m_req_valid),
        .m_tcdm_req_write_o     (wo_m_req_write),
        .m_tcdm_req_addr_o      (wo_m_req_addr),
        .m_tcdm_req_wdata_o     (wo_m_req_wdata),
        .m_tcdm_req_strb_o      (wo_m_req_strb),
        .m_tcdm_req_ready_i     (wo_m_req_ready)
    );

    skid_buffer #(
        .DATA_WIDTH (72),
        .DEPTH      (16)
    ) u_fp4_skid (
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

    assign fp4_skid_pop  = wo_fp4_ready;
    assign wo_fp4_data   = fp4_skid_dout;

    multi_bank_tcdm_72b #(
        .NUM_BANKS  (32),
        .DATA_WIDTH (72),
        .STRB_WIDTH (9),
        .BANK_DEPTH (1024)
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
        .NUM_BANKS   (32),
        .DATA_WIDTH  (72),
        .ADDR_WIDTH  (32),
        .BANK_DEPTH  (1024)
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

    // -------------------------------------------------------------------------
    // Interconnect wiring
    // -------------------------------------------------------------------------
    assign m_req_valid[0]  = host_req_valid;
    assign m_req_addr[0]   = host_req_addr;
    assign m_req_write[0]  = host_req_write;
    assign m_req_wdata[0]  = host_req_wdata;
    assign m_req_strb[0]   = host_req_strb;
    assign host_req_ready  = m_req_ready[0];
    assign host_rsp_valid  = m_rsp_valid[0];
    assign host_rsp_rdata  = m_rsp_rdata[0];

    assign m_req_valid[1]  = act_m_req_valid;
    assign m_req_addr[1]   = act_m_req_addr;
    assign m_req_write[1]  = act_m_req_write;
    assign m_req_wdata[1]  = act_m_req_wdata;
    assign m_req_strb[1]   = act_m_req_strb;
    assign act_m_req_ready = m_req_ready[1];
    assign act_m_rsp_valid = m_rsp_valid[1];
    assign act_m_rsp_rdata = m_rsp_rdata[1];

    genvar gw;
    generate
        for (gw = 0; gw < 16; gw++) begin : GEN_WT_XBAR
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

    assign m_req_valid[18] = wo_m_req_valid;
    assign m_req_addr[18]  = wo_m_req_addr;
    assign m_req_write[18] = wo_m_req_write;
    assign m_req_wdata[18] = wo_m_req_wdata;
    assign m_req_strb[18]  = wo_m_req_strb;
    assign wo_m_req_ready  = m_req_ready[18];

    // Core stream connections
    assign ext_act_broadcast = act_stream_data;
    assign ext_act_valid     = act_stream_valid;
    assign wt_valid_all      = &wt_stream_valid;

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    function automatic [31:0] tcdm_addr(input int word, input int bank);
        tcdm_addr = (word << 9) | (bank << 4);
    endfunction

    function automatic [31:0] q_word(input int hh, input int qq, input int kkb);
        q_word = hh * (N_Q*K_BLOCKS) + qq * K_BLOCKS + kkb;
    endfunction

    function automatic [31:0] k_word(input int hh, input int kkj, input int kkb);
        k_word = hh * (KV_TILES*K_BLOCKS) + kkj * K_BLOCKS + kkb;
    endfunction

    function automatic [31:0] v_word(input int hh, input int ddvi, input int kkj);
        v_word = 16 + hh * (DV_TILES*KV_TILES) + ddvi * KV_TILES + kkj;
    endfunction

    task automatic wait_cycles(input int n);
        repeat (n) @(posedge clk);
    endtask

    task automatic host_write(input [31:0] a, input [71:0] d);
    begin
        @(negedge clk);
        host_req_valid = 1'b1;
        host_req_addr  = a;
        host_req_write = 1'b1;
        host_req_wdata = d;
        host_req_strb  = 9'h1FF;
        while (!host_req_ready) @(negedge clk);
        @(posedge clk);
        @(negedge clk);
        host_req_valid = 1'b0;
        host_req_write = 1'b0;
        host_req_strb  = 9'h000;
    end
    endtask

    task automatic host_read(input [31:0] a, output [71:0] d);
    begin
        @(negedge clk);
        host_req_valid = 1'b1;
        host_req_addr  = a;
        host_req_write = 1'b0;
        host_req_wdata = '0;
        host_req_strb  = 9'h000;
        while (!host_req_ready) @(negedge clk);
        @(posedge clk);
        @(negedge clk);
        host_req_valid = 1'b0;
        while (!host_rsp_valid) @(negedge clk);
        d = host_rsp_rdata;
    end
    endtask

    task automatic start_act(input [31:0] base, input int kblocks, input int kstride,
                             input int tokens, input int tstride);
    begin
        @(negedge clk);
        act_freeze          = 1'b0;
        act_freeze_cycles   = 32'd0;
        act_base            = base;
        act_k_blocks        = kblocks;
        act_k_stride        = kstride;
        act_tokens          = tokens;
        act_token_stride    = tstride;
        act_start           = 1'b1;
        @(posedge clk);
        @(negedge clk);
        act_start           = 1'b0;
        $display("DBG start_act base=0x%08X kb=%0d ks=%0d tok=%0d ts=0x%08X busy=%b state=%0d",
                 base, kblocks, kstride, tokens, tstride, act_busy,
                 u_act_streamer.state_q);
    end
    endtask

    task automatic start_wt(input [31:0] base, input int rep, input int kblocks,
                            input int kstride, input int ntiles, input int tstep);
    begin
        @(negedge clk);
        wt_base       = base;
        wt_repeat     = rep;
        wt_k_blocks   = kblocks;
        wt_k_stride   = kstride;
        wt_num_tiles  = ntiles;
        wt_tile_step  = tstep;
        wt_start      = 1'b1;
        @(posedge clk);
        @(negedge clk);
        wt_start      = 1'b0;
    end
    endtask

    task automatic start_wo(input [31:0] base, input int num_vectors, input int stride);
    begin
        @(negedge clk);
        wo_transpose   = 1'b0; // linear FP4 write mode
        wo_base        = base;
        wo_num_vectors = num_vectors;
        wo_line_stride = stride;
        wo_start       = 1'b1;
        @(posedge clk);
        @(negedge clk);
        wo_start       = 1'b0;
    end
    endtask

    task automatic qk_fire(input [3:0] slot);
    begin
        @(negedge clk);
        dbg_wait = 0;
        while (!(act_stream_valid && (&wt_stream_valid))) begin
            @(negedge clk);
            dbg_wait = dbg_wait + 1;
            if ((dbg_wait <= 10) || ((dbg_wait % 500) == 0)) begin
                $display("DBG QK wait=%0d act_v=%b act_busy=%b act_state=%0d act_req=%b act_rdy=%b act_rsp=%b skid_v=%b skid_r=%b lvl=%0d rspd=%h | wt_v=%h wt_busy=%b wt_state=%0d rep=%b wt_req0=%b wt_rdy0=%b wt_rsp0=%b",
                         dbg_wait, act_stream_valid, act_busy, u_act_streamer.state_q,
                         u_act_streamer.m_tcdm_req_valid_o,
                         u_act_streamer.m_tcdm_req_ready_i,
                         u_act_streamer.m_tcdm_rsp_valid_i,
                         u_act_streamer.skid_valid, u_act_streamer.skid_ready,
                         u_act_streamer.skid_level, u_act_streamer.m_tcdm_rsp_rdata_i,
                         wt_stream_valid, wt_busy, u_wt_streamer.state_q,
                         u_wt_streamer.replay_active_q,
                         u_wt_streamer.m_tcdm_req_valid_o[0],
                         u_wt_streamer.m_tcdm_req_ready_i[0],
                         u_wt_streamer.m_tcdm_rsp_valid_i[0]);
            end
        end
        cfg_mode         = 2'b01;
        lane_acc_idx     = slot;
        lane_valid       = 16'hFFFF;
        act_stream_ready = 1'b1;
        wt_stream_ready  = 16'hFFFF;
        @(posedge clk);
        @(negedge clk);
        lane_valid       = 16'h0000;
        act_stream_ready = 1'b0;
        wt_stream_ready  = 16'h0000;
    end
    endtask

    task automatic pv_fire(input [3:0] slot);
    begin
        @(negedge clk);
        while (!((&wt_stream_valid) && !p_fifo_empty)) @(negedge clk);
        cfg_mode        = 2'b10;
        lane_acc_idx    = slot;
        lane_valid      = 16'hFFFF;
        wt_stream_ready = 16'hFFFF;
        act_stream_ready = 1'b0;
        @(posedge clk);
        @(negedge clk);
        lane_valid      = 16'h0000;
        wt_stream_ready = 16'h0000;
    end
    endtask

    task automatic drain_s(input [3:0] idx, input bit row_start);
    begin
        @(negedge clk);
        tile_drain_req      = 1'b1;
        tile_drain_idx      = idx;
        tile_clear_on_drain = 1'b1;
        @(posedge clk);
        @(negedge clk);
        tile_drain_req      = 1'b0;
        tile_clear_on_drain = 1'b0;

        if (row_start) begin
            cfg_row_start = 1'b1;
            @(posedge clk);
            @(negedge clk);
            cfg_row_start = 1'b0;
        end
    end
    endtask

    task automatic drain_o_one(input [3:0] idx);
    begin
        @(negedge clk);
        tile_drain_req      = 1'b1;
        tile_drain_idx      = idx;
        tile_clear_on_drain = 1'b1;
        @(posedge clk);
    end
    endtask

    task automatic wait_act_idle;
    begin
        while (act_busy) @(negedge clk);
    end
    endtask

    task automatic wait_wt_idle;
    begin
        while (wt_busy) @(negedge clk);
    end
    endtask

    // -------------------------------------------------------------------------
    // Output monitor: final_normalizer output is 16 x FP16 packed per query.
    // -------------------------------------------------------------------------
    always @(posedge clk) begin
        if (rst_n && core_fp16_valid) begin
            // Output order is fixed by the loop nest:
            //   h -> qi -> dvi -> query m (0..7), 16 lanes each.
            // Derive the expected pass from a global counter instead of the
            // live loop variables, which may already have advanced.
            exp_m   = out_total % M_Q;
            exp_dvi = (out_total / M_Q) % DV_TILES;
            exp_qi  = (out_total / (M_Q*DV_TILES)) % Q_TILES;
            exp_h   = out_total / (M_Q*DV_TILES*Q_TILES);

            mon_q_global = exp_qi * M_Q + exp_m;
            for (mon_l = 0; mon_l < 16; mon_l = mon_l + 1) begin
                mon_v_col      = exp_dvi * 16 + mon_l;
                mon_golden_idx = (exp_h * N_Q + mon_q_global) * D_V + mon_v_col;
                mon_actual     = core_fp16_packed[mon_l*16 +: 16];
                mon_golden     = mem_golden[mon_golden_idx];

                mon_actual_val = fp16_to_real(mon_actual);
                mon_golden_val = fp16_to_real(mon_golden);
                mon_diff       = mon_actual_val - mon_golden_val;
                mon_abs_diff   = (mon_diff < 0.0) ? -mon_diff : mon_diff;
                sum_abs_err = sum_abs_err + mon_abs_diff;
                sum_sq_err  = sum_sq_err + (mon_diff * mon_diff);
                if (mon_abs_diff > max_abs_err) max_abs_err = mon_abs_diff;

                if (mon_actual == mon_golden) begin
                    match_count = match_count + 1;
                end else begin
                    mismatch_count = mismatch_count + 1;
                    $display("  [MISMATCH] h=%0d qi=%0d dvi=%0d q=%0d lane=%0d act=0x%04X gold=0x%04X",
                             exp_h, exp_qi, exp_dvi, exp_m, mon_l, mon_actual, mon_golden);
                end
            end
            out_total = out_total + 1;
        end
    end

    function automatic real fp16_to_real(input logic [15:0] hv);
        logic s;
        logic [4:0] e;
        logic [9:0] mf;
        real v;
        begin
            s = hv[15];
            e = hv[14:10];
            mf = hv[9:0];
            if (e == 5'd0 && mf == 10'd0) return 0.0;
            if (e == 5'b11111) return s ? -999999.0 : 999999.0;
            if (e == 5'd0) v = real'(mf) / 16777216.0;
            else v = (1.0 + real'(mf) / 1024.0) * $itor(32'd1 << e) / 32768.0;
            return s ? -v : v;
        end
    endfunction

    // -------------------------------------------------------------------------
    // Main sequence
    // -------------------------------------------------------------------------
    initial begin
        $display("\n========================================================================");
        $display("   SNAX FLASH SYSTEM: CORE + ACT STREAMER + WT STREAMER + TCDM + OUT  ");
        $display("   H=%0d Q_TILES=%0d DV_TILES=%0d KV_TILES=%0d M_Q=%0d D_K=%0d D_V=%0d",
                 H, Q_TILES, DV_TILES, KV_TILES, M_Q, D_K, D_V);
        $display("========================================================================\n");

        $readmemh("sim_snax_core_flash/hex/q_packets.hex", mem_q);
        $readmemh("sim_snax_core_flash/hex/k_packets.hex", mem_k);
        $readmemh("sim_snax_core_flash/hex/v_packets.hex", mem_v);
        $readmemh("sim_snax_core_flash/hex/golden_core_fp16.hex", mem_golden);

        rst_n = 1'b0;
        cfg_mode = 2'b00;
        cfg_row_start = 1'b0;
        cfg_key_mask = 16'hFFFF;
        cfg_enable_slq = 1'b1;
        tile_drain_req = 1'b0;
        tile_drain_idx = 4'd0;
        tile_clear_on_drain = 1'b0;
        lane_valid = 16'h0000;
        lane_acc_idx = 4'd0;
        host_req_valid = 1'b0;
        host_req_addr = 32'd0;
        host_req_write = 1'b0;
        host_req_wdata = 72'd0;
        host_req_strb = 9'h000;
        act_start = 1'b0; act_freeze = 1'b0; act_freeze_cycles = 32'd0;
        act_base = 32'd0; act_k_blocks = 32'd0; act_k_stride = 32'd0;
        act_tokens = 32'd0; act_token_stride = 32'd0; act_stream_ready = 1'b0;
        wt_start = 1'b0; wt_base = 32'd0; wt_repeat = 32'd0; wt_k_blocks = 32'd0;
        wt_k_stride = 32'd0; wt_num_tiles = 32'd0; wt_tile_step = 32'd0;
        wt_stream_ready = 16'h0000;
        wo_start = 1'b0; wo_transpose = 1'b0; wo_base = 32'd0;
        wo_num_vectors = 32'd0; wo_line_stride = 32'd0;

        match_count = 0; mismatch_count = 0; out_count = 0;
        out_total = 0; out_base = 0;
        max_abs_err = 0.0; sum_abs_err = 0.0; sum_sq_err = 0.0;

        wait_cycles(5);
        rst_n = 1'b1;
        wait_cycles(5);

        // ---------------------------------------------------------------------
        // Preload Q/K/V into TCDM
        // Q: bank 16, word = h*32 + q*2 + kb
        // K: bank 0..15, word = h*4 + kj*2 + kb
        // V: bank 0..15, word = 16 + h*4 + dvi*2 + kj
        // ---------------------------------------------------------------------
        $display("[%0t ns] Preloading Q/K/V into TCDM...", $time);
        for (h = 0; h < H; h = h + 1) begin
            for (int q = 0; q < N_Q; q = q + 1) begin
                for (int kkb = 0; kkb < K_BLOCKS; kkb = kkb + 1) begin
                    host_write(tcdm_addr(q_word(h, q, kkb), 16),
                               mem_q[h*(N_Q*K_BLOCKS) + q*K_BLOCKS + kkb]);
                end
            end
        end
        for (h = 0; h < H; h = h + 1) begin
            for (int kkj = 0; kkj < KV_TILES; kkj = kkj + 1) begin
                for (int kkb = 0; kkb < K_BLOCKS; kkb = kkb + 1) begin
                    for (int lane = 0; lane < 16; lane = lane + 1) begin
                        host_write(tcdm_addr(k_word(h, kkj, kkb), lane),
                                   mem_k[h*(KV_TILES*K_BLOCKS*16) + kkj*(K_BLOCKS*16) + kkb*16 + lane]);
                    end
                end
            end
        end
        for (h = 0; h < H; h = h + 1) begin
            for (int ddvi = 0; ddvi < DV_TILES; ddvi = ddvi + 1) begin
                for (int kkj = 0; kkj < KV_TILES; kkj = kkj + 1) begin
                    for (int lane = 0; lane < 16; lane = lane + 1) begin
                        host_write(tcdm_addr(v_word(h, ddvi, kkj), lane),
                                   mem_v[h*(DV_TILES*KV_TILES*16) + ddvi*(KV_TILES*16) + kkj*16 + lane]);
                    end
                end
            end
        end
        wait_cycles(10);
        $display("[%0t ns] Preload complete.", $time);

        // Output streamer: 8 final drains x 8 queries = 64 FP4 packets.
        start_wo(tcdm_addr(0, 17), H*Q_TILES*DV_TILES*M_Q, WORD_STRIDE);

        // ---------------------------------------------------------------------
        // Main loop nest
        // ---------------------------------------------------------------------
        for (h = 0; h < H; h = h + 1) begin
            for (qi = 0; qi < Q_TILES; qi = qi + 1) begin
                for (dvi = 0; dvi < DV_TILES; dvi = dvi + 1) begin
                    $display("[%0t ns] h=%0d qi=%0d dvi=%0d", $time, h, qi, dvi);
                    for (kj = 0; kj < KV_TILES; kj = kj + 1) begin
                        // ---------------- Mode 01: QK ----------------
                        start_wt(tcdm_addr(k_word(h, kj, 0), 0),
                                 8, K_BLOCKS, WORD_STRIDE, 1, 0);
                        for (kb = 0; kb < K_BLOCKS; kb = kb + 1) begin
                            start_act(tcdm_addr(q_word(h, qi*M_Q, kb), 16),
                                      1, 0, M_Q, 2*WORD_STRIDE);
                            for (m = 0; m < M_Q; m = m + 1) begin
                                qk_fire(4'd8 + m[3:0]);
                            end
                            wait_act_idle();
                        end
                        wait_wt_idle();

                        // ---------------- softmax / S drain ----------------
                        for (m = 0; m < M_Q; m = m + 1) begin
                            drain_s(4'd8 + m[3:0], (kj == 0));
                        end
                        wait_cycles(4);

                        // ---------------- Mode 10: P*V ----------------
                        start_wt(tcdm_addr(v_word(h, dvi, kj), 0),
                                 8, 1, 0, 1, 0);
                        for (m = 0; m < M_Q; m = m + 1) begin
                            pv_fire(4'd0 + m[3:0]);
                        end
                        wait_wt_idle();
                        wait_cycles(3);
                    end

                    // ---------------- final drain / normalize ----------------
                    out_base = out_total;
                    for (m = 0; m < M_Q; m = m + 1) begin
                        drain_o_one(4'd0 + m[3:0]);
                    end
                    @(negedge clk);
                    tile_drain_req      = 1'b0;
                    tile_clear_on_drain = 1'b0;
                    while ((out_total - out_base) < M_Q) @(negedge clk);
                    wait_cycles(3);
                end
            end
        end

        // Wait for the output streamer to flush all FP4 packets.
        while (wo_busy) @(negedge clk);
        wait_cycles(5);

        // Read back two FP4 packets from bank 17 as a write-path smoke check.
        begin
            logic [71:0] rd0, rd1;
            host_read(tcdm_addr(0, 17), rd0);
            host_read(tcdm_addr(1, 17), rd1);
            $display("[%0t ns] write_out readback: [0]=0x%018X [1]=0x%018X", $time, rd0, rd1);
        end

        $display("\n========================================================================");
        $display("              SNAX FLASH SYSTEM INTEGRATION RESULT                      ");
        $display("========================================================================");
        $display(" Total outputs checked : %0d", match_count + mismatch_count);
        $display(" BIT-EXACT matches     : %0d", match_count);
        $display(" Mismatches            : %0d", mismatch_count);
        $display(" Max |Error|           : %0.7f", max_abs_err);
        $display(" Mean |Error|          : %0.7f", sum_abs_err / (match_count + mismatch_count));
        $display(" RMSE                  : %0.7f", $sqrt(sum_sq_err / (match_count + mismatch_count)));
        if (mismatch_count == 0 && match_count == H*Q_TILES*DV_TILES*M_Q*16)
            $display("\n >>> [PASS] SYSTEM-LEVEL FLASH TILING BIT-EXACT! <<<");
        else
            $display("\n >>> [FAIL] SYSTEM MISMATCH OR TIMEOUT <<<");
        $display("========================================================================\n");
        $finish;
    end

    initial begin
        #(CLK_PERIOD * 2000000);
        $display("\n[ERROR] System TB watchdog timeout!");
        $finish;
    end

endmodule
