// =============================================================================
// tb_snax_llm_top_flash.sv
//
// Top-level TB for snax_llm_top_flash.
// The TB acts as the host:
//   1. preload Q/K/V packets into TCDM through host DMA
//   2. write CSR start
//   3. wait for done, monitor the debug FP16 output and compare 1024 values
//   4. read back FP4 output packets from TCDM bank 17
// =============================================================================
`timescale 1ns/1ps

module tb_snax_llm_top_flash;

    localparam int H        = 2;
    localparam int N_Q      = 16;
    localparam int N_KV     = 32;
    localparam int D_K      = 32;
    localparam int D_V      = 32;
    localparam int M_Q      = 8;
    localparam int K_BLOCKS = D_K / 16;
    localparam int Q_TILES  = N_Q / M_Q;
    localparam int DV_TILES = D_V / 16;
    localparam int KV_TILES = N_KV / 16;
    localparam int CLK_PERIOD = 10;
    localparam int WORD_STRIDE = 32'h200;

    logic clk, rst_n;

    logic        csr_we;
    logic [11:0] csr_addr;
    logic [31:0] csr_wdata;
    logic [31:0] csr_rdata;

    logic        host_req_valid, host_req_ready;
    logic [31:0] host_req_addr;
    logic        host_req_write;
    logic [71:0] host_req_wdata;
    logic [8:0]  host_req_strb;
    logic        host_rsp_valid;
    logic [71:0] host_rsp_rdata;

    logic        busy, done_irq;
    logic [255:0] dbg_fp16;
    logic         dbg_fp16_valid;

    logic [71:0] mem_q [0:H*N_Q*K_BLOCKS - 1];
    logic [71:0] mem_k [0:H*KV_TILES*K_BLOCKS*16 - 1];
    logic [71:0] mem_v [0:H*DV_TILES*KV_TILES*16 - 1];
    logic [15:0] mem_golden [0:H*N_Q*D_V - 1];

    int out_total, match_count, mismatch_count;
    int dbg_init = 0;
    int mon_l, exp_h, exp_qi, exp_dvi, exp_m;
    int mon_q_global, mon_v_col, mon_golden_idx;
    logic [15:0] mon_actual, mon_golden;
    real mon_actual_val, mon_golden_val, mon_diff, mon_abs_diff;
    real max_abs_err, sum_abs_err, sum_sq_err;
    int t, c, w, k, lane;

    snax_llm_top_flash dut (
        .clk                    (clk),
        .rst_n                  (rst_n),
        .host_csr_we_i          (csr_we),
        .host_csr_addr_i        (csr_addr),
        .host_csr_wdata_i       (csr_wdata),
        .host_csr_rdata_o       (csr_rdata),
        .host_mem_req_valid_i   (host_req_valid),
        .host_mem_req_ready_o   (host_req_ready),
        .host_mem_req_addr_i    (host_req_addr),
        .host_mem_req_write_i   (host_req_write),
        .host_mem_req_wdata_i   (host_req_wdata),
        .host_mem_req_strb_i    (host_req_strb),
        .host_mem_rsp_valid_o   (host_rsp_valid),
        .host_mem_rsp_rdata_o   (host_rsp_rdata),
        .subsystem_busy_o       (busy),
        .subsystem_done_irq_o   (done_irq),
        .dbg_fp16_packed_o      (dbg_fp16),
        .dbg_fp16_valid_o       (dbg_fp16_valid)
    );

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

    task automatic csr_write(input [11:0] a, input [31:0] d);
    begin
        @(negedge clk);
        csr_we = 1'b1; csr_addr = a; csr_wdata = d;
        @(posedge clk);
        @(negedge clk);
        csr_we = 1'b0;
    end
    endtask

    task automatic csr_read(input [11:0] a, output [31:0] d);
    begin
        @(negedge clk);
        csr_we = 1'b0; csr_addr = a;
        #1 d = csr_rdata;
        @(negedge clk);
    end
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

    int dbg_cnt;
    always @(posedge clk) begin
        if (rst_n && busy) begin
            dbg_cnt = dbg_cnt + 1;
            if ((dbg_cnt % 200) == 0) begin
                $display("DBG top cyc=%0d seq=%0d h=%0d qi=%0d dvi=%0d kj=%0d kb=%0d m=%0d act_busy=%b act_v=%b wt_busy=%b wt_v=%h p_empty=%b lane=%h",
                         dbg_cnt, dut.u_seq.state_q, dut.u_seq.h, dut.u_seq.qi,
                         dut.u_seq.dvi, dut.u_seq.kj, dut.u_seq.kb, dut.u_seq.m,
                         dut.act_busy, dut.act_stream_valid, dut.wt_busy,
                         dut.wt_stream_valid, dut.p_fifo_empty, dut.lane_valid);
                $display("DBG act state=%0d tok=%0d k=%0d off=%0d skidv=%b skidr=%b lvl=%0d rdy=%b | seq_actrdy=%b",
                         dut.u_act_streamer.state_q, dut.u_act_streamer.tok_cnt_q,
                         dut.u_act_streamer.k_cnt_q, dut.u_act_streamer.outstanding_q,
                         dut.u_act_streamer.skid_valid, dut.u_act_streamer.skid_ready,
                         dut.u_act_streamer.count, dut.act_stream_ready,
                         dut.u_seq.act_ready_o);
                $display("DBG wt state=%0d rep=%0d kb=%0d tile=%0d replay=%b skidv=%h skidr=%h popg=%b poplast=%b rdy=%h | seq_wtrdy=%h",
                         dut.u_wt_streamer.state_q, dut.u_wt_streamer.rep_cnt_q,
                         dut.u_wt_streamer.kb_cnt_q, dut.u_wt_streamer.tile_cnt_q,
                         dut.u_wt_streamer.replay_active_q, dut.u_wt_streamer.skid_valid,
                         dut.u_wt_streamer.skid_ready, dut.u_wt_streamer.core_pop_group,
                         dut.u_wt_streamer.core_pop_last, dut.wt_stream_ready,
                         dut.u_seq.wt_ready_o);
            end
        end
    end

    always @(posedge clk) begin
        if (rst_n && dut.wt_start) begin
            $display("DBG WTSTART seq=%0d base=%h kb_out=%0d rep=%0d wt_state=%0d wt_cfg_kb=%0d wt_busy=%b",
                     dut.u_seq.state_q, dut.wt_base, dut.wt_k_blocks, dut.wt_repeat,
                     dut.u_wt_streamer.state_q, dut.u_wt_streamer.cfg_k_blocks_i,
                     dut.wt_busy);
        end
        if (rst_n && (dut.u_seq.state_q == 15 || dut.u_seq.state_q == 16)) begin
            $display("DBG SEQCFG st=%0d h=%0d qi=%0d dvi=%0d kj=%0d kb=%0d m=%0d wt_start=%b wt_base=%h wt_kb=%0d wt_rep=%0d | wt state=%0d busy=%b kb=%0d rep=%0d replay=%b skidv=%h",
                     dut.u_seq.state_q, dut.u_seq.h, dut.u_seq.qi, dut.u_seq.dvi,
                     dut.u_seq.kj, dut.u_seq.kb, dut.u_seq.m, dut.wt_start,
                     dut.wt_base, dut.wt_k_blocks, dut.wt_repeat,
                     dut.u_wt_streamer.state_q, dut.wt_busy,
                     dut.u_wt_streamer.kb_cnt_q, dut.u_wt_streamer.rep_cnt_q,
                     dut.u_wt_streamer.replay_active_q, dut.u_wt_streamer.skid_valid);
        end
    end

    // -------------------------------------------------------------------------
    // Top debug output monitor
    // -------------------------------------------------------------------------
    always @(posedge clk) begin
        if (rst_n && dbg_fp16_valid) begin
            exp_m   = out_total % M_Q;
            exp_dvi = (out_total / M_Q) % DV_TILES;
            exp_qi  = (out_total / (M_Q*DV_TILES)) % Q_TILES;
            exp_h   = out_total / (M_Q*DV_TILES*Q_TILES);

            mon_q_global = exp_qi * M_Q + exp_m;
            for (mon_l = 0; mon_l < 16; mon_l = mon_l + 1) begin
                mon_v_col      = exp_dvi * 16 + mon_l;
                mon_golden_idx = (exp_h * N_Q + mon_q_global) * D_V + mon_v_col;
                mon_actual     = dbg_fp16[mon_l*16 +: 16];
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

    initial begin
        $display("\n========================================================================");
        $display("   TOP-LEVEL SNAX FP4 FLASH SUBSYSTEM TEST                               ");
        $display("   host DMA preload + CSR start + streamers + TCDM + core + write_out   ");
        $display("========================================================================\n");

        $readmemh("sim_snax_core_flash/hex/q_packets.hex", mem_q);
        $readmemh("sim_snax_core_flash/hex/k_packets.hex", mem_k);
        $readmemh("sim_snax_core_flash/hex/v_packets.hex", mem_v);
        $readmemh("sim_snax_core_flash/hex/golden_core_fp16.hex", mem_golden);

        rst_n = 1'b0;
        csr_we = 1'b0; csr_addr = 12'h0; csr_wdata = 32'h0;
        host_req_valid = 1'b0; host_req_addr = 32'h0; host_req_write = 1'b0;
        host_req_wdata = 72'h0; host_req_strb = 9'h0;
        match_count = 0; mismatch_count = 0; out_total = 0; dbg_cnt = 0;
        max_abs_err = 0.0; sum_abs_err = 0.0; sum_sq_err = 0.0;

        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        repeat (5) @(posedge clk);

        // ---------------------------------------------------------------------
        // Preload Q/K/V into TCDM
        // ---------------------------------------------------------------------
        $display("[%0t ns] Host DMA preload Q/K/V...", $time);
        for (int hh = 0; hh < H; hh = hh + 1) begin
            for (int qq = 0; qq < N_Q; qq = qq + 1) begin
                for (int kkb = 0; kkb < K_BLOCKS; kkb = kkb + 1) begin
                    host_write(tcdm_addr(q_word(hh, qq, kkb), 16),
                               mem_q[hh*(N_Q*K_BLOCKS) + qq*K_BLOCKS + kkb]);
                end
            end
        end
        for (int hh = 0; hh < H; hh = hh + 1) begin
            for (int kkj = 0; kkj < KV_TILES; kkj = kkj + 1) begin
                for (int kkb = 0; kkb < K_BLOCKS; kkb = kkb + 1) begin
                    for (int ll = 0; ll < 16; ll = ll + 1) begin
                        host_write(tcdm_addr(k_word(hh, kkj, kkb), ll),
                                   mem_k[hh*(KV_TILES*K_BLOCKS*16) + kkj*(K_BLOCKS*16) + kkb*16 + ll]);
                    end
                end
            end
        end
        for (int hh = 0; hh < H; hh = hh + 1) begin
            for (int ddvi = 0; ddvi < DV_TILES; ddvi = ddvi + 1) begin
                for (int kkj = 0; kkj < KV_TILES; kkj = kkj + 1) begin
                    for (int ll = 0; ll < 16; ll = ll + 1) begin
                        host_write(tcdm_addr(v_word(hh, ddvi, kkj), ll),
                                   mem_v[hh*(DV_TILES*KV_TILES*16) + ddvi*(KV_TILES*16) + kkj*16 + ll]);
                    end
                end
            end
        end
        repeat (10) @(posedge clk);
        $display("[%0t ns] Preload complete.", $time);

        // ---------------------------------------------------------------------
        // Start the hardware sequencer through CSR
        // ---------------------------------------------------------------------
        csr_write(12'h000, 32'h0000_0001);
        $display("[%0t ns] CSR start written; waiting for done...", $time);

        while (!done_irq && !(out_total >= H*Q_TILES*DV_TILES*M_Q)) @(negedge clk);
        // Let the last FP4 packets drain through the output streamer.
        repeat (200) @(posedge clk);

        begin
            logic [71:0] rd0, rd1;
            logic [31:0] status;
            csr_read(12'h004, status);
            host_read(tcdm_addr(0, 17), rd0);
            host_read(tcdm_addr(1, 17), rd1);
            $display("[%0t ns] CSR status=0x%08X busy=%0b done=%0b", $time, status, busy, done_irq);
            $display("[%0t ns] FP4 write_out readback: [0]=0x%018X [1]=0x%018X", $time, rd0, rd1);
        end

        $display("\n========================================================================");
        $display("              TOP-LEVEL SNAX FLASH TEST RESULT                          ");
        $display("========================================================================");
        $display(" Total outputs checked : %0d", match_count + mismatch_count);
        $display(" BIT-EXACT matches     : %0d", match_count);
        $display(" Mismatches            : %0d", mismatch_count);
        $display(" Max |Error|           : %0.7f", max_abs_err);
        $display(" Mean |Error|          : %0.7f", sum_abs_err / (match_count + mismatch_count));
        $display(" RMSE                  : %0.7f", $sqrt(sum_sq_err / (match_count + mismatch_count)));
        if (mismatch_count == 0 && match_count == H*Q_TILES*DV_TILES*M_Q*16)
            $display("\n >>> [PASS] TOP-LEVEL SNAX FP4 FLASH BIT-EXACT! <<<");
        else
            $display("\n >>> [FAIL] TOP-LEVEL SYSTEM MISMATCH/TIMEOUT <<<");
        $display("========================================================================\n");
        $finish;
    end

    initial begin
        #(CLK_PERIOD * 2000000);
        $display("\n[ERROR] Top-level TB watchdog timeout!");
        $finish;
    end

endmodule
