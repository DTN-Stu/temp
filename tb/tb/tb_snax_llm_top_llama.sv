// =============================================================================
// tb_snax_llm_top_llama.sv
//
// Full-system testbench for snax_llm_top_clean with REAL-WORLD LLM SHAPE:
//   H = 1, N_Q = 16, N_KV = 64, D_K = 128, D_V = 128, M_Q = 8
//   (LLaMA-2/3, Mistral-7B single-head attention shape)
//
// Stimulus & Golden produced by tb/generate_snax_llm_top_llama_data.py
// =============================================================================
`timescale 1ns/1ps

module tb_snax_llm_top_llama;

    // -------------------------------------------------------------------------
    // Problem shape: Real LLM Dimension (LLaMA Attention Head)
    // -------------------------------------------------------------------------
    localparam int H          = 1;
    localparam int N_Q        = 16;
    localparam int N_KV       = 64;
    localparam int D_K        = 128;
    localparam int D_V        = 128;
    localparam int M_Q        = 8;
    localparam int K_BLOCKS   = D_K / 16;   // 8
    localparam int Q_TILES    = N_Q / M_Q;  // 2
    localparam int DV_TILES   = D_V / 16;   // 8
    localparam int KV_TILES   = N_KV / 16;  // 4
    localparam int NUM_VEC    = H * Q_TILES * DV_TILES * M_Q;      // 128
    localparam int NUM_WORD   = NUM_VEC * 4;                       // 512
    localparam int NUM_FP16   = NUM_WORD * 4;                      // 2048
    localparam int CLK_PERIOD = 10;

    localparam int OUT_BANK   = 17;
    localparam logic [31:0] Q_BASE   = 32'h0000_0100;  // bank 16, word 0
    localparam logic [31:0] OUT_BASE = 32'h0000_0110;  // bank 17, word 0
    localparam logic [31:0] RING_BASE= 32'h0000_0000;  // bank 0,  word 0

    // -------------------------------------------------------------------------
    // CSR register indices (snax_csr_manager)
    // -------------------------------------------------------------------------
    localparam logic [11:0] A_CTRL      = 12'd0;
    localparam logic [11:0] A_WO_BASE   = 12'd13;
    localparam logic [11:0] A_WO_NV     = 12'd14;
    localparam logic [11:0] A_WO_LS     = 12'd15;
    localparam logic [11:0] A_SEQ_CTRL  = 12'd31;
    localparam logic [11:0] A_ATTN_H    = 12'd32;
    localparam logic [11:0] A_ATTN_QT   = 12'd33;
    localparam logic [11:0] A_ATTN_MQ   = 12'd34;
    localparam logic [11:0] A_ATTN_DV   = 12'd35;
    localparam logic [11:0] A_ATTN_KV   = 12'd36;
    localparam logic [11:0] A_ATTN_KB   = 12'd37;
    localparam logic [11:0] A_RING_BASE = 12'd41;
    localparam logic [11:0] A_Q_BASE    = 12'd42;
    localparam logic [11:0] A_OUT_BASE  = 12'd43;
    localparam logic [11:0] A_DRAM_KV   = 12'd44;
    localparam logic [11:0] A_TILE_ST   = 12'd45;
    localparam logic [11:0] A_START     = 12'd47;

    localparam logic [11:0] A_RO_SEQ    = 12'd48;
    localparam logic [11:0] A_RO_WO     = 12'd51;
    localparam logic [11:0] A_RO_ID     = 12'd55;

    localparam logic [31:0] CSR_ID_VALUE = 32'h534E_4146;

    // -------------------------------------------------------------------------
    // DUT signals
    // -------------------------------------------------------------------------
    logic clk, rst_n;

    logic        csr_req_valid, csr_req_ready, csr_req_write;
    logic [11:0] csr_req_addr;
    logic [31:0] csr_req_data;
    logic        csr_rsp_valid, csr_rsp_ready;
    logic [31:0] csr_rsp_data;

    logic        cpu_mem_req_valid, cpu_mem_req_ready;
    logic [31:0] cpu_mem_req_addr;
    logic        cpu_mem_req_write;
    logic [71:0] cpu_mem_req_wdata;
    logic [8:0]  cpu_mem_req_strb;
    logic        cpu_mem_rsp_valid;
    logic [71:0] cpu_mem_rsp_rdata;

    logic        dma_mem_req_valid, dma_mem_req_ready;
    logic [31:0] dma_mem_req_addr;
    logic        dma_mem_req_write;
    logic [71:0] dma_mem_req_wdata;
    logic [8:0]  dma_mem_req_strb;
    logic        dma_mem_rsp_valid;
    logic [71:0] dma_mem_rsp_rdata;

    logic        dma_desc_valid, dma_desc_ready;
    logic [1:0]  dma_desc_dir;
    logic [31:0] dma_desc_dram_addr, dma_desc_tcdm_addr, dma_desc_size;
    logic        dma_desc_done;

    logic        busy, done_irq;

    // -------------------------------------------------------------------------
    // Clock & reset
    // -------------------------------------------------------------------------
    initial begin
        clk = 1'b0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    initial begin
        rst_n = 1'b0;
        #(CLK_PERIOD * 5);
        rst_n = 1'b1;
    end

    // -------------------------------------------------------------------------
    // DUT
    // -------------------------------------------------------------------------
    snax_llm_top_clean dut (
        .clk                     (clk),
        .rst_n                   (rst_n),

        .cpu_csr_req_valid_i     (csr_req_valid),
        .cpu_csr_req_ready_o     (csr_req_ready),
        .cpu_csr_req_write_i     (csr_req_write),
        .cpu_csr_req_addr_i      (csr_req_addr),
        .cpu_csr_req_data_i      (csr_req_data),
        .cpu_csr_rsp_valid_o     (csr_rsp_valid),
        .cpu_csr_rsp_ready_i     (csr_rsp_ready),
        .cpu_csr_rsp_data_o      (csr_rsp_data),

        .cpu_mem_req_valid_i     (cpu_mem_req_valid),
        .cpu_mem_req_ready_o     (cpu_mem_req_ready),
        .cpu_mem_req_addr_i      (cpu_mem_req_addr),
        .cpu_mem_req_write_i     (cpu_mem_req_write),
        .cpu_mem_req_wdata_i     (cpu_mem_req_wdata),
        .cpu_mem_req_strb_i      (cpu_mem_req_strb),
        .cpu_mem_rsp_valid_o     (cpu_mem_rsp_valid),
        .cpu_mem_rsp_rdata_o     (cpu_mem_rsp_rdata),

        .dma_mem_req_valid_i     (dma_mem_req_valid),
        .dma_mem_req_ready_o     (dma_mem_req_ready),
        .dma_mem_req_addr_i      (dma_mem_req_addr),
        .dma_mem_req_write_i     (dma_mem_req_write),
        .dma_mem_req_wdata_i     (dma_mem_req_wdata),
        .dma_mem_req_strb_i      (dma_mem_req_strb),
        .dma_mem_rsp_valid_o     (dma_mem_rsp_valid),
        .dma_mem_rsp_rdata_o     (dma_mem_rsp_rdata),

        .dma_desc_valid_o        (dma_desc_valid),
        .dma_desc_ready_i        (dma_desc_ready),
        .dma_desc_dir_o          (dma_desc_dir),
        .dma_desc_dram_addr_o    (dma_desc_dram_addr),
        .dma_desc_tcdm_addr_o    (dma_desc_tcdm_addr),
        .dma_desc_size_o         (dma_desc_size),
        .dma_desc_done_i         (dma_desc_done),

        .subsystem_busy_o        (busy),
        .subsystem_done_irq_o    (done_irq)
    );

    // -------------------------------------------------------------------------
    // Optional waveform dump (build with -DFST_TRACE, run with --trace)
    // -------------------------------------------------------------------------
    initial begin
        if (!$test$plusargs("NO_WAVE")) begin
`ifdef FST_TRACE
            $dumpfile("tb_snax_llm_top_llama.fst");
`else
            $dumpfile("tb_snax_llm_top_llama.vcd");
`endif
            if ($test$plusargs("WAVE_DEEP"))
                $dumpvars(0, tb_snax_llm_top_llama);
            else
                $dumpvars(0, dut);
        end
    end

    // -------------------------------------------------------------------------
    // Reference data
    // -------------------------------------------------------------------------
    localparam int NUM_PRELOAD = H*N_Q*K_BLOCKS
                               + H*KV_TILES*K_BLOCKS*16
                               + H*DV_TILES*KV_TILES*16;   // 1152

    logic [31:0] pre_addr  [0:NUM_PRELOAD - 1];
    logic [71:0] pre_data  [0:NUM_PRELOAD - 1];
    logic [15:0] mem_golden[0:NUM_FP16 - 1];
    logic [71:0] exp_word  [0:NUM_WORD - 1];
    logic [71:0] out_words [0:NUM_WORD - 1];

    int  checks_pass, checks_fail;
    int  fp16_match, fp16_mismatch;
    real max_abs_err, sum_abs_err;

    function automatic logic [31:0] tcdm_addr(input int word, input int bank);
        tcdm_addr = (32'(word) << 9) | (32'(bank) << 4);
    endfunction

    function automatic real fp16_to_real(input logic [15:0] hv);
        logic       s;
        logic [4:0] e;
        logic [9:0] mf;
        real        v;
        s  = hv[15];
        e  = hv[14:10];
        mf = hv[9:0];
        if (e == 5'd0 && mf == 10'd0) return 0.0;
        if (e == 5'b11111) return s ? -999999.0 : 999999.0;
        if (e == 5'd0) v = real'(mf) / 16777216.0;
        else           v = (1.0 + real'(mf) / 1024.0) * $itor(32'd1 << e) / 32768.0;
        return s ? -v : v;
    endfunction

    task automatic check(input bit cond, input string name);
        if (cond) begin
            checks_pass = checks_pass + 1;
        end else begin
            checks_fail = checks_fail + 1;
            $display("    [FAIL] %s", name);
        end
    endtask

    // -------------------------------------------------------------------------
    // Bus driver tasks (Driven at negedge to match TCDM protocol cleanly)
    // -------------------------------------------------------------------------
    task automatic csr_write(input logic [11:0] a, input logic [31:0] d);
        begin
            @(negedge clk);
            csr_req_valid = 1'b1;
            csr_req_write = 1'b1;
            csr_req_addr  = a;
            csr_req_data  = d;
            #1;
            while (!csr_req_ready) begin
                @(negedge clk); #1;
            end
            @(posedge clk);
            @(negedge clk);
            csr_req_valid = 1'b0;
        end
    endtask

    task automatic csr_read(input logic [11:0] a, output logic [31:0] d);
        begin
            @(negedge clk);
            csr_req_valid = 1'b1;
            csr_req_write = 1'b0;
            csr_req_addr  = a;
            csr_req_data  = 32'd0;
            #1;
            while (!csr_req_ready) begin
                @(negedge clk); #1;
            end
            @(posedge clk);
            @(negedge clk);
            csr_req_valid = 1'b0;
            d = csr_rsp_data;
        end
    endtask

    task automatic dma_write(input logic [31:0] a, input logic [71:0] d);
        begin
            @(negedge clk);
            dma_mem_req_valid = 1'b1;
            dma_mem_req_write = 1'b1;
            dma_mem_req_addr  = a;
            dma_mem_req_wdata = d;
            dma_mem_req_strb  = 9'h1FF;
            #1;
            while (!dma_mem_req_ready) begin
                @(negedge clk); #1;
            end
            @(posedge clk);
            @(negedge clk);
            dma_mem_req_valid = 1'b0;
            dma_mem_req_write = 1'b0;
            dma_mem_req_strb  = 9'h000;
        end
    endtask

    task automatic dma_read(input logic [31:0] a, output logic [71:0] d);
        begin
            @(negedge clk);
            dma_mem_req_valid = 1'b1;
            dma_mem_req_write = 1'b0;
            dma_mem_req_addr  = a;
            dma_mem_req_wdata = '0;
            dma_mem_req_strb  = 9'h000;
            #1;
            while (!dma_mem_req_ready) begin
                @(negedge clk); #1;
            end
            @(posedge clk);
            @(negedge clk);
            dma_mem_req_valid = 1'b0;
            d = dma_mem_rsp_rdata;
        end
    endtask

    // -------------------------------------------------------------------------
    // Main test sequence
    // -------------------------------------------------------------------------
    int i, cycle_cnt;
    logic [31:0] rd32;
    logic [71:0] rd72;
    bit seen_done;

    initial begin
        csr_req_valid     = 1'b0;
        csr_req_write     = 1'b0;
        csr_req_addr      = '0;
        csr_req_data      = '0;
        csr_rsp_ready     = 1'b1;
        cpu_mem_req_valid = 1'b0;
        cpu_mem_req_write = 1'b0;
        cpu_mem_req_addr  = '0;
        cpu_mem_req_wdata = '0;
        cpu_mem_req_strb  = '0;
        dma_mem_req_valid = 1'b0;
        dma_mem_req_write = 1'b0;
        dma_mem_req_addr  = '0;
        dma_mem_req_wdata = '0;
        dma_mem_req_strb  = '0;
        dma_desc_ready    = 1'b1;
        dma_desc_done     = 1'b0;

        checks_pass   = 0;
        checks_fail   = 0;
        fp16_match    = 0;
        fp16_mismatch = 0;
        max_abs_err   = 0.0;
        sum_abs_err   = 0.0;

        // Load reference files (Relative to sim/ directory)
        $readmemh("sim_snax_llm_top_llama/hex/tcdm_preload_addr.hex", pre_addr);
        $readmemh("sim_snax_llm_top_llama/hex/tcdm_preload_data.hex", pre_data);
        $readmemh("sim_snax_llm_top_llama/hex/golden_core_fp16.hex",   mem_golden);
        $readmemh("sim_snax_llm_top_llama/hex/expected_out_words.hex", exp_word);

        wait (rst_n == 1'b1);
        repeat (10) @(posedge clk);

        $display("\n=======================================================");
        $display("  tb_snax_llm_top_llama: Real-World LLaMA Attention Head");
        $display("  H=%0d N_Q=%0d N_KV=%0d D_K=%0d D_V=%0d (Q_T=%0d, KV_T=%0d, DV_T=%0d, KB=%0d)",
                 H, N_Q, N_KV, D_K, D_V, Q_TILES, KV_TILES, DV_TILES, K_BLOCKS);
        $display("  Output: %0d vectors = %0d words = %0d FP16 values", NUM_VEC, NUM_WORD, NUM_FP16);
        $display("=======================================================");

        // 1. ID register check
        csr_read(A_RO_ID, rd32);
        check(rd32 == CSR_ID_VALUE, "CSR hardware ID (SNAF)");

        // 2. Preload TCDM with Q, K, V via DMA
        $display("[2] Preloading %0d TCDM words via DMA ...", NUM_PRELOAD);
        for (i = 0; i < NUM_PRELOAD; i++) begin
            dma_write(pre_addr[i], pre_data[i]);
        end

        // 3. Program sequencer via CSR
        $display("[3] Programming sequencer descriptor via CSR ...");
        csr_write(A_CTRL,      32'h0);            // src_sel=0 -> sequencer owns control
        csr_write(A_SEQ_CTRL,  32'h1);            // op_mode=01 FlashAttention, no DMA
        csr_write(A_ATTN_H,    H);
        csr_write(A_ATTN_QT,   Q_TILES);
        csr_write(A_ATTN_MQ,   M_Q);
        csr_write(A_ATTN_DV,   DV_TILES);
        csr_write(A_ATTN_KV,   KV_TILES);
        csr_write(A_ATTN_KB,   K_BLOCKS);
        csr_write(A_RING_BASE, RING_BASE);
        csr_write(A_Q_BASE,    Q_BASE);
        csr_write(A_OUT_BASE,  OUT_BASE);
        csr_write(A_DRAM_KV,   32'h0);
        csr_write(A_TILE_ST,   32'h200);
        csr_write(A_WO_BASE,   OUT_BASE);
        csr_write(A_WO_NV,     NUM_VEC);
        csr_write(A_WO_LS,     32'h200);

        // 4. Start FlashAttention
        $display("[4] Launching FlashAttention through CSR ...");
        csr_write(A_START, 32'h0000_0003);

        seen_done = 1'b0;
        cycle_cnt = 0;
        while (!seen_done && cycle_cnt < 500000) begin
            @(posedge clk);
            cycle_cnt = cycle_cnt + 1;
            if (done_irq && !busy) seen_done = 1'b1;
        end
        $display("    Completed in %0d cycles (busy=%0b done_irq=%0b)", cycle_cnt, busy, done_irq);
        check(seen_done, "Sequencer finished without timeout");

        csr_read(A_RO_SEQ, rd32);
        check(rd32[0] == 1'b0, "Sequencer idle after done");

        repeat (50) @(posedge clk);

        // 5. Read back results from Bank 17
        $display("[5] Reading back %0d result words from bank %0d ...", NUM_WORD, OUT_BANK);
        for (i = 0; i < NUM_WORD; i++) begin
            dma_read(tcdm_addr(i, OUT_BANK), out_words[i]);
        end

        // 6. Compare with Golden Reference
        $display("[6] Bit-exact verification against Golden FP16 ...");
        for (i = 0; i < NUM_WORD; i++) begin
            for (int e = 0; e < 4; e++) begin
                logic [15:0] act_fp, exp_fp;
                real act_r, exp_r, diff;
                act_fp = out_words[i][e*16 +: 16];
                exp_fp = exp_word[i][e*16 +: 16];
                act_r  = fp16_to_real(act_fp);
                exp_r  = fp16_to_real(exp_fp);
                diff   = act_r - exp_r;
                if (diff < 0.0) diff = -diff;
                if (diff > max_abs_err) max_abs_err = diff;
                sum_abs_err = sum_abs_err + diff;

                if (act_fp === exp_fp) begin
                    fp16_match = fp16_match + 1;
                end else begin
                    fp16_mismatch = fp16_mismatch + 1;
                    if (fp16_mismatch <= 10) begin
                        $display("    Mismatch word=%0d elem=%0d (idx=%0d): act=0x%04X (%f) exp=0x%04X (%f) diff=%f",
                                 i, e, i*4+e, act_fp, act_r, exp_fp, exp_r, diff);
                    end
                end
            end
        end

        $display("\n=======================================================");
        $display("  SIMULATION RESULTS: LLaMA ATTENTION REAL-SIZE");
        $display("=======================================================");
        $display("  Total FP16 Values  : %0d", NUM_FP16);
        $display("  FP16 Matches       : %0d / %0d (%.2f%%)", fp16_match, NUM_FP16, 100.0 * fp16_match / NUM_FP16);
        $display("  FP16 Mismatches    : %0d", fp16_mismatch);
        $display("  Max Absolute Error : %f", max_abs_err);
        $display("  Total Cycles       : %0d", cycle_cnt);
        $display("  Assertion Checks   : %0d Passed, %0d Failed", checks_pass, checks_fail);
        $display("=======================================================\n");

        if (fp16_mismatch == 0 && checks_fail == 0) begin
            $display(">>> ALL TESTS PASSED: 100%% BIT-EXACT MATCH FOR LLM SIZE! <<<\n");
        end else begin
            $display(">>> SIMULATION FAILED WITH %0d MISMATCHES! <<<\n", fp16_mismatch);
        end

        $finish;
    end

endmodule
