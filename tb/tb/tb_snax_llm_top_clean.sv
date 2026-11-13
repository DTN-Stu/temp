// =============================================================================
// tb_snax_llm_top_clean.sv
//
// Full-system testbench for snax_llm_top_clean (21-master variant).
//
// The TB plays three roles simultaneously, matching the block diagram:
//   * RV32I  : CSR request/response driver + TCDM master 1 driver
//   * DMA    : TCDM master 0 driver (preload Q/K/V, readback results)
//   * host   : waits for subsystem_done_irq_o
//
// Test sequence
//   1. ID register check
//   2. TCDM reachability from DMA (master 0)
//   3. TCDM reachability from RV32I (master 1)
//   4. Preload Q / K / V packets through the DMA port
//   5. Program the sequencer descriptor through CSR, start FlashAttention
//   6. Wait for completion, then read the FP16 result words back through DMA
//   7. Bit-exact comparison against the golden FP16 reference
//
// TCDM address map used by snax_llm_sequencer (identical to the core TBs):
//   Q  : bank 16, word = h*32 + q*2 + kb
//   K  : bank 0..15, word = h*4 + kj*2 + kb
//   V  : bank 0..15, word = 16 + h*4 + dvi*2 + kj
//   OUT: bank 17, word = 0..  (one 256b FP16 vector = 4 consecutive words)
// =============================================================================
`timescale 1ns/1ps

module tb_snax_llm_top_clean;

    // -------------------------------------------------------------------------
    // Problem shape
    // -------------------------------------------------------------------------
    localparam int H          = 2;
    localparam int N_Q        = 16;
    localparam int N_KV       = 32;
    localparam int D_K        = 32;
    localparam int D_V        = 32;
    localparam int M_Q        = 8;
    localparam int K_BLOCKS   = D_K / 16;   // 2
    localparam int Q_TILES    = N_Q / M_Q;  // 2
    localparam int DV_TILES   = D_V / 16;   // 2
    localparam int KV_TILES   = N_KV / 16;  // 2
    localparam int NUM_VEC    = H * Q_TILES * DV_TILES * M_Q;      // 64
    localparam int NUM_WORD   = NUM_VEC * 4;                       // 256
    localparam int NUM_FP16   = NUM_WORD * 4;                      // 1024
    localparam int CLK_PERIOD = 10;

    localparam int OUT_BANK   = 17;
    localparam logic [31:0] Q_BASE   = 32'h0000_0100;  // bank 16, word 0
    localparam logic [31:0] OUT_BASE = 32'h0000_0110;  // bank 17, word 0
    localparam logic [31:0] RING_BASE= 32'h0000_0000;  // bank 0,  word 0

    // -------------------------------------------------------------------------
    // CSR register indices (snax_csr_manager)
    //   RW  0..46  , START = 47 , RO 48..
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

    logic busy, done_irq;

    snax_llm_top_clean dut (
        .clk                     (clk),
        .rst_n                   (rst_n),

        .cpu_csr_req_valid_i     (csr_req_valid),
        .cpu_csr_req_ready_o     (csr_req_ready),
        .cpu_csr_req_write_i     (csr_req_write),
        .cpu_csr_req_addr_i      (csr_req_addr),
        .cpu_csr_req_data_i      (csr_req_data),
        .cpu_csr_rsp_valid_o     (csr_rsp_valid),
        .cpu_csr_rsp_data_o      (csr_rsp_data),
        .cpu_csr_rsp_ready_i     (csr_rsp_ready),

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
    // Clock / reset
    // -------------------------------------------------------------------------
    initial begin
        clk = 1'b0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    // -------------------------------------------------------------------------
    // Waveform dump
    //
    //   default      : the DUT scope (all top-level interface traffic)
    //   +WAVE_DEEP   : the whole testbench, including the stimulus arrays
    //   +NO_WAVE     : skip the dump entirely (fastest)
    //
    // Format follows the simulator: define FST_TRACE (run.sh --fst) to write
    // FST instead of VCD.  FST is roughly 13x smaller for this design.
    //
    // Note: Verilator flattens modules, so a scope-limited dump still stores
    // every signal of the design; the level argument has no effect there.
    // Use --fst for Verilator, the VCD size is dominated by the TCDM arrays.
    // -------------------------------------------------------------------------
    initial begin
        if (!$test$plusargs("NO_WAVE")) begin
`ifdef FST_TRACE
            $dumpfile("tb_snax_llm_top_clean.fst");
`else
            $dumpfile("tb_snax_llm_top_clean.vcd");
`endif
            if ($test$plusargs("WAVE_DEEP"))
                $dumpvars(0, tb_snax_llm_top_clean);
            else
                $dumpvars(0, dut);
        end
    end

    // Optional heartbeat so a stalled run is visible in the waveform.
    initial begin
        if ($test$plusargs("WAVE_DEEP"))
            $display("[wave] deep dump enabled (+WAVE_DEEP)");
    end

    // -------------------------------------------------------------------------
    // Reference data
    // -------------------------------------------------------------------------
    // -------------------------------------------------------------------------
    // Stimulus / golden images, produced by
    //   tb/generate_snax_llm_top_clean_data.py
    // -------------------------------------------------------------------------
    localparam int NUM_PRELOAD = H*N_Q*K_BLOCKS
                               + H*KV_TILES*K_BLOCKS*16
                               + H*DV_TILES*KV_TILES*16;   // 320

    logic [31:0] pre_addr  [0:NUM_PRELOAD - 1];
    logic [71:0] pre_data  [0:NUM_PRELOAD - 1];
    logic [15:0] mem_golden[0:NUM_FP16 - 1];
    logic [71:0] exp_word  [0:NUM_WORD - 1];

    logic [71:0] out_words [0:NUM_WORD - 1];

    // -------------------------------------------------------------------------
    // Scoreboard
    // -------------------------------------------------------------------------
    int  checks_pass, checks_fail;
    int  fp16_match, fp16_mismatch;
    real max_abs_err, sum_abs_err;

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------
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
            $display("  [FAIL] %s", name);
        end
    endtask

    // -------------------------------------------------------------------------
    // CSR driver
    //
    // Handshake discipline: drive at negedge, sample `ready` combinationally,
    // commit at the following posedge, then deassert at the next negedge.  The
    // response is valid in the negedge right after the committing posedge, so
    // it is captured there (rsp_ready is held high, which keeps a single-cycle
    // response window).
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

    // -------------------------------------------------------------------------
    // DMA master 0 driver
    // -------------------------------------------------------------------------
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
    // RV32I master 1 driver
    // -------------------------------------------------------------------------
    task automatic cpu_write(input logic [31:0] a, input logic [71:0] d);
        begin
            @(negedge clk);
            cpu_mem_req_valid = 1'b1;
            cpu_mem_req_write = 1'b1;
            cpu_mem_req_addr  = a;
            cpu_mem_req_wdata = d;
            cpu_mem_req_strb  = 9'h1FF;
            #1;
            while (!cpu_mem_req_ready) begin
                @(negedge clk); #1;
            end
            @(posedge clk);
            @(negedge clk);
            cpu_mem_req_valid = 1'b0;
            cpu_mem_req_write = 1'b0;
            cpu_mem_req_strb  = 9'h000;
        end
    endtask

    task automatic cpu_read(input logic [31:0] a, output logic [71:0] d);
        begin
            @(negedge clk);
            cpu_mem_req_valid = 1'b1;
            cpu_mem_req_write = 1'b0;
            cpu_mem_req_addr  = a;
            cpu_mem_req_wdata = '0;
            cpu_mem_req_strb  = 9'h000;
            #1;
            while (!cpu_mem_req_ready) begin
                @(negedge clk); #1;
            end
            @(posedge clk);
            @(negedge clk);
            cpu_mem_req_valid = 1'b0;
            d = cpu_mem_rsp_rdata;
        end
    endtask

    // -------------------------------------------------------------------------
    // Main sequence
    // -------------------------------------------------------------------------
    logic [31:0] rd32;
    logic [71:0] rd72, rd72b;
    int          i, v, w, j, lane, vcol;
    int          exp_h, exp_qi, exp_dvi, exp_m, golden_idx;
    int          v_hits;
    int          word_mismatch;
    int          csv_fd;
    logic [15:0] act16, gold16;
    real         a_val, g_val, diff, adiff;
    logic        seen_done;
    int          cycle_cnt;

    initial begin
        checks_pass = 0; checks_fail = 0;
        fp16_match = 0;  fp16_mismatch = 0;
        max_abs_err = 0.0; sum_abs_err = 0.0;

        csr_req_valid = 1'b0; csr_req_write = 1'b0;
        csr_req_addr  = 12'd0; csr_req_data = 32'd0;
        csr_rsp_ready = 1'b1;

        cpu_mem_req_valid = 1'b0; cpu_mem_req_write = 1'b0;
        cpu_mem_req_addr  = 32'd0; cpu_mem_req_wdata = '0;
        cpu_mem_req_strb  = 9'd0;

        dma_mem_req_valid = 1'b0; dma_mem_req_write = 1'b0;
        dma_mem_req_addr  = 32'd0; dma_mem_req_wdata = '0;
        dma_mem_req_strb  = 9'd0;

        dma_desc_ready = 1'b1;   // no external DMA in this TB
        dma_desc_done  = 1'b0;

        rst_n = 1'b0;
        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        repeat (5) @(posedge clk);

        $display("\n========================================================================");
        $display("   SNAX FP4 LLM TOP (21 masters) - FULL SYSTEM TEST                      ");
        $display("   DMA(0) + RV32I(1) + act(2) + wt(3..18) + wo(19) + rd(20)            ");
        $display("========================================================================\n");

        // ---------------------------------------------------------------------
        // 1. Identification
        // ---------------------------------------------------------------------
        $display("[1] CSR identification");
        csr_read(A_RO_ID, rd32);
        $display("    ID = 0x%08X (expected 0x%08X)", rd32, CSR_ID_VALUE);
        check(rd32 == CSR_ID_VALUE, "CSR ID register");

        // ---------------------------------------------------------------------
        // 2. DMA master 0 reachability
        // ---------------------------------------------------------------------
        $display("[2] TCDM access through DMA (master 0)");
        dma_write(tcdm_addr(0, OUT_BANK), 72'hA5A5_1111_2222_3333_44);
        dma_read (tcdm_addr(0, OUT_BANK), rd72);
        $display("    wrote 0x%018X  read 0x%018X", 72'hA5A5_1111_2222_3333_44, rd72);
        check(rd72 == 72'hA5A5_1111_2222_3333_44, "DMA write/read bank 17");

        // ---------------------------------------------------------------------
        // 3. RV32I master 1 reachability (same word, different master)
        // ---------------------------------------------------------------------
        $display("[3] TCDM access through RV32I (master 1)");
        cpu_read(tcdm_addr(0, OUT_BANK), rd72b);
        check(rd72b == 72'hA5A5_1111_2222_3333_44, "RV32I read sees DMA write");
        cpu_write(tcdm_addr(0, OUT_BANK), 72'h5A5A_4444_3333_2222_11);
        dma_read (tcdm_addr(0, OUT_BANK), rd72);
        $display("    CPU wrote 0x%018X  DMA read 0x%018X", 72'h5A5A_4444_3333_2222_11, rd72);
        check(rd72 == 72'h5A5A_4444_3333_2222_11, "DMA read sees RV32I write");

        // Clean the scratch word so it cannot be mistaken for output.
        dma_write(tcdm_addr(0, OUT_BANK), 72'h0);

        // ---------------------------------------------------------------------
        // 4. Preload Q / K / V
        // ---------------------------------------------------------------------
        $display("[4] Preloading Q / K / V through the DMA port");
        $readmemh("sim_snax_llm_top_clean/hex/tcdm_preload_addr.hex", pre_addr);
        $readmemh("sim_snax_llm_top_clean/hex/tcdm_preload_data.hex", pre_data);
        $readmemh("sim_snax_llm_top_clean/hex/golden_core_fp16.hex",  mem_golden);
        $readmemh("sim_snax_llm_top_clean/hex/expected_out_words.hex", exp_word);

        for (i = 0; i < NUM_PRELOAD; i++)
            dma_write(pre_addr[i], pre_data[i]);

        // Spot-check the first and last preload word.
        dma_read(pre_addr[0], rd72);
        check(rd72 == pre_data[0], "first preload word round-trip");
        dma_read(pre_addr[NUM_PRELOAD-1], rd72);
        check(rd72 == pre_data[NUM_PRELOAD-1], "last preload word round-trip");

        // ---------------------------------------------------------------------
        // 5. Program the sequencer descriptor
        // ---------------------------------------------------------------------
        $display("[5] Programming the sequencer descriptor via CSR");

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

        // Read a few back to prove the CSR file latched them.
        csr_read(A_ATTN_H, rd32);
        check(rd32[7:0] == H, "CSR attn_h readback");
        csr_read(A_ATTN_QT, rd32);
        check(rd32[15:0] == Q_TILES, "CSR attn_q_tiles readback");
        csr_read(A_OUT_BASE, rd32);
        check(rd32 == OUT_BASE, "CSR out_base readback");

        // ---------------------------------------------------------------------
        // 6. Start and wait for completion
        // ---------------------------------------------------------------------
        $display("[6] Starting FlashAttention through CSR (start mask, seq bit)");
        csr_write(A_START, 32'h0000_0003);   // bit0 commit, bit1 sequencer

        seen_done  = 1'b0;
        cycle_cnt  = 0;
        while (!seen_done && cycle_cnt < 400000) begin
            @(posedge clk);
            cycle_cnt = cycle_cnt + 1;
            if (done_irq && !busy) seen_done = 1'b1;
        end
        $display("    %s after %0d cycles (busy=%0b done_irq=%0b)",
                 seen_done ? "Completed" : "TIMEOUT", cycle_cnt, busy, done_irq);
        check(seen_done, "sequencer completion");

        csr_read(A_RO_SEQ, rd32);
        $display("    status: seq_busy=%0b seq_done=%0b", rd32[0], rd32[1]);
        check(rd32[0] == 1'b0, "sequencer idle after done");

        // Let the write-out pipeline drain completely.
        repeat (200) @(posedge clk);

        // ---------------------------------------------------------------------
        // 7. Collect FP16 results from TCDM
        // ---------------------------------------------------------------------
        $display("[7] Reading back %0d result words from bank %0d", NUM_WORD, OUT_BANK);
        for (i = 0; i < NUM_WORD; i++) begin
            dma_read(tcdm_addr(i, OUT_BANK), out_words[i]);
        end

        // Dump the raw captured words so a mismatch can be inspected offline.
        csv_fd = $fopen("sim_snax_llm_top_clean/csv/top_out_words.csv", "w");
        $fdisplay(csv_fd, "word,actual_hex,expected_hex,match");
        for (i = 0; i < NUM_WORD; i++)
            $fdisplay(csv_fd, "%0d,0x%018X,0x%018X,%0d",
                      i, out_words[i], exp_word[i], (out_words[i] == exp_word[i]));
        $fclose(csv_fd);
        $display("    wrote sim_snax_llm_top_clean/csv/top_out_words.csv");

        // Word-level check against the generator's packed vectors.
        word_mismatch = 0;
        for (i = 0; i < NUM_WORD; i++)
            if (out_words[i] !== exp_word[i]) word_mismatch = word_mismatch + 1;
        $display("    word-exact: %0d/%0d", NUM_WORD - word_mismatch, NUM_WORD);
        check(word_mismatch == 0, "all output words bit-exact");

        // ---------------------------------------------------------------------
        // 8. Bit-exact comparison
        // ---------------------------------------------------------------------
        $display("[8] Comparing against the golden FP16 reference");
        csv_fd = $fopen("sim_snax_llm_top_clean/csv/top_compare.csv", "w");
        $fdisplay(csv_fd, "head,query_tile,dv_slice,m,query_global,lane,v_col_global,actual_hex,golden_hex,actual_real,golden_real,match");
        for (v = 0; v < NUM_VEC; v++) begin
            exp_m   = v % M_Q;
            exp_dvi = (v / M_Q) % DV_TILES;
            exp_qi  = (v / (M_Q * DV_TILES)) % Q_TILES;
            exp_h   = v / (M_Q * DV_TILES * Q_TILES);

            v_hits = 0;
            for (w = 0; w < 4; w++) begin
                for (j = 0; j < 4; j++) begin
                    lane     = w * 4 + j;
                    vcol     = exp_dvi * 16 + lane;
                    golden_idx = (exp_h * N_Q + exp_qi * M_Q + exp_m) * D_V + vcol;

                    act16  = out_words[v * 4 + w][j * 16 +: 16];
                    gold16 = mem_golden[golden_idx];

                    a_val = fp16_to_real(act16);
                    g_val = fp16_to_real(gold16);
                    diff  = a_val - g_val;
                    adiff = (diff < 0.0) ? -diff : diff;
                    sum_abs_err = sum_abs_err + adiff;
                    if (adiff > max_abs_err) max_abs_err = adiff;

                    if (act16 == gold16) begin
                        fp16_match = fp16_match + 1;
                        v_hits = v_hits + 1;
                    end else begin
                        fp16_mismatch = fp16_mismatch + 1;
                        if (fp16_mismatch <= 10)
                            $display("    [MISMATCH] h=%0d qi=%0d dvi=%0d m=%0d lane=%0d act=0x%04X gold=0x%04X",
                                     exp_h, exp_qi, exp_dvi, exp_m, lane, act16, gold16);
                    end
                    $fdisplay(csv_fd, "%0d,%0d,%0d,%0d,%0d,%0d,%0d,0x%04X,0x%04X,%.7f,%.7f,%0d",
                              exp_h, exp_qi, exp_dvi, exp_m, exp_qi*M_Q + exp_m,
                              lane, vcol, act16, gold16,
                              a_val, g_val, (act16 == gold16));
                end
            end
            if (v < 16)
                $display("    vector %0d (h=%0d qi=%0d dvi=%0d m=%0d): %0d/16 lanes exact  first_lane=0x%04X",
                         v, exp_h, exp_qi, exp_dvi, exp_m, v_hits,
                         out_words[v*4][15:0]);
        end

        $fclose(csv_fd);
        $display("    wrote sim_snax_llm_top_clean/csv/top_compare.csv");

        // ---------------------------------------------------------------------
        // 8b. Diagnostic: is the output a shifted / permuted copy of golden?
        // ---------------------------------------------------------------------
        begin
            int found_at, probe;
            int hits;
            hits = 0;
            for (probe = 0; probe < 8; probe++) begin
                found_at = -1;
                for (i = 0; i < NUM_FP16; i++)
                    if (mem_golden[i] == out_words[probe * 4][15:0]) begin
                        found_at = i;
                        i = NUM_FP16;
                    end
                if (found_at >= 0) hits = hits + 1;
                $display("    probe word %0d: actual[15:0]=0x%04X  golden_index_of_that_value=%0d",
                         probe, out_words[probe * 4][15:0], found_at);
            end
            $display("    direct-value hits in golden: %0d/8", hits);
        end

        // ---------------------------------------------------------------------
        // 8c. Permutation search: for each actual vector, which golden vector
        //     inside the same (h,qi,dvi) group does it match?
        // ---------------------------------------------------------------------
        begin
            int av, gv, ww, jj, ln, hits, best_gv, best_hits;
            int gi;
            for (av = 0; av < 16; av++) begin
                best_gv = -1; best_hits = -1;
                exp_dvi = (av / M_Q) % DV_TILES;
                exp_qi  = (av / (M_Q * DV_TILES)) % Q_TILES;
                exp_h   = av / (M_Q * DV_TILES * Q_TILES);
                for (gv = 0; gv < M_Q; gv++) begin
                    hits = 0;
                    for (ww = 0; ww < 4; ww++)
                        for (jj = 0; jj < 4; jj++) begin
                            ln = ww * 4 + jj;
                            vcol = exp_dvi * 16 + ln;
                            gi = (exp_h * N_Q + exp_qi * M_Q + gv) * D_V + vcol;
                            if (out_words[av * 4 + ww][jj * 16 +: 16] == mem_golden[gi])
                                hits = hits + 1;
                        end
                    if (hits > best_hits) begin
                        best_hits = hits;
                        best_gv   = gv;
                    end
                end
                $display("    actual vec %0d (expected m=%0d) best matches golden m=%0d (%0d/16 lanes)",
                         av, av % M_Q, best_gv, best_hits);
            end
        end

        // ---------------------------------------------------------------------
        // 9. Report
        // ---------------------------------------------------------------------
        $display("\n========================================================================");
        $display("                        TEST RESULT                                     ");
        $display("========================================================================");
        $display(" Structural checks : %0d passed, %0d failed", checks_pass, checks_fail);
        $display(" FP16 compared     : %0d", fp16_match + fp16_mismatch);
        $display("   bit-exact       : %0d", fp16_match);
        $display("   mismatched      : %0d", fp16_mismatch);
        $display(" Max |error|       : %0.7f", max_abs_err);
        if (fp16_match + fp16_mismatch > 0)
            $display(" Mean |error|      : %0.7f",
                     sum_abs_err / (fp16_match + fp16_mismatch));
        if (checks_fail == 0 && fp16_mismatch == 0 &&
            (fp16_match + fp16_mismatch) == NUM_FP16)
            $display("\n >>> [PASS] SNAX LLM TOP - FULL SYSTEM BIT-EXACT <<<");
        else
            $display("\n >>> [FAIL] see details above <<<");
        $display("========================================================================\n");
        $finish;
    end

    // -------------------------------------------------------------------------
    // Watchdog
    // -------------------------------------------------------------------------
    initial begin
        #(CLK_PERIOD * 4_000_000);
        $display("\n[FATAL] tb_snax_llm_top_clean watchdog timeout");
        $finish;
    end

    // -------------------------------------------------------------------------
    // Progress monitor
    // -------------------------------------------------------------------------
    int mon_cycles;
    initial mon_cycles = 0;
    always @(posedge clk) begin
        if (rst_n && busy) begin
            mon_cycles = mon_cycles + 1;
            if ((mon_cycles % 20000) == 0) begin
                $display("    ... busy %0d cyc | seq=%0d h=%0d qi=%0d dvi=%0d kj=%0d kb=%0d m=%0d",
                         mon_cycles, dut.u_seq.state_q, dut.u_seq.cnt_h_q,
                         dut.u_seq.cnt_qi_q, dut.u_seq.cnt_dvi_q, dut.u_seq.cnt_kj_q,
                         dut.u_wt_streamer.kb_cnt_q, dut.u_seq.cnt_m_q);
                $display("        act_busy=%0b act_v=%0b act_rdy=%0b | wt_busy=%0b wt_v=%h wt_rdy=%h",
                         dut.act_busy, dut.act_stream_valid, dut.act_stream_ready,
                         dut.wt_busy, dut.wt_stream_valid, dut.wt_stream_ready);
                $display("        act st=%0d tok=%0d k=%0d outstd=%0d | wt st=%0d kb=%0d rep=%0d replay=%0b",
                         dut.u_act_streamer.state_q, dut.u_act_streamer.tok_cnt_q,
                         dut.u_act_streamer.k_cnt_q, dut.u_act_streamer.outstanding_q,
                         dut.u_wt_streamer.state_q, dut.u_wt_streamer.kb_cnt_q,
                         dut.u_wt_streamer.rep_cnt_q, dut.u_wt_streamer.replay_active_q);
                $display("        lane_valid=%h p_empty=%0b cp_fire=%0b wo_busy=%0b wo_fp4_v=%0b wo_fp4_rdy=%0b",
                         dut.lane_valid, dut.p_fifo_empty,
                         dut.u_seq.qk_compute_fire, dut.wo_busy,
                         dut.wo_fp4_valid, dut.wo_fp4_ready);
                $display("        wo out_writes=%0d wo_done=%0b rd_busy=%0b v_busy=%0b",
                         out_writes, dut.wo_done, dut.rd_busy, dut.v_busy);
            end
        end
    end

    // -------------------------------------------------------------------------
    // QK beat utilisation: how many QK_RUN cycles actually produce a beat, and
    // which operand was missing on the cycles that did not.
    //   state 4 = ST_ATTN_QK_RUN (see the [SEQ] trace for the encoding)
    // -------------------------------------------------------------------------
    int u_qk_cyc, u_qk_fire, u_qk_noact, u_qk_nowt;
    int u_qk_drain, u_qk_both, u_drain_any, u_drain_qk, u_drain_sm, u_drain_other;
    initial begin
        u_qk_cyc = 0; u_qk_fire = 0; u_qk_noact = 0; u_qk_nowt = 0;
        u_qk_drain = 0; u_qk_both = 0;
        u_drain_any = 0; u_drain_qk = 0; u_drain_sm = 0; u_drain_other = 0;
    end
    always @(posedge clk) begin
        if (rst_n) begin
            if (dut.u_seq.tile_drain_req_o) begin
                u_drain_any = u_drain_any + 1;
                if (dut.u_seq.state_q == 5'd4)      u_drain_qk    = u_drain_qk + 1;
                else if (dut.u_seq.state_q == 5'd5) u_drain_sm    = u_drain_sm + 1;
                else                                u_drain_other = u_drain_other + 1;
            end
            if (dut.u_seq.state_q == 5'd4) begin
                u_qk_cyc = u_qk_cyc + 1;
                if (dut.u_seq.qk_compute_fire) begin
                    u_qk_fire = u_qk_fire + 1;
                end else if (!dut.act_stream_valid) begin
                    u_qk_noact = u_qk_noact + 1;
                end else if (!(&dut.wt_stream_valid)) begin
                    u_qk_nowt = u_qk_nowt + 1;
                end
                if (dut.u_seq.tile_drain_req_o) begin
                    u_qk_drain = u_qk_drain + 1;
                    if (dut.u_seq.qk_compute_fire) u_qk_both = u_qk_both + 1;
                end
            end
        end
    end

    final begin
        $display("\n[QK UTIL] QK_RUN cycles = %0d, beats = %0d, utilisation = %0d%%",
                 u_qk_cyc, u_qk_fire, (u_qk_cyc > 0) ? (100 * u_qk_fire / u_qk_cyc) : 0);
        $display("          idle because act_valid=0 : %0d", u_qk_noact);
        $display("          idle because wt_valid=0  : %0d", u_qk_nowt);
        $display("          idle for other reasons   : %0d",
                 u_qk_cyc - u_qk_fire - u_qk_noact - u_qk_nowt);
        $display("          drain pulses in QK_RUN   : %0d (of which same-cycle as a beat: %0d)",
                 u_qk_drain, u_qk_both);
        $display("[DRAIN]   total pulses = %0d  (QK_RUN %0d, SM_DRAIN %0d, other %0d)",
                 u_drain_any, u_drain_qk, u_drain_sm, u_drain_other);
    end

    // -------------------------------------------------------------------------
    // Sequencer state residency: where the 1152 cycles actually go.
    // The state encoding is the order of the state_t enum.
    // -------------------------------------------------------------------------
    int st_cyc [0:31];
    int st_i;
    initial for (st_i = 0; st_i < 32; st_i++) st_cyc[st_i] = 0;
    always @(posedge clk)
        if (rst_n) st_cyc[dut.u_seq.state_q] = st_cyc[dut.u_seq.state_q] + 1;

    final begin
        $display("\n[STATE RESIDENCY]");
        for (st_i = 0; st_i < 32; st_i++)
            if (st_cyc[st_i] != 0)
                $display("  state %0d : %0d cycles", st_i, st_cyc[st_i]);
    end

    // -------------------------------------------------------------------------
    // Per-cycle overlap trace (first 140 busy cycles)
    // -------------------------------------------------------------------------
    always @(posedge clk) begin
        if (rst_n && dut.u_seq.tile_drain_req_o)
            $display("  [DRN %0d] st=%0d didx=%h nrm=%0d fp16v=%0b fp16r=%0b",
                     mon_cycles, dut.u_seq.state_q, dut.u_seq.tile_drain_idx_o,
                     dut.u_seq.nrm_state_q, dut.core_fp16_valid, dut.core_fp16_ready);
    end

    // -------------------------------------------------------------------------
    // Sequencer start-pulse monitor
    // -------------------------------------------------------------------------
    always @(posedge clk) begin
        if (rst_n && dut.seq_wt_start)
            $display("  [WTPULSE] cyc=%0d seq=%0d base=%h kb=%0d rep=%0d nt=%0d | wt_state=%0d wt_busy=%0b",
                     mon_cycles, dut.u_seq.state_q, dut.seq_wt_base, dut.seq_wt_k_blocks,
                     dut.seq_wt_repeat, dut.seq_wt_num_tiles,
                     dut.u_wt_streamer.state_q, dut.wt_busy);
        if (rst_n && dut.seq_act_start)
            $display("  [ACTPULSE] cyc=%0d seq=%0d base=%h tok=%0d tstr=%h kb=%0d | act_state=%0d act_busy=%0b",
                     mon_cycles, dut.u_seq.state_q, dut.seq_act_base, dut.seq_act_tokens,
                     dut.seq_act_token_stride, dut.seq_act_k_blocks,
                     dut.u_act_streamer.state_q, dut.act_busy);
    end

    // -------------------------------------------------------------------------
    // Sequencer state-transition monitor
    // -------------------------------------------------------------------------
    logic [4:0] prev_seq_state;
    initial prev_seq_state = 5'd0;
    always @(posedge clk) begin
        if (rst_n && (dut.u_seq.state_q != prev_seq_state)) begin
            $display("  [SEQ] cyc=%0d  %0d -> %0d (h=%0d qi=%0d dvi=%0d kj=%0d kb=%0d m=%0d dr=%0d)",
                     mon_cycles, prev_seq_state, dut.u_seq.state_q,
                     dut.u_seq.cnt_h_q, dut.u_seq.cnt_qi_q, dut.u_seq.cnt_dvi_q,
                     dut.u_seq.cnt_kj_q, dut.u_wt_streamer.kb_cnt_q, dut.u_seq.cnt_m_q,
                     dut.u_seq.cnt_dr_q);
            prev_seq_state = dut.u_seq.state_q;
        end
    end
    // -------------------------------------------------------------------------
    // Per-group trace: print the group indices at the first drain of each
    // output-drain group, so a skipped (h, qi, dvi) group is visible.
    // -------------------------------------------------------------------------
    always @(posedge clk) begin
        if (rst_n && dut.u_seq.state_q == 5'd15 && dut.u_seq.cnt_dr_q == 4'd0)
            $display("  [GROUP] h=%0d qi=%0d dvi=%0d kj=%0d | norm_seen=%0b fp16_v=%0b",
                     dut.u_seq.cnt_h_q, dut.u_seq.cnt_qi_q, dut.u_seq.cnt_dvi_q,
                     dut.u_seq.cnt_kj_q, dut.u_seq.nrm_state_q, dut.core_fp16_valid);
        if (rst_n && dut.u_seq.state_q == 5'd9 && dut.u_seq.cnt_dr_q == 4'd0)
            $display("  [SMGRP] h=%0d qi=%0d dvi=%0d kj=%0d",
                     dut.u_seq.cnt_h_q, dut.u_seq.cnt_qi_q, dut.u_seq.cnt_dvi_q,
                     dut.u_seq.cnt_kj_q);
    end

    // -------------------------------------------------------------------------
    // Pipeline timing probe: offset between a softmax drain request and the
    // resulting P push, together with the query index seen by the core.
    // -------------------------------------------------------------------------
    int     t_drain_s, t_drain_o;
    int     cyc;
    initial begin t_drain_s = -1; t_drain_o = -1; cyc = 0; end
    always @(posedge clk) begin
        if (rst_n) begin
            cyc = cyc + 1;
            if (dut.tile_drain_req && dut.tile_drain_idx[3]) begin
                t_drain_s = cyc;
                $display("  [DRAIN_S] cyc=%0d idx=%0d (m=%0d)", cyc,
                         dut.tile_drain_idx, dut.tile_drain_idx[2:0]);
            end
            if (dut.tile_drain_req && !dut.tile_drain_idx[3]) begin
                t_drain_o = cyc;
                $display("  [DRAIN_O] cyc=%0d idx=%0d (m=%0d) inv_l_mant_q[m]=0x%04X",
                         cyc, dut.tile_drain_idx, dut.tile_drain_idx[2:0],
                         dut.u_core.inv_l_mant_q[dut.tile_drain_idx[2:0]]);
            end
            if (dut.u_core.raw_drained_valid)
                $display("      raw_drained  cyc=%0d  (+%0d from S drain) drain_query_idx=%0d",
                         cyc, cyc - t_drain_s, dut.u_core.drain_query_idx);
            if (dut.u_core.sm_out_valid)
                $display("      sm_out_valid cyc=%0d  (+%0d) q_d2=%0d  drain_query_idx=%0d",
                         cyc, cyc - t_drain_s,
                         dut.u_core.sm_query_idx_d2, dut.u_core.drain_query_idx);
        end
    end

    // -------------------------------------------------------------------------
    // Data probe: which Q row and K block the core actually receives
    // -------------------------------------------------------------------------
    int beat_cnt;
    initial beat_cnt = 0;
    always @(posedge clk) begin
        if (rst_n && dut.u_seq.qk_compute_fire && beat_cnt < 20) begin
            $display("  [QKBEAT %0d] m=%0d kb=%0d | act=0x%04X  wt_l0=0x%04X",
                     beat_cnt, dut.u_seq.cnt_m_q, dut.u_wt_streamer.kb_cnt_q,
                     dut.act_stream_data[15:0], dut.wt_stream_data[0][15:0]);
            beat_cnt = beat_cnt + 1;
        end
    end

    // -------------------------------------------------------------------------
    // Write-out descriptor + address trace
    // -------------------------------------------------------------------------
    int wo_dumped;
    initial wo_dumped = 0;
    always @(posedge clk) begin
        if (rst_n && dut.u_write_out.start_pulse && wo_dumped < 1) begin
            $display("  [WO_START] mode=%0d base=%h nvec=%0d lstride=%h nbatch=%0d bstride=%h",
                     dut.u_write_out.cfg_mode_i, dut.u_write_out.cfg_base_addr_i,
                     dut.u_write_out.cfg_num_vectors_i, dut.u_write_out.cfg_line_stride_i,
                     dut.u_write_out.cfg_num_batches_i, dut.u_write_out.cfg_batch_stride_i);
            wo_dumped = 1;
        end
        if (rst_n && dut.wo_m_req_valid && dut.wo_m_req_ready && dut.wo_m_req_write && mon_cycles < 200)
            $display("  [WO_WR] addr=%h bank=%0d word=%0d", dut.wo_m_req_addr,
                     dut.wo_m_req_addr[8:4], dut.wo_m_req_addr[18:9]);
    end

    int out_writes;
    initial out_writes = 0;
    always @(posedge clk) begin
        if (rst_n && dut.wo_m_req_valid && dut.wo_m_req_ready && dut.wo_m_req_write) begin
            if (dut.wo_m_req_addr[8:4] == OUT_BANK[4:0])
                out_writes = out_writes + 1;
        end
    end

    // -------------------------------------------------------------------------
    // Cycle-accurate trace of the first 80 busy cycles after start
    // -------------------------------------------------------------------------
    always @(posedge clk) begin
        if (rst_n && mon_cycles > 0 && mon_cycles < 80)
            $display("  [CYC %0d] seq=%0d m=%0d kb=%0d | act v=%0b r=%0b tok=%0d busy=%0b | wt v=%h r=%h rep=%0d kb=%0d busy=%0b fire=%0b",
                     mon_cycles, dut.u_seq.state_q, dut.u_seq.cnt_m_q, dut.u_wt_streamer.kb_cnt_q,
                     dut.act_stream_valid, dut.act_stream_ready,
                     dut.u_act_streamer.tok_cnt_q, dut.act_busy,
                     dut.wt_stream_valid, dut.wt_stream_ready,
                     dut.u_wt_streamer.rep_cnt_q, dut.u_wt_streamer.kb_cnt_q,
                     dut.wt_busy, dut.u_seq.qk_compute_fire);
    end

    // -------------------------------------------------------------------------
    // Event counters: locate where the pipeline loses work
    // -------------------------------------------------------------------------
    int n_core_fp16, n_drain, n_drain_s, n_drain_o, n_p_push, n_p_pop, n_pv_fire, n_qk_fire;
    int n_raw, n_norm, n_gemm, n_fp16_valid, n_fp16_ready, n_hold;
    int n_drop, n_push, n_pop;
    initial begin
        n_core_fp16 = 0; n_drain = 0; n_drain_s = 0; n_drain_o = 0;
        n_p_push = 0; n_p_pop = 0; n_pv_fire = 0; n_qk_fire = 0;
        n_raw = 0; n_norm = 0; n_gemm = 0; n_fp16_valid = 0; n_fp16_ready = 0; n_hold = 0;
        n_drop = 0; n_push = 0; n_pop = 0;
    end
    always @(posedge clk) begin
        if (rst_n) begin
            if (dut.core_fp16_valid && dut.core_fp16_ready)    n_core_fp16 = n_core_fp16 + 1;
            if (dut.tile_drain_req) begin
                n_drain = n_drain + 1;
                if (dut.tile_drain_idx[3]) n_drain_s = n_drain_s + 1;
                else                       n_drain_o = n_drain_o + 1;
            end
            if (dut.u_core.sm_out_valid)                        n_p_push = n_p_push + 1;
            if (dut.u_core.p_fifo_pop)                          n_p_pop  = n_p_pop + 1;
            if (dut.u_seq.pv_compute_fire)                      n_pv_fire = n_pv_fire + 1;
            if (dut.u_seq.qk_compute_fire)                      n_qk_fire = n_qk_fire + 1;
            if (dut.u_core.raw_drained_valid)                   n_raw = n_raw + 1;
            if (dut.u_core.norm_vec_valid)                      n_norm = n_norm + 1;
            if (dut.u_core.gemm_vec_valid)                      n_gemm = n_gemm + 1;
            if (dut.core_fp16_valid)                            n_fp16_valid = n_fp16_valid + 1;
            if (dut.core_fp16_ready)                            n_fp16_ready = n_fp16_ready + 1;
            if (dut.u_core.out_hold_q)                          n_hold = n_hold + 1;
            if (dut.u_core.src_valid)                           n_push = n_push + 1;
        end
    end

    final begin
        $display(" counters: qk_fire=%0d pv_fire=%0d core_fp16=%0d",
                 n_qk_fire, n_pv_fire, n_core_fp16);
        $display("           drain total=%0d (S=%0d O=%0d) p_push=%0d p_pop=%0d",
                 n_drain, n_drain_s, n_drain_o, n_p_push, n_p_pop);
        $display("           raw_drained=%0d norm_vec=%0d gemm_vec=%0d fp16_valid=%0d fp16_ready=%0d hold=%0d",
                 n_raw, n_norm, n_gemm, n_fp16_valid, n_fp16_ready, n_hold);
        $display("           out fifo: push=%0d pop=%0d dropped=%0d", n_push, n_pop, n_drop);
    end

    final begin
        $display(" write-out committed %0d words to bank %0d (expected %0d)",
                 out_writes, OUT_BANK, NUM_WORD);
    end

endmodule
