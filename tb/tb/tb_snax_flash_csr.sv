`timescale 1ns/1ps

module tb_snax_flash_csr;

    localparam int CLK_PERIOD = 10;
    localparam int N_REGS = 28;

    logic clk, rst_n;
    logic csr_we;
    logic [11:0] csr_addr;
    logic [31:0] csr_wdata;
    logic [31:0] csr_rdata;

    logic seq_busy, seq_done;
    logic act_busy, wt_busy, wo_busy, v_busy, v_done;
    logic p_fifo_full, p_fifo_empty, l_ready, core_fp16_valid;

    logic start_pulse, src_sel, act_ready_force, wt_ready_force;
    logic act_start, wt_start, wo_start, wo_mode_fp16, v_start;
    logic [31:0] act_base, act_k_blocks, act_k_stride, act_tokens, act_token_stride;
    logic [31:0] wt_base, wt_repeat, wt_k_blocks, wt_k_stride, wt_num_tiles, wt_tile_step;
    logic [31:0] wo_base, wo_num_vectors, wo_line_stride;
    logic [1:0] core_mode;
    logic core_row_start, core_enable_slq, core_tile_drain_req, core_tile_clear;
    logic [15:0] core_key_mask, core_lane_valid;
    logic [3:0] core_tile_drain_idx, core_lane_acc_idx;
    logic [31:0] v_base, v_num_batches;

    logic [11:0] addr_list [0:N_REGS-1];
    logic [31:0] data_list [0:N_REGS-1];

    int i;
    int match_count, mismatch_count;
    logic [31:0] rd;

    snax_flash_csr dut (
        .clk(clk), .rst_n(rst_n),
        .csr_we(csr_we), .csr_addr(csr_addr), .csr_wdata(csr_wdata),
        .csr_rdata(csr_rdata),
        .seq_busy(seq_busy), .seq_done(seq_done),
        .act_busy(act_busy), .wt_busy(wt_busy), .wo_busy(wo_busy),
        .v_busy(v_busy), .v_done(v_done),
        .p_fifo_full(p_fifo_full), .p_fifo_empty(p_fifo_empty),
        .l_ready(l_ready), .core_fp16_valid(core_fp16_valid),
        .start_pulse(start_pulse), .src_sel(src_sel),
        .act_ready_force(act_ready_force), .wt_ready_force(wt_ready_force),
        .act_start(act_start), .act_base(act_base),
        .act_k_blocks(act_k_blocks), .act_k_stride(act_k_stride),
        .act_tokens(act_tokens), .act_token_stride(act_token_stride),
        .wt_start(wt_start), .wt_base(wt_base), .wt_repeat(wt_repeat),
        .wt_k_blocks(wt_k_blocks), .wt_k_stride(wt_k_stride),
        .wt_num_tiles(wt_num_tiles), .wt_tile_step(wt_tile_step),
        .wo_start(wo_start), .wo_mode_fp16(wo_mode_fp16),
        .wo_base(wo_base), .wo_num_vectors(wo_num_vectors),
        .wo_line_stride(wo_line_stride),
        .core_mode(core_mode), .core_row_start(core_row_start),
        .core_key_mask(core_key_mask), .core_enable_slq(core_enable_slq),
        .core_tile_drain_req(core_tile_drain_req),
        .core_tile_drain_idx(core_tile_drain_idx),
        .core_tile_clear_on_drain(core_tile_clear),
        .core_lane_valid(core_lane_valid), .core_lane_acc_idx(core_lane_acc_idx),
        .v_start(v_start), .v_base(v_base), .v_num_batches(v_num_batches)
    );

    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    task automatic wr(input [11:0] a, input [31:0] d);
        @(negedge clk);
        csr_we = 1'b1; csr_addr = a; csr_wdata = d;
        @(posedge clk);
        @(negedge clk);
        csr_we = 1'b0;
    endtask

    task automatic rd_csr(input [11:0] a, output [31:0] d);
        @(negedge clk);
        csr_we = 1'b0; csr_addr = a;
        #1 d = csr_rdata;
        @(negedge clk);
    endtask

    initial begin
        $display("\n=====================================================");
        $display("   FLASH CSR FILE UNIT TEST                            ");
        $display("=====================================================\n");

        rst_n = 1'b0;
        csr_we = 1'b0; csr_addr = 12'h0; csr_wdata = 32'h0;
        seq_busy = 1'b0; seq_done = 1'b0;
        act_busy = 1'b0; wt_busy = 1'b0; wo_busy = 1'b0;
        v_busy = 1'b0; v_done = 1'b0;
        p_fifo_full = 1'b0; p_fifo_empty = 1'b1;
        l_ready = 1'b1; core_fp16_valid = 1'b0;
        match_count = 0; mismatch_count = 0;

        addr_list[0] = 12'h014; data_list[0] = 32'h1111_0000;
        addr_list[1] = 12'h018; data_list[1] = 32'd2;
        addr_list[2] = 12'h01C; data_list[2] = 32'h200;
        addr_list[3] = 12'h020; data_list[3] = 32'd8;
        addr_list[4] = 12'h024; data_list[4] = 32'h100;
        addr_list[5] = 12'h034; data_list[5] = 32'h2222_0000;
        addr_list[6] = 12'h038; data_list[6] = 32'd16;
        addr_list[7] = 12'h03C; data_list[7] = 32'd2;
        addr_list[8] = 12'h040; data_list[8] = 32'h200;
        addr_list[9] = 12'h044; data_list[9] = 32'd2;
        addr_list[10] = 12'h048; data_list[10] = 32'h400;
        addr_list[11] = 12'h054; data_list[11] = 32'h3333_0000;
        addr_list[12] = 12'h058; data_list[12] = 32'd64;
        addr_list[13] = 12'h05C; data_list[13] = 32'h200;
        addr_list[14] = 12'h060; data_list[14] = 32'h1;
        addr_list[15] = 12'h070; data_list[15] = 32'h2;
        addr_list[16] = 12'h074; data_list[16] = 32'h1;
        addr_list[17] = 12'h078; data_list[17] = 32'h00FF;
        addr_list[18] = 12'h07C; data_list[18] = 32'h1;
        addr_list[19] = 12'h080; data_list[19] = 32'h1;
        addr_list[20] = 12'h084; data_list[20] = 32'hA;
        addr_list[21] = 12'h088; data_list[21] = 32'h1;
        addr_list[22] = 12'h08C; data_list[22] = 32'hFFFF;
        addr_list[23] = 12'h090; data_list[23] = 32'hB;
        addr_list[24] = 12'h0A4; data_list[24] = 32'h4444_0000;
        addr_list[25] = 12'h0A8; data_list[25] = 32'd3;
        addr_list[26] = 12'h00C; data_list[26] = 32'h7;
        addr_list[27] = 12'h008; data_list[27] = 32'h534E_4146;

        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        for (i = 0; i < N_REGS; i = i + 1) begin
            if (addr_list[i] != 12'h008)
                wr(addr_list[i], data_list[i]);
        end

        for (i = 0; i < N_REGS; i = i + 1) begin
            rd_csr(addr_list[i], rd);
            if (rd === data_list[i]) begin
                match_count = match_count + 1;
            end else begin
                mismatch_count = mismatch_count + 1;
                $display("  [MISMATCH] addr=0x%03h actual=0x%08h expect=0x%08h",
                         addr_list[i], rd, data_list[i]);
            end
        end

        seq_busy = 1'b1; seq_done = 1'b1;
        v_busy = 1'b1; v_done = 1'b1;
        rd_csr(12'h004, rd);
        if (rd === 32'h3) match_count = match_count + 1;
        else begin
            mismatch_count = mismatch_count + 1;
            $display("  [MISMATCH] global status=0x%08h expect=0x3", rd);
        end
        rd_csr(12'h0AC, rd);
        if (rd === 32'h3) match_count = match_count + 1;
        else begin
            mismatch_count = mismatch_count + 1;
            $display("  [MISMATCH] v status=0x%08h expect=0x3", rd);
        end

        $display("\n-----------------------------------------------------");
        $display(" Checks  : %0d", match_count + mismatch_count);
        $display(" Matches : %0d", match_count);
        $display(" Mismatches: %0d", mismatch_count);
        if (mismatch_count == 0)
            $display(" >>> [PASS] FLASH CSR FILE <<<");
        else
            $display(" >>> [FAIL] FLASH CSR FILE <<<");
        $display("-----------------------------------------------------\n");
        $finish;
    end

endmodule
