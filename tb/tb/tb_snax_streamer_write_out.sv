// =============================================================================
// tb_snax_streamer_write_out.sv
// Unit test for snax_streamer_write_out linear FP4 and FP16 modes.
// =============================================================================
`timescale 1ns/1ps

module tb_snax_streamer_write_out;

    localparam int DATA_WIDTH = 72;
    localparam int ADDR_WIDTH = 32;
    localparam int NUM_LANES  = 16;
    localparam int STRB_WIDTH = 9;
    localparam int CLK_PERIOD = 10;

    logic clk, rst_n;

    logic        csr_start;
    logic        busy, done;
    logic [1:0]  cfg_mode;
    logic [31:0] cfg_base, cfg_num_vectors, cfg_line_stride;
    logic [31:0] cfg_num_batches, cfg_batch_stride;

    logic [NUM_LANES*16-1:0] core_fp16;
    logic                    core_fp16_valid;
    logic                    core_fp16_ready;

    logic [DATA_WIDTH-1:0] core_fp4;
    logic                  core_fp4_valid;
    logic                  core_fp4_ready;

    logic                  m_req_valid, m_req_write, m_req_ready;
    logic [ADDR_WIDTH-1:0] m_req_addr;
    logic [DATA_WIDTH-1:0] m_req_wdata;
    logic [STRB_WIDTH-1:0] m_req_strb;

    logic [DATA_WIDTH-1:0] tcdm [0:31][0:1023];

    int n, v, w, c;
    int match_count, mismatch_count;
    logic [DATA_WIDTH-1:0] expect_pkt;

    snax_streamer_write_out #(
        .DATA_WIDTH (DATA_WIDTH),
        .ADDR_WIDTH (ADDR_WIDTH),
        .NUM_LANES  (NUM_LANES),
        .STRB_WIDTH (STRB_WIDTH)
    ) dut (
        .clk                    (clk),
        .rst_n                  (rst_n),
        .csr_start_i            (csr_start),
        .busy_o                 (busy),
        .done_irq_o             (done),
        .cfg_mode_i             (cfg_mode),
        .cfg_base_addr_i        (cfg_base),
        .cfg_num_vectors_i      (cfg_num_vectors),
        .cfg_line_stride_i      (cfg_line_stride),
        .cfg_num_batches_i      (cfg_num_batches),
        .cfg_batch_stride_i     (cfg_batch_stride),
        .core_out_fp16_data_i   (core_fp16),
        .core_out_fp16_valid_i  (core_fp16_valid),
        .core_out_fp16_ready_o  (core_fp16_ready),
        .core_out_fp4_data_i    (core_fp4),
        .core_out_fp4_valid_i   (core_fp4_valid),
        .core_out_fp4_ready_o   (core_fp4_ready),
        .m_tcdm_req_valid_o     (m_req_valid),
        .m_tcdm_req_write_o     (m_req_write),
        .m_tcdm_req_addr_o      (m_req_addr),
        .m_tcdm_req_wdata_o     (m_req_wdata),
        .m_tcdm_req_strb_o      (m_req_strb),
        .m_tcdm_req_ready_i     (m_req_ready)
    );

    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    function automatic [15:0] pat16(input int vec, input int ch);
        pat16 = 16'hB000 + vec*16 + ch;
    endfunction

    always @(posedge clk) begin
        if (rst_n && m_req_valid && m_req_write && m_req_ready) begin
            tcdm[m_req_addr[8:4]][m_req_addr[18:9]] <= m_req_wdata;
        end
    end

    task automatic wait_cycles(input int ncycles);
        repeat (ncycles) @(posedge clk);
    endtask

    initial begin
        $display("\n=====================================================");
        $display("   WRITE-OUT LINEAR FP4 / FP16 UNIT TEST               ");
        $display("=====================================================\n");

        rst_n = 1'b0;
        csr_start = 1'b0;
        cfg_mode = 2'd0;
        cfg_base = 32'h0000_0000;
        cfg_num_vectors = 32'd4;
        cfg_line_stride = 32'h0000_0010;
        cfg_num_batches = 32'd1;
        cfg_batch_stride = 32'd0;
        core_fp16 = '0;
        core_fp16_valid = 1'b0;
        core_fp4 = '0;
        core_fp4_valid = 1'b0;
        m_req_ready = 1'b1;
        match_count = 0;
        mismatch_count = 0;

        wait_cycles(5);
        rst_n = 1'b1;
        wait_cycles(5);

        @(negedge clk);
        csr_start = 1'b1;
        @(posedge clk);
        @(negedge clk);
        csr_start = 1'b0;
        wait_cycles(2);

        for (n = 0; n < 4; n = n + 1) begin
            @(negedge clk);
            core_fp4 = {8'hA0 + n[7:0], 64'h0123_4567_89AB_CDEF};
            core_fp4_valid = 1'b1;
            @(posedge clk);
        end

        @(negedge clk);
        core_fp4_valid = 1'b0;
        core_fp4 = '0;

        while (busy) @(negedge clk);
        wait_cycles(3);

        for (n = 0; n < 4; n = n + 1) begin
            expect_pkt = {8'hA0 + n[7:0], 64'h0123_4567_89AB_CDEF};
            if (tcdm[n][0] === expect_pkt) begin
                match_count = match_count + 1;
            end else begin
                mismatch_count = mismatch_count + 1;
                $display("  [MISMATCH] FP4 bank=%0d word=0 actual=%018h expect=%018h",
                         n, tcdm[n][0], expect_pkt);
            end
        end

        cfg_mode = 2'd1;
        cfg_base = 32'h0000_0000;
        cfg_num_vectors = 32'd2;
        cfg_line_stride = 32'h0000_0010;
        cfg_num_batches = 32'd1;
        cfg_batch_stride = 32'd0;

        @(negedge clk);
        csr_start = 1'b1;
        @(posedge clk);
        @(negedge clk);
        csr_start = 1'b0;
        wait_cycles(2);

        for (v = 0; v < 2; v = v + 1) begin
            while (!core_fp16_ready) @(negedge clk);
            @(negedge clk);
            for (c = 0; c < NUM_LANES; c = c + 1) begin
                core_fp16[c*16 +: 16] = pat16(v, c);
            end
            core_fp16_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            core_fp16_valid = 1'b0;
            wait_cycles(6);
        end

        while (busy) @(negedge clk);
        wait_cycles(3);

        for (v = 0; v < 2; v = v + 1) begin
            for (w = 0; w < 4; w = w + 1) begin
                expect_pkt = {8'h00,
                              pat16(v, w*4 + 3),
                              pat16(v, w*4 + 2),
                              pat16(v, w*4 + 1),
                              pat16(v, w*4 + 0)};
                if (tcdm[v*4 + w][0] === expect_pkt) begin
                    match_count = match_count + 1;
                end else begin
                    mismatch_count = mismatch_count + 1;
                    $display("  [MISMATCH] FP16 bank=%0d word=0 actual=%018h expect=%018h",
                             v*4 + w, tcdm[v*4 + w][0], expect_pkt);
                end
            end
        end

        // ------------------------------------------------------------------
        // 3) V FP4 batch: 16 channels x 2 batches
        //    base=0x1000 -> bank0/word8, line=0x10 -> bank, batch=0x200 -> word
        // ------------------------------------------------------------------
        cfg_mode = 2'd2;
        cfg_base = 32'h0000_1000;
        cfg_num_vectors = 32'd16;
        cfg_line_stride = 32'h0000_0010;
        cfg_num_batches = 32'd2;
        cfg_batch_stride = 32'h0000_0200;

        @(negedge clk);
        csr_start = 1'b1;
        @(posedge clk);
        @(negedge clk);
        csr_start = 1'b0;
        wait_cycles(2);

        for (n = 0; n < 32; n = n + 1) begin
            @(negedge clk);
            core_fp4 = {8'hC0 + n[7:0], 64'h0123_4567_89AB_CDEF};
            core_fp4_valid = 1'b1;
            @(posedge clk);
        end
        @(negedge clk);
        core_fp4_valid = 1'b0;
        core_fp4 = '0;

        while (busy) @(negedge clk);
        wait_cycles(3);

        for (n = 0; n < 32; n = n + 1) begin
            expect_pkt = {8'hC0 + n[7:0], 64'h0123_4567_89AB_CDEF};
            if (tcdm[n % 16][8 + (n / 16)] === expect_pkt) begin
                match_count = match_count + 1;
            end else begin
                mismatch_count = mismatch_count + 1;
                $display("  [MISMATCH] V bank=%0d word=%0d actual=%018h expect=%018h",
                         n % 16, 8 + (n / 16),
                         tcdm[n % 16][8 + (n / 16)], expect_pkt);
            end
        end

        $display("\n-----------------------------------------------------");
        $display(" Values checked : %0d", match_count + mismatch_count);
        $display(" Matches        : %0d", match_count);
        $display(" Mismatches     : %0d", mismatch_count);
        if (mismatch_count == 0 && match_count == 44)
            $display(" >>> [PASS] FP4 + FP16 + V WRITE <<<");
        else
            $display(" >>> [FAIL] FP4 + FP16 + V WRITE <<<");
        $display("-----------------------------------------------------\n");
        $finish;
    end

endmodule
