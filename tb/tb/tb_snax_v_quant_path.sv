// =============================================================================
// tb_snax_v_quant_path.sv
// 16 tokens x 16 channels -> transpose -> log_quantizer -> 16 FP4 packets
// =============================================================================
`timescale 1ns/1ps

module tb_snax_v_quant_path;

    localparam int DATA_WIDTH = 72;
    localparam int ADDR_WIDTH = 32;
    localparam int NUM_LANES  = 16;
    localparam int STRB_WIDTH = 9;
    localparam int CLK_PERIOD = 10;

    logic clk, rst_n;

    logic        csr_start, busy, done;
    logic [31:0] cfg_base, cfg_num_batches;

    logic [NUM_LANES*16-1:0] core_fp16;
    logic                    core_fp16_valid;
    logic                    core_fp16_ready;

    logic                  m_req_valid, m_req_write, m_req_ready;
    logic [ADDR_WIDTH-1:0] m_req_addr;
    logic [DATA_WIDTH-1:0] m_req_wdata;
    logic [STRB_WIDTH-1:0] m_req_strb;

    logic [DATA_WIDTH-1:0] tcdm [0:31][0:1023];

    logic [NUM_LANES-1:0][15:0] ref_fp16;
    logic                       ref_valid;
    logic                       ref_in_ready;
    logic [DATA_WIDTH-1:0]      ref_packet;
    logic                       ref_out_valid;
    logic [DATA_WIDTH-1:0]      expected [0:15];

    int t, c;
    int match_count, mismatch_count;

    snax_v_quant_path #(
        .NUM_LANES  (NUM_LANES),
        .DATA_WIDTH (DATA_WIDTH),
        .ADDR_WIDTH (ADDR_WIDTH),
        .STRB_WIDTH (STRB_WIDTH)
    ) dut (
        .clk                    (clk),
        .rst_n                  (rst_n),
        .csr_start_i            (csr_start),
        .busy_o                 (busy),
        .done_irq_o             (done),
        .cfg_base_addr_i        (cfg_base),
        .cfg_num_batches_i      (cfg_num_batches),
        .core_out_fp16_data_i   (core_fp16),
        .core_out_fp16_valid_i  (core_fp16_valid),
        .core_out_fp16_ready_o  (core_fp16_ready),
        .m_tcdm_req_valid_o     (m_req_valid),
        .m_tcdm_req_write_o     (m_req_write),
        .m_tcdm_req_addr_o      (m_req_addr),
        .m_tcdm_req_wdata_o     (m_req_wdata),
        .m_tcdm_req_strb_o      (m_req_strb),
        .m_tcdm_req_ready_i     (m_req_ready)
    );

    log_quantizer #(
        .BLOCK_SIZE   (NUM_LANES),
        .PACKET_WIDTH (DATA_WIDTH)
    ) u_ref_logq (
        .clk        (clk),
        .reset_n    (rst_n),
        .fp16_in    (ref_fp16),
        .in_valid   (ref_valid),
        .in_ready   (ref_in_ready),
        .packed_vec (ref_packet),
        .out_valid  (ref_out_valid),
        .out_ready  (1'b1)
    );

    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    function automatic [15:0] pat(input int tok, input int ch);
        pat = 16'hA000 + tok*16 + ch;
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
        $display("   V TRANSPOSE -> LOG QUANT -> FP4 UNIT TEST          ");
        $display("=====================================================\n");

        rst_n = 1'b0;
        csr_start = 1'b0;
        cfg_base = 32'h0000_0000;
        cfg_num_batches = 32'd1;
        core_fp16 = '0;
        core_fp16_valid = 1'b0;
        m_req_ready = 1'b1;
        ref_fp16 = '0;
        ref_valid = 1'b0;
        match_count = 0;
        mismatch_count = 0;

        wait_cycles(5);
        rst_n = 1'b1;
        wait_cycles(5);

        for (c = 0; c < NUM_LANES; c = c + 1) begin
            @(negedge clk);
            for (t = 0; t < NUM_LANES; t = t + 1) begin
                ref_fp16[t] = pat(t, c);
            end
            ref_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            ref_valid = 1'b0;

            while (!ref_out_valid) @(posedge clk);
            expected[c] = ref_packet;
            @(negedge clk);
        end

        @(negedge clk);
        csr_start = 1'b1;
        @(posedge clk);
        @(negedge clk);
        csr_start = 1'b0;
        wait_cycles(2);

        for (t = 0; t < NUM_LANES; t = t + 1) begin
            @(negedge clk);
            for (c = 0; c < NUM_LANES; c = c + 1) begin
                core_fp16[c*16 +: 16] = pat(t, c);
            end
            core_fp16_valid = 1'b1;
            @(posedge clk);
        end

        @(negedge clk);
        core_fp16_valid = 1'b0;
        core_fp16 = '0;

        while (busy) @(negedge clk);
        wait_cycles(3);

        for (c = 0; c < NUM_LANES; c = c + 1) begin
            if (tcdm[c][0] === expected[c]) begin
                match_count = match_count + 1;
            end else begin
                mismatch_count = mismatch_count + 1;
                $display("  [MISMATCH] channel=%0d actual=%018h expect=%018h",
                         c, tcdm[c][0], expected[c]);
            end
        end

        $display("\n-----------------------------------------------------");
        $display(" Packets checked : %0d", match_count + mismatch_count);
        $display(" Matches         : %0d", match_count);
        $display(" Mismatches      : %0d", mismatch_count);
        if (mismatch_count == 0 && match_count == NUM_LANES)
            $display(" >>> [PASS] V TRANSPOSE + LOG QUANT + FP4 WRITE <<<");
        else
            $display(" >>> [FAIL] V TRANSPOSE + LOG QUANT + FP4 WRITE <<<");
        $display("-----------------------------------------------------\n");
        $finish;
    end

endmodule
