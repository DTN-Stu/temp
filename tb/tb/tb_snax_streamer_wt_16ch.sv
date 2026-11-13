`timescale 1ns/1ps

// =============================================================================
// Functional TB for snax_streamer_wt_16ch FIFO-head replay
//
// Config: repeat_tokens=4, k_blocks=2, num_tiles=1, k_stride=0x200
// Expected:
//   - exactly 2 TCDM vector reads: 0x1000 and 0x1200
//   - exactly 8 core vector handshakes: 4x 0x1000 data, 4x 0x1200 data
// =============================================================================
module tb_snax_streamer_wt_16ch;

    localparam int DATA_WIDTH = 72;
    localparam int ADDR_WIDTH = 32;
    localparam int NUM_LANES  = 16;

    localparam logic [31:0] BASE_ADDR = 32'h0000_1000;
    localparam logic [31:0] K_STRIDE  = 32'h0000_0200;
    localparam int REPEAT_TOKENS = 4;
    localparam int K_BLOCKS      = 2;
    localparam int NUM_TILES     = 1;

    logic clk;
    logic rst_n;

    logic csr_start;
    logic busy;
    logic done_irq;

    logic [31:0] cfg_repeat_tokens;
    logic [31:0] cfg_k_blocks;
    logic [31:0] cfg_k_stride;
    logic [31:0] cfg_num_tiles;
    logic [31:0] cfg_tile_step;

    logic [NUM_LANES-1:0]                 m_req_valid;
    logic [NUM_LANES-1:0]                 m_req_write;
    logic [NUM_LANES-1:0][ADDR_WIDTH-1:0] m_req_addr;
    logic [NUM_LANES-1:0][DATA_WIDTH-1:0] m_req_wdata;
    logic [NUM_LANES-1:0][8:0]            m_req_strb;
    logic [NUM_LANES-1:0]                 m_req_ready;

    logic [NUM_LANES-1:0][DATA_WIDTH-1:0] m_rsp_rdata;
    logic [NUM_LANES-1:0]                 m_rsp_valid;

    logic [NUM_LANES-1:0][DATA_WIDTH-1:0] wt_data;
    logic [NUM_LANES-1:0]                 wt_valid;
    logic [NUM_LANES-1:0]                 wt_ready;

    int tcdm_vec_count;
    int core_vec_count;
    int errors;
    bit done_seen;

    // Synthetic TCDM data: unique per lane and address.
    function automatic logic [DATA_WIDTH-1:0] make_data(
        input logic [31:0] addr,
        input int          lane
    );
        return {8'hA5, addr[31:0], 16'h1234, lane[7:0], 8'h5A};
    endfunction

    snax_streamer_wt_16ch #(
        .DATA_WIDTH (DATA_WIDTH),
        .ADDR_WIDTH (ADDR_WIDTH),
        .NUM_LANES  (NUM_LANES),
        .STRB_WIDTH (9)
    ) dut (
        .clk                 (clk),
        .rst_n               (rst_n),
        .csr_start_i         (csr_start),
        .busy_o              (busy),
        .done_irq_o          (done_irq),
        .cfg_base_addr_i     (BASE_ADDR),
        .cfg_repeat_tokens_i (cfg_repeat_tokens),
        .cfg_k_blocks_i      (cfg_k_blocks),
        .cfg_k_stride_i      (cfg_k_stride),
        .cfg_num_tiles_i     (cfg_num_tiles),
        .cfg_tile_step_i     (cfg_tile_step),
        .m_tcdm_req_valid_o  (m_req_valid),
        .m_tcdm_req_write_o  (m_req_write),
        .m_tcdm_req_addr_o   (m_req_addr),
        .m_tcdm_req_wdata_o  (m_req_wdata),
        .m_tcdm_req_strb_o   (m_req_strb),
        .m_tcdm_req_ready_i  (m_req_ready),
        .m_tcdm_rsp_rdata_i  (m_rsp_rdata),
        .m_tcdm_rsp_valid_i  (m_rsp_valid),
        .wt_stream_data_o    (wt_data),
        .wt_stream_valid_o   (wt_valid),
        .wt_stream_ready_i   (wt_ready)
    );

    always #5 clk = ~clk;

    // -------------------------------------------------------------------------
    // TCDM model: accept every request, return data with 1-cycle latency.
    // -------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_rsp_valid <= '0;
            m_rsp_rdata <= '0;
        end
        else begin
            m_rsp_valid <= '0;
            for (int w = 0; w < NUM_LANES; w++) begin
                if (m_req_valid[w] && m_req_ready[w]) begin
                    m_rsp_valid[w] <= 1'b1;
                    m_rsp_rdata[w] <= make_data(m_req_addr[w], w);
                end
            end
        end
    end

    // -------------------------------------------------------------------------
    // Scoreboard
    // -------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tcdm_vec_count <= 0;
            core_vec_count <= 0;
            errors         <= 0;
            done_seen      <= 1'b0;
        end
        else begin
            if (done_irq) done_seen <= 1'b1;

            // One vector request: all lanes should carry the same row base.
            if (m_req_valid[0] && m_req_ready[0]) begin
                logic [31:0] exp_addr;
                exp_addr = BASE_ADDR + tcdm_vec_count * K_STRIDE;
                for (int w = 0; w < NUM_LANES; w++) begin
                    if (m_req_addr[w] !== (exp_addr + w * 16)) begin
                        errors <= errors + 1;
                        $display("[FAIL] TCDM vec=%0d lane=%0d addr=%h exp=%h",
                                 tcdm_vec_count, w, m_req_addr[w],
                                 exp_addr + w * 16);
                    end
                end
                tcdm_vec_count <= tcdm_vec_count + 1;
            end

            // One core vector handshake: replay group must match expected data.
            if (wt_valid == '1 && wt_ready == '1) begin
                logic [31:0] exp_addr;
                int vec_idx;
                vec_idx  = core_vec_count / REPEAT_TOKENS;
                exp_addr = BASE_ADDR + vec_idx * K_STRIDE;
                for (int w = 0; w < NUM_LANES; w++) begin
                    if (wt_data[w] !== make_data(exp_addr + w * 16, w)) begin
                        errors <= errors + 1;
                        $display("[FAIL] core vec=%0d lane=%0d data=%h exp=%h",
                                 core_vec_count, w, wt_data[w],
                                 make_data(exp_addr + w * 16, w));
                    end
                end
                core_vec_count <= core_vec_count + 1;
            end
        end
    end

    // -------------------------------------------------------------------------
    // Stimulus
    // -------------------------------------------------------------------------
    initial begin
        clk               = 1'b0;
        rst_n             = 1'b0;
        csr_start         = 1'b0;
        cfg_repeat_tokens = REPEAT_TOKENS;
        cfg_k_blocks      = K_BLOCKS;
        cfg_k_stride      = K_STRIDE;
        cfg_num_tiles     = NUM_TILES;
        cfg_tile_step     = '0;
        m_req_ready       = '1;
        wt_ready          = '1;

        repeat (5) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        @(negedge clk);
        csr_start = 1'b1;
        @(negedge clk);
        csr_start = 1'b0;

        // Wait for all core handshakes, then drain/done.
        wait (core_vec_count == REPEAT_TOKENS * K_BLOCKS * NUM_TILES);
        repeat (10) @(posedge clk);

        $display("");
        $display("TCDM vector requests : %0d (expected %0d)",
                 tcdm_vec_count, K_BLOCKS * NUM_TILES);
        $display("Core vector handshakes: %0d (expected %0d)",
                 core_vec_count, REPEAT_TOKENS * K_BLOCKS * NUM_TILES);
        $display("done_irq seen        : %0d", done_seen);
        $display("busy after done      : %0d", busy);
        $display("Errors               : %0d", errors);

        if (errors == 0 &&
            tcdm_vec_count == K_BLOCKS * NUM_TILES &&
            core_vec_count == REPEAT_TOKENS * K_BLOCKS * NUM_TILES &&
            done_seen && !busy)
            $display(">>> [PASS] WT STREAMER FIFO-HEAD REPLAY <<<");
        else
            $display(">>> [FAIL] WT STREAMER FIFO-HEAD REPLAY <<<");

        $finish;
    end

endmodule
