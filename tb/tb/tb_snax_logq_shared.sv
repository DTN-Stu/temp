`timescale 1ns/1ps

// =============================================================================
// Unit TB for snax_logq_shared
//   - pushes N vectors from each of the three sources
//   - always drains all three output ports (a packet may still be in flight
//     when the source changes)
//   - checks every packet against a bit-exact reference of log_quantizer
//   - verifies the V > readback > core priority
// =============================================================================
module tb_snax_logq_shared;

    localparam int NUM_LANES = 16;
    localparam int PKT_W     = 72;
    localparam int N         = 8;
    localparam int MAXN      = 16;

    logic clk, rst_n;

    logic [NUM_LANES*16-1:0] core_data, v_data, rd_data;
    logic core_valid, v_valid, rd_valid;
    logic core_ready, v_ready, rd_ready;

    logic [PKT_W-1:0] core_pkt, v_pkt, rd_pkt;
    logic core_pkt_valid, v_pkt_valid, rd_pkt_valid;
    logic core_pkt_ready, v_pkt_ready, rd_pkt_ready;
    logic [1:0] active_src;

    logic [PKT_W-1:0] exp_core [0:MAXN-1];
    logic [PKT_W-1:0] exp_v    [0:MAXN-1];
    logic [PKT_W-1:0] exp_rd   [0:MAXN-1];

    int got_core, got_v, got_rd;
    int checks, errors;

    snax_logq_shared #(
        .NUM_LANES  (NUM_LANES),
        .PKT_W      (PKT_W),
        .FIFO_DEPTH (16)
    ) dut (
        .clk             (clk),
        .rst_n           (rst_n),
        .core_fp16_data  (core_data),
        .core_fp16_valid (core_valid),
        .core_fp16_ready (core_ready),
        .v_col_data      (v_data),
        .v_col_valid     (v_valid),
        .v_col_ready     (v_ready),
        .rd_fp16_data    (rd_data),
        .rd_fp16_valid   (rd_valid),
        .rd_fp16_ready   (rd_ready),
        .core_pkt_data   (core_pkt),
        .core_pkt_valid  (core_pkt_valid),
        .core_pkt_ready  (core_pkt_ready),
        .v_pkt_data      (v_pkt),
        .v_pkt_valid     (v_pkt_valid),
        .v_pkt_ready     (v_pkt_ready),
        .rd_pkt_data     (rd_pkt),
        .rd_pkt_valid    (rd_pkt_valid),
        .rd_pkt_ready    (rd_pkt_ready),
        .active_src_o    (active_src)
    );

    always #5 clk = ~clk;

    // Watchdog: never let the TB hang silently.
    initial begin
        #200000;
        $display("[FAIL] watchdog timeout: core=%0d v=%0d rd=%0d checks=%0d",
                 got_core, got_v, got_rd, checks);
        $finish;
    end

    assign core_pkt_ready = 1'b1;
    assign v_pkt_ready    = 1'b1;
    assign rd_pkt_ready   = 1'b1;

    // ------------------------------------------------------------------
    // Bit-exact reference of log_quantizer
    // ------------------------------------------------------------------
    function automatic logic [7:0] mu_of(input logic [15:0] v);
        logic [4:0] e;
        logic [9:0] m;
        e = v[14:10];
        m = v[9:0];
        if (e == 5'd0)       return 8'd0;
        else if (e == 5'd31) return 8'd247;
        else                 return {e, m[9:7]};
    endfunction

    function automatic logic [2:0] mag_of(input int diff);
        if      (diff <= 2)  return 3'b111;
        else if (diff <= 6)  return 3'b110;
        else if (diff <= 10) return 3'b101;
        else if (diff <= 14) return 3'b100;
        else if (diff <= 18) return 3'b011;
        else if (diff <= 24) return 3'b010;
        else if (diff <= 36) return 3'b001;
        else                 return 3'b000;
    endfunction

    function automatic logic [PKT_W-1:0] model_pkt(
        input logic [NUM_LANES*16-1:0] vec
    );
        logic [7:0]        mus [0:NUM_LANES-1];
        logic signed [8:0] raw;
        logic signed [7:0] scale;
        logic [7:0]        mu_max;
        logic [PKT_W-1:0]  pkt;
        logic [2:0]        mag;
        logic [3:0]        code;
        int i, diff;
        begin
            mu_max = 8'd0;
            for (i = 0; i < NUM_LANES; i++) begin
                mus[i] = mu_of(vec[i*16 +: 16]);
                if (mus[i] > mu_max) mu_max = mus[i];
            end

            if (mu_max == 8'd0) scale = 8'sd0;
            else begin
                raw = $signed({1'b0, mu_max}) - 9'sd140;
                if      (raw >  9'sd127) scale = 8'sd127;
                else if (raw < -9'sd128) scale = 8'sh80;
                else                     scale = raw[7:0];
            end

            pkt = '0;
            for (i = 0; i < NUM_LANES; i++) begin
                diff = mu_max - mus[i];
                mag  = (mus[i] == 8'd0 || mu_max == 8'd0) ? 3'b000 : mag_of(diff);
                code = (mag == 3'b000) ? 4'b0000 : {vec[i*16+15], mag};
                pkt[4*i +: 4] = code;
            end
            pkt[PKT_W-1 -: 8] = scale;
        end
        return pkt;
    endfunction

    function automatic logic [NUM_LANES*16-1:0] make_vec(
        input int src, input int idx
    );
        logic [NUM_LANES*16-1:0] v;
        for (int i = 0; i < NUM_LANES; i++)
            v[i*16 +: 16] = 16'h3C00 + ((src * 37 + idx * 11 + i * 3) % 96);
        return v;
    endfunction

    // ------------------------------------------------------------------
    // Monitor: drain every output port on every cycle
    // ------------------------------------------------------------------
    always @(negedge clk) begin
        if (rst_n) begin
            if (core_pkt_valid && core_pkt_ready) begin
                checks++;
                if (got_core >= MAXN || core_pkt !== exp_core[got_core]) begin
                    errors++;
                    $display("[FAIL] core[%0d] act=%018h exp=%018h",
                             got_core, core_pkt,
                             (got_core < MAXN) ? exp_core[got_core] : '0);
                end
                got_core++;
            end
            if (v_pkt_valid && v_pkt_ready) begin
                checks++;
                if (got_v >= MAXN || v_pkt !== exp_v[got_v]) begin
                    errors++;
                    $display("[FAIL] v[%0d] act=%018h exp=%018h",
                             got_v, v_pkt,
                             (got_v < MAXN) ? exp_v[got_v] : '0);
                end
                got_v++;
            end
            if (rd_pkt_valid && rd_pkt_ready) begin
                checks++;
                if (got_rd >= MAXN || rd_pkt !== exp_rd[got_rd]) begin
                    errors++;
                    $display("[FAIL] rd[%0d] act=%018h exp=%018h",
                             got_rd, rd_pkt,
                             (got_rd < MAXN) ? exp_rd[got_rd] : '0);
                end
                got_rd++;
            end
        end
    end

    // Push one vector of `src` on a cycle where the source is granted.
    task automatic push_one(input int src, input int idx);
        logic done;
        logic [NUM_LANES*16-1:0] vec;
        done = 1'b0;
        vec  = make_vec(src, idx);
        while (!done) begin
            @(negedge clk);
            core_valid = 1'b0;
            v_valid    = 1'b0;
            rd_valid   = 1'b0;
            case (src)
                0: if (core_ready) begin
                       core_data  = vec; core_valid = 1'b1;
                       exp_core[idx] = model_pkt(vec);
                       done = 1'b1;
                   end
                1: if (v_ready) begin
                       v_data  = vec; v_valid = 1'b1;
                       exp_v[idx] = model_pkt(vec);
                       done = 1'b1;
                   end
                default: if (rd_ready) begin
                       rd_data  = vec; rd_valid = 1'b1;
                       exp_rd[idx] = model_pkt(vec);
                       done = 1'b1;
                   end
            endcase
            if (done) @(negedge clk);
        end
        core_valid = 1'b0;
        v_valid    = 1'b0;
        rd_valid   = 1'b0;
    endtask

    initial begin
        clk = 1'b0;
        rst_n = 1'b0;
        core_data = '0; v_data = '0; rd_data = '0;
        core_valid = 1'b0; v_valid = 1'b0; rd_valid = 1'b0;
        got_core = 0; got_v = 0; got_rd = 0;
        checks = 0; errors = 0;

        repeat (3) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        for (int i = 0; i < N; i++) push_one(0, i);
        for (int i = 0; i < N; i++) push_one(1, i);
        for (int i = 0; i < N; i++) push_one(2, i);

        // let the last packets drain (quantizer latency + FIFO)
        repeat (12) @(negedge clk);

        if (got_core != N || got_v != N || got_rd != N) begin
            errors++;
            $display("[FAIL] counts: core=%0d v=%0d rd=%0d (expect %0d each)",
                     got_core, got_v, got_rd, N);
        end

        // Priority: V must win over core when both request together
        @(negedge clk);
        core_data  = make_vec(0, 12);
        core_valid = 1'b1;
        v_data     = make_vec(1, 12);
        v_valid    = 1'b1;
        exp_v[12]  = model_pkt(v_data);
        exp_core[12] = model_pkt(core_data);
        #1;
        if (!v_ready) begin
            errors++;
            $display("[FAIL] priority: V not granted while core requests");
        end else begin
            checks++;
            $display("[INFO] priority: V granted (core_ready=%0b)", core_ready);
        end
        @(negedge clk);
        core_valid = 1'b0;
        v_valid    = 1'b0;
        repeat (12) @(negedge clk);

        $display("");
        $display("Packets checked : %0d", checks);
        $display("Errors          : %0d", errors);
        if (errors == 0 && checks >= 25)
            $display(">>> [PASS] SHARED LOG-QUANTIZER 3:1 ROUTING <<<");
        else
            $display(">>> [FAIL] SHARED LOG-QUANTIZER <<<");
        $finish;
    end

endmodule
