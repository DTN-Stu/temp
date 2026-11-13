`timescale 1ns/1ps

// =============================================================================
// Standalone TB for log_quantizer (slq_encoder.sv)
//
// Checks:
//   1. cfg_enable_slq_i=0 must block all FP4 output valid pulses.
//   2. One vector at a time: packet + out_valid timing (2-cycle latency,
//      exactly one valid cycle, then deassert).
//   3. Back-to-back streaming at 1 vector/cycle: every output packet matches
//      its golden, and valid is asserted for every output.
//   4. cfg_enable_slq_i=0 after reset/drain: no leakage.
//
// Signal names out_vec_fp4_packet_o / out_vec_fp4_valid_o match the core ports
// that are wired directly from log_quantizer.packed_vec / out_valid.
// =============================================================================
module tb_slq_encoder;

    localparam int BLOCK_SIZE = 16;
    localparam int PKT_W      = 72;
    localparam int MAX_VECS   = 2048;

    logic clk;
    logic reset_n;

    logic [BLOCK_SIZE-1:0][15:0] fp16_in;
    logic                        fp16_valid;
    logic                        cfg_enable_slq_i;
    logic                        in_valid;
    logic                        in_ready;

    logic [PKT_W-1:0]            out_vec_fp4_packet_o;
    logic                        out_vec_fp4_valid_o;

    logic [255:0] vecs [0:MAX_VECS-1];
    logic [PKT_W-1:0] exps [0:MAX_VECS-1];

    int num_vecs;
    int fd;
    int rc;
    int v;
    int errors;
    int valid_errors;
    int checked_one;
    int checked_stream;

    assign in_valid = fp16_valid & cfg_enable_slq_i & in_ready;

    log_quantizer #(
        .BLOCK_SIZE    (BLOCK_SIZE),
        .PACKET_WIDTH  (PKT_W)
    ) dut (
        .clk        (clk),
        .reset_n    (reset_n),
        .fp16_in    (fp16_in),
        .in_valid   (in_valid),
        .in_ready   (in_ready),
        .packed_vec (out_vec_fp4_packet_o),
        .out_valid  (out_vec_fp4_valid_o),
        .out_ready  (1'b1)
    );

    always #5 clk = ~clk;

    task automatic check_packet(
        input int          idx,
        input logic [71:0] exp,
        input logic [71:0] act,
        input string       tag
    );
        if (act !== exp) begin
            errors++;
            $display("[FAIL] %s vec=%0d exp=%h act=%h", tag, idx, exp, act);
            $display("       exp scale=%02h code0=%h | act scale=%02h code0=%h",
                     exp[71:64], exp[3:0], act[71:64], act[3:0]);
        end
    endtask

    initial begin
        clk = 1'b0;
        reset_n = 1'b0;
        fp16_in = '0;
        fp16_valid = 1'b0;
        cfg_enable_slq_i = 1'b0;
        errors = 0;
        valid_errors = 0;
        checked_one = 0;
        checked_stream = 0;

        $readmemh("sim_slq_encoder/hex/slq_input.hex", vecs);
        $readmemh("sim_slq_encoder/hex/slq_expected.hex", exps);

        fd = $fopen("sim_slq_encoder/hex/slq_count.txt", "r");
        if (fd == 0) begin
            $display("[FATAL] cannot open sim_slq_encoder/hex/slq_count.txt");
            $finish;
        end
        rc = $fscanf(fd, "%d", num_vecs);
        $fclose(fd);
        if (rc != 1 || num_vecs <= 0 || num_vecs > MAX_VECS) begin
            $display("[FATAL] bad vector count rc=%0d num=%0d", rc, num_vecs);
            $finish;
        end

        $display("=== SLQ encoder standalone test: %0d vectors ===", num_vecs);

        repeat (5) @(posedge clk);
        @(negedge clk);
        reset_n = 1'b1;

        // ------------------------------------------------------------------
        // Phase 0: cfg_enable_slq_i=0 must suppress valid
        // ------------------------------------------------------------------
        @(negedge clk);
        cfg_enable_slq_i = 1'b0;
        fp16_in = vecs[0];
        fp16_valid = 1'b1;
        repeat (5) begin
            @(negedge clk);
            if (out_vec_fp4_valid_o !== 1'b0) begin
                valid_errors++;
                $display("[FAIL] valid=1 while cfg_enable_slq_i=0");
            end
        end
        fp16_valid = 1'b0;
        repeat (2) @(negedge clk);
        cfg_enable_slq_i = 1'b1;
        repeat (2) @(negedge clk);

        // ------------------------------------------------------------------
        // Phase 1: one vector at a time, check latency + packet + deassert
        // ------------------------------------------------------------------
        for (v = 0; v < num_vecs; v++) begin
            @(negedge clk);
            fp16_in = vecs[v];
            fp16_valid = 1'b1;

            @(posedge clk);       // stage 1 samples input
            @(negedge clk);
            fp16_valid = 1'b0;

            @(posedge clk);       // stage 2 produces output
            @(negedge clk);
            if (out_vec_fp4_valid_o !== 1'b1) begin
                valid_errors++;
                $display("[FAIL] one-shot vec=%0d: out_valid=0", v);
            end
            else begin
                checked_one++;
                check_packet(v, exps[v], out_vec_fp4_packet_o, "one-shot");
            end

            @(posedge clk);       // drain
            @(negedge clk);
            if (out_vec_fp4_valid_o !== 1'b0) begin
                valid_errors++;
                $display("[FAIL] one-shot vec=%0d: out_valid stuck high", v);
            end
        end

        // ------------------------------------------------------------------
        // Phase 2: back-to-back streaming, 1 vector/cycle
        // ------------------------------------------------------------------
        @(negedge clk);
        for (v = 0; v < num_vecs; v++) begin
            fp16_in = vecs[v];
            fp16_valid = 1'b1;

            // At this negedge the DUT output is vector v-2.
            if (v >= 2) begin
                if (out_vec_fp4_valid_o !== 1'b1) begin
                    valid_errors++;
                    $display("[FAIL] stream vec=%0d: out_valid=0", v - 2);
                end
                else begin
                    checked_stream++;
                    check_packet(v - 2, exps[v - 2], out_vec_fp4_packet_o,
                                 "stream");
                end
            end
            @(negedge clk);
        end

        // Last two outputs, then the final deassert.
        fp16_valid = 1'b0;

        if (out_vec_fp4_valid_o !== 1'b1) begin
            valid_errors++;
            $display("[FAIL] stream tail vec=%0d: out_valid=0", num_vecs - 2);
        end
        else begin
            checked_stream++;
            check_packet(num_vecs - 2, exps[num_vecs - 2],
                         out_vec_fp4_packet_o, "stream-tail0");
        end

        @(negedge clk);
        if (out_vec_fp4_valid_o !== 1'b1) begin
            valid_errors++;
            $display("[FAIL] stream tail vec=%0d: out_valid=0", num_vecs - 1);
        end
        else begin
            checked_stream++;
            check_packet(num_vecs - 1, exps[num_vecs - 1],
                         out_vec_fp4_packet_o, "stream-tail1");
        end

        @(negedge clk);
        if (out_vec_fp4_valid_o !== 1'b0) begin
            valid_errors++;
            $display("[FAIL] stream drain: out_valid stuck high");
        end

        // ------------------------------------------------------------------
        // Phase 3: disable again after streaming, check no leakage
        // ------------------------------------------------------------------
        cfg_enable_slq_i = 1'b0;
        fp16_in = vecs[0];
        fp16_valid = 1'b1;
        repeat (4) begin
            @(negedge clk);
            if (out_vec_fp4_valid_o !== 1'b0) begin
                valid_errors++;
                $display("[FAIL] valid=1 after disabling cfg_enable_slq_i");
            end
        end
        fp16_valid = 1'b0;
        cfg_enable_slq_i = 1'b1;
        repeat (2) @(negedge clk);

        $display("");
        $display("Vector count        : %0d", num_vecs);
        $display("One-shot checked    : %0d", checked_one);
        $display("Streaming checked   : %0d", checked_stream);
        $display("Packet mismatches   : %0d", errors);
        $display("Valid timing errors : %0d", valid_errors);

        if (errors == 0 && valid_errors == 0 &&
            checked_one == num_vecs && checked_stream == num_vecs)
            $display(">>> [PASS] SLQ ENCODER PACKET + VALID TIMING <<<");
        else
            $display(">>> [FAIL] SLQ ENCODER <<<");

        $finish;
    end

endmodule
