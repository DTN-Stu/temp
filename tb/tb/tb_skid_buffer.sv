`timescale 1ns/1ps

// =============================================================================
// Unit TB for skid_buffer (DEPTH=2)
// =============================================================================
module tb_skid_buffer;

    localparam int DATA_WIDTH = 72;

    logic clk;
    logic reset_n;

    logic                push;
    logic [DATA_WIDTH-1:0] data_in;
    logic                pop;
    logic [DATA_WIDTH-1:0] data_out;
    logic                ready;
    logic                valid;
    logic [1:0]          level;

    int checks;
    int errors;

    localparam logic [DATA_WIDTH-1:0] A = 72'h11_2222_3333_4444_5555;
    localparam logic [DATA_WIDTH-1:0] B = 72'hAA_BBBB_CCCC_DDDD_EEEE;
    localparam logic [DATA_WIDTH-1:0] C = 72'h01_2345_6789_ABCD_EF01;
    localparam logic [DATA_WIDTH-1:0] D = 72'hFE_DCBA_9876_5432_10FE;

    skid_buffer #(
        .DATA_WIDTH (DATA_WIDTH),
        .DEPTH      (2)
    ) dut (
        .clk      (clk),
        .rst_n    (reset_n),
        .push     (push),
        .data_in  (data_in),
        .pop      (pop),
        .data_out (data_out),
        .ready    (ready),
        .valid    (valid),
        .level    (level)
    );

    always #5 clk = ~clk;

    task automatic check(
        input string name,
        input logic  exp_ready,
        input logic  exp_valid,
        input logic [1:0] exp_level,
        input logic  check_data,
        input logic [DATA_WIDTH-1:0] exp_data
    );
        checks++;
        if (ready !== exp_ready || valid !== exp_valid || level !== exp_level) begin
            errors++;
            $display("[FAIL] %s: ready=%b/%b valid=%b/%b level=%0d/%0d",
                     name, ready, exp_ready, valid, exp_valid, level, exp_level);
        end
        if (check_data && data_out !== exp_data) begin
            errors++;
            $display("[FAIL] %s: data_out=%h expected=%h", name, data_out, exp_data);
        end
    endtask

    task automatic cycle(input logic p, input logic q,
                         input logic [DATA_WIDTH-1:0] d);
        @(negedge clk);
        push    = p;
        pop     = q;
        data_in = d;
        @(posedge clk);
        @(negedge clk);
        push    = 1'b0;
        pop     = 1'b0;
        data_in = '0;
    endtask

    initial begin
        clk     = 1'b0;
        reset_n = 1'b0;
        push    = 1'b0;
        pop     = 1'b0;
        data_in = '0;
        checks  = 0;
        errors  = 0;

        repeat (3) @(posedge clk);
        @(negedge clk);
        reset_n = 1'b1;
        @(negedge clk);

        check("reset", 1'b1, 1'b0, 2'd0, 1'b0, '0);

        cycle(1'b1, 1'b0, A);
        check("push A", 1'b1, 1'b1, 2'd1, 1'b1, A);

        cycle(1'b1, 1'b0, B);
        check("push B", 1'b0, 1'b1, 2'd2, 1'b1, A);

        cycle(1'b1, 1'b0, C);
        check("push while full", 1'b0, 1'b1, 2'd2, 1'b1, A);

        cycle(1'b0, 1'b1, '0);
        check("pop A", 1'b1, 1'b1, 2'd1, 1'b1, B);

        cycle(1'b1, 1'b1, C);
        check("push C + pop B", 1'b1, 1'b1, 2'd1, 1'b1, C);

        cycle(1'b1, 1'b0, D);
        check("push D", 1'b0, 1'b1, 2'd2, 1'b1, C);

        cycle(1'b0, 1'b1, '0);
        check("pop C", 1'b1, 1'b1, 2'd1, 1'b1, D);

        cycle(1'b0, 1'b1, '0);
        check("pop D", 1'b1, 1'b0, 2'd0, 1'b0, '0);

        cycle(1'b0, 1'b1, '0);
        check("pop while empty", 1'b1, 1'b0, 2'd0, 1'b0, '0);

        cycle(1'b1, 1'b0, A);
        cycle(1'b1, 1'b0, B);
        check("fill before reset", 1'b0, 1'b1, 2'd2, 1'b1, A);

        @(negedge clk);
        reset_n = 1'b0;
        @(posedge clk);
        @(negedge clk);
        check("reset while full", 1'b1, 1'b0, 2'd0, 1'b0, '0);

        reset_n = 1'b1;
        @(negedge clk);
        check("after reset", 1'b1, 1'b0, 2'd0, 1'b0, '0);

        $display("");
        $display("Checks : %0d", checks);
        $display("Errors : %0d", errors);
        if (errors == 0)
            $display(">>> [PASS] SKID BUFFER UNIT TEST <<<");
        else
            $display(">>> [FAIL] SKID BUFFER UNIT TEST <<<");

        $finish;
    end

endmodule
