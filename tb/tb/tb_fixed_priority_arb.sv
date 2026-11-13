`timescale 1ns/1ps

module tb_fixed_priority_arb;
    localparam int N = 19;

    logic [N-1:0] req;
    logic [N-1:0] grant;
    int errors;

    fixed_priority_arb_n #(.NUM_REQS(N), .HOST_MASTER(0)) dut (
        .req_i   (req),
        .grant_o (grant)
    );

    task automatic check(input [N-1:0] exp, input string name);
    begin
        #1;
        if (grant !== exp) begin
            errors = errors + 1;
            $display("[FAIL] %0s: grant=0x%05X exp=0x%05X", name, grant, exp);
        end else begin
            $display("[PASS] %0s: grant=0x%05X", name, grant);
        end
    end
    endtask

    initial begin
        errors = 0;
        req = '0;      check('0, "idle");
        req[0] = 1;    check(19'h00001, "host only");
        req[1] = 1;    check(19'h00002, "act beats host");
        req[2] = 1;    check(19'h00002, "act beats wt");
        req[1] = 0;    check(19'h00004, "wt only");
        req[18] = 1;   check(19'h00004, "wt beats out");
        req[2] = 0;    check(19'h40000, "out beats host");
        req[0] = 0;    check(19'h40000, "out only");
        req[1] = 1;    check(19'h00002, "act beats out");
        req[1] = 0;
        req[0] = 1;
        req[18] = 0;   check(19'h00001, "host when no streamer");

        $display("\nErrors: %0d", errors);
        if (errors == 0)
            $display(">>> [PASS] FIXED-PRIORITY ARBITER <<<");
        else
            $display(">>> [FAIL] FIXED-PRIORITY ARBITER <<<");
        $finish;
    end
endmodule
