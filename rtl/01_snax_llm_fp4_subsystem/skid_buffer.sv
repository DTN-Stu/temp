// =============================================================================
// Module: skid_buffer
// Description: Reusable elastic FIFO / skid buffer with ready/valid handshake.
//              DEPTH=2 reproduces the original 2-entry streamer buffer.
// =============================================================================
`timescale 1ns/1ps

module skid_buffer #(
    parameter int unsigned DATA_WIDTH = 72,
    parameter int unsigned DEPTH      = 2
) (
    input  logic                            clk,
    input  logic                            rst_n,

    input  logic                            push,
    input  logic [DATA_WIDTH-1:0]           data_in,
    input  logic                            pop,
    output logic [DATA_WIDTH-1:0]           data_out,

    output logic                            ready,   // push can be accepted
    output logic                            valid,   // data_out is valid
    output logic [$clog2(DEPTH+1)-1:0]      level
);

    localparam int unsigned PTR_W   = (DEPTH > 1) ? $clog2(DEPTH) : 1;
    localparam int unsigned LEVEL_W = $clog2(DEPTH + 1);

    logic [DATA_WIDTH-1:0] mem [0:DEPTH-1];
    logic [PTR_W-1:0]      wr_ptr;
    logic [PTR_W-1:0]      rd_ptr;
    logic [LEVEL_W-1:0]    count;

    wire push_fire = push && ready;
    wire pop_fire  = pop  && valid;

    assign ready    = (count < DEPTH);
    assign valid    = (count != 0);
    assign data_out = mem[rd_ptr];
    assign level    = count;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr <= '0;
            rd_ptr <= '0;
            count  <= '0;
        end
        else begin
            if (push_fire) begin
                mem[wr_ptr] <= data_in;
                if (wr_ptr == DEPTH - 1)
                    wr_ptr <= '0;
                else
                    wr_ptr <= wr_ptr + 1'b1;
            end

            if (pop_fire) begin
                if (rd_ptr == DEPTH - 1)
                    rd_ptr <= '0;
                else
                    rd_ptr <= rd_ptr + 1'b1;
            end

            if (push_fire && !pop_fire) begin
                count <= count + 1'b1;
            end
            else if (!push_fire && pop_fire) begin
                count <= count - 1'b1;
            end
        end
    end

endmodule
