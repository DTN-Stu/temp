`timescale 1ns/1ps

// =============================================================================
// Module: p_fifo
// Description: Transient Ping-Pong / Ring FIFO for 72-bit FP4 Probability Packets.
//              Decouples Softmax Requantizer from MatMul 2 (P * V), keeping P
//              entirely on-chip without touching the Shared SRAM.
// =============================================================================
module p_fifo #(
    parameter int DATA_WIDTH = 72,
    parameter int DEPTH      = 4
)(
    input  logic                   clk,
    input  logic                   rst_n,
    input  logic                   clear,

    // Write Port (from softmax_requantizer)
    input  logic                   push,
    input  logic [DATA_WIDTH-1:0]  wdata,
    output logic                   full,

    // Read Port (to vsq_matmul_engine act_broadcast)
    input  logic                   pop,
    output logic [DATA_WIDTH-1:0]  rdata,
    output logic                   empty,
    output logic [$clog2(DEPTH):0] count
);

    localparam int AW = $clog2(DEPTH);

    logic [DATA_WIDTH-1:0] mem [0:DEPTH-1];
    logic [AW-1:0]         wptr, rptr;
    logic [$clog2(DEPTH):0] cnt;
    logic [DATA_WIDTH-1:0] rdata_q;

    assign empty = (cnt == 0);
    assign full  = (cnt == DEPTH);
    assign count = cnt;
    assign rdata = rdata_q;

    wire [AW-1:0] next_rptr = (rptr == AW'(DEPTH-1)) ? '0 : rptr + 1'b1;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wptr    <= '0;
            rptr    <= '0;
            cnt     <= '0;
            rdata_q <= '0;
        end else if (clear) begin
            wptr    <= '0;
            rptr    <= '0;
            cnt     <= '0;
            rdata_q <= '0;
        end else begin
            case ({push & ~full, pop & ~empty})
                2'b10: begin // Push only
                    mem[wptr] <= wdata;
                    wptr      <= (wptr == AW'(DEPTH-1)) ? '0 : wptr + 1'b1;
                    cnt       <= cnt + 1'b1;
                    if (cnt == 0) begin
                        rdata_q <= wdata;
                    end
                end
                2'b01: begin // Pop only
                    rptr <= next_rptr;
                    cnt  <= cnt - 1'b1;
                    if (cnt > 1) begin
                        rdata_q <= mem[next_rptr];
                    end
                end
                2'b11: begin // Simultaneous push and pop
                    mem[wptr] <= wdata;
                    wptr      <= (wptr == AW'(DEPTH-1)) ? '0 : wptr + 1'b1;
                    rptr      <= next_rptr;
                    if (cnt == 1) begin
                        rdata_q <= wdata;
                    end else begin
                        rdata_q <= mem[next_rptr];
                    end
                end
                default: ; // Idle
            endcase
        end
    end

endmodule
