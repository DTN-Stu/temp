`timescale 1ns/1ps

module snax_transpose_buffer #(
    parameter int unsigned TOKENS    = 16,
    parameter int unsigned CHANNELS  = 16,
    parameter int unsigned DATA_WIDTH = 16
)(
    input  logic                                    clk,
    input  logic                                    rst_n,

    input  logic                                    wr_en,
    input  logic [$clog2(TOKENS)-1:0]               wr_row,
    input  logic [CHANNELS-1:0][DATA_WIDTH-1:0]     wr_data,

    input  logic [$clog2(CHANNELS)-1:0]             rd_col,
    output logic [TOKENS-1:0][DATA_WIDTH-1:0]       rd_data
);

    logic [DATA_WIDTH-1:0] mem [0:TOKENS-1][0:CHANNELS-1];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int t = 0; t < TOKENS; t++) begin
                for (int c = 0; c < CHANNELS; c++) begin
                    mem[t][c] <= '0;
                end
            end
        end else if (wr_en) begin
            for (int c = 0; c < CHANNELS; c++) begin
                mem[wr_row][c] <= wr_data[c];
            end
        end
    end

    always_comb begin
        for (int t = 0; t < TOKENS; t++) begin
            rd_data[t] = mem[t][rd_col];
        end
    end

endmodule
