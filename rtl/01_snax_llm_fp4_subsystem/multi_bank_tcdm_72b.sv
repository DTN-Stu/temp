// =============================================================================
// Module: multi_bank_tcdm_72b.sv
// Description: Multi-Bank Scratchpad Memory (TCDM) 32 Banks x 72-bit Data Width
//              - Mỗi Bank là 1 macro Single-Port (SP) SRAM đồng bộ (6T Bitcell)
//              - Hỗ trợ Byte Strobe 9-bit (72-bit = 9 bytes x 8 bits)
//              - Độ trễ đọc: đúng 1 chu kỳ clock (Single-cycle Read Latency)
// =============================================================================
`timescale 1ns/1ps

module sram_bank_72b #(
    parameter int unsigned DATA_WIDTH  = 72,
    parameter int unsigned STRB_WIDTH  = 9,
    parameter int unsigned BANK_DEPTH  = 1024,
    localparam int unsigned BANK_ADDR_W = $clog2(BANK_DEPTH)
)(
    input  logic                   clk,
    input  logic                   rst_n,

    input  logic                   req_i,
    input  logic                   write_i,
    input  logic [BANK_ADDR_W-1:0] addr_i,
    input  logic [DATA_WIDTH-1:0]  wdata_i,
    input  logic [STRB_WIDTH-1:0]  strb_i,

    output logic [DATA_WIDTH-1:0]  rdata_o,
    output logic                   rvalid_o
);

    // Mảng bộ nhớ SRAM (mỗi word gồm 9 bytes)
    logic [STRB_WIDTH-1:0][7:0] mem [BANK_DEPTH];

    // Pipeline register đọc 1 cycle
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rdata_o  <= '0;
            rvalid_o <= 1'b0;
        end else begin
            rvalid_o <= req_i && !write_i;
            if (req_i) begin
                if (write_i) begin
                    for (int b = 0; b < STRB_WIDTH; b++) begin
                        if (strb_i[b]) begin
                            mem[addr_i][b] <= wdata_i[b*8 +: 8];
                        end
                    end
                end else begin
                    rdata_o <= mem[addr_i];
                end
            end
        end
    end

endmodule


module multi_bank_tcdm_72b #(
    parameter int unsigned NUM_BANKS   = 32,
    parameter int unsigned DATA_WIDTH  = 72,
    parameter int unsigned STRB_WIDTH  = 9,
    parameter int unsigned BANK_DEPTH  = 1024,
    localparam int unsigned BANK_ADDR_W = $clog2(BANK_DEPTH)
)(
    input  logic                                          clk,
    input  logic                                          rst_n,

    // Giao tiếp kết nối từ Interconnect tới từng Bank
    input  logic [NUM_BANKS-1:0]                          bank_req_i,
    input  logic [NUM_BANKS-1:0]                          bank_write_i,
    input  logic [NUM_BANKS-1:0][BANK_ADDR_W-1:0]         bank_addr_i,
    input  logic [NUM_BANKS-1:0][DATA_WIDTH-1:0]          bank_wdata_i,
    input  logic [NUM_BANKS-1:0][STRB_WIDTH-1:0]          bank_strb_i,

    output logic [NUM_BANKS-1:0][DATA_WIDTH-1:0]          bank_rdata_o,
    output logic [NUM_BANKS-1:0]                          bank_rvalid_o
);

    genvar b;
    generate
        for (b = 0; b < NUM_BANKS; b++) begin : GEN_BANKS
            sram_bank_72b #(
                .DATA_WIDTH (DATA_WIDTH),
                .STRB_WIDTH (STRB_WIDTH),
                .BANK_DEPTH (BANK_DEPTH)
            ) u_bank (
                .clk      (clk),
                .rst_n    (rst_n),
                .req_i    (bank_req_i[b]),
                .write_i  (bank_write_i[b]),
                .addr_i   (bank_addr_i[b]),
                .wdata_i  (bank_wdata_i[b]),
                .strb_i   (bank_strb_i[b]),
                .rdata_o  (bank_rdata_o[b]),
                .rvalid_o (bank_rvalid_o[b])
            );
        end
    endgenerate

endmodule
