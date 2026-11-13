// =============================================================================
// Module: snax_llm_top.sv
// Description: Top-Level Complete Subsystem tích hợp:
//              - 32-Bank Single-Port SRAM (DW=72, Strb=9, Tổng 288 KB)
//              - Interleaved Crossbar Interconnect (18 Masters x 32 Banks)
//              - 16x Weight Streamer Channels (1152 bits/cycle)
//              - 1x Activation Streamer Channel (72 bits/cycle với Freeze Mode)
//              - 1x Output Write Streamer (72-bit Packet)
//              - SNAX Fused LLM Shell (FP4 DotProduct + Exp-Free Softmax + Normalizer)
// =============================================================================
`timescale 1ns/1ps
`include "pe_defines.svh"

module snax_llm_top #(
    parameter int unsigned DATA_WIDTH     = 72,   // 72-bit Packet (64b FP4 + 8b Exp)
    parameter int unsigned STRB_WIDTH     = 9,    // 9 bytes per word
    parameter int unsigned ADDR_WIDTH     = 32,   // 32-bit Address
    parameter int unsigned NUM_BANKS      = 32,   // 32 Banks SRAM TCDM
    parameter int unsigned BANK_DEPTH     = 1024, // 1024 words x 9 bytes = 9 KB / bank (Tổng 288 KB)
    parameter int unsigned NUM_LANES_P    = NUM_LANES,        // 16 Lanes
    parameter int unsigned ACC_DEPTH_P    = ACCUMULATOR_DEPTH // 64 / 128
)(
    input  logic                                  clk,
    input  logic                                  rst_n,

    // -------------------------------------------------------------------------
    // 1. Host CSR Interface (Từ RISC-V Snitch Core / CPU Host)
    // -------------------------------------------------------------------------
    input  logic                                  host_csr_we_i,
    input  logic [11:0]                           host_csr_addr_i,
    input  logic [31:0]                           host_csr_wdata_i,
    output logic [31:0]                           host_csr_rdata_o,

    // -------------------------------------------------------------------------
    // 2. Host Direct Memory Access / iDMA Port (Nạp dữ liệu ngoài DRAM vào TCDM)
    // -------------------------------------------------------------------------
    input  logic                                  host_mem_req_valid_i,
    output logic                                  host_mem_req_ready_o,
    input  logic [ADDR_WIDTH-1:0]                 host_mem_req_addr_i,
    input  logic                                  host_mem_req_write_i,
    input  logic [DATA_WIDTH-1:0]                 host_mem_req_wdata_i,
    input  logic [STRB_WIDTH-1:0]                 host_mem_req_strb_i,

    output logic                                  host_mem_rsp_valid_o,
    output logic [DATA_WIDTH-1:0]                 host_mem_rsp_rdata_o,

    // -------------------------------------------------------------------------
    // 3. Status & Interrupts
    // -------------------------------------------------------------------------
    output logic                                  subsystem_busy_o,
    output logic                                  subsystem_done_irq_o
);

    localparam int unsigned NUM_MASTERS = 18;
    localparam int unsigned BANK_ADDR_W = $clog2(BANK_DEPTH);

    // =========================================================================
    // PHẦN 1: GIẢI MÃ ĐỊA CHỈ CSR CHO HỆ THỐNG
    // =========================================================================
    // 0x000 - 0x07F : Activation Streamer CSRs
    // 0x080 - 0x0FF : Weight Streamer CSRs
    // 0x100 - 0x17F : Output Streamer CSRs
    // 0x180 - 0x1FF : LLM Shell Wrapper CSRs

    logic [1:0] csr_target_sel;
    assign csr_target_sel = host_csr_addr_i[8:7];

    logic        csr_we_shell;
    logic [31:0] csr_rdata_shell;

    assign csr_we_shell = host_csr_we_i && (host_csr_addr_i[11:8] == 4'h1);

    always_comb begin
        if (host_csr_addr_i[11:8] == 4'h1) begin
            host_csr_rdata_o = csr_rdata_shell;
        end else begin
            host_csr_rdata_o = '0;
        end
    end

    // =========================================================================
    // PHẦN 2: TCDM INTERCONNECT WIRES (18 Masters x 32 Banks)
    // =========================================================================
    logic [NUM_MASTERS-1:0]                  m_req_valid;
    logic [NUM_MASTERS-1:0]                  m_req_ready;
    logic [NUM_MASTERS-1:0][ADDR_WIDTH-1:0]  m_req_addr;
    logic [NUM_MASTERS-1:0]                  m_req_write;
    logic [NUM_MASTERS-1:0][DATA_WIDTH-1:0]  m_req_wdata;
    logic [NUM_MASTERS-1:0][STRB_WIDTH-1:0]  m_req_strb;

    logic [NUM_MASTERS-1:0]                  m_rsp_valid;
    logic [NUM_MASTERS-1:0][DATA_WIDTH-1:0]  m_rsp_rdata;

    // SRAM Bank Ports
    logic [NUM_BANKS-1:0]                    b_req;
    logic [NUM_BANKS-1:0]                    b_write;
    logic [NUM_BANKS-1:0][BANK_ADDR_W-1:0]   b_addr;
    logic [NUM_BANKS-1:0][DATA_WIDTH-1:0]    b_wdata;
    logic [NUM_BANKS-1:0][STRB_WIDTH-1:0]    b_strb;
    logic [NUM_BANKS-1:0][DATA_WIDTH-1:0]    b_rdata;
    logic [NUM_BANKS-1:0]                    b_rvalid;

    // --- Master 0: Host / iDMA AXI Port ---
    assign m_req_valid[0]       = host_mem_req_valid_i;
    assign host_mem_req_ready_o = m_req_ready[0];
    assign m_req_addr[0]        = host_mem_req_addr_i;
    assign m_req_write[0]       = host_mem_req_write_i;
    assign m_req_wdata[0]       = host_mem_req_wdata_i;
    assign m_req_strb[0]        = host_mem_req_strb_i;
    assign host_mem_rsp_valid_o = m_rsp_valid[0];
    assign host_mem_rsp_rdata_o = m_rsp_rdata[0];

    // =========================================================================
    // PHẦN 3: KẾT NỐI STREAMERS VÀ CORE ACCELERATOR
    // =========================================================================
    logic [DATA_WIDTH-1:0]  act_stream_data;
    logic                   act_stream_valid;
    logic                   act_stream_ready;

    logic [NUM_LANES_P-1:0][DATA_WIDTH-1:0] wt_stream_data;
    logic [NUM_LANES_P-1:0] wt_stream_valid;
    logic [NUM_LANES_P-1:0] wt_stream_ready;

    logic [NUM_LANES_P*16-1:0] out_fp16_data;
    logic                      out_fp16_valid;
    logic [DATA_WIDTH-1:0]     out_fp4_data;
    logic                      out_fp4_valid;

    logic shell_busy, shell_done;

    assign subsystem_busy_o     = shell_busy;
    assign subsystem_done_irq_o = shell_done;

    // --- Master 1: Kênh Activation Streamer (72-bit) ---
    // Nối trực tiếp Response từ Interconnect vào Accelerator Core
    assign act_stream_data  = m_rsp_rdata[1];
    assign act_stream_valid = m_rsp_valid[1];

    // Phát request đọc Activation từ TCDM
    assign m_req_valid[1] = act_stream_ready;
    assign m_req_write[1] = 1'b0;
    assign m_req_addr[1]  = 32'h0000_1000; // Địa chỉ base của Activation (từ AGU)
    assign m_req_wdata[1] = '0;
    assign m_req_strb[1]  = '0;

    // --- Masters 2..17: 16 Kênh Weight Streamer (16 x 72-bit = 1152 bits/cycle) ---
    genvar w;
    generate
        for (w = 0; w < NUM_LANES_P; w++) begin : GEN_WT_STREAM_PORTS
            localparam int unsigned M_IDX = 2 + w;

            assign wt_stream_data[w]  = m_rsp_rdata[M_IDX];
            assign wt_stream_valid[w] = m_rsp_valid[M_IDX];

            // Mỗi lane yêu cầu 1 word 72-bit từ TCDM song song
            assign m_req_valid[M_IDX] = wt_stream_ready[w];
            assign m_req_write[M_IDX] = 1'b0;
            // Địa chỉ cách nhau theo bước nhảy block (tương thích Interleaved addressing)
            assign m_req_addr[M_IDX]  = 32'h0000_2000 + (w * 16); 
            assign m_req_wdata[M_IDX] = '0;
            assign m_req_strb[M_IDX]  = '0;
        end
    endgenerate

    // =========================================================================
    // PHẦN 4: KHỞI TẠO SHELL WRAPPER (snax_llm_shell)
    // =========================================================================
    snax_llm_shell #(
        .NUM_LANES_P (NUM_LANES_P),
        .ACC_DEPTH_P (ACC_DEPTH_P),
        .PKT_W       (DATA_WIDTH)
    ) u_llm_shell (
        .clk                (clk),
        .rst_n              (rst_n),
        .csr_we_i           (csr_we_shell),
        .csr_addr_i         (host_csr_addr_i[7:0]),
        .csr_wdata_i        (host_csr_wdata_i),
        .csr_rdata_o        (csr_rdata_shell),
        .act_stream_data_i  (act_stream_data),
        .act_stream_valid_i (act_stream_valid),
        .act_stream_ready_o (act_stream_ready),
        .wt_stream_data_i   (wt_stream_data),
        .wt_stream_valid_i  (wt_stream_valid),
        .wt_stream_ready_o  (wt_stream_ready),
        .out_fp16_data_o    (out_fp16_data),
        .out_fp16_valid_o   (out_fp16_valid),
        .out_fp16_ready_i   (1'b1),
        .out_fp4_data_o     (out_fp4_data),
        .out_fp4_valid_o    (out_fp4_valid),
        .out_fp4_ready_i    (1'b1),
        .engine_busy_o      (shell_busy),
        .engine_done_irq_o  (shell_done)
    );

    // =========================================================================
    // PHẦN 5: INTERLEAVED INTERCONNECT (18 Masters x 32 Banks)
    // =========================================================================
    interleaved_interconnect_72b #(
        .NUM_MASTERS (NUM_MASTERS),
        .NUM_BANKS   (NUM_BANKS),
        .DATA_WIDTH  (DATA_WIDTH),
        .ADDR_WIDTH  (ADDR_WIDTH),
        .BANK_DEPTH  (BANK_DEPTH)
    ) u_interconnect_32b (
        .clk           (clk),
        .rst_n         (rst_n),
        .m_req_valid_i (m_req_valid),
        .m_req_ready_o (m_req_ready),
        .m_req_addr_i  (m_req_addr),
        .m_req_write_i (m_req_write),
        .m_req_wdata_i (m_req_wdata),
        .m_req_strb_i  (m_req_strb),
        .m_rsp_valid_o (m_rsp_valid),
        .m_rsp_rdata_o (m_rsp_rdata),
        .b_req_o       (b_req),
        .b_write_o     (b_write),
        .b_addr_o      (b_addr),
        .b_wdata_o     (b_wdata),
        .b_strb_o      (b_strb),
        .b_rdata_i     (b_rdata),
        .b_rvalid_i    (b_rvalid)
    );

    // =========================================================================
    // PHẦN 6: MULTI-BANK TCDM SRAM (32 Banks x 72-bit Single-Port)
    // =========================================================================
    multi_bank_tcdm_72b #(
        .NUM_BANKS  (NUM_BANKS),
        .DATA_WIDTH (DATA_WIDTH),
        .STRB_WIDTH (STRB_WIDTH),
        .BANK_DEPTH (BANK_DEPTH)
    ) u_tcdm_sram_32b (
        .clk          (clk),
        .rst_n        (rst_n),
        .bank_req_i   (b_req),
        .bank_write_i (b_write),
        .bank_addr_i  (b_addr),
        .bank_wdata_i (b_wdata),
        .bank_strb_i  (b_strb),
        .bank_rdata_o (b_rdata),
        .bank_rvalid_o(b_rvalid)
    );

endmodule
