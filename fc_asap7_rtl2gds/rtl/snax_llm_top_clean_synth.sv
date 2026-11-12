// =============================================================================
// snax_llm_top_clean_synth.sv
//
// Synthesis wrapper around the real top, snax_llm_top_clean.
// It only shrinks the TCDM so the design can be built with a standard-cell-only
// library (ASAP7 ships no SRAM macros).  snax_llm_top_clean itself infers
// NUM_BANKS x BANK_DEPTH x 72 = 32 x 1024 x 72 = 2.36 Mbit of storage.
//
// Defaults here: 8 banks x 64 words x 72 bit = 36,864 storage bits.
// Set USE_WRAPPER=0 in config.tcl (and supply SRAM macros) to synthesize the
// unmodified snax_llm_top_clean instead.
// =============================================================================
`timescale 1ns/1ps

module snax_llm_top_clean_synth #(
    parameter int unsigned NUM_BANKS  = 8,
    parameter int unsigned BANK_DEPTH = 64,
    parameter int unsigned NUM_LANES  = 16,
    parameter int unsigned ACC_DEPTH  = 16
)(
    input  logic          clk,
    input  logic          rst_n,

    // CPU CSR
    input  logic          cpu_csr_req_valid_i,
    output logic          cpu_csr_req_ready_o,
    input  logic          cpu_csr_req_write_i,
    input  logic [11:0]   cpu_csr_req_addr_i,
    input  logic [31:0]   cpu_csr_req_data_i,
    output logic          cpu_csr_rsp_valid_o,
    output logic [31:0]   cpu_csr_rsp_data_o,
    input  logic          cpu_csr_rsp_ready_i,

    // CPU TCDM master port
    input  logic          cpu_mem_req_valid_i,
    output logic          cpu_mem_req_ready_o,
    input  logic [31:0]   cpu_mem_req_addr_i,
    input  logic          cpu_mem_req_write_i,
    input  logic [71:0]   cpu_mem_req_wdata_i,
    input  logic [8:0]    cpu_mem_req_strb_i,
    output logic          cpu_mem_rsp_valid_o,
    output logic [71:0]   cpu_mem_rsp_rdata_o,

    // DMA TCDM master port
    input  logic          dma_mem_req_valid_i,
    output logic          dma_mem_req_ready_o,
    input  logic [31:0]   dma_mem_req_addr_i,
    input  logic          dma_mem_req_write_i,
    input  logic [71:0]   dma_mem_req_wdata_i,
    input  logic [8:0]    dma_mem_req_strb_i,
    output logic          dma_mem_rsp_valid_o,
    output logic [71:0]   dma_mem_rsp_rdata_o,

    output logic          subsystem_busy_o,
    output logic          subsystem_done_irq_o
);

    snax_llm_top_clean #(
        .DATA_WIDTH  (72),
        .STRB_WIDTH  (9),
        .ADDR_WIDTH  (32),
        .NUM_BANKS   (NUM_BANKS),
        .BANK_DEPTH  (BANK_DEPTH),
        .NUM_LANES   (NUM_LANES),
        .ACC_DEPTH   (ACC_DEPTH)
    ) u_snax_llm_top_clean (
        .clk                    (clk),
        .rst_n                  (rst_n),

        .cpu_csr_req_valid_i    (cpu_csr_req_valid_i),
        .cpu_csr_req_ready_o    (cpu_csr_req_ready_o),
        .cpu_csr_req_write_i    (cpu_csr_req_write_i),
        .cpu_csr_req_addr_i     (cpu_csr_req_addr_i),
        .cpu_csr_req_data_i     (cpu_csr_req_data_i),
        .cpu_csr_rsp_valid_o    (cpu_csr_rsp_valid_o),
        .cpu_csr_rsp_data_o     (cpu_csr_rsp_data_o),
        .cpu_csr_rsp_ready_i    (cpu_csr_rsp_ready_i),

        .cpu_mem_req_valid_i    (cpu_mem_req_valid_i),
        .cpu_mem_req_ready_o    (cpu_mem_req_ready_o),
        .cpu_mem_req_addr_i     (cpu_mem_req_addr_i),
        .cpu_mem_req_write_i    (cpu_mem_req_write_i),
        .cpu_mem_req_wdata_i    (cpu_mem_req_wdata_i),
        .cpu_mem_req_strb_i     (cpu_mem_req_strb_i),
        .cpu_mem_rsp_valid_o    (cpu_mem_rsp_valid_o),
        .cpu_mem_rsp_rdata_o    (cpu_mem_rsp_rdata_o),

        .dma_mem_req_valid_i    (dma_mem_req_valid_i),
        .dma_mem_req_ready_o    (dma_mem_req_ready_o),
        .dma_mem_req_addr_i     (dma_mem_req_addr_i),
        .dma_mem_req_write_i    (dma_mem_req_write_i),
        .dma_mem_req_wdata_i    (dma_mem_req_wdata_i),
        .dma_mem_req_strb_i     (dma_mem_req_strb_i),
        .dma_mem_rsp_valid_o    (dma_mem_rsp_valid_o),
        .dma_mem_rsp_rdata_o    (dma_mem_rsp_rdata_o),

        .subsystem_busy_o       (subsystem_busy_o),
        .subsystem_done_irq_o   (subsystem_done_irq_o)
    );

endmodule
