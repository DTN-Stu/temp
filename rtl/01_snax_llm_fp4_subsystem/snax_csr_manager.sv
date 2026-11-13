`timescale 1ns/1ps

module snax_csr_manager #(
    parameter int unsigned NUM_RW     = 32,
    parameter int unsigned NUM_RO     = 8,
    parameter int unsigned ADDR_WIDTH = 12
)(
    input  logic                         clk,
    input  logic                         rst_n,

    // CPU / Snitch CSR request channel
    input  logic                         csr_req_valid_i,
    output logic                         csr_req_ready_o,
    input  logic                         csr_req_write_i,
    input  logic [ADDR_WIDTH-1:0]        csr_req_addr_i,
    input  logic [31:0]                  csr_req_data_i,

    // CPU / Snitch CSR response channel
    output logic                         csr_rsp_valid_o,
    output logic [31:0]                  csr_rsp_data_o,
    input  logic                         csr_rsp_ready_i,

    // Accelerator side: active RW configuration and bulk-commit
    output logic [NUM_RW-1:0][31:0]      cfg_rw_o,
    output logic [31:0]                  start_mask_o,
    output logic                         cfg_valid_o,
    input  logic                         cfg_ready_i,
    output logic                         cfg_commit_o,

    // Accelerator side: read-only registers
    input  logic [NUM_RO-1:0][31:0]      ro_i
);

    localparam logic [ADDR_WIDTH-1:0] ADDR_START  = ADDR_WIDTH'(NUM_RW - 1);
    localparam logic [ADDR_WIDTH-1:0] ADDR_RO0    = ADDR_WIDTH'(NUM_RW);
    localparam logic [ADDR_WIDTH-1:0] ADDR_RO_END = ADDR_WIDTH'(NUM_RW + NUM_RO);

    logic [NUM_RW-1:0][31:0] shadow_q;
    logic [NUM_RW-1:0][31:0] active_q;
    logic                    pending_q;
    logic                    commit_q;
    logic                    rsp_valid_q;
    logic [31:0]             rsp_data_q;

    wire req_fire = csr_req_valid_i && csr_req_ready_o;
    wire addr_is_rw   = (csr_req_addr_i < ADDR_START);
    wire addr_is_start= (csr_req_addr_i == ADDR_START);
    wire addr_is_ro   = (csr_req_addr_i >= ADDR_RO0) && (csr_req_addr_i < ADDR_RO_END);

    assign csr_req_ready_o = !rsp_valid_q && !pending_q;
    assign csr_rsp_valid_o = rsp_valid_q;
    assign csr_rsp_data_o  = rsp_data_q;

    assign cfg_rw_o     = active_q;
    assign start_mask_o = active_q[ADDR_START];
    assign cfg_valid_o  = pending_q;
    assign cfg_commit_o = commit_q;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < NUM_RW; i++) begin
                shadow_q[i] <= '0;
                active_q[i] <= '0;
            end
            pending_q  <= 1'b0;
            commit_q   <= 1'b0;
            rsp_valid_q <= 1'b0;
            rsp_data_q  <= '0;
        end else begin
            commit_q <= 1'b0;

            if (pending_q && cfg_ready_i) begin
                active_q  <= shadow_q;
                pending_q <= 1'b0;
                commit_q  <= 1'b1;
            end

            if (rsp_valid_q && csr_rsp_ready_i) begin
                rsp_valid_q <= 1'b0;
            end

            if (req_fire) begin
                rsp_valid_q <= 1'b1;

                if (csr_req_write_i) begin
                    rsp_data_q <= 32'd0;

                    if (addr_is_rw) begin
                        shadow_q[csr_req_addr_i] <= csr_req_data_i;
                    end else if (addr_is_start) begin
                        shadow_q[ADDR_START] <= csr_req_data_i;
                        pending_q <= csr_req_data_i[0];
                    end
                end else begin
                    if (addr_is_rw) begin
                        rsp_data_q <= shadow_q[csr_req_addr_i];
                    end else if (addr_is_start) begin
                        rsp_data_q <= shadow_q[ADDR_START];
                    end else if (addr_is_ro) begin
                        rsp_data_q <= ro_i[csr_req_addr_i - ADDR_RO0];
                    end else begin
                        rsp_data_q <= 32'd0;
                    end
                end
            end
        end
    end

endmodule
