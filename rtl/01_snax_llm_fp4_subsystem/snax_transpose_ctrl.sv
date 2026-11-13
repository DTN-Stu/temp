// =============================================================================
// Module: snax_transpose_ctrl
// Description: Address/sequencing controller for a 16x16 FP16 transpose buffer.
//
//   A transpose cannot stream: every TOKENS row must be written before any
//   column can be read out.  This block owns that phase sequence:
//
//     IDLE   -> start_i
//     GATHER -> accept TOKENS rows, one per row_valid_i handshake
//     ISSUE  -> request one transposed column (rd_col_o)
//     WAIT   -> hold until the quantizer pipeline is idle again
//     back to ISSUE for the next column; after TOKENS columns either the next
//     batch (GATHER) or done_o.
//
//   The buffer itself is a plain register file (snax_transpose_buffer); this
//   block only generates its write address, read address and enables.
//
//   col_valid_o is a single-cycle request: it is asserted in the cycle that
//   col_accept_i is observed, because the log-quantizer latches on in_valid
//   and has no ready, so a level would be sampled as several packets.
// =============================================================================
`timescale 1ns/1ps

module snax_transpose_ctrl #(
    parameter int unsigned TOKENS = 16,
    parameter int unsigned IDX_W  = $clog2(TOKENS)
)(
    input  logic              clk,
    input  logic              rst_n,

    // Job control
    input  logic              start_i,
    input  logic [31:0]       num_batches_i,
    output logic              busy_o,
    output logic              done_o,

    // Gather side (core FP16 rows in)
    input  logic              row_valid_i,
    output logic              row_ready_o,
    output logic              wr_en_o,
    output logic [IDX_W-1:0]  wr_row_o,

    // Emit side (transposed columns out to the shared quantizer)
    output logic              col_valid_o,
    input  logic              col_accept_i,
    input  logic              pipe_idle_i,
    output logic [IDX_W-1:0]  rd_col_o
);

    typedef enum logic [1:0] {
        IDLE, GATHER, ISSUE, WAIT
    } state_t;

    state_t             state_q, state_d;
    logic [IDX_W-1:0]   row_q,   row_d;
    logic [IDX_W-1:0]   col_q,   col_d;
    logic [31:0]        batch_q, batch_d;

    wire last_row   = (row_q   == TOKENS - 1);
    wire last_col   = (col_q   == TOKENS - 1);
    wire last_batch = (batch_q == num_batches_i - 1);

    always_comb begin
        state_d     = state_q;
        row_d       = row_q;
        col_d       = col_q;
        batch_d     = batch_q;
        done_o      = 1'b0;
        row_ready_o = 1'b0;
        col_valid_o = 1'b0;

        case (state_q)
            IDLE: begin
                if (start_i) begin
                    row_d   = '0;
                    col_d   = '0;
                    batch_d = '0;
                    if (num_batches_i == 0) begin
                        done_o = 1'b1;
                    end else begin
                        state_d = GATHER;
                    end
                end
            end

            GATHER: begin
                row_ready_o = 1'b1;
                if (row_valid_i) begin
                    if (last_row) begin
                        row_d   = '0;
                        col_d   = '0;
                        state_d = ISSUE;
                    end else begin
                        row_d = row_q + 1'b1;
                    end
                end
            end

            ISSUE: begin
                if (col_accept_i) begin
                    col_valid_o = 1'b1;
                    state_d     = WAIT;
                end
            end

            WAIT: begin
                if (pipe_idle_i) begin
                    if (last_col) begin
                        col_d = '0;
                        if (last_batch) begin
                            done_o  = 1'b1;
                            state_d = IDLE;
                        end else begin
                            batch_d = batch_q + 1'b1;
                            state_d = GATHER;
                        end
                    end else begin
                        col_d   = col_q + 1'b1;
                        state_d = ISSUE;
                    end
                end
            end

            default: state_d = IDLE;
        endcase
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q <= IDLE;
            row_q   <= '0;
            col_q   <= '0;
            batch_q <= '0;
        end else begin
            state_q <= state_d;
            row_q   <= row_d;
            col_q   <= col_d;
            batch_q <= batch_d;
        end
    end

    assign busy_o   = (state_q != IDLE);
    assign wr_en_o  = (state_q == GATHER) && row_valid_i;
    assign wr_row_o = row_q;
    assign rd_col_o = col_q;

endmodule
