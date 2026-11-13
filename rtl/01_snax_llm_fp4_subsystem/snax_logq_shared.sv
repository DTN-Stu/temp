// =============================================================================
// Module: snax_logq_shared
// Description: Single log_quantizer shared by three FP16 256b sources.
//
//   source 0 : core row FP16      -> activation FP4 packet
//   source 1 : V transpose column -> V FP4 packet
//   source 2 : TCDM readback FP16 -> requantized FP4 packet
//
// Arbitration is automatic with priority V > readback > core, because a V
// column is produced inside a transpose batch and cannot be stalled, while
// core row and readback traffic are buffered.
//
// Pipeline:
//   mux -> log_quantizer (latency 2) -> tagged FIFO -> demux by tag
// The tag travels with the packet so the output is routed to the source that
// produced it, regardless of arbitration changes while packets are in flight.
// =============================================================================
`timescale 1ns/1ps

module snax_logq_shared #(
    parameter int unsigned NUM_LANES  = 16,
    parameter int unsigned PKT_W      = 72,
    parameter int unsigned FIFO_DEPTH = 16
)(
    input  logic clk,
    input  logic rst_n,

    // source 0: core row FP16
    input  logic [NUM_LANES*16-1:0] core_fp16_data,
    input  logic                    core_fp16_valid,
    output logic                    core_fp16_ready,

    // source 1: V transpose column
    input  logic [NUM_LANES*16-1:0] v_col_data,
    input  logic                    v_col_valid,
    output logic                    v_col_ready,

    // source 2: TCDM readback FP16
    input  logic [NUM_LANES*16-1:0] rd_fp16_data,
    input  logic                    rd_fp16_valid,
    output logic                    rd_fp16_ready,

    // quantized packet outputs
    output logic [PKT_W-1:0]        core_pkt_data,
    output logic                    core_pkt_valid,
    input  logic                    core_pkt_ready,

    output logic [PKT_W-1:0]        v_pkt_data,
    output logic                    v_pkt_valid,
    input  logic                    v_pkt_ready,

    output logic [PKT_W-1:0]        rd_pkt_data,
    output logic                    rd_pkt_valid,
    input  logic                    rd_pkt_ready,

    // monitor: source owning the current FIFO head
    output logic [1:0]              active_src_o
);

    localparam int TAG_W  = 2;
    localparam int FIFO_W = PKT_W + TAG_W;

    // -------------------------------------------------------------------------
    // Source mux (priority V > readback > core)
    // -------------------------------------------------------------------------
    logic [1:0]              src_sel;
    logic                    src_valid;
    logic [NUM_LANES*16-1:0] src_data;

    always_comb begin
        if (v_col_valid) begin
            src_sel   = 2'd1;
            src_data  = v_col_data;
            src_valid = 1'b1;
        end else if (rd_fp16_valid) begin
            src_sel   = 2'd2;
            src_data  = rd_fp16_data;
            src_valid = 1'b1;
        end else begin
            src_sel   = 2'd0;
            src_data  = core_fp16_data;
            src_valid = core_fp16_valid;
        end
    end

    // -------------------------------------------------------------------------
    // Tagged output FIFO (generic elastic FIFO, unrelated to the P FIFO that
    // sits between softmax and the P*V matmul in the core).
    // -------------------------------------------------------------------------
    logic [FIFO_W-1:0] fifo_rdata;
    logic              fifo_push, fifo_pop, fifo_ready, fifo_valid;
    logic [4:0]        fifo_count;

    logic [PKT_W-1:0]  q_packet;
    logic              q_valid;

    // Reserve two slots for the quantizer latency: at most two packets can be
    // in flight outside the FIFO.
    wire issue_ok = (fifo_count < (FIFO_DEPTH - 2));
    wire issue    = src_valid && issue_ok;

    assign core_fp16_ready = (src_sel == 2'd0) && issue_ok;
    assign v_col_ready     = (src_sel == 2'd1) && issue_ok;
    assign rd_fp16_ready   = (src_sel == 2'd2) && issue_ok;

    logic [1:0] tag_d1, tag_d2;
    logic       issue_d1;

    // The tag pipeline must follow the quantizer's internal valid pipeline.
    // Advancing it unconditionally would attach the wrong tag whenever the
    // selected source changes while a packet is still in flight.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            issue_d1 <= 1'b0;
            tag_d1   <= 2'd0;
            tag_d2   <= 2'd0;
        end else begin
            issue_d1 <= issue;
            if (issue)
                tag_d1 <= src_sel;
            if (issue_d1)
                tag_d2 <= tag_d1;
        end
    end

    logic                  q_ready;

    log_quantizer #(
        .BLOCK_SIZE   (NUM_LANES),
        .PACKET_WIDTH (PKT_W)
    ) u_log_quantizer (
        .clk        (clk),
        .reset_n    (rst_n),
        .fp16_in    (src_data),
        .in_valid   (issue),
        .in_ready   (q_ready),
        .packed_vec (q_packet),
        .out_valid  (q_valid),
        .out_ready  (fifo_ready)
    );

    assign fifo_push = q_valid;

    skid_buffer #(
        .DATA_WIDTH (FIFO_W),
        .DEPTH      (FIFO_DEPTH)
    ) u_out_fifo (
        .clk      (clk),
        .rst_n    (rst_n),
        .push     (fifo_push),
        .data_in  ({tag_d2, q_packet}),
        .pop      (fifo_pop),
        .data_out (fifo_rdata),
        .ready    (fifo_ready),
        .valid    (fifo_valid),
        .level    (fifo_count)
    );

    wire fifo_empty = !fifo_valid;

    // -------------------------------------------------------------------------
    // Demux by tagged head
    // -------------------------------------------------------------------------
    wire [1:0]       head_tag = fifo_rdata[FIFO_W-1 -: TAG_W];
    wire [PKT_W-1:0] head_pkt = fifo_rdata[PKT_W-1:0];

    assign core_pkt_data = head_pkt;
    assign v_pkt_data    = head_pkt;
    assign rd_pkt_data   = head_pkt;

    assign core_pkt_valid = !fifo_empty && (head_tag == 2'd0);
    assign v_pkt_valid    = !fifo_empty && (head_tag == 2'd1);
    assign rd_pkt_valid   = !fifo_empty && (head_tag == 2'd2);

    assign fifo_pop = (head_tag == 2'd0) ? (core_pkt_valid && core_pkt_ready) :
                      (head_tag == 2'd1) ? (v_pkt_valid    && v_pkt_ready)    :
                      (head_tag == 2'd2) ? (rd_pkt_valid   && rd_pkt_ready)   :
                                           1'b0;

    assign active_src_o = head_tag;

endmodule
