`timescale 1ns/1ps

// =============================================================================
// Module: snax_v_quant_path
// Description: V transpose front-end.  It gathers 16 core FP16 vectors into a
//              16x16 transpose buffer, then emits one column (16 tokens of one
//              output channel) per request to the shared log-quantizer and
//              forwards the returned FP4 packet to the write-out streamer.
//
// The log-quantizer itself lives in snax_logq_shared at top level.
// =============================================================================
module snax_v_quant_path #(
    parameter int unsigned NUM_LANES  = 16,
    parameter int unsigned DATA_WIDTH = 72,
    parameter int unsigned ADDR_WIDTH = 32,
    parameter int unsigned STRB_WIDTH = 9
)(
    input  logic                              clk,
    input  logic                              rst_n,

    input  logic                              csr_start_i,
    output logic                              busy_o,
    output logic                              done_irq_o,

    input  logic [ADDR_WIDTH-1:0]             cfg_base_addr_i,
    input  logic [31:0]                       cfg_num_batches_i,

    input  logic [NUM_LANES*16-1:0]           core_out_fp16_data_i,
    input  logic                              core_out_fp16_valid_i,
    output logic                              core_out_fp16_ready_o,

    // Column request to the shared log-quantizer
    output logic [NUM_LANES*16-1:0]           col_data_o,
    output logic                              col_valid_o,
    input  logic                              col_ready_i,

    // Quantized packet returned by the shared log-quantizer
    input  logic [DATA_WIDTH-1:0]             pkt_data_i,
    input  logic                              pkt_valid_i,
    output logic                              pkt_ready_o,

    // Packet forwarded to snax_streamer_write_out (V mode)
    output logic [DATA_WIDTH-1:0]             wo_pkt_data_o,
    output logic                              wo_pkt_valid_o,
    input  logic                              wo_pkt_ready_i
);

    typedef enum logic [2:0] {
        IDLE,
        GATHER,
        QUANT_REQ,
        QUANT_WAIT,
        WRITE
    } state_t;

    state_t state_q, state_d;

    logic [3:0]            gather_cnt_q, gather_cnt_d;
    logic [3:0]            ch_cnt_q,     ch_cnt_d;
    logic [31:0]           batch_cnt_q,  batch_cnt_d;
    logic [ADDR_WIDTH-1:0] base_q,       base_d;
    logic [DATA_WIDTH-1:0] pkt_q,        pkt_d;
    logic                  start_q;

    logic [NUM_LANES-1:0][15:0] col_data;
    wire start_pulse = csr_start_i && !start_q;

    snax_transpose_buffer #(
        .TOKENS     (16),
        .CHANNELS   (NUM_LANES),
        .DATA_WIDTH (16)
    ) u_transpose_buffer (
        .clk     (clk),
        .rst_n   (rst_n),
        .wr_en   (state_q == GATHER && core_out_fp16_valid_i),
        .wr_row  (gather_cnt_q),
        .wr_data (core_out_fp16_data_i),
        .rd_col  (ch_cnt_q),
        .rd_data (col_data)
    );

    assign col_data_o     = col_data;
    assign col_valid_o    = (state_q == QUANT_REQ);
    assign pkt_ready_o    = (state_q == QUANT_WAIT);
    assign wo_pkt_data_o  = pkt_q;
    assign wo_pkt_valid_o = (state_q == WRITE);

    always_comb begin
        state_d               = state_q;
        gather_cnt_d          = gather_cnt_q;
        ch_cnt_d              = ch_cnt_q;
        batch_cnt_d           = batch_cnt_q;
        base_d                = base_q;
        pkt_d                 = pkt_q;
        done_irq_o            = 1'b0;
        core_out_fp16_ready_o = 1'b0;

        case (state_q)
            IDLE: begin
                if (start_pulse) begin
                    if (cfg_num_batches_i == 0) begin
                        done_irq_o = 1'b1;
                    end else begin
                        gather_cnt_d = '0;
                        ch_cnt_d     = '0;
                        batch_cnt_d  = '0;
                        base_d       = cfg_base_addr_i;
                        pkt_d        = '0;
                        state_d      = GATHER;
                    end
                end
            end

            GATHER: begin
                core_out_fp16_ready_o = 1'b1;

                if (core_out_fp16_valid_i) begin
                    if (gather_cnt_q < 4'd15) begin
                        gather_cnt_d = gather_cnt_q + 1'b1;
                    end else begin
                        gather_cnt_d = '0;
                        ch_cnt_d     = '0;
                        state_d      = QUANT_REQ;
                    end
                end
            end

            QUANT_REQ: begin
                if (col_ready_i)
                    state_d = QUANT_WAIT;
            end

            QUANT_WAIT: begin
                if (pkt_valid_i) begin
                    pkt_d   = pkt_data_i;
                    state_d = WRITE;
                end
            end

            WRITE: begin
                if (wo_pkt_ready_i) begin
                    if (ch_cnt_q < 4'd15) begin
                        ch_cnt_d = ch_cnt_q + 1'b1;
                        state_d  = QUANT_REQ;
                    end else begin
                        ch_cnt_d = '0;

                        if (batch_cnt_q < cfg_num_batches_i - 1'b1) begin
                            batch_cnt_d = batch_cnt_q + 1'b1;
                            state_d     = GATHER;
                        end else begin
                            done_irq_o = 1'b1;
                            state_d    = IDLE;
                        end
                    end
                end
            end

            default: state_d = IDLE;
        endcase
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q      <= IDLE;
            gather_cnt_q <= '0;
            ch_cnt_q     <= '0;
            batch_cnt_q  <= '0;
            base_q       <= '0;
            pkt_q        <= '0;
            start_q      <= 1'b0;
        end else begin
            state_q      <= state_d;
            gather_cnt_q <= gather_cnt_d;
            ch_cnt_q     <= ch_cnt_d;
            batch_cnt_q  <= batch_cnt_d;
            base_q       <= base_d;
            pkt_q        <= pkt_d;
            start_q      <= csr_start_i;
        end
    end

    assign busy_o = (state_q != IDLE);

endmodule
