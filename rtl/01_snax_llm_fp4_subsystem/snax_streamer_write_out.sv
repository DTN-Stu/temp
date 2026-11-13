// =============================================================================
// Module: snax_streamer_write_out
// Description: Linear FP4 / FP16 output writer plus V FP4 batch writer.
//
// Modes:
//   2'd0 : FP4 linear   - one 72b packet per vector
//   2'd1 : FP16 linear  - one 256b FP16 vector -> 4 x 72b words
//   2'd2 : V FP4 batch  - cfg_num_vectors channel packets per batch
//
// V mode address:
//   addr = cfg_base_addr_i + batch * cfg_batch_stride_i
//                        + channel * cfg_line_stride_i
//   Typical V layout: cfg_num_vectors=16, cfg_line_stride=16,
//                     cfg_batch_stride=0x200.
//
// INPUT FIFO (this variant):
//   A VEC_W-wide elastic FIFO (skid_buffer) sits between the producer (core /
//   log-quantizer) and the TCDM write engine.  `core_out_*_ready_o` now reflects
//   FIFO space instead of the instantaneous TCDM write state, so the consumer
//   side can push vectors back-to-back while the engine drains 4 words per FP16
//   vector into TCDM.  This removes the per-vector handshake bubbles seen when
//   the write engine back-pressures the core on every 4-word burst.
//
//   FP4 / V records use the low DATA_WIDTH bits; FP16 records use the full
//   VEC_W bits.  The mode is fixed per job, so a single FIFO suffices.
// =============================================================================
`timescale 1ns/1ps

module snax_streamer_write_out #(
    parameter int unsigned DATA_WIDTH = 72,
    parameter int unsigned ADDR_WIDTH = 32,
    parameter int unsigned NUM_LANES  = 16,
    parameter int unsigned STRB_WIDTH = 9,
    parameter int unsigned FIFO_DEPTH = 16       // input vector FIFO depth
)(
    input  logic                            clk,
    input  logic                            rst_n,

    input  logic                            csr_start_i,
    output logic                            busy_o,
    output logic                            done_irq_o,

    input  logic [1:0]                      cfg_mode_i,          // 0 FP4, 1 FP16, 2 V FP4
    input  logic [ADDR_WIDTH-1:0]           cfg_base_addr_i,
    input  logic [31:0]                     cfg_num_vectors_i,   // packets per batch / linear vectors
    input  logic [31:0]                     cfg_line_stride_i,    // per packet / per FP16 word / per channel
    input  logic [31:0]                     cfg_num_batches_i,    // V mode batches
    input  logic [31:0]                     cfg_batch_stride_i,   // V mode batch stride

    input  logic [NUM_LANES*16-1:0]         core_out_fp16_data_i,
    input  logic                            core_out_fp16_valid_i,
    output logic                            core_out_fp16_ready_o,

    input  logic [DATA_WIDTH-1:0]           core_out_fp4_data_i,
    input  logic                            core_out_fp4_valid_i,
    output logic                            core_out_fp4_ready_o,

    output logic                            m_tcdm_req_valid_o,
    output logic                            m_tcdm_req_write_o,
    output logic [ADDR_WIDTH-1:0]           m_tcdm_req_addr_o,
    output logic [DATA_WIDTH-1:0]           m_tcdm_req_wdata_o,
    output logic [STRB_WIDTH-1:0]           m_tcdm_req_strb_o,
    input  logic                            m_tcdm_req_ready_i
);

    localparam logic [1:0] MODE_FP4  = 2'd0;
    localparam logic [1:0] MODE_FP16 = 2'd1;
    localparam logic [1:0] MODE_V    = 2'd2;

    localparam int VEC_W = NUM_LANES * 16;   // 256b full FP16 vector

    typedef enum logic [1:0] {
        IDLE,
        WRITE_FP4,
        WRITE_FP16,
        WRITE_V
    } state_t;

    state_t state_q, state_d;

    logic [31:0]           vec_cnt_q,    vec_cnt_d;
    logic [31:0]           batch_cnt_q,  batch_cnt_d;
    logic [ADDR_WIDTH-1:0] write_addr_q, write_addr_d;
    logic [ADDR_WIDTH-1:0] batch_base_q, batch_base_d;
    logic [1:0]            word_cnt_q,   word_cnt_d;
    logic                  start_q;

    // Job descriptor, sampled once at start.  The producer is free to change
    // cfg_* afterwards (the sequencer only presents the descriptor during its
    // launch state), so the engine must not read these live.
    logic [1:0]            mode_q;
    logic [ADDR_WIDTH-1:0] base_q;
    logic [31:0]           nvec_q;
    logic [31:0]           lstride_q;
    logic [31:0]           nbatch_q;
    logic [31:0]           bstride_q;

    wire tcdm_fire   = m_tcdm_req_valid_o && m_tcdm_req_ready_i;
    wire start_pulse = csr_start_i && !start_q;

    // Inside IDLE the live descriptor is used (it is what the launch samples);
    // once the job runs, only the latched copy is read.
    wire [1:0]            mode    = (state_q == IDLE) ? cfg_mode_i         : mode_q;
    wire [ADDR_WIDTH-1:0] base    = (state_q == IDLE) ? cfg_base_addr_i    : base_q;
    wire [31:0]           nvec    = (state_q == IDLE) ? cfg_num_vectors_i  : nvec_q;
    wire [31:0]           lstride = (state_q == IDLE) ? cfg_line_stride_i  : lstride_q;
    wire [31:0]           nbatch  = (state_q == IDLE) ? cfg_num_batches_i  : nbatch_q;
    wire [31:0]           bstride = (state_q == IDLE) ? cfg_batch_stride_i : bstride_q;

    // -------------------------------------------------------------------------
    // Input vector FIFO: decouples the producer from the 4-word TCDM burst.
    // -------------------------------------------------------------------------
    logic            in_fifo_pop;
    wire             in_fifo_ready;
    wire             in_fifo_valid;
    wire [VEC_W-1:0] in_fifo_rdata;
    wire [$clog2(FIFO_DEPTH+1)-1:0] in_fifo_level;

    wire             push_is_fp16 = (mode == MODE_FP16);
    wire             push_valid   = push_is_fp16 ? core_out_fp16_valid_i
                                                 : core_out_fp4_valid_i;
    wire [VEC_W-1:0] push_data    = push_is_fp16
        ? core_out_fp16_data_i
        : {{(VEC_W-DATA_WIDTH){1'b0}}, core_out_fp4_data_i};

    // Accept only while a job is active; ready mirrors FIFO space so the
    // producer can stream back-to-back until the FIFO fills.
    wire             push_en = (state_q != IDLE) && push_valid && in_fifo_ready;

    skid_buffer #(
        .DATA_WIDTH (VEC_W),
        .DEPTH      (FIFO_DEPTH)
    ) u_in_fifo (
        .clk      (clk),
        .rst_n    (rst_n),
        .push     (push_en),
        .data_in  (push_data),
        .pop      (in_fifo_pop),
        .data_out (in_fifo_rdata),
        .ready    (in_fifo_ready),
        .valid    (in_fifo_valid),
        .level    (in_fifo_level)
    );

    assign core_out_fp16_ready_o = (state_q != IDLE) && push_is_fp16 && in_fifo_ready;
    assign core_out_fp4_ready_o  = (state_q != IDLE) && !push_is_fp16 && in_fifo_ready;

    function automatic logic [DATA_WIDTH-1:0] pack_fp16_word(
        input logic [NUM_LANES*16-1:0] vec,
        input logic [1:0] word
    );
        pack_fp16_word = {8'h00,
                          vec[(word*4 + 3)*16 +: 16],
                          vec[(word*4 + 2)*16 +: 16],
                          vec[(word*4 + 1)*16 +: 16],
                          vec[(word*4 + 0)*16 +: 16]};
    endfunction

    always_comb begin
        state_d       = state_q;
        vec_cnt_d     = vec_cnt_q;
        batch_cnt_d   = batch_cnt_q;
        write_addr_d  = write_addr_q;
        batch_base_d  = batch_base_q;
        word_cnt_d    = word_cnt_q;
        done_irq_o    = 1'b0;
        in_fifo_pop   = 1'b0;

        m_tcdm_req_valid_o = 1'b0;
        m_tcdm_req_write_o = 1'b1;
        m_tcdm_req_addr_o  = write_addr_q;
        m_tcdm_req_wdata_o = '0;
        m_tcdm_req_strb_o  = '1;

        case (state_q)
            IDLE: begin
                if (start_pulse) begin
                    if ((nvec == 0) ||
                        ((mode == MODE_V) && (nbatch == 0))) begin
                        done_irq_o = 1'b1;
                    end else begin
                        vec_cnt_d     = '0;
                        batch_cnt_d   = '0;
                        write_addr_d  = base;
                        batch_base_d  = base;
                        word_cnt_d    = '0;
                        case (mode)
                            MODE_FP16: state_d = WRITE_FP16;
                            MODE_V:    state_d = WRITE_V;
                            default:   state_d = WRITE_FP4;
                        endcase
                    end
                end
            end
            // -------------------------------------------------------------
            // Linear FP4: one 72b packet per vector
            // -------------------------------------------------------------
            WRITE_FP4: begin
                if (in_fifo_valid) begin
                    m_tcdm_req_valid_o = 1'b1;
                    m_tcdm_req_wdata_o = in_fifo_rdata[DATA_WIDTH-1:0];

                    if (tcdm_fire) begin
                        in_fifo_pop  = 1'b1;
                        write_addr_d = write_addr_q + lstride;

                        if (vec_cnt_q < nvec - 1'b1) begin
                            vec_cnt_d = vec_cnt_q + 1'b1;
                        end else begin
                            done_irq_o = 1'b1;
                            state_d    = IDLE;
                        end
                    end
                end
            end

            // -------------------------------------------------------------
            // Linear FP16: one 256b vector -> 4 x 72b words.
            // The FIFO head is held (not popped) until the 4th word commits,
            // so the four words write back-to-back with no re-fetch bubble.
            // -------------------------------------------------------------
            WRITE_FP16: begin
                if (in_fifo_valid) begin
                    m_tcdm_req_valid_o = 1'b1;
                    m_tcdm_req_wdata_o = pack_fp16_word(in_fifo_rdata, word_cnt_q);

                    if (tcdm_fire) begin
                        write_addr_d = write_addr_q + lstride;

                        if (word_cnt_q < 2'd3) begin
                            word_cnt_d = word_cnt_q + 1'b1;
                        end else begin
                            word_cnt_d  = '0;
                            in_fifo_pop = 1'b1;   // whole vector committed

                            if (vec_cnt_q < nvec - 1'b1) begin
                                vec_cnt_d = vec_cnt_q + 1'b1;
                            end else begin
                                done_irq_o = 1'b1;
                                state_d    = IDLE;
                            end
                        end
                    end
                end
            end

            // -------------------------------------------------------------
            // V FP4 batch: cfg_num_vectors channel packets per batch
            // -------------------------------------------------------------
            WRITE_V: begin
                if (in_fifo_valid) begin
                    m_tcdm_req_valid_o = 1'b1;
                    m_tcdm_req_wdata_o = in_fifo_rdata[DATA_WIDTH-1:0];

                    if (tcdm_fire) begin
                        in_fifo_pop  = 1'b1;
                        write_addr_d = write_addr_q + lstride;

                        if (vec_cnt_q < nvec - 1'b1) begin
                            vec_cnt_d = vec_cnt_q + 1'b1;
                        end else begin
                            vec_cnt_d = '0;

                            if (batch_cnt_q < nbatch - 1'b1) begin
                                batch_cnt_d  = batch_cnt_q + 1'b1;
                                batch_base_d = batch_base_q + bstride;
                                write_addr_d = batch_base_q + bstride;
                            end else begin
                                done_irq_o = 1'b1;
                                state_d    = IDLE;
                            end
                        end
                    end
                end
            end

            default: state_d = IDLE;
        endcase
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q       <= IDLE;
            vec_cnt_q     <= '0;
            batch_cnt_q   <= '0;
            write_addr_q  <= '0;
            batch_base_q  <= '0;
            word_cnt_q    <= '0;
            start_q       <= 1'b0;
        end else begin
            state_q       <= state_d;
            vec_cnt_q     <= vec_cnt_d;
            batch_cnt_q   <= batch_cnt_d;
            write_addr_q  <= write_addr_d;
            batch_base_q  <= batch_base_d;
            word_cnt_q    <= word_cnt_d;
            start_q       <= csr_start_i;

            // Sample the whole descriptor on the launch edge; it stays valid for
            // the entire job even if the producer changes it afterwards.
            if (start_pulse) begin
                mode_q    <= cfg_mode_i;
                base_q    <= cfg_base_addr_i;
                nvec_q    <= cfg_num_vectors_i;
                lstride_q <= cfg_line_stride_i;
                nbatch_q  <= cfg_num_batches_i;
                bstride_q <= cfg_batch_stride_i;
            end
        end
    end

    assign busy_o = (state_q != IDLE) || (in_fifo_level != '0);

endmodule
