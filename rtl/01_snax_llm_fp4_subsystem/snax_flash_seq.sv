// =============================================================================
// Module: snax_flash_seq
// Description: Synthesizable sequencer for the FlashAttention tiling loop nest
//              used by the FP4 core.  It replaces the TB task sequence:
//
//   for h, qi, dvi, kj:
//     start K streamer; start Q streamer per K-block; 8 QK beats
//     drain S slots 8..7+8 -> softmax
//     start V streamer; 8 P*V beats
//     final drain O slots 0..7 -> 1/l -> FP16
//
// The TCDM address map is fixed and matches tb_snax_core_flash_system:
//   Q  : bank 16, word = h*32 + q*2 + kb
//   K  : bank 0..15, word = h*4 + kj*2 + kb
//   V  : bank 0..15, word = 16 + h*4 + dvi*2 + kj
//   OUT: bank 17, word = 0..
// =============================================================================
`timescale 1ns/1ps

module snax_flash_seq #(
    parameter int H        = 2,
    parameter int N_Q      = 16,
    parameter int M_Q      = 8,
    parameter int D_K      = 32,
    parameter int D_V      = 32,
    parameter int KV_TILES = 2
)(
    input  logic        clk,
    input  logic        rst_n,

    input  logic        start_i,
    output logic        busy_o,
    output logic        done_o,

    // Activation / Q streamer
    output logic        act_start_o,
    output logic [31:0] act_base_o,
    output logic [31:0] act_k_blocks_o,
    output logic [31:0] act_k_stride_o,
    output logic [31:0] act_tokens_o,
    output logic [31:0] act_token_stride_o,
    input  logic        act_busy_i,
    input  logic        act_valid_i,
    output logic        act_ready_o,

    // 16-channel K/V streamer
    output logic        wt_start_o,
    output logic [31:0] wt_base_o,
    output logic [31:0] wt_repeat_o,
    output logic [31:0] wt_k_blocks_o,
    output logic [31:0] wt_k_stride_o,
    output logic [31:0] wt_num_tiles_o,
    output logic [31:0] wt_tile_step_o,
    input  logic        wt_busy_i,
    input  logic [15:0] wt_valid_i,
    output logic [15:0] wt_ready_o,

    // Core control
    output logic [1:0]  cfg_mode_o,
    output logic        cfg_row_start_o,
    output logic [15:0] cfg_key_mask_o,
    output logic        cfg_enable_slq_o,
    output logic        tile_drain_req_o,
    output logic [3:0]  tile_drain_idx_o,
    output logic        tile_clear_on_drain_o,
    output logic [15:0] lane_valid_o,
    output logic [3:0]  lane_acc_idx_o,
    input  logic        p_fifo_empty_i,
    input  logic        core_fp16_valid_i,

    // Output streamer
    output logic        wo_start_o,
    output logic [31:0] wo_base_o,
    output logic [31:0] wo_num_vectors_o,
    output logic [31:0] wo_line_stride_o
);

    localparam int K_BLOCKS = D_K / 16; // 2
    localparam int Q_TILES  = N_Q / M_Q; // 2
    localparam int DV_TILES = D_V / 16;  // 2
    localparam int WORD_STRIDE = 32'h200;
    localparam int NUM_OUT_VEC = H * Q_TILES * DV_TILES * M_Q; // 64

    typedef enum logic [4:0] {
        IDLE, INIT, INIT2,
        QK_WT, QK_WT2, QK_ACT, QK_RUN, QK_FIRE, QK_NEXT, QK_ACT_DRAIN, QK_WT_DRAIN,
        S_REQ, S_DEASSERT, S_ROW, S_WAIT,
        PV_WT, PV_WT2, PV_RUN, PV_FIRE, PV_NEXT, PV_DRAIN,
        FIN_REQ, FIN_NEXT, FIN_WAIT, DONE
    } state_t;

    state_t state_q;

    // The streamer configuration ports are live inputs, not latched by the
    // streamer.  Keep the V configuration stable for the whole PV phase.
    wire pv_phase = (state_q == PV_WT)   || (state_q == PV_WT2) ||
                    (state_q == PV_RUN)  || (state_q == PV_FIRE) ||
                    (state_q == PV_NEXT) || (state_q == PV_DRAIN);

    logic [1:0] h, qi, dvi, kj, kb;
    logic [2:0] m;
    logic [2:0] wait_cnt;
    logic [3:0] fin_cnt;
    logic       done_q;

    // -------------------------------------------------------------------------
    // Address helpers (fixed TCDM map)
    // -------------------------------------------------------------------------
    function automatic [31:0] q_addr(input logic [1:0] hh, input logic [4:0] qq,
                                     input logic [1:0] kkb);
        q_addr = (((hh * 32 + qq * 2 + kkb) << 9) | 32'h0000_0100);
    endfunction

    function automatic [31:0] k_addr(input logic [1:0] hh, input logic [1:0] kkj,
                                     input logic [1:0] kkb);
        k_addr = ((hh * 4 + kkj * 2 + kkb) << 9);
    endfunction

    function automatic [31:0] v_addr(input logic [1:0] hh, input logic [1:0] ddvi,
                                     input logic [1:0] kkj);
        v_addr = ((16 + hh * 4 + ddvi * 2 + kkj) << 9);
    endfunction

    // -------------------------------------------------------------------------
    // Outputs (combinational by state)
    // -------------------------------------------------------------------------
    always_comb begin
        act_start_o         = 1'b0;
        act_base_o          = q_addr(h, qi*M_Q, kb);
        act_k_blocks_o      = 32'd1;
        act_k_stride_o      = 32'd0;
        act_tokens_o        = M_Q;
        act_token_stride_o  = 2*WORD_STRIDE;
        act_ready_o         = 1'b0;

        wt_start_o          = 1'b0;
        wt_base_o           = pv_phase ? v_addr(h, dvi, kj) : k_addr(h, kj, 0);
        wt_repeat_o         = M_Q;
        wt_k_blocks_o       = pv_phase ? 32'd1 : K_BLOCKS;
        wt_k_stride_o       = pv_phase ? 32'd0 : WORD_STRIDE;
        wt_num_tiles_o      = 32'd1;
        wt_tile_step_o      = 32'd0;
        wt_ready_o          = 16'h0000;

        cfg_mode_o          = 2'b00;
        cfg_row_start_o     = 1'b0;
        cfg_key_mask_o      = 16'hFFFF;
        cfg_enable_slq_o    = 1'b1;

        tile_drain_req_o    = 1'b0;
        tile_drain_idx_o    = 4'd0;
        tile_clear_on_drain_o = 1'b0;

        lane_valid_o        = 16'h0000;
        lane_acc_idx_o      = 4'd0;

        wo_start_o          = 1'b0;
        wo_base_o           = 32'h0000_0110; // bank 17, word 0
        wo_num_vectors_o    = NUM_OUT_VEC;
        wo_line_stride_o    = WORD_STRIDE;

        case (state_q)
            INIT, INIT2: begin
                wo_start_o = 1'b1;
            end

            QK_WT, QK_WT2: begin
                wt_start_o     = 1'b1;
                wt_base_o      = k_addr(h, kj, 0);
                wt_repeat_o    = M_Q;
                wt_k_blocks_o  = K_BLOCKS;
                wt_k_stride_o  = WORD_STRIDE;
                wt_num_tiles_o = 32'd1;
                wt_tile_step_o = 32'd0;
                cfg_mode_o     = 2'b01;
            end

            QK_ACT: begin
                act_start_o        = 1'b1;
                act_base_o         = q_addr(h, qi*M_Q, kb);
                act_k_blocks_o     = 32'd1;
                act_k_stride_o     = 32'd0;
                act_tokens_o       = M_Q;
                act_token_stride_o = 2*WORD_STRIDE;
                cfg_mode_o         = 2'b01;
            end

            QK_RUN, QK_NEXT, QK_ACT_DRAIN, QK_WT_DRAIN: begin
                cfg_mode_o = 2'b01;
            end

            QK_FIRE: begin
                cfg_mode_o     = 2'b01;
                lane_valid_o   = 16'hFFFF;
                lane_acc_idx_o = {1'b1, m};
                act_ready_o    = 1'b1;
                wt_ready_o     = 16'hFFFF;
            end

            S_REQ: begin
                cfg_mode_o            = 2'b01;
                tile_drain_req_o      = 1'b1;
                tile_drain_idx_o      = {1'b1, m};
                tile_clear_on_drain_o = 1'b1;
            end

            S_DEASSERT: begin
                cfg_mode_o      = 2'b01;
                cfg_row_start_o = (kj == 2'd0);
            end

            S_ROW, S_WAIT: begin
                cfg_mode_o = 2'b01;
            end

            PV_WT, PV_WT2: begin
                cfg_mode_o     = 2'b10;
                wt_start_o     = 1'b1;
                wt_base_o      = v_addr(h, dvi, kj);
                wt_repeat_o    = M_Q;
                wt_k_blocks_o  = 32'd1;
                wt_k_stride_o  = 32'd0;
                wt_num_tiles_o = 32'd1;
                wt_tile_step_o = 32'd0;
            end

            PV_RUN, PV_NEXT, PV_DRAIN: begin
                cfg_mode_o = 2'b10;
            end

            PV_FIRE: begin
                cfg_mode_o     = 2'b10;
                lane_valid_o   = 16'hFFFF;
                lane_acc_idx_o = {1'b0, m};
                wt_ready_o     = 16'hFFFF;
            end

            FIN_REQ: begin
                cfg_mode_o            = 2'b10;
                tile_drain_req_o      = 1'b1;
                tile_drain_idx_o      = {1'b0, m};
                tile_clear_on_drain_o = 1'b1;
            end

            FIN_NEXT, FIN_WAIT: begin
                cfg_mode_o = 2'b10;
            end

            default: begin
            end
        endcase
    end

    // -------------------------------------------------------------------------
    // FSM
    // -------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q  <= IDLE;
            h <= '0; qi <= '0; dvi <= '0; kj <= '0; kb <= '0; m <= '0;
            wait_cnt <= '0; fin_cnt <= '0; done_q <= 1'b0;
        end else begin
            if (core_fp16_valid_i &&
                (state_q == FIN_REQ || state_q == FIN_NEXT || state_q == FIN_WAIT))
                fin_cnt <= fin_cnt + 1'b1;

            case (state_q)
                IDLE: begin
                    if (start_i) begin
                        h <= '0; qi <= '0; dvi <= '0; kj <= '0; kb <= '0; m <= '0;
                        wait_cnt <= '0; fin_cnt <= '0; done_q <= 1'b0;
                        state_q <= INIT;
                    end
                end

                INIT: begin
                    state_q <= INIT2;
                end

                INIT2: begin
                    state_q <= QK_WT;
                end

                QK_WT: begin
                    state_q <= QK_WT2;
                end

                QK_WT2: begin
                    state_q <= QK_ACT;
                end

                QK_ACT: begin
                    m <= '0;
                    state_q <= QK_RUN;
                end

                QK_RUN: begin
                    if (act_valid_i && (&wt_valid_i))
                        state_q <= QK_FIRE;
                end

                QK_FIRE: begin
                    state_q <= QK_NEXT;
                end

                QK_NEXT: begin
                    if (m < 3'd7) begin
                        m <= m + 1'b1;
                        state_q <= QK_RUN;
                    end else begin
                        state_q <= QK_ACT_DRAIN;
                    end
                end

                QK_ACT_DRAIN: begin
                    if (!act_busy_i) begin
                        if (kb < K_BLOCKS - 1) begin
                            kb <= kb + 1'b1;
                            state_q <= QK_ACT;
                        end else begin
                            state_q <= QK_WT_DRAIN;
                        end
                    end
                end

                QK_WT_DRAIN: begin
                    if (!wt_busy_i) begin
                        m <= '0;
                        state_q <= S_REQ;
                    end
                end

                S_REQ: begin
                    state_q <= S_DEASSERT;
                end

                S_DEASSERT: begin
                    state_q <= S_ROW;
                end

                S_ROW: begin
                    if (m < 3'd7) begin
                        m <= m + 1'b1;
                        state_q <= S_REQ;
                    end else begin
                        wait_cnt <= '0;
                        state_q <= S_WAIT;
                    end
                end

                S_WAIT: begin
                    if (wait_cnt < 3'd3) begin
                        wait_cnt <= wait_cnt + 1'b1;
                    end else begin
                        m <= '0;
                        state_q <= PV_WT;
                    end
                end

                PV_WT: begin
                    state_q <= PV_WT2;
                end

                PV_WT2: begin
                    m <= '0;
                    state_q <= PV_RUN;
                end

                PV_RUN: begin
                    if ((&wt_valid_i) && !p_fifo_empty_i)
                        state_q <= PV_FIRE;
                end

                PV_FIRE: begin
                    state_q <= PV_NEXT;
                end

                PV_NEXT: begin
                    if (m < 3'd7) begin
                        m <= m + 1'b1;
                        state_q <= PV_RUN;
                    end else begin
                        state_q <= PV_DRAIN;
                    end
                end

                PV_DRAIN: begin
                    if (!wt_busy_i) begin
                        if (kj < KV_TILES-1) begin
                            kj <= kj + 1'b1;
                            kb <= '0;
                            m <= '0;
                            state_q <= QK_WT;
                        end else begin
                            fin_cnt <= '0;
                            m <= '0;
                            state_q <= FIN_REQ;
                        end
                    end
                end

                FIN_REQ: begin
                    state_q <= FIN_NEXT;
                end

                FIN_NEXT: begin
                    if (m < 3'd7) begin
                        m <= m + 1'b1;
                        state_q <= FIN_REQ;
                    end else begin
                        state_q <= FIN_WAIT;
                    end
                end

                FIN_WAIT: begin
                    if (fin_cnt >= M_Q) begin
                        if (dvi < DV_TILES-1) begin
                            dvi <= dvi + 1'b1;
                            kj <= '0;
                            kb <= '0;
                            m <= '0;
                            state_q <= QK_WT;
                        end else if (qi < Q_TILES-1) begin
                            qi <= qi + 1'b1;
                            dvi <= '0;
                            kj <= '0;
                            kb <= '0;
                            m <= '0;
                            state_q <= QK_WT;
                        end else if (h < H-1) begin
                            h <= h + 1'b1;
                            qi <= '0;
                            dvi <= '0;
                            kj <= '0;
                            kb <= '0;
                            m <= '0;
                            state_q <= QK_WT;
                        end else begin
                            done_q <= 1'b1;
                            state_q <= DONE;
                        end
                    end
                end

                DONE: begin
                    if (start_i) begin
                        h <= '0; qi <= '0; dvi <= '0; kj <= '0; kb <= '0; m <= '0;
                        wait_cnt <= '0; fin_cnt <= '0; done_q <= 1'b0;
                        state_q <= INIT;
                    end
                end

                default: state_q <= IDLE;
            endcase
        end
    end

    assign busy_o = (state_q != IDLE) && (state_q != DONE);
    assign done_o = done_q;

endmodule
