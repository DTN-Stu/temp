`timescale 1ns/1ps

module snax_flash_csr_snax (
    input  logic        clk,
    input  logic        rst_n,

    // SNAX CSR request / response
    input  logic        csr_req_valid_i,
    output logic        csr_req_ready_o,
    input  logic        csr_req_write_i,
    input  logic [11:0] csr_req_addr_i,
    input  logic [31:0] csr_req_data_i,
    output logic        csr_rsp_valid_o,
    output logic [31:0] csr_rsp_data_o,
    input  logic        csr_rsp_ready_i,

    input  logic        cfg_ready_i,

    input  logic        seq_busy,
    input  logic        seq_done,
    input  logic        act_busy,
    input  logic        wt_busy,
    input  logic        wo_busy,
    input  logic        v_busy,
    input  logic        v_done,
    input  logic        rd_busy,
    input  logic        rd_done,
    input  logic        p_fifo_full,
    input  logic        p_fifo_empty,
    input  logic        p_fifo_almost_full,
    input  logic        l_ready,
    input  logic        core_fp16_valid,

    output logic        start_pulse,
    output logic        src_sel,
    output logic        act_ready_force,
    output logic        wt_ready_force,

    output logic        act_start,
    output logic [31:0] act_base,
    output logic [31:0] act_k_blocks,
    output logic [31:0] act_k_stride,
    output logic [31:0] act_tokens,
    output logic [31:0] act_token_stride,

    output logic        wt_start,
    output logic [31:0] wt_base,
    output logic [31:0] wt_repeat,
    output logic [31:0] wt_k_blocks,
    output logic [31:0] wt_k_stride,
    output logic [31:0] wt_num_tiles,
    output logic [31:0] wt_tile_step,

    output logic        wo_start,
    output logic        wo_mode_fp16,
    output logic [31:0] wo_base,
    output logic [31:0] wo_num_vectors,
    output logic [31:0] wo_line_stride,

    output logic [1:0]  core_mode,
    output logic        core_row_start,
    output logic [15:0] core_key_mask,
    output logic        core_enable_slq,
    output logic        core_tile_drain_req,
    output logic [3:0]  core_tile_drain_idx,
    output logic        core_tile_clear_on_drain,
    output logic [15:0] core_lane_valid,
    output logic [3:0]  core_lane_acc_idx,

    output logic        v_start,
    output logic [31:0] v_base,
    output logic [31:0] v_num_batches,

    output logic        rd_start,
    output logic [31:0] rd_base,
    output logic [31:0] rd_num_vectors,
    output logic [31:0] rd_line_stride,

    output logic [1:0]  quant_src_sel,

    // Sequencer descriptor (all CPU-programmable)
    output logic [1:0]  seq_op_mode,        // 00 GEMM, 01 FlashAttention
    output logic        seq_is_causal,
    output logic        seq_enable_dma,
    output logic [7:0]  seq_attn_h,
    output logic [15:0] seq_attn_q_tiles,
    output logic [3:0]  seq_attn_m_q,
    output logic [15:0] seq_attn_dv_tiles,
    output logic [15:0] seq_attn_kv_tiles,
    output logic [15:0] seq_attn_k_blocks,
    output logic [15:0] seq_gemm_n_tiles,
    output logic [15:0] seq_gemm_m_tiles,
    output logic [15:0] seq_gemm_k_blocks,
    output logic [31:0] seq_tcdm_ring_base,
    output logic [31:0] seq_tcdm_q_base,
    output logic [31:0] seq_tcdm_out_base,
    output logic [31:0] seq_dram_kv_base,
    output logic [31:0] seq_stride_tile
);

    localparam int NUM_RW         = 48;
    localparam int NUM_RO         = 9;

    localparam int RW_CTRL        = 0;
    localparam int RW_ACT_BASE    = 1;
    localparam int RW_ACT_KB      = 2;
    localparam int RW_ACT_KS      = 3;
    localparam int RW_ACT_TOK     = 4;
    localparam int RW_ACT_TSTR    = 5;
    localparam int RW_WT_BASE     = 6;
    localparam int RW_WT_REP      = 7;
    localparam int RW_WT_KB       = 8;
    localparam int RW_WT_KS       = 9;
    localparam int RW_WT_NT       = 10;
    localparam int RW_WT_TS       = 11;
    localparam int RW_WO_MODE     = 12;
    localparam int RW_WO_BASE     = 13;
    localparam int RW_WO_NV       = 14;
    localparam int RW_WO_LS       = 15;
    localparam int RW_CORE_MODE   = 16;
    localparam int RW_CORE_ROW    = 17;
    localparam int RW_CORE_MASK   = 18;
    localparam int RW_CORE_SLQ    = 19;
    localparam int RW_CORE_DR     = 20;
    localparam int RW_CORE_DI     = 21;
    localparam int RW_CORE_CLR    = 22;
    localparam int RW_CORE_LV     = 23;
    localparam int RW_CORE_LA     = 24;
    localparam int RW_V_BASE      = 25;
    localparam int RW_V_NB        = 26;
    localparam int RW_RD_BASE     = 27;
    localparam int RW_RD_NV       = 28;
    localparam int RW_RD_LS       = 29;
    localparam int RW_QUANT       = 30;

    // Sequencer descriptor registers (index 31..45), start moves to 47.
    localparam int RW_SEQ_CTRL    = 31;
    localparam int RW_SEQ_ATTN_H  = 32;
    localparam int RW_SEQ_ATTN_QT = 33;
    localparam int RW_SEQ_ATTN_MQ = 34;
    localparam int RW_SEQ_ATTN_DV = 35;
    localparam int RW_SEQ_ATTN_KV = 36;
    localparam int RW_SEQ_ATTN_KB = 37;
    localparam int RW_SEQ_GEMM_NT = 38;
    localparam int RW_SEQ_GEMM_MT = 39;
    localparam int RW_SEQ_GEMM_KB = 40;
    localparam int RW_SEQ_RING    = 41;
    localparam int RW_SEQ_Q_BASE  = 42;
    localparam int RW_SEQ_OUT     = 43;
    localparam int RW_SEQ_DRAM_KV = 44;
    localparam int RW_SEQ_TILE_ST = 45;

    localparam int RW_START       = 47;

    logic [NUM_RW-1:0][31:0] cfg_rw;
    logic [31:0]             start_mask;
    logic                    cfg_valid;
    logic                    cfg_commit;
    logic [NUM_RO-1:0][31:0] ro;

    assign ro[0] = {30'd0, seq_done, seq_busy};
    assign ro[1] = {31'd0, act_busy};
    assign ro[2] = {31'd0, wt_busy};
    assign ro[3] = {31'd0, wo_busy};
    assign ro[4] = {30'd0, v_done, v_busy};
    assign ro[5] = {30'd0, rd_done, rd_busy};
    assign ro[6] = {28'd0, core_fp16_valid, l_ready, p_fifo_empty, p_fifo_full};
    assign ro[7] = 32'h534E_4146;
    assign ro[8] = {31'd0, p_fifo_almost_full};

    snax_csr_manager #(
        .NUM_RW     (NUM_RW),
        .NUM_RO     (NUM_RO),
        .ADDR_WIDTH (12)
    ) u_manager (
        .clk                (clk),
        .rst_n              (rst_n),
        
        .csr_req_valid_i     (csr_req_valid_i),
        .csr_req_ready_o     (csr_req_ready_o),
        .csr_req_write_i     (csr_req_write_i),
        .csr_req_addr_i      (csr_req_addr_i),
        .csr_req_data_i      (csr_req_data_i),
        .csr_rsp_valid_o     (csr_rsp_valid_o),
        .csr_rsp_data_o      (csr_rsp_data_o),
        .csr_rsp_ready_i     (csr_rsp_ready_i),
        .cfg_rw_o            (cfg_rw),
        .start_mask_o        (start_mask),
        .cfg_valid_o         (cfg_valid),
        .cfg_ready_i         (cfg_ready_i),
        .cfg_commit_o        (cfg_commit),
        .ro_i                (ro)
    );

    assign src_sel         = cfg_rw[RW_CTRL][0];
    assign act_ready_force = cfg_rw[RW_CTRL][1];
    assign wt_ready_force  = cfg_rw[RW_CTRL][2];

    // Last RW word: bit0 = commit, bit1 = seq, bit2 = act,
    // bit3 = wt, bit4 = wo, bit5 = v, bit6 = read.
    assign start_pulse = cfg_commit && start_mask[1];
    assign act_start   = cfg_commit && start_mask[2];
    assign wt_start    = cfg_commit && start_mask[3];
    assign wo_start    = cfg_commit && start_mask[4];
    assign v_start     = cfg_commit && start_mask[5];
    assign rd_start    = cfg_commit && start_mask[6];

    assign act_base         = cfg_rw[RW_ACT_BASE];
    assign act_k_blocks     = cfg_rw[RW_ACT_KB];
    assign act_k_stride     = cfg_rw[RW_ACT_KS];
    assign act_tokens       = cfg_rw[RW_ACT_TOK];
    assign act_token_stride = cfg_rw[RW_ACT_TSTR];

    assign wt_base       = cfg_rw[RW_WT_BASE];
    assign wt_repeat     = cfg_rw[RW_WT_REP];
    assign wt_k_blocks   = cfg_rw[RW_WT_KB];
    assign wt_k_stride   = cfg_rw[RW_WT_KS];
    assign wt_num_tiles  = cfg_rw[RW_WT_NT];
    assign wt_tile_step  = cfg_rw[RW_WT_TS];

    assign wo_mode_fp16   = cfg_rw[RW_WO_MODE][0];
    assign wo_base        = cfg_rw[RW_WO_BASE];
    assign wo_num_vectors = cfg_rw[RW_WO_NV];
    assign wo_line_stride = cfg_rw[RW_WO_LS];

    assign core_mode              = cfg_rw[RW_CORE_MODE][1:0];
    assign core_row_start         = cfg_rw[RW_CORE_ROW][0];
    assign core_key_mask          = cfg_rw[RW_CORE_MASK][15:0];
    assign core_enable_slq        = cfg_rw[RW_CORE_SLQ][0];
    assign core_tile_drain_req    = cfg_rw[RW_CORE_DR][0];
    assign core_tile_drain_idx    = cfg_rw[RW_CORE_DI][3:0];
    assign core_tile_clear_on_drain = cfg_rw[RW_CORE_CLR][0];
    assign core_lane_valid        = cfg_rw[RW_CORE_LV][15:0];
    assign core_lane_acc_idx      = cfg_rw[RW_CORE_LA][3:0];

    assign v_base        = cfg_rw[RW_V_BASE];
    assign v_num_batches = cfg_rw[RW_V_NB];
    assign rd_base       = cfg_rw[RW_RD_BASE];
    assign rd_num_vectors= cfg_rw[RW_RD_NV];
    assign rd_line_stride= cfg_rw[RW_RD_LS];
    assign quant_src_sel = cfg_rw[RW_QUANT][1:0];

    // Sequencer descriptor
    assign seq_op_mode      = cfg_rw[RW_SEQ_CTRL][1:0];
    assign seq_is_causal    = cfg_rw[RW_SEQ_CTRL][2];
    assign seq_enable_dma   = cfg_rw[RW_SEQ_CTRL][3];
    assign seq_attn_h       = cfg_rw[RW_SEQ_ATTN_H][7:0];
    assign seq_attn_q_tiles = cfg_rw[RW_SEQ_ATTN_QT][15:0];
    assign seq_attn_m_q     = cfg_rw[RW_SEQ_ATTN_MQ][3:0];
    assign seq_attn_dv_tiles= cfg_rw[RW_SEQ_ATTN_DV][15:0];
    assign seq_attn_kv_tiles= cfg_rw[RW_SEQ_ATTN_KV][15:0];
    assign seq_attn_k_blocks= cfg_rw[RW_SEQ_ATTN_KB][15:0];
    assign seq_gemm_n_tiles = cfg_rw[RW_SEQ_GEMM_NT][15:0];
    assign seq_gemm_m_tiles = cfg_rw[RW_SEQ_GEMM_MT][15:0];
    assign seq_gemm_k_blocks= cfg_rw[RW_SEQ_GEMM_KB][15:0];
    assign seq_tcdm_ring_base= cfg_rw[RW_SEQ_RING];
    assign seq_tcdm_q_base  = cfg_rw[RW_SEQ_Q_BASE];
    assign seq_tcdm_out_base= cfg_rw[RW_SEQ_OUT];
    assign seq_dram_kv_base = cfg_rw[RW_SEQ_DRAM_KV];
    assign seq_stride_tile  = cfg_rw[RW_SEQ_TILE_ST];

endmodule
