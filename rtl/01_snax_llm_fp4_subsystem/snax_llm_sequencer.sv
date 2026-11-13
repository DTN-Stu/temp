// =============================================================================
// Module: snax_llm_sequencer.sv
// Description: Complete, Production-Grade Hardware Orchestration Sequencer for
//              SNAX FP4 LLM Subsystem.
//
// Key Features:
//   1. Multi-Mode Workload Support:
//      - OP_ATTN (FlashAttention: 4-level nested loops: Head, Q-tile, DV-slice, KV-tile)
//      - OP_GEMM (Dense GeMM: Output-Stationary over 16 Lanes x 16 Latch Slots)
//   2. Credit-Based TCDM Ring Buffer Management:
//      - Treats TCDM as an N-slot circular sliding window.
//      - Asynchronously issues DMA prefetch commands to hide DRAM latency.
//      - Credit tracking guarantees zero RAW / WAR hazards without lockstep stalls.
//   3. Dynamic Hardware Causal Mask Generator:
//      - Computes 16-bit causal masks on-the-fly from (q_idx, kv_idx).
//   4. Decoupled Event-Driven Execution:
//      - Zero-bubble handshakes, robust p_fifo backpressure handling.
// =============================================================================

`timescale 1ns/1ps

module snax_llm_sequencer #(
    parameter int unsigned ADDR_WIDTH      = 32,
    parameter int unsigned NUM_LANES       = 16,
    parameter int unsigned NUM_RING_SLOTS  = 4,   // TCDM Circular Buffer Depth
    parameter int unsigned SLOT_SIZE_BYTES = 16 * 1024 // 16 KB per Ring Slot
)(
    input  logic                                clk,
    input  logic                                rst_n,

    // =========================================================================
    // 1. TOP-LEVEL JOB CONTROL & CONFIGURATION DESCRIPTOR
    // =========================================================================
    input  logic                                start_i,
    output logic                                busy_o,
    output logic                                done_o,

    // Opcode selection
    input  logic [1:0]                          cfg_op_mode_i,       // 00: GEMM, 01: FlashAttention, 10: Transpose/Quant, 11: readback
    input  logic                                cfg_is_causal_i,     // 1: Enable causal autoregressive masking
    input  logic                                cfg_enable_dma_i,    // 1: Enable autonomous DMA prefetching via Ring Buffer

    // Attention Dimensions
    input  logic [7:0]                          cfg_attn_h_i,        // Number of Heads (H)
    input  logic [15:0]                         cfg_attn_q_tiles_i,  // Total Q-Tiles (N_Q / M_Q)
    input  logic [3:0]                          cfg_attn_m_q_i,      // Query tokens per tile (Default: 8)
    input  logic [15:0]                         cfg_attn_dv_tiles_i, // D_V Slices (D_V / 16)
    input  logic [15:0]                         cfg_attn_kv_tiles_i, // KV Tiles (SeqLen_KV / 16)
    input  logic [15:0]                         cfg_attn_k_blocks_i, // D_K Blocks (D_K / 16)

    // GEMM Dimensions
    input  logic [15:0]                         cfg_gemm_n_tiles_i,  // N Output Channel Tiles (N / 16)
    input  logic [15:0]                         cfg_gemm_m_tiles_i,  // M Token Tiles (M / 16)
    input  logic [15:0]                         cfg_gemm_k_blocks_i, // K Reduction Blocks (K / 16)

    // Base Addresses & Strides (TCDM & DRAM)
    input  logic [ADDR_WIDTH-1:0]               cfg_tcdm_ring_base_i,// TCDM Base for circular buffer
    input  logic [ADDR_WIDTH-1:0]               cfg_tcdm_q_base_i,   // TCDM Base for Q/Act
    input  logic [ADDR_WIDTH-1:0]               cfg_tcdm_out_base_i, // TCDM Base for Result Output
    input  logic [ADDR_WIDTH-1:0]               cfg_dram_kv_base_i,  // DRAM Base for KV-Cache / Weights
    input  logic [31:0]                         cfg_stride_tile_i,   // Stride between tiles in bytes

    // V transpose / readback descriptors
    input  logic [ADDR_WIDTH-1:0]               cfg_v_base_i,
    input  logic [31:0]                         cfg_v_num_batches_i,
    input  logic [ADDR_WIDTH-1:0]               cfg_v_out_base_i,
    input  logic [ADDR_WIDTH-1:0]               cfg_rd_base_i,
    input  logic [31:0]                         cfg_rd_num_vectors_i,
    input  logic [31:0]                         cfg_rd_line_stride_i,
    input  logic [ADDR_WIDTH-1:0]               cfg_rd_out_base_i,

    // =========================================================================
    // 2. DMA MASTER INTERFACE (MASTER 0 CO-PROCESSOR HANDSHAKE)
    // =========================================================================
    output logic                                dma_req_valid_o,
    input  logic                                dma_req_ready_i,
    output logic [1:0]                          dma_req_dir_o,       // 0: DRAM -> TCDM, 1: TCDM -> DRAM
    output logic [ADDR_WIDTH-1:0]               dma_req_dram_addr_o,
    output logic [ADDR_WIDTH-1:0]               dma_req_tcdm_addr_o,
    output logic [31:0]                         dma_req_size_bytes_o,
    input  logic                                dma_done_i,

    // =========================================================================
    // 3. ACTIVATION STREAMER INTERFACE (MASTER 2)
    // =========================================================================
    output logic                                act_start_o,
    output logic [ADDR_WIDTH-1:0]               act_base_o,
    output logic [31:0]                         act_k_blocks_o,
    output logic [31:0]                         act_k_stride_o,
    output logic [31:0]                         act_tokens_o,
    output logic [31:0]                         act_token_stride_o,
    input  logic                                act_busy_i,
    input  logic                                act_valid_i,
    output logic                                act_ready_o,

    // =========================================================================
    // 4. 16-CHANNEL WEIGHT / KV STREAMER INTERFACE (MASTERS 3..18)
    // =========================================================================
    output logic                                wt_start_o,
    output logic [ADDR_WIDTH-1:0]               wt_base_o,
    output logic [31:0]                         wt_repeat_o,
    output logic [31:0]                         wt_k_blocks_o,
    output logic [31:0]                         wt_k_stride_o,
    output logic [31:0]                         wt_num_tiles_o,
    output logic [31:0]                         wt_tile_step_o,
    input  logic                                wt_busy_i,
    input  logic [NUM_LANES-1:0]                wt_valid_i,
    output logic [NUM_LANES-1:0]                wt_ready_o,

    // =========================================================================
    // 5. WRITE-OUT STREAMER INTERFACE (MASTER 19)
    //    The sequencer owns the full descriptor: the top-level must not derive
    //    any of these from the mode, because the engine samples them live for
    //    the entire job.
    // =========================================================================
    output logic                                wo_start_o,
    output logic [1:0]                          wo_mode_o,           // 0: FP4 linear, 1: FP16 linear, 2: V FP4 batch
    output logic [ADDR_WIDTH-1:0]               wo_base_o,
    output logic [31:0]                         wo_num_vectors_o,
    output logic [31:0]                         wo_line_stride_o,
    output logic [31:0]                         wo_num_batches_o,
    output logic [31:0]                         wo_batch_stride_o,
    input  logic                                wo_busy_i,
    input  logic                                wo_done_i,

    // =========================================================================
    // 6. V TRANSPOSE / TCDM READBACK ENGINES (also sequenced here so that the
    //    write-out descriptor has exactly one owner)
    // =========================================================================
    output logic                                v_start_o,
    input  logic                                v_busy_i,
    input  logic                                v_done_i,
    output logic                                rd_start_o,
    input  logic                                rd_busy_i,
    input  logic                                rd_done_i,

    // =========================================================================
    // 6. SNAX FUSED LLM COMPUTE CORE CONTROL
    // =========================================================================
    output logic [1:0]                          cfg_mode_o,          // 00: GEMM, 01: QK, 10: PV
    output logic                                cfg_row_start_o,
    output logic [NUM_LANES-1:0]                cfg_key_mask_o,
    output logic                                cfg_enable_slq_o,
    output logic                                tile_drain_req_o,
    output logic [3:0]                          tile_drain_idx_o,
    output logic                                tile_clear_on_drain_o,
    output logic [NUM_LANES-1:0]                lane_valid_o,
    output logic [3:0]                          lane_acc_idx_o,
    input  logic                                p_fifo_empty_i,
    input  logic                                p_fifo_almost_full_i,
    input  logic                                core_fp16_valid_i,
    // Core output FIFO can still take the vector a new drain would produce.
    input  logic                                core_out_fifo_ready_i
);

    // =========================================================================
    // LOCAL TYPEDEFS & PARAMETERS
    // =========================================================================
    localparam int unsigned PTR_W = (NUM_RING_SLOTS > 1) ? $clog2(NUM_RING_SLOTS) : 1;
    localparam int unsigned CNT_W = $clog2(NUM_RING_SLOTS + 1);

    typedef enum logic [4:0] {
        ST_IDLE,
        ST_PREP_DRAIN,
        ST_PREFETCH_INIT,
        // FlashAttention Phases
        ST_ATTN_QK_LAUNCH,
        ST_ATTN_QK_RUN,
        ST_ATTN_SM_DRAIN,
        ST_ATTN_SM_WAIT,
        ST_ATTN_PV_LAUNCH,
        ST_ATTN_PV_RUN,
        ST_ATTN_PV_DRAIN,
        ST_ATTN_LAST_DRAIN,
        // GEMM Phases
        ST_GEMM_LAUNCH,
        ST_GEMM_RUN,
        ST_GEMM_DRAIN,
        ST_GEMM_DRAIN_WAIT,
        // Standalone engine opcodes
        ST_V_LAUNCH,
        ST_V_RUN,
        ST_RD_LAUNCH,
        ST_RD_RUN,
        // Epilogue
        ST_WAIT_WRITE_OUT,
        ST_DONE
    } state_t;

    state_t state_q, state_d;

    // =========================================================================
    // 1. CREDIT-BASED TCDM RING BUFFER CONTROLLER
    // =========================================================================
    logic [PTR_W-1:0] ring_wr_ptr_q;
    logic [PTR_W-1:0] ring_rd_ptr_q;
    logic [CNT_W-1:0] free_credits_q;
    logic [CNT_W-1:0] ready_tiles_q;

    wire dma_can_push = (free_credits_q > '0) && cfg_enable_dma_i;
    wire core_can_pop = (ready_tiles_q > '0)  || !cfg_enable_dma_i; // Bypass if DMA disabled

    logic dma_issue_fire;
    logic core_release_credit_pulse;
    logic [ADDR_WIDTH-1:0] dma_curr_tcdm_addr;
    logic [ADDR_WIDTH-1:0] core_curr_tcdm_addr;

    assign dma_curr_tcdm_addr  = cfg_tcdm_ring_base_i + ({16'd0, ring_wr_ptr_q} * SLOT_SIZE_BYTES);
    assign core_curr_tcdm_addr = cfg_enable_dma_i ? (cfg_tcdm_ring_base_i + ({16'd0, ring_rd_ptr_q} * SLOT_SIZE_BYTES)): cfg_tcdm_ring_base_i; // Fallback to direct base

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ring_wr_ptr_q   <= '0;
            ring_rd_ptr_q   <= '0;
            free_credits_q  <= NUM_RING_SLOTS[CNT_W-1:0];
            ready_tiles_q   <= '0;
        end else if (start_i) begin
            ring_wr_ptr_q   <= '0;
            ring_rd_ptr_q   <= '0;
            free_credits_q  <= NUM_RING_SLOTS[CNT_W-1:0];
            ready_tiles_q   <= '0;
        end else begin
            // Producer (DMA) credit consumption
            if (dma_issue_fire && !core_release_credit_pulse)
                free_credits_q <= free_credits_q - 1'b1;
            else if (!dma_issue_fire && core_release_credit_pulse)
                free_credits_q <= free_credits_q + 1'b1;

            if (dma_issue_fire)
                ring_wr_ptr_q <= (ring_wr_ptr_q == PTR_W'(NUM_RING_SLOTS - 1)) ? '0 : ring_wr_ptr_q + 1'b1;

            // Consumer (Core) ready tile update
            if (dma_done_i && !core_release_credit_pulse)
                ready_tiles_q <= ready_tiles_q + 1'b1;
            else if (!dma_done_i && core_release_credit_pulse)
                ready_tiles_q <= ready_tiles_q - 1'b1;

            if (core_release_credit_pulse)
                ring_rd_ptr_q <= (ring_rd_ptr_q == PTR_W'(NUM_RING_SLOTS - 1)) ? '0 : ring_rd_ptr_q + 1'b1;
        end
    end

    // =========================================================================
    // 2. 5D HIERARCHICAL LOOP NEST ENGINE
    // =========================================================================
    // Attention Counters
    logic [7:0]   cnt_h_q;
    logic [15:0]  cnt_qi_q;
    logic [15:0]  cnt_dvi_q;
    logic [15:0]  cnt_kj_q;
    logic [3:0]   cnt_m_q;
    logic [3:0]   cnt_dr_q;      // per-slot drain index (softmax / output)
    logic         kv_first_q;    // the current KV tile is the first of a row
    logic         last_group_q;  // the group just retired was the final one

    // -------------------------------------------------------------------------
    // Background O-slot drain engine.
    //
    // The core exposes the accumulator drain on a second read port and does not
    // gate the write port with it, so the O vectors of group N can be drained
    // while the S slots of group N+1 are being filled.  O lives in slots 0..7
    // and S in 8..15, so the two engines never address the same entry, and each
    // drain masks its own slot on completion (tile_clear_on_drain).
    //
    // The SM drain owns tile_drain_req while it runs; the engine waits for a
    // free cycle instead of stealing the port.
    // -------------------------------------------------------------------------
    typedef enum logic [1:0] { NRM_IDLE, NRM_REQ, NRM_WAIT } nrm_state_t;
    nrm_state_t nrm_state_q, nrm_state_d;

    logic [3:0] nrm_dr_q, nrm_dr_d;
    logic       nrm_seen_q, nrm_seen_d;
    logic       nrm_req;        // drain pulse generated by the engine
    logic       nrm_start_o;    // main FSM hands a finished group over

    wire nrm_busy = (nrm_state_q != NRM_IDLE);

    always_comb begin
        nrm_state_d = nrm_state_q;
        nrm_dr_d    = nrm_dr_q;
        nrm_seen_d  = nrm_seen_q;
        nrm_req     = 1'b0;

        case (nrm_state_q)
            NRM_IDLE: begin
                if (nrm_start_o) begin
                    nrm_dr_d    = '0;
                    nrm_seen_d  = 1'b0;
                    nrm_state_d = NRM_REQ;
                end
            end
            NRM_REQ: begin
                if (state_q != ST_ATTN_SM_DRAIN) begin
                    nrm_req     = 1'b1;
                    nrm_seen_d  = 1'b0;
                    nrm_state_d = NRM_WAIT;
                end
            end
            NRM_WAIT: begin
                // The core holds one FP16 vector: wait for it to appear and then
                // be consumed before requesting the next slot.
                if (core_fp16_valid_i) nrm_seen_d = 1'b1;
                if (nrm_seen_q && !core_fp16_valid_i) begin
                    if (nrm_dr_q == cfg_attn_m_q_i - 4'd1) begin
                        nrm_dr_d    = '0;
                        nrm_state_d = NRM_IDLE;
                    end else begin
                        nrm_dr_d    = nrm_dr_q + 4'd1;
                        nrm_state_d = NRM_REQ;
                    end
                end
            end
            default: nrm_state_d = NRM_IDLE;
        endcase
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            nrm_state_q <= NRM_IDLE;
            nrm_dr_q    <= '0;
            nrm_seen_q  <= 1'b0;
        end else if (start_i) begin
            nrm_state_q <= NRM_IDLE;
            nrm_dr_q    <= '0;
            nrm_seen_q  <= 1'b0;
        end else begin
            nrm_state_q <= nrm_state_d;
            nrm_dr_q    <= nrm_dr_d;
            nrm_seen_q  <= nrm_seen_d;
        end
    end

    // Attention Addresses.
    //
    // Everything is a multiple of the 0x200-byte tile block, and one K block
    // holds 16 D_K elements, so both the per-tile and per-head strides scale
    // with cfg_attn_k_blocks_i.  The K-block term is NOT part of these bases:
    // the streamers walk the K blocks themselves through their own AGUs, using
    // wt_k_stride / act_k_stride, so D_K is not capped by the address map.
    //
    //   Q[h][qi][m][kb] : q_base    + (h*Q_TILES*M_Q*KB + qi*M_Q*KB + m*KB + kb)*0x200
    //   K[h][kj][kb]    : ring_base + (h*KV_TILES*KB   + kj*KB + kb)*0x200
    //   V[h][dvi][kj]   : ring_base + (V_BASE_BLOCK + h*DV_TILES*KV_TILES + dvi*KV_TILES + kj)*0x200
    //
    // V_BASE_BLOCK must stay >= H*KV_TILES*KB, i.e. past the K region.
    wire [31:0] q_blk_per_head = 32'(cfg_attn_q_tiles_i) * 32'(cfg_attn_m_q_i) * 32'(cfg_attn_k_blocks_i);
    wire [31:0] q_blk_per_tile = 32'(cfg_attn_m_q_i) * 32'(cfg_attn_k_blocks_i);
    wire [31:0] k_blk_per_head = 32'(cfg_attn_kv_tiles_i) * 32'(cfg_attn_k_blocks_i);
    wire [31:0] v_blk_per_head = 32'(cfg_attn_dv_tiles_i) * 32'(cfg_attn_kv_tiles_i);

    wire [31:0] auto_v_base  = 32'(cfg_attn_h_i) * k_blk_per_head;
    wire [31:0] v_base_block = (auto_v_base > 32'd16) ? auto_v_base : 32'd16;

    wire [31:0] q_addr_calc = cfg_tcdm_q_base_i +
                             ((32'(cnt_h_q) * q_blk_per_head +
                               32'(cnt_qi_q) * q_blk_per_tile) * 32'h200);

    wire [31:0] k_addr_calc = cfg_tcdm_ring_base_i +
                             ((32'(cnt_h_q) * k_blk_per_head +
                               32'(cnt_kj_q) * 32'(cfg_attn_k_blocks_i)) * 32'h200);

    wire [31:0] v_addr_calc = cfg_tcdm_ring_base_i +
                             ((v_base_block +
                               32'(cnt_h_q)   * v_blk_per_head +
                               32'(cnt_dvi_q) * 32'(cfg_attn_kv_tiles_i) +
                               32'(cnt_kj_q)) * 32'h200);

    // GEMM Counters
    logic [15:0]  cnt_gn_q; // GEMM N-tile
    logic [15:0]  cnt_gm_q; // GEMM M-tile
    logic [15:0]  cnt_gk_q; // GEMM K-block
    logic [3:0]   cnt_gbeat_q;
    logic         gemm_drain_seen_q, gemm_drain_seen_d;

    // Background GEMM drain engine (Port B).  When a tile finishes its K
    // accumulation, this engine drains slots 0..15 one per cycle while the next
    // tile accumulates into the SAME slots on Port A.  The 2R1W latch array and
    // the slot_cleared mask make this safe as long as each slot is drained (and
    // cleared) before the next tile's Stage-2 write to that slot.
    logic         gemm_drain_active_q;
    logic [4:0]   gdr_idx_q;            // 0..15 = slot being issued, 16 = done

    wire [31:0] gemm_wt_addr = cfg_tcdm_ring_base_i +
                              ((32'(cnt_gn_q) * 32'(cfg_gemm_k_blocks_i) +
                                32'(cnt_gk_q)) * 32'h200);

    wire [31:0] gemm_act_addr = cfg_tcdm_q_base_i +
                               (((32'(cnt_gm_q) * 32'd16) * 32'(cfg_gemm_k_blocks_i) +
                                 32'(cnt_gk_q)) * 32'h200);

    wire last_m_beat  = (cnt_m_q == (cfg_attn_m_q_i - 4'd1));
    wire last_kj      = (cnt_kj_q == (cfg_attn_kv_tiles_i - 16'd1));
    wire last_dvi     = (cnt_dvi_q == (cfg_attn_dv_tiles_i - 16'd1));
    wire last_qi      = (cnt_qi_q == (cfg_attn_q_tiles_i - 16'd1));
    wire last_h       = (cnt_h_q == (cfg_attn_h_i - 8'd1));

    wire last_gbeat   = (cnt_gbeat_q == 4'd15);
    wire last_gk      = (cnt_gk_q == (cfg_gemm_k_blocks_i - 16'd1));
    wire last_gm      = (cnt_gm_q == (cfg_gemm_m_tiles_i - 16'd1));
    wire last_gn      = (cnt_gn_q == (cfg_gemm_n_tiles_i - 16'd1));

    // =========================================================================
    // 3. HARDWARE DYNAMIC CAUSAL MASK GENERATOR
    // =========================================================================
    logic [NUM_LANES-1:0] auto_causal_mask;
    always_comb begin
        if (!cfg_is_causal_i) begin
            auto_causal_mask = 16'hFFFF;
        end else begin
            for (int l = 0; l < 16; l++) begin
                // Causal logic: key_absolute_index <= query_absolute_index
                // Key index = kj * 16 + l
                // Query index = qi * M_Q + m
                auto_causal_mask[l] = ((32'(cnt_kj_q) * 32'd16 + 32'(l)) <= 
                                       (32'(cnt_qi_q) * 32'(cfg_attn_m_q_i) + 32'(cnt_m_q)));
            end
        end
    end

    // =========================================================================
    // 4. MICRO-FSM EXECUTION COORDINATOR
    // =========================================================================
    logic qk_compute_fire;
    logic pv_compute_fire;
    logic gemm_compute_fire;

    // GEMM compute beat t is gated on drain progress: slot t must already be
    // drained+cleared (t < gdr_idx_q) before the next tile's Stage-2 write to
    // that slot.  If the drain stalls (FIFO full), the compute beat stalls too.
    wire gemm_beat_ok = !gemm_drain_active_q || ({1'b0, cnt_gbeat_q} < gdr_idx_q);

    assign qk_compute_fire   = act_valid_i && (&wt_valid_i);
    assign pv_compute_fire   = !p_fifo_empty_i && (&wt_valid_i);
    assign gemm_compute_fire = act_valid_i && (&wt_valid_i) && gemm_beat_ok;

    // The last beat of the last K block: the tile's accumulators are final.
    wire gemm_k_last_fire = (state_q == ST_GEMM_RUN) && gemm_compute_fire &&
                            last_gbeat && last_gk;

    always_comb begin
        state_d                   = state_q;
        gemm_drain_seen_d         = gemm_drain_seen_q;
        dma_req_valid_o           = 1'b0;
        dma_req_dir_o             = 2'd0; // DRAM -> TCDM
        dma_req_dram_addr_o       = cfg_dram_kv_base_i + (32'(cnt_kj_q) * cfg_stride_tile_i);
        dma_req_tcdm_addr_o       = dma_curr_tcdm_addr;
        dma_req_size_bytes_o      = SLOT_SIZE_BYTES;
        dma_issue_fire            = 1'b0;
        core_release_credit_pulse = 1'b0;

        act_start_o               = 1'b0;
        act_base_o                = cfg_tcdm_q_base_i;
        act_k_blocks_o            = 32'd1;
        act_k_stride_o            = 32'd0;
        act_tokens_o              = 32'(cfg_attn_m_q_i);
        act_token_stride_o        = 32'h200;
        act_ready_o               = 1'b0;

        wt_start_o                = 1'b0;
        wt_base_o                 = core_curr_tcdm_addr;
        wt_repeat_o               = 32'(cfg_attn_m_q_i);
        wt_k_blocks_o             = 32'd1;
        wt_k_stride_o             = 32'h200;
        wt_num_tiles_o            = 32'd1;
        wt_tile_step_o            = 32'd0;
        wt_ready_o                = 16'h0000;

        wo_start_o                = 1'b0;
        wo_mode_o                 = 2'd1; // Default FP16
        wo_base_o                 = cfg_tcdm_out_base_i;
        wo_num_vectors_o          = 32'd16;
        wo_line_stride_o          = 32'h200;
        wo_num_batches_o          = 32'd1;
        wo_batch_stride_o         = 32'd0;

        v_start_o                 = 1'b0;
        rd_start_o                = 1'b0;

        cfg_mode_o                = 2'b00;
        cfg_row_start_o           = 1'b0;
        cfg_key_mask_o            = auto_causal_mask;
        cfg_enable_slq_o          = 1'b1;

        tile_drain_req_o          = 1'b0;
        tile_drain_idx_o          = 4'd0;
        tile_clear_on_drain_o     = 1'b0;
        nrm_start_o               = 1'b0;
        lane_valid_o              = 16'h0000;
        lane_acc_idx_o            = 4'd0;

        case (state_q)
            ST_IDLE: begin
                if (start_i) begin
                    state_d = ST_PREP_DRAIN;
                end
            end
            ST_PREP_DRAIN: begin
                // Pre-configure Write-Out Streamer for results
                wo_start_o       = 1'b1;
                wo_mode_o        = 2'd1; // FP16 out
                wo_base_o        = cfg_tcdm_out_base_i;
                wo_num_vectors_o = (cfg_op_mode_i == 2'b00)
                                 ? (32'(cfg_gemm_m_tiles_i) * 32'(cfg_gemm_n_tiles_i) * 32'd16)
                                 : (32'(cfg_attn_h_i) * 32'(cfg_attn_q_tiles_i) * 32'(cfg_attn_dv_tiles_i) * 32'(cfg_attn_m_q_i));
                state_d = ST_PREFETCH_INIT;
            end

            ST_PREFETCH_INIT: begin
                // Kick off initial DMA prefetch if enabled
                if (cfg_enable_dma_i && dma_can_push) begin
                    dma_req_valid_o      = 1'b1;
                    dma_req_dram_addr_o  = cfg_dram_kv_base_i;
                    if (dma_req_ready_i) begin
                        dma_issue_fire   = 1'b1;
                        state_d          = (cfg_op_mode_i == 2'b00) ? ST_GEMM_LAUNCH :
                                           (cfg_op_mode_i == 2'b10) ? ST_V_LAUNCH  :
                                           (cfg_op_mode_i == 2'b11) ? ST_RD_LAUNCH :
                                                                      ST_ATTN_QK_LAUNCH;
                    end
                end else begin
                    state_d = (cfg_op_mode_i == 2'b00) ? ST_GEMM_LAUNCH :
                              (cfg_op_mode_i == 2'b10) ? ST_V_LAUNCH  :
                              (cfg_op_mode_i == 2'b11) ? ST_RD_LAUNCH :
                                                         ST_ATTN_QK_LAUNCH;
                end
            end

            // =================================================================
            // STANDALONE V TRANSPOSE (opcode 10)
            // =================================================================
            ST_V_LAUNCH: begin
                if (!v_busy_i && !wo_busy_i) begin
                    v_start_o          = 1'b1;
                    wo_start_o         = 1'b1;
                    wo_mode_o          = 2'd2;             // V FP4 batch
                    wo_base_o          = cfg_v_out_base_i;
                    wo_num_vectors_o   = 32'd16;           // 16 channels
                    wo_line_stride_o   = 32'd16;           // channel stride (bank = channel)
                    wo_num_batches_o   = cfg_v_num_batches_i;
                    wo_batch_stride_o  = 32'h0000_0200;
                    state_d            = ST_V_RUN;
                end
            end

            ST_V_RUN: begin
                if (v_done_i || (!v_busy_i && !wo_busy_i))
                    state_d = ST_WAIT_WRITE_OUT;
            end

            // =================================================================
            // STANDALONE TCDM READBACK + REQUANTIZE (opcode 11)
            // =================================================================
            ST_RD_LAUNCH: begin
                if (!rd_busy_i && !wo_busy_i) begin
                    cfg_enable_slq_o   = 1'b1;
                    rd_start_o         = 1'b1;
                    wo_start_o         = 1'b1;
                    wo_mode_o          = 2'd0;             // FP4 linear
                    wo_base_o          = cfg_rd_out_base_i;
                    wo_num_vectors_o   = cfg_rd_num_vectors_i;
                    wo_line_stride_o   = cfg_rd_line_stride_i;
                    wo_num_batches_o   = 32'd1;
                    wo_batch_stride_o  = 32'd0;
                    state_d            = ST_RD_RUN;
                end
            end

            ST_RD_RUN: begin
                cfg_enable_slq_o = 1'b1;
                if (rd_done_i && !wo_busy_i)
                    state_d = ST_WAIT_WRITE_OUT;
            end

            // =================================================================
            // FLASHATTENTION PATH
            //
            // One launch per KV tile.  Both streamers carry a 2-D AGU, so the
            // K-block loop (D_K / 16) runs inside them:
            //   WT : replays one K vector cfg_attn_m_q_i times, then advances
            //        by wt_k_stride and repeats, for cfg_attn_k_blocks_i blocks.
            //   ACT: sweeps cfg_attn_m_q_i tokens, then advances by act_k_stride
            //        to the next K block.
            // Beat ordering is identical on both sides (K block outer, token /
            // replay inner), so the accumulator slot {1'b1, cnt_m_q[2:0]} sees
            // the same (m, kb) pair from Q and from K and accumulates D_K.
            //
            //   QK:  launch WT+ACT -> M_Q*KB beats -> softmax drain M_Q slots
            //   PV:  launch WT     -> M_Q beats    -> next kv tile
            //        after the last kv tile, drain O for M_Q slots
            // =================================================================
            ST_ATTN_QK_LAUNCH: begin
                cfg_mode_o = 2'b01;

                wt_start_o    = 1'b1;
                wt_base_o     = cfg_enable_dma_i ? core_curr_tcdm_addr : k_addr_calc;
                wt_repeat_o   = 32'(cfg_attn_m_q_i);       // replays per K block
                wt_k_blocks_o = 32'(cfg_attn_k_blocks_i);  // K blocks walked by the AGU
                wt_k_stride_o = 32'h200;

                act_start_o        = 1'b1;
                act_base_o         = q_addr_calc;
                act_tokens_o       = 32'(cfg_attn_m_q_i);
                act_token_stride_o = 32'(cfg_attn_k_blocks_i) * 32'h200;
                act_k_blocks_o     = 32'(cfg_attn_k_blocks_i);
                act_k_stride_o     = 32'h200;

                state_d = ST_ATTN_QK_RUN;
            end

            ST_ATTN_QK_RUN: begin
                cfg_mode_o = 2'b01;
                if (qk_compute_fire) begin
                    lane_valid_o   = 16'hFFFF;
                    lane_acc_idx_o = {1'b1, cnt_m_q[2:0]};  // S slots 8..15
                    act_ready_o    = 1'b1;
                    wt_ready_o     = 16'hFFFF;
                end
                // Both AGUs retire on their own once every K block has been
                // streamed, so busy deasserting is the end of the whole QK
                // phase. Wait for previous group's normalizer drain to retire
                // so l_accumulator is not overwritten while O is draining.
                if (!act_busy_i && !wt_busy_i && !nrm_busy)
                    state_d = ST_ATTN_SM_DRAIN;
            end

            ST_ATTN_SM_DRAIN: begin
                cfg_mode_o            = 2'b01;
                tile_drain_req_o      = 1'b1;
                tile_drain_idx_o      = {1'b1, cnt_dr_q[2:0]};
                tile_clear_on_drain_o = 1'b1;
                cfg_row_start_o       = kv_first_q;
                if (cnt_dr_q == cfg_attn_m_q_i - 4'd1)
                    state_d = ST_ATTN_PV_LAUNCH;
                else
                    state_d = ST_ATTN_SM_DRAIN;
            end

            ST_ATTN_SM_WAIT: begin
                // Retained for backward-compatibility; bypassed in 1-cycle-per-slot drain mode
                cfg_mode_o      = 2'b01;
                cfg_row_start_o = kv_first_q;
                state_d         = (cnt_dr_q == cfg_attn_m_q_i - 4'd1)
                                ? ST_ATTN_PV_LAUNCH : ST_ATTN_SM_DRAIN;
            end

            ST_ATTN_PV_LAUNCH: begin
                // Keep attention mode for this cycle: the core registers the
                // last drained score one cycle after the final SM drain pulse and
                // softmax only accepts it while cfg_mode is 2'b01.  Nothing
                // accumulates here (lane_valid_o is low), so this is free.
                cfg_mode_o    = 2'b01;
                // The O slots may still be drained by the background engine from
                // the previous group; PV would overwrite them.
                if (!nrm_busy) begin
                wt_start_o    = 1'b1;
                wt_base_o     = cfg_enable_dma_i ? (core_curr_tcdm_addr + 32'h1000)
                                                 : v_addr_calc;
                wt_repeat_o   = 32'(cfg_attn_m_q_i);
                wt_k_blocks_o = 32'd1;   // V is reduced over keys, not over D_K
                wt_k_stride_o = 32'd0;
                state_d       = ST_ATTN_PV_RUN;
                end
            end

            ST_ATTN_PV_RUN: begin
                cfg_mode_o = 2'b10;
                if (pv_compute_fire) begin
                    lane_valid_o   = 16'hFFFF;
                    lane_acc_idx_o = {1'b0, cnt_m_q[2:0]};  // O slots 0..7
                    wt_ready_o     = 16'hFFFF;
                    state_d        = (cnt_m_q == cfg_attn_m_q_i - 4'd1)
                                   ? ST_ATTN_PV_DRAIN : ST_ATTN_PV_RUN;
                end
            end

            ST_ATTN_PV_DRAIN: begin
                cfg_mode_o = 2'b10;
                // Keep the streamer draining: it still holds the last replay
                // until the core acknowledges it, so ready must stay asserted
                // even though no new beat is being issued.
                wt_ready_o = 16'hFFFF;
                if (!wt_busy_i) begin
                    core_release_credit_pulse = cfg_enable_dma_i;
                    // cnt_kj_q wraps back to 0 only after the last KV tile.
                    if (cnt_kj_q == 16'd0) begin
                        // Last KV tile of this (h, qi, dvi) group.  Hand the O
                        // slots to the background drain engine and continue with
                        // the next group's QK straight away: QK fills the S slots
                        // (8..15) while the engine drains O (0..7), so the two
                        // never touch the same accumulator entry.
                        nrm_start_o = 1'b1;
                        state_d     = last_group_q ? ST_ATTN_LAST_DRAIN
                                                   : ST_ATTN_QK_LAUNCH;
                    end else begin
                        state_d = ST_ATTN_QK_LAUNCH;   // next KV tile
                    end
                end
            end

            // The last group has no following QK phase to hide its O drain
            // behind, and the core only drives the FP16 output register from the
            // normalizer while cfg_mode is 2'b10, so hold PV mode until the
            // drain engine retires.
            ST_ATTN_LAST_DRAIN: begin
                cfg_mode_o = 2'b10;
                if (!nrm_busy)
                    state_d = ST_WAIT_WRITE_OUT;
            end

            // =================================================================
            // DENSE GEMM PATH
            // =================================================================
            ST_GEMM_LAUNCH: begin
                if (core_can_pop && !act_busy_i && !wt_busy_i) begin
                    cfg_mode_o    = 2'b00; // Mode 00: GEMM
                    // Launch Weight Streamer ONCE for the whole tile; its 3-D AGU
                    // walks all K blocks internally (k_stride = one 72b word),
                    // replaying each K vector 16 times (tokens).  No per-K relaunch.
                    wt_start_o    = 1'b1;
                    wt_base_o     = cfg_enable_dma_i ? core_curr_tcdm_addr : gemm_wt_addr;
                    wt_repeat_o   = 32'd16; // 16 Tokens
                    wt_k_blocks_o = cfg_gemm_k_blocks_i;
                    wt_k_stride_o = 32'h200;

                    // Launch Act Streamer ONCE: token inner, K-block outer.
                    act_start_o        = 1'b1;
                    act_base_o         = gemm_act_addr;
                    act_tokens_o       = 32'd16;
                    act_token_stride_o = 32'(cfg_gemm_k_blocks_i) * 32'h200;
                    act_k_blocks_o     = cfg_gemm_k_blocks_i;
                    act_k_stride_o     = 32'h200;

                    state_d       = ST_GEMM_RUN;
                end
            end

            ST_GEMM_RUN: begin
                cfg_mode_o = 2'b00;
                if (gemm_compute_fire) begin
                    lane_valid_o   = 16'hFFFF;
                    lane_acc_idx_o = cnt_gbeat_q; // Full 16 slots [0..15]
                    act_ready_o    = 1'b1;
                    wt_ready_o     = 16'hFFFF;

                    // The streamers walk all K blocks internally, so stay in RUN
                    // between K blocks (no relaunch/fill bubble).  Only the last
                    // beat of the last K block ends the tile.
                    if (last_gbeat && last_gk) begin
                        // Tile fully accumulated: hand it to the background drain
                        // engine and start the next tile immediately, so drain
                        // (Port B) overlaps next-tile compute.
                        core_release_credit_pulse = cfg_enable_dma_i;
                        state_d = (last_gm && last_gn) ? ST_WAIT_WRITE_OUT
                                                       : ST_GEMM_LAUNCH;
                    end
                end
            end

            ST_GEMM_DRAIN: begin
                cfg_mode_o = 2'b00;
                // Wait for the downstream Write-Out Streamer to be ready for the next vector
                if (core_out_fifo_ready_i) begin
                    tile_drain_req_o      = 1'b1;
                    tile_drain_idx_o      = cnt_gbeat_q;
                    tile_clear_on_drain_o = 1'b1;
                    gemm_drain_seen_d     = 1'b0;
                    state_d               = ST_GEMM_DRAIN_WAIT;
                end
            end

            ST_GEMM_DRAIN_WAIT: begin
                cfg_mode_o = 2'b00;
                if (core_fp16_valid_i) gemm_drain_seen_d = 1'b1;
                // Once the vector has been transferred to write-out streamer (handshake complete)
                if (gemm_drain_seen_q && !core_fp16_valid_i) begin
                    if (last_gbeat) begin
                        core_release_credit_pulse = cfg_enable_dma_i;
                        if (last_gm && last_gn) begin
                            state_d = ST_WAIT_WRITE_OUT;
                        end else begin
                            state_d = ST_GEMM_LAUNCH;
                        end
                    end else begin
                        state_d = ST_GEMM_DRAIN;
                    end
                end
            end

            ST_WAIT_WRITE_OUT: begin
                if ((wo_done_i || !wo_busy_i) && !nrm_busy && !gemm_drain_active_q) begin
                    state_d = ST_DONE;
                end
            end

            ST_DONE: begin
                // Hold DONE until the next start so `done_o` is observable.
                if (!start_i)
                    state_d = ST_DONE;
            end

            default: state_d = ST_IDLE;
        endcase

        // The background O drain shares tile_drain_req with the SM drain; the
        // engine never asserts nrm_req on a cycle the SM drain owns the port.
        if (nrm_req) begin
            tile_drain_req_o      = 1'b1;
            tile_drain_idx_o      = nrm_dr_q;
            tile_clear_on_drain_o = 1'b1;
        end

        // Background GEMM drain: Port B reads slot gdr_idx_q while the next
        // tile accumulates on Port A.  Gated by write-out FIFO space so the
        // single core output register never overruns.
        if (gemm_drain_active_q && core_out_fifo_ready_i) begin
            tile_drain_req_o      = 1'b1;
            tile_drain_idx_o      = gdr_idx_q[3:0];
            tile_clear_on_drain_o = 1'b1;
        end
    end

    // Sequential state & counter register updates
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q           <= ST_IDLE;
            cnt_h_q           <= '0;
            cnt_qi_q          <= '0;
            cnt_dvi_q         <= '0;
            cnt_kj_q          <= '0;
            cnt_m_q           <= '0;
            cnt_gn_q          <= '0;
            cnt_gm_q          <= '0;
            cnt_gk_q          <= '0;
            cnt_gbeat_q       <= '0;
            gemm_drain_seen_q <= 1'b0;
            gemm_drain_active_q <= 1'b0;
            gdr_idx_q           <= 5'd0;
        end else begin
            state_q <= state_d;

            if (start_i) begin
                cnt_h_q           <= '0;
                cnt_qi_q          <= '0;
                cnt_dvi_q         <= '0;
                cnt_kj_q          <= '0;
                cnt_m_q           <= '0;
                cnt_dr_q          <= '0;
                cnt_gn_q          <= '0;
                cnt_gm_q          <= '0;
                cnt_gk_q          <= '0;
                cnt_gbeat_q       <= '0;
                gemm_drain_seen_q <= 1'b0;
                gemm_drain_active_q <= 1'b0;
                gdr_idx_q           <= 5'd0;
                kv_first_q        <= 1'b1;
            end else begin
                // ---------------------------------------------------------
                // QK beat counter.  Only the accumulator slot index is needed
                // here: the K-block loop runs inside the streamers and the
                // phase ends when both of them retire, so there is no kb
                // counter and no per-block re-arm.  cnt_m_q is restarted at
                // every launch so the S slot mapping is always the same.
                // ---------------------------------------------------------
                if (state_q == ST_ATTN_QK_LAUNCH || state_q == ST_ATTN_PV_LAUNCH)
                    cnt_m_q <= '0;
                else if (state_q == ST_ATTN_QK_RUN && qk_compute_fire)
                    cnt_m_q <= (cnt_m_q == cfg_attn_m_q_i - 4'd1) ? '0 : cnt_m_q + 4'd1;

                // ---------------------------------------------------------
                // PV loop nest: m (beat).  The last beat of a KV tile
                // advances kj; the last kj moves to the output drain.
                // ---------------------------------------------------------
                if (state_q == ST_ATTN_PV_RUN && pv_compute_fire) begin
                    if (cnt_m_q == cfg_attn_m_q_i - 4'd1) begin
                        cnt_m_q <= '0;
                        if (cnt_kj_q == cfg_attn_kv_tiles_i - 16'd1) begin
                            cnt_kj_q <= '0;
                            // Latch the termination decision before the group
                            // counters advance, so the drain phase that follows
                            // knows whether this was the final group.
                            last_group_q <= (cnt_dvi_q == cfg_attn_dv_tiles_i - 16'd1) &&
                                            (cnt_qi_q  == cfg_attn_q_tiles_i  - 16'd1) &&
                                            (cnt_h_q   == cfg_attn_h_i        - 8'd1);
                            // dvi / qi / h advance only when the whole KV
                            // sequence for this dvi slice is complete.
                            if (cnt_dvi_q == cfg_attn_dv_tiles_i - 16'd1) begin
                                cnt_dvi_q <= '0;
                                if (cnt_qi_q == cfg_attn_q_tiles_i - 16'd1) begin
                                    cnt_qi_q <= '0;
                                    if (cnt_h_q == cfg_attn_h_i - 8'd1)
                                        cnt_h_q <= '0;
                                    else
                                        cnt_h_q <= cnt_h_q + 8'd1;
                                end else begin
                                    cnt_qi_q <= cnt_qi_q + 16'd1;
                                end
                            end else begin
                                cnt_dvi_q <= cnt_dvi_q + 16'd1;
                            end
                        end else begin
                            cnt_kj_q <= cnt_kj_q + 16'd1;
                        end
                    end else begin
                        cnt_m_q <= cnt_m_q + 4'd1;
                    end
                end

                // GEMM Counter Progression
                if (state_q == ST_GEMM_RUN && gemm_compute_fire) begin
                    if (last_gbeat) begin
                        cnt_gbeat_q <= '0;
                        if (last_gk) begin
                            cnt_gk_q <= '0;
                            // Output tile finished: advance (gm, gn) now so the
                            // next launch uses the correct tile while this one
                            // is drained in the background.
                            if (last_gm) begin
                                cnt_gm_q <= '0;
                                if (last_gn) cnt_gn_q <= '0;
                                else         cnt_gn_q <= cnt_gn_q + 1'b1;
                            end else begin
                                cnt_gm_q <= cnt_gm_q + 1'b1;
                            end
                        end else begin
                            cnt_gk_q <= cnt_gk_q + 1'b1;
                        end
                    end else begin
                        cnt_gbeat_q <= cnt_gbeat_q + 1'b1;
                    end
                end

                // Background GEMM drain engine: start on the tile's K_last and
                // issue slot 0..15 one per cycle, gated by write-out FIFO space.
                if (gemm_k_last_fire) begin
                    gemm_drain_active_q <= 1'b1;
                    gdr_idx_q           <= 5'd0;
                end else if (gemm_drain_active_q && core_out_fifo_ready_i) begin
                    if (gdr_idx_q == 5'd15) begin
                        gemm_drain_active_q <= 1'b0;
                        gdr_idx_q           <= 5'd16;
                    end else begin
                        gdr_idx_q <= gdr_idx_q + 5'd1;
                    end
                end

                if (state_q == ST_GEMM_DRAIN_WAIT) begin
                    if (gemm_drain_seen_q && !core_fp16_valid_i) begin
                        gemm_drain_seen_q <= 1'b0;
                        if (last_gbeat) begin
                            cnt_gbeat_q <= '0;
                            if (last_gm) begin
                                cnt_gm_q <= '0;
                                if (last_gn) cnt_gn_q <= '0;
                                else         cnt_gn_q <= cnt_gn_q + 1'b1;
                            end else begin
                                cnt_gm_q <= cnt_gm_q + 1'b1;
                            end
                        end else begin
                            cnt_gbeat_q <= cnt_gbeat_q + 1'b1;
                        end
                    end else begin
                        gemm_drain_seen_q <= gemm_drain_seen_d;
                    end
                end else begin
                    gemm_drain_seen_q <= 1'b0;
                end

                // ---------------------------------------------------------
                // Softmax drain index: advances once per slot, restarts for
                // every new drain group.  The O-slot drain index lives in the
                // background engine.
                // ---------------------------------------------------------
                if (state_q == ST_ATTN_QK_RUN && !act_busy_i && !wt_busy_i)
                    cnt_dr_q <= '0;
                else if (state_q == ST_ATTN_PV_DRAIN)
                    cnt_dr_q <= '0;
                else if (state_q == ST_ATTN_SM_DRAIN || state_q == ST_ATTN_SM_WAIT)
                    cnt_dr_q <= cnt_dr_q + 4'd1;

                // kv_first_q marks the first KV tile of a query row.  It is set
                // again as soon as the last tile of a group retires, which is
                // exactly when cnt_kj_q wraps back to 0.
                if (state_q == ST_ATTN_PV_DRAIN)
                    kv_first_q <= (cnt_kj_q == 16'd0);
            end
        end
    end

    // DONE is a terminal state that is not "busy": the job has finished and the
    // result is available, the FSM only waits there for the next start.
    assign busy_o = ((state_q != ST_IDLE) && (state_q != ST_DONE)) || nrm_busy ||
                    gemm_drain_active_q;
    assign done_o = (state_q == ST_DONE) && !nrm_busy && !gemm_drain_active_q;

endmodule
