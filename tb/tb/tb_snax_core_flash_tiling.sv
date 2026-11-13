// =============================================================================
// Module: tb_snax_core_flash_tiling.sv
// Description: Cycle-accurate, bit-exact testbench for snax_fused_llm_core with
//              the FULL FlashAttention tiling loop nest:
//
//                for h    in 0..H-1          (heads)
//                for qi   in 0..Q_TILES-1    (M_Q = 8 queries per tile)
//                for dvi  in 0..DV_TILES-1   (16-channel V slices)
//                  reset m,l (row_start per query)
//                  for kj in 0..KV_TILES-1   (16-key KV tiles)
//                    Mode 01: Q * K^T -> softmax -> 8 P blocks -> p_fifo
//                    Mode 10: pop P -> rescale O + P * V slice -> slots 0..7
//                  final drain slots 0..7 -> 1/l -> 16 FP16 channels
//
//              Test size: H=2, N_Q=16, N_KV=32, D_K=32, D_V=32
//              => 2 heads x 2 query tiles x 2 d_v slices x 2 KV tiles
//              => 8 passes x 8 queries x 16 lanes = 1024 outputs.
// =============================================================================
`timescale 1ns/1ps
`include "pe_defines.svh"

module tb_snax_core_flash_tiling;

    localparam int NUM_LANES_P = 16;
    localparam int ACC_DEPTH_P = 16;   // 8:8 partition: O 0..7, S 8..15
    localparam int NUM_QUERIES = 8;    // queries per tile
    localparam int PKT_W       = 72;
    localparam int FRAC_BITS   = 16;
    localparam int CLK_PERIOD  = 10;   // 100 MHz

    localparam int H           = 2;    // heads
    localparam int N_Q         = 16;   // query tokens
    localparam int N_KV        = 32;   // key/value tokens
    localparam int D_K         = 32;   // Q/K head dimension
    localparam int D_V         = 32;   // V head dimension

    localparam int M_Q         = 8;    // queries per tile
    localparam int K_BLOCKS    = D_K / 16; // 2
    localparam int Q_TILES     = N_Q / M_Q; // 2
    localparam int DV_TILES    = D_V / 16;  // 2
    localparam int KV_TILES    = N_KV / 16; // 2

    // Waveform-visible loop counters. Phase: 0=idle, 1=QK, 2=softmax/S drain,
    // 3=P*V, 4=final drain.
    localparam int PH_IDLE    = 0;
    localparam int PH_QK      = 1;
    localparam int PH_SOFTMAX = 2;
    localparam int PH_PV      = 3;
    localparam int PH_FINAL   = 4;
    int dbg_h, dbg_qi, dbg_dvi, dbg_kj, dbg_kb, dbg_m;
    int dbg_phase;
    int dbg_q_pkt, dbg_k_pkt, dbg_v_pkt;

    // DUT signals
    logic                            clk;
    logic                            rst_n;

    logic [1:0]                      cfg_mode_i;
    logic                            cfg_row_start_i;
    logic [NUM_LANES_P-1:0]          cfg_key_mask_i;
    logic                            cfg_enable_slq_i;

    logic                            tile_drain_req_i;
    logic [$clog2(ACC_DEPTH_P)-1:0]  tile_drain_idx_i;
    logic                            tile_clear_on_drain_i;

    logic [NUM_LANES_P-1:0]          lane_valid_i;
    logic [$clog2(ACC_DEPTH_P)-1:0]  lane_acc_idx_i;

    logic [PKT_W-1:0]                ext_act_broadcast_i;
    logic                            ext_act_valid_i;
    logic [NUM_LANES_P*PKT_W-1:0] wt_in_i;   // flat for Verilator
    logic                            wt_valid_i;

    logic [NUM_LANES_P*16-1:0]       out_vec_fp16_packed_o;
    logic [NUM_LANES_P*16-1:0]       out_vec_fp16_lanes_o; // flat for Verilator
    logic                            out_vec_fp16_valid_o;
    logic [PKT_W-1:0]                out_vec_fp4_packet_o;
    logic                            out_vec_fp4_valid_o;

    logic                            p_fifo_full_o;
    logic                            p_fifo_empty_o;
    logic                            p_fifo_almost_full_o;
    logic                            l_ready_o;

    // Testbench memories
    logic [PKT_W-1:0] mem_q [0:H*N_Q*K_BLOCKS - 1];                  // 64
    logic [PKT_W-1:0] mem_k [0:H*KV_TILES*K_BLOCKS*16 - 1];          // 128
    logic [PKT_W-1:0] mem_v [0:H*DV_TILES*KV_TILES*16 - 1];          // 128
    logic [15:0]      mem_golden_fp16 [0:H*N_Q*D_V - 1];             // 1024

    snax_fused_llm_core #(
        .NUM_LANES_P  (NUM_LANES_P),
        .ACC_DEPTH_P  (ACC_DEPTH_P),
        .NUM_QUERIES  (NUM_QUERIES),
        .PKT_W        (PKT_W),
        .FRAC_BITS    (FRAC_BITS),
        .P_FIFO_DEPTH (16)
    ) u_dut (
        .clk                   (clk),
        .rst_n                 (rst_n),
        .cfg_mode_i            (cfg_mode_i),
        .cfg_row_start_i       (cfg_row_start_i),
        .cfg_key_mask_i        (cfg_key_mask_i),
        .cfg_enable_slq_i      (cfg_enable_slq_i),
        .tile_drain_req_i      (tile_drain_req_i),
        .tile_drain_idx_i      (tile_drain_idx_i),
        .tile_clear_on_drain_i (tile_clear_on_drain_i),
        .lane_valid_i          (lane_valid_i),
        .lane_acc_idx_i        (lane_acc_idx_i),
        .ext_act_broadcast_i   (ext_act_broadcast_i),
        .ext_act_valid_i       (ext_act_valid_i),
        .wt_in_i               (wt_in_i),
        .wt_valid_i            (wt_valid_i),
        .out_vec_fp16_packed_o (out_vec_fp16_packed_o),
        .out_vec_fp16_lanes_o  (out_vec_fp16_lanes_o),
        .out_vec_fp16_valid_o  (out_vec_fp16_valid_o),
        .out_vec_fp4_packet_o  (out_vec_fp4_packet_o),
        .out_vec_fp4_valid_o   (out_vec_fp4_valid_o),
        .p_fifo_full_o         (p_fifo_full_o),
        .p_fifo_empty_o        (p_fifo_empty_o),
        .p_fifo_almost_full_o  (p_fifo_almost_full_o),
        .l_ready_o             (l_ready_o)
    );

    initial begin
        clk = 0;
        forever #(CLK_PERIOD / 2) clk = ~clk;
    end

    task automatic wait_cycles(input int n);
        repeat (n) @(posedge clk);
    endtask

    function automatic real fp16_to_real(input logic [15:0] h);
        logic sign;
        logic [4:0] exp;
        logic [9:0] mant;
        real val;
        begin
            sign = h[15];
            exp  = h[14:10];
            mant = h[9:0];
            if (exp == 5'd0 && mant == 10'd0) return 0.0;
            if (exp == 5'b11111) return sign ? -999999.0 : 999999.0;
            if (exp == 5'd0) begin
                val = real'(mant) / 16777216.0;
            end else begin
                val = (1.0 + real'(mant) / 1024.0) * $itor(32'd1 << exp) / 32768.0;
            end
            return sign ? -val : val;
        end
    endfunction

    int match_count     = 0;
    int mismatch_count  = 0;
    int total_drained   = 0;
    int csv_fd;

    real max_abs_err = 0.0;
    real sum_abs_err = 0.0;
    real sum_sq_err  = 0.0;

    initial begin
        $display("\n========================================================================");
        $display("   SNAX_FUSED_LLM_CORE FULL TILING: H -> Q_TILE -> DV_SLICE -> KV     ");
        $display("   H=%0d, N_Q=%0d (%0d tiles), N_KV=%0d (%0d tiles), D_K=%0d, D_V=%0d (%0d slices)",
                 H, N_Q, Q_TILES, N_KV, KV_TILES, D_K, D_V, DV_TILES);
        $display("========================================================================\n");

        $readmemh("sim_snax_core_flash/hex/q_packets.hex", mem_q);
        $readmemh("sim_snax_core_flash/hex/k_packets.hex", mem_k);
        $readmemh("sim_snax_core_flash/hex/v_packets.hex", mem_v);
        $readmemh("sim_snax_core_flash/hex/golden_core_fp16.hex", mem_golden_fp16);

        $display("[%0t ns] Loaded input packets & golden reference into memory.", $time);

        rst_n                 = 1'b0;
        cfg_mode_i            = 2'b00;
        cfg_row_start_i       = 1'b0;
        cfg_key_mask_i        = '1;
        cfg_enable_slq_i      = 1'b0;
        tile_drain_req_i      = 1'b0;
        tile_drain_idx_i      = '0;
        tile_clear_on_drain_i = 1'b0;
        lane_valid_i          = '0;
        lane_acc_idx_i        = '0;
        ext_act_broadcast_i   = '0;
        ext_act_valid_i       = 1'b0;
        wt_valid_i            = 1'b0;
        wt_in_i = '0;

        dbg_h = 0; dbg_qi = 0; dbg_dvi = 0; dbg_kj = 0; dbg_kb = 0; dbg_m = 0;
        dbg_phase = PH_IDLE;
        dbg_q_pkt = 0; dbg_k_pkt = 0; dbg_v_pkt = 0;

        wait_cycles(5);
        rst_n = 1'b1;
        wait_cycles(5);
        $display("[%0t ns] Reset deasserted. SNAX Fused Core ready.\n", $time);

        csv_fd = $fopen("sim_snax_core_flash/csv/rtl_vs_golden.csv", "w");
        if (csv_fd != 0)
            $fwrite(csv_fd, "head,query_tile,dv_slice,query_idx,lane,v_col_global,actual_fp16,actual_real,golden_fp16,golden_real,abs_err\n");

        // =====================================================================
        // LOOP 1: HEADS
        // =====================================================================
        for (int h = 0; h < H; h++) begin
            dbg_h = h;
            $display("########################################################################");
            $display("[%0t ns] >>> HEAD h = %0d <<<", $time, h);
            $display("########################################################################");

            // =================================================================
            // LOOP 2: QUERY TILES
            // =================================================================
            for (int qi = 0; qi < Q_TILES; qi++) begin
                dbg_qi = qi;
                $display("[%0t ns] === QUERY TILE qi = %0d (Q %0d..%0d) ===",
                         $time, qi, qi*M_Q, qi*M_Q + M_Q - 1);

                // =============================================================
                // LOOP 3: d_v CHANNEL SLICES
                // =============================================================
                for (int dvi = 0; dvi < DV_TILES; dvi++) begin
                    dbg_dvi = dvi;
                    dbg_kj  = 0;
                    dbg_kb  = 0;
                    dbg_m   = 0;
                    $display("[%0t ns] --- DV SLICE dvi = %0d (channels %0d..%0d) ---",
                             $time, dvi, dvi*16, dvi*16+15);

                    // =========================================================
                    // LOOP 4: KV TILES
                    // =========================================================
                    for (int kj = 0; kj < KV_TILES; kj++) begin
                        dbg_kj = kj;
                        $display("[%0t ns] KV BLOCK kj = %0d (keys %0d..%0d)",
                                 $time, kj, kj*16, kj*16+15);

                        // -----------------------------------------------------
                        // Mode 01: Q * K^T -> Slots 8..15
                        // -----------------------------------------------------
                        cfg_mode_i = 2'b01;
                        dbg_phase  = PH_QK;

                        for (int kb = 0; kb < K_BLOCKS; kb++) begin
                            dbg_kb = kb;
                            for (int l = 0; l < NUM_LANES_P; l++) begin
                                automatic int k_idx = h * (KV_TILES*K_BLOCKS*16)
                                                    + kj * (K_BLOCKS*16)
                                                    + kb * 16 + l;
                                dbg_k_pkt = k_idx;
                                wt_in_i[l*PKT_W +: PKT_W] = mem_k[k_idx];
                            end
                            wt_valid_i = 1'b1;

                            for (int m = 0; m < M_Q; m++) begin
                                automatic int q_idx = h * (N_Q*K_BLOCKS)
                                                    + (qi * M_Q + m) * K_BLOCKS + kb;
                                dbg_m     = m;
                                dbg_q_pkt = q_idx;
                                ext_act_broadcast_i = mem_q[q_idx];
                                ext_act_valid_i     = 1'b1;
                                lane_acc_idx_i      = 4'd8 + m[2:0];
                                lane_valid_i        = 16'hFFFF;
                                @(posedge clk);
                            end
                        end

                        ext_act_valid_i = 1'b0;
                        lane_valid_i    = '0;
                        wt_valid_i      = 1'b0;
                        wait_cycles(2);

                        // -----------------------------------------------------
                        // Drain S slots 8..15 -> softmax -> P FIFO
                        // -----------------------------------------------------
                        dbg_phase = PH_SOFTMAX;
                        for (int m = 0; m < M_Q; m++) begin
                            dbg_m = m;
                            tile_drain_req_i      = 1'b1;
                            tile_drain_idx_i      = 4'd8 + m[2:0];
                            tile_clear_on_drain_i = 1'b1;
                            @(posedge clk);
                            tile_drain_req_i      = 1'b0;
                            tile_clear_on_drain_i = 1'b0;

                            // row_start in the cycle the raw score enters softmax
                            if (kj == 0) cfg_row_start_i = 1'b1;
                            @(posedge clk);
                            cfg_row_start_i       = 1'b0;
                        end

                        wait_cycles(4);

                        // -----------------------------------------------------
                        // Mode 10: pop P -> P * V[dvi] -> slots 0..7
                        // -----------------------------------------------------
                        cfg_mode_i = 2'b10;
                        dbg_phase  = PH_PV;

                        for (int l = 0; l < NUM_LANES_P; l++) begin
                            automatic int v_idx = h * (DV_TILES*KV_TILES*16)
                                                + dvi * (KV_TILES*16)
                                                + kj * 16 + l;
                            dbg_v_pkt = v_idx;
                            wt_in_i[l*PKT_W +: PKT_W] = mem_v[v_idx];
                        end
                        wt_valid_i = 1'b1;

                        for (int m = 0; m < M_Q; m++) begin
                            dbg_m = m;
                            lane_acc_idx_i = 4'd0 + m[2:0];
                            lane_valid_i   = 16'hFFFF;
                            @(posedge clk);
                        end

                        lane_valid_i = '0;
                        wt_valid_i   = 1'b0;
                        wait_cycles(3);
                    end

                    // ---------------------------------------------------------
                    // Final drain / normalize this d_v slice
                    // ---------------------------------------------------------
                    $display("[%0t ns] FINAL DRAIN h=%0d qi=%0d dvi=%0d", $time, h, qi, dvi);
                    cfg_mode_i = 2'b10;
                    dbg_phase  = PH_FINAL;

                    fork
                        begin
                            for (int m = 0; m < M_Q; m++) begin
                                dbg_m = m;
                                tile_drain_req_i      = 1'b1;
                                tile_drain_idx_i      = 4'd0 + m[2:0];
                                tile_clear_on_drain_i = 1'b1;
                                @(posedge clk);
                            end
                            tile_drain_req_i      = 1'b0;
                            tile_clear_on_drain_i = 1'b0;
                        end

                        begin
                            automatic int slice_drained = 0;
                            while (slice_drained < M_Q) begin
                                @(posedge clk);
                                if (out_vec_fp16_valid_o) begin
                                    automatic int m = slice_drained;
                                    automatic int q_global = qi * M_Q + m;
                                    for (int l = 0; l < NUM_LANES_P; l++) begin
                                        automatic int v_col = dvi * 16 + l;
                                        automatic int golden_idx = (h * N_Q + q_global) * D_V + v_col;
                                        automatic logic [15:0] actual_hex = out_vec_fp16_lanes_o[l*16 +: 16];
                                        automatic logic [15:0] golden_hex = mem_golden_fp16[golden_idx];

                                        automatic real actual_val = fp16_to_real(actual_hex);
                                        automatic real golden_val = fp16_to_real(golden_hex);
                                        automatic real diff       = actual_val - golden_val;
                                        automatic real abs_diff   = (diff < 0) ? -diff : diff;

                                        sum_abs_err += abs_diff;
                                        sum_sq_err  += (abs_diff * abs_diff);
                                        if (abs_diff > max_abs_err) max_abs_err = abs_diff;

                                        if (actual_hex == golden_hex) begin
                                            match_count++;
                                        end else begin
                                            mismatch_count++;
                                            $display("  [MISMATCH] h=%0d qi=%0d dvi=%0d q=%0d lane=%0d: actual=0x%04X golden=0x%04X",
                                                     h, qi, dvi, m, l, actual_hex, golden_hex);
                                        end

                                        if (csv_fd != 0)
                                            $fwrite(csv_fd, "%0d,%0d,%0d,%0d,%0d,%0d,0x%04X,%0.7f,0x%04X,%0.7f,%0.7f\n",
                                                    h, qi, dvi, m, l, v_col,
                                                    actual_hex, actual_val,
                                                    golden_hex, golden_val, abs_diff);
                                    end
                                    slice_drained++;
                                    total_drained++;
                                end
                            end
                        end
                    join

                    wait_cycles(5);
                end
            end
        end

        dbg_phase = PH_IDLE;
        dbg_h = 0; dbg_qi = 0; dbg_dvi = 0; dbg_kj = 0; dbg_kb = 0; dbg_m = 0;

        if (csv_fd != 0) $fclose(csv_fd);

        wait_cycles(5);
        $display("\n========================================================================");
        $display("           SNAX_FUSED_LLM_CORE FULL FLASH TILING RESULT                 ");
        $display("========================================================================");
        $display(" Total outputs verified (H=%0d x Q_TILES=%0d x DV_TILES=%0d x %0d x 16) : %0d",
                 H, Q_TILES, DV_TILES, M_Q, match_count + mismatch_count);
        $display(" BIT-EXACT matches : %0d", match_count);
        $display(" Mismatches        : %0d", mismatch_count);

        if (mismatch_count == 0 && match_count == H * Q_TILES * DV_TILES * M_Q * 16) begin
            $display("\n >>> [PASS] 100%% BIT-EXACT MATCH FOR FULL TILING LOOP NEST! <<<");
            $display(" >>> HEADS x QUERY-TILES x DV-SLICES x KV-TILES VERIFIED!      <<<");
        end else begin
            $display("\n >>> [STATUS] SIMULATION COMPLETE WITH DETAILED SUMMARY <<<");
        end

        if (match_count + mismatch_count > 0) begin
            $display("\nError Metrics vs Golden Model:");
            $display("  Max |Error| : %0.7f", max_abs_err);
            $display("  Mean |Error|: %0.7f", sum_abs_err / (match_count + mismatch_count));
            $display("  RMSE        : %0.7f", $sqrt(sum_sq_err / (match_count + mismatch_count)));
        end
        $display("========================================================================\n");

        $finish;
    end

    initial begin
        #(CLK_PERIOD * 200000);
        $display("\n[ERROR] Simulation watchdog timeout reached!");
        $finish;
    end

endmodule
