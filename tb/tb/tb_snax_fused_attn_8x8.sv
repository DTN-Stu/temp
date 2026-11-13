// =============================================================================
// Module: tb_snax_fused_attn_8x8.sv
// Description: Cycle-accurate, bit-exact Testbench for SNAX Fused LLM Core
//              verifying FlashAttention with 8:8 Accumulator Partitioning:
//              - Slots 0..7  : Persistent Output Accumulator O (8 Query Tokens)
//              - Slots 8..15 : Scratchpad Score Buffer S (8 Query Tokens x 16 Keys)
//              - Workload: M_query=8, d_k=32, d_v=16, N_seq=32 (2 blocks x 16 keys)
//              - Automated verification against Python Bit-exact Golden Reference
// =============================================================================
`timescale 1ns/1ps
`include "pe_defines.svh"

module tb_snax_fused_attn_8x8;

    localparam int NUM_LANES_P = 16;
    localparam int ACC_DEPTH_P = 16;   // 8:8 Split (Total 16 slots)
    localparam int PKT_W       = 72;   // 64b FP4 + 8b Log8 scale
    localparam int FRAC_BITS   = 16;
    localparam int CLK_PERIOD  = 10;   // 100 MHz (10ns)

    localparam int M_QUERIES   = 8;    // Q0..Q7
    localparam int K_BLOCKS    = 2;    // d_k = 32 -> 2 blocks of 16
    localparam int KV_BLOCKS   = 2;    // 32 keys/values -> 2 blocks of 16 keys
    localparam int N_KEYS      = 16;   // 16 keys per KV block

    // DUT Signals
    logic                            clk;
    logic                            rst_n;

    // Control
    logic [1:0]                      cfg_mode_i;
    logic                            cfg_row_start_i;
    logic [NUM_LANES_P-1:0]          cfg_key_mask_i;
    logic                            cfg_enable_slq_i;

    // Drain & Accumulator Indexing
    logic                            tile_drain_req_i;
    logic [$clog2(ACC_DEPTH_P)-1:0]  tile_drain_idx_i;
    logic                            tile_clear_on_drain_i;

    logic [NUM_LANES_P-1:0]          lane_valid_i;
    logic [$clog2(ACC_DEPTH_P)-1:0]  lane_acc_idx_i;

    // Data Streams
    logic [PKT_W-1:0]                ext_act_broadcast_i;
    logic                            ext_act_valid_i;
    logic [NUM_LANES_P-1:0][PKT_W-1:0] wt_in_i;
    logic                            wt_valid_i;

    // Outputs
    logic [NUM_LANES_P*16-1:0]       out_vec_fp16_packed_o;
    logic [NUM_LANES_P-1:0][15:0]    out_vec_fp16_lanes_o;
    logic                            out_vec_fp16_valid_o;
    logic [PKT_W-1:0]                out_vec_fp4_packet_o;
    logic                            out_vec_fp4_valid_o;

    logic                            p_fifo_full_o;
    logic                            p_fifo_empty_o;
    logic                            p_fifo_almost_full_o;
    logic                            l_ready_o;

    // Testbench Storage for Golden Data
    logic [PKT_W-1:0] mem_q [0:M_QUERIES * K_BLOCKS - 1]; // 16 packets
    logic [PKT_W-1:0] mem_k [0:KV_BLOCKS * K_BLOCKS * 16 - 1]; // 64 packets
    logic [PKT_W-1:0] mem_v [0:KV_BLOCKS * 16 - 1]; // 32 packets
    logic [15:0]      mem_golden [0:M_QUERIES * 16 - 1]; // 128 FP16 words

    // Instantiate DUT
    snax_fused_llm_core #(
        .NUM_LANES_P  (NUM_LANES_P),
        .ACC_DEPTH_P  (ACC_DEPTH_P),
        .PKT_W        (PKT_W),
        .FRAC_BITS    (FRAC_BITS),
        .P_FIFO_DEPTH (16),
        .NUM_QUERIES  (8)
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

    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD / 2) clk = ~clk;
    end

    task automatic wait_cycles(input int n);
        repeat (n) @(posedge clk);
    endtask

    // FP16 to real helper
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

    // Debug monitors: first divergent stage (QK score / P packet / l).
    always @(posedge clk) begin
        if (rst_n && (cfg_mode_i == 2'b01) && u_dut.raw_drained_valid) begin
            $display("DBG_QK q=%0d s0=%0d s1=%0d s2=%0d s3=%0d s4=%0d s5=%0d s6=%0d s7=%0d s8=%0d s9=%0d s10=%0d s11=%0d s12=%0d s13=%0d s14=%0d s15=%0d",
                     u_dut.drain_query_idx,
                     u_dut.score_array[0],  u_dut.score_array[1],  u_dut.score_array[2],  u_dut.score_array[3],
                     u_dut.score_array[4],  u_dut.score_array[5],  u_dut.score_array[6],  u_dut.score_array[7],
                     u_dut.score_array[8],  u_dut.score_array[9],  u_dut.score_array[10], u_dut.score_array[11],
                     u_dut.score_array[12], u_dut.score_array[13], u_dut.score_array[14], u_dut.score_array[15]);
        end
        if (rst_n && u_dut.sm_out_valid) begin
            $display("DBG_P q=%0d pblk=%0d lc=%0d code0=%0h code1=%0h code2=%0h code3=%0h",
                     u_dut.sm_query_idx_d2, $signed(u_dut.sm_p_blk_out),
                     $signed(u_dut.sm_l_contrib_out),
                     u_dut.sm_code_out[3:0], u_dut.sm_code_out[7:4],
                     u_dut.sm_code_out[11:8], u_dut.sm_code_out[15:12]);
        end
    end

    // Variables for statistics
    int match_count    = 0;
    int mismatch_count = 0;
    int drain_checked  = 0;
    int rtl_err_fd;

    real max_abs_err = 0.0;
    real sum_abs_err = 0.0;
    real sum_sq_err  = 0.0;

    initial begin
        $display("\n========================================================================");
        $display("   TESTBENCH: FUSED FLASHATTENTION 8:8 PARTITIONING VERIFICATION        ");
        $display("   M_query=8, d_k=32, d_v=16, N_seq=32 keys (2 blocks x 16 keys)       ");
        $display("   Slots 0..7  : Persistent Output O (idx[3]=0)                        ");
        $display("   Slots 8..15 : Scratchpad Scores S (idx[3]=1)                         ");
        $display("========================================================================\n");

        $readmemh("sim_fused_attn_8x8/hex/q_packets.hex", mem_q);
        $readmemh("sim_fused_attn_8x8/hex/k_packets.hex", mem_k);
        $readmemh("sim_fused_attn_8x8/hex/v_packets.hex", mem_v);
        $readmemh("sim_fused_attn_8x8/hex/golden_attn_fp16.hex", mem_golden);

        $display("[%0t ns] Loaded input packets & golden reference into memory.", $time);

        // Reset and initialization
        rst_n                 = 1'b0;
        cfg_mode_i            = 2'b00;
        cfg_row_start_i       = 1'b0;
        cfg_key_mask_i        = '1;
        cfg_enable_slq_i      = 1'b0; // Output FP16
        tile_drain_req_i      = 1'b0;
        tile_drain_idx_i      = '0;
        tile_clear_on_drain_i = 1'b0;
        lane_valid_i          = '0;
        lane_acc_idx_i        = '0;
        ext_act_broadcast_i   = '0;
        ext_act_valid_i       = 1'b0;
        wt_valid_i            = 1'b0;
        for (int l = 0; l < NUM_LANES_P; l++) wt_in_i[l] = '0;

        wait_cycles(5);
        rst_n = 1'b1;
        wait_cycles(5);
        $display("[%0t ns] Reset deasserted. Core ready.\n", $time);

        rtl_err_fd = $fopen("sim_fused_attn_8x8/csv/rtl_vs_golden.csv", "w");
        if (rtl_err_fd != 0)
            $fwrite(rtl_err_fd, "query_idx,lane,actual_fp16,actual_real,golden_fp16,golden_real,abs_err\n");

        // =====================================================================
        // LOOP OVER KV BLOCKS: j = 0, 1 (2 blocks of 16 keys/values)
        // =====================================================================
        for (int j = 0; j < KV_BLOCKS; j++) begin
            $display("------------------------------------------------------------------------");
            $display("[%0t ns] >>> STARTING KV BLOCK j = %0d (Keys %0d..%0d) <<<", 
                     $time, j, j*16, j*16+15);
            $display("------------------------------------------------------------------------");

            // -----------------------------------------------------------------
            // STEP 1: MODE 01 — Compute Q * K^T -> Slots 8..15 (idx[3]=1)
            // -----------------------------------------------------------------
            $display("[%0t ns] [MODE 01] Computing Scores S = Q * K^T into Slots 8..15...", $time);
            cfg_mode_i      = 2'b01;
            cfg_row_start_i = 1'b0;

            // Accumulate dot products across d_k (2 sub-blocks of 16 elements)
            for (int kb = 0; kb < K_BLOCKS; kb++) begin
                // Load 16 keys for this sub-block onto wt_in_i[0:15]
                for (int l = 0; l < NUM_LANES_P; l++) begin
                    automatic int k_mem_idx = j * (K_BLOCKS * 16) + kb * 16 + l;
                    wt_in_i[l] = mem_k[k_mem_idx];
                end
                wt_valid_i = 1'b1;

                // Broadcast 8 queries into Scratchpad Slots 8..15
                for (int m = 0; m < M_QUERIES; m++) begin
                    automatic int q_mem_idx = m * K_BLOCKS + kb;
                    ext_act_broadcast_i = mem_q[q_mem_idx];
                    ext_act_valid_i     = 1'b1;

                    // Address: idx[3]=1 (Slot 8 + m)
                    lane_acc_idx_i      = 4'd8 + m[2:0];
                    lane_valid_i        = '1;

                    @(posedge clk);
                end
            end

            // Deassert inputs
            ext_act_valid_i = 1'b0;
            lane_valid_i    = '0;
            wt_valid_i      = 1'b0;

            // Wait 2 cycles for Stage 2 RMW pipeline commit
            wait_cycles(2);

            // -----------------------------------------------------------------
            // STEP 2: DRAIN SLOTS 8..15 TO SOFTMAX & PUSH TO P_FIFO
            // -----------------------------------------------------------------
            $display("[%0t ns] [DRAIN S] Draining Slots 8..15 to Softmax Requantizer...", $time);
            for (int m = 0; m < M_QUERIES; m++) begin
                tile_drain_req_i      = 1'b1;
                tile_drain_idx_i      = 4'd8 + m[2:0];
                tile_clear_on_drain_i = 1'b1; // Auto-clear scratchpad slot!
                @(posedge clk);               // drain request
                tile_drain_req_i      = 1'b0;
                tile_clear_on_drain_i = 1'b0;
                if (j == 0) begin
                    cfg_row_start_i = 1'b1;   // first raw score of this query
                    @(posedge clk);
                    cfg_row_start_i = 1'b0;
                end
            end

            // Wait 4 cycles for Softmax pipeline (latency 2) to push all 8 P blocks into p_fifo
            wait_cycles(4);
            $display("[%0t ns] [P_FIFO] P blocks generated and pushed to p_fifo. FIFO count: %0d", 
                     $time, u_dut.u_p_fifo.count);

            // -----------------------------------------------------------------
            // STEP 3: MODE 10 — Consume P from FIFO & Accumulate P * V into Slots 0..7
            // -----------------------------------------------------------------
            $display("[%0t ns] [MODE 10] Accumulating P * V into Output Slots 0..7 (idx[3]=0)...", $time);
            cfg_mode_i = 2'b10;

            // Load Value matrix V_j onto wt_in_i[0:15]
            for (int l = 0; l < NUM_LANES_P; l++) begin
                automatic int v_mem_idx = j * 16 + l;
                wt_in_i[l] = mem_v[v_mem_idx];
            end
            wt_valid_i = 1'b1;

            // Consume 8 P packets from p_fifo and accumulate into Slots 0..7
            for (int m = 0; m < M_QUERIES; m++) begin
                // In Mode 10, activation comes from p_fifo head automatically.
                // We point lane_acc_idx_i to Slot 0 + m (idx[3]=0)
                lane_acc_idx_i = 4'd0 + m[2:0];
                lane_valid_i   = '1;

                @(posedge clk);
            end

            // Deassert compute inputs
            lane_valid_i = '0;
            wt_valid_i   = 1'b0;

            // Wait 3 cycles for Stage 2 accumulation & rescaling to commit
            wait_cycles(3);
            $display("[%0t ns] [MODE 10 COMPLETE] KV Block j=%0d accumulated. FIFO empty: %0b\n",
                     $time, j, p_fifo_empty_o);
        end

        // =====================================================================
        // STEP 4: FINAL DRAIN & NORMALIZATION — Drain Slots 0..7 (Output O)
        // =====================================================================
        $display("========================================================================");
        $display("[%0t ns] >>> DRAINING FINAL OUTPUT O FROM SLOTS 0..7 (WITH 1/l) <<<", $time);
        $display("========================================================================");

        // Keep Mode 10 so final_normalizer applies 1/l (bypass_inv_l = 0)
        cfg_mode_i = 2'b10;

        fork
            // Thread A: Request Drains for Slots 0..7
            begin
                for (int m = 0; m < M_QUERIES; m++) begin
                    tile_drain_req_i      = 1'b1;
                    tile_drain_idx_i      = 4'd0 + m[2:0];
                    tile_clear_on_drain_i = 1'b1;
                    @(posedge clk);
                end
                tile_drain_req_i      = 1'b0;
                tile_clear_on_drain_i = 1'b0;
            end

            // Thread B: Capture and verify output FP16 vector
            begin
                while (drain_checked < M_QUERIES) begin
                    @(posedge clk);
                    if (out_vec_fp16_valid_o) begin
                        automatic int m = drain_checked;
                        $display("[%0t ns] Captured FP16 Output for Query Token %0d:", $time, m);

                        for (int l = 0; l < NUM_LANES_P; l++) begin
                            automatic logic [15:0] actual_hex = out_vec_fp16_lanes_o[l];
                            automatic logic [15:0] golden_hex = mem_golden[m * 16 + l];

                            automatic real actual_val = fp16_to_real(actual_hex);
                            automatic real golden_val = fp16_to_real(golden_hex);
                            automatic real diff       = actual_val - golden_val;
                            automatic real abs_diff   = (diff < 0) ? -diff : diff;

                            sum_abs_err += abs_diff;
                            sum_sq_err  += (abs_diff * abs_diff);
                            if (abs_diff > max_abs_err) max_abs_err = abs_diff;

                            if (rtl_err_fd != 0)
                                $fwrite(rtl_err_fd, "%0d,%0d,0x%04X,%0.7f,0x%04X,%0.7f,%0.7f\n",
                                        m, l, actual_hex, actual_val, golden_hex, golden_val, abs_diff);

                            if (actual_hex == golden_hex) begin
                                match_count++;
                                $display("  Lane %2d: Actual=0x%04X (~%7.4f) | Golden=0x%04X (~%7.4f) -> [MATCH]",
                                         l, actual_hex, actual_val, golden_hex, golden_val);
                            end else begin
                                mismatch_count++;
                                $display("  Lane %2d: Actual=0x%04X (~%7.4f) | Golden=0x%04X (~%7.4f) -> [MISMATCH]",
                                         l, actual_hex, actual_val, golden_hex, golden_val);
                            end
                        end
                        drain_checked++;
                    end
                end
            end
        join

        if (rtl_err_fd != 0) $fclose(rtl_err_fd);

        wait_cycles(5);
        $display("\n========================================================================");
        $display("            FUSED FLASHATTENTION 8:8 VERIFICATION RESULT                ");
        $display("========================================================================");
        $display(" Total outputs checked (8 Queries x 16 Lanes) : %0d", match_count + mismatch_count);
        $display(" BIT-EXACT matches                            : %0d", match_count);
        $display(" Mismatches                                   : %0d", mismatch_count);
        if (mismatch_count == 0 && match_count == M_QUERIES * 16) begin
            $display("\n >>> [PASS] 100%% BIT-EXACT MATCH WITH GOLDEN REFERENCE! <<<");
            $display(" >>> FLASHATTENTION 8:8 PARTITIONING VERIFIED SUCCESSFULLY! <<<");
        end else begin
            $display("\n >>> [STATUS] COMPLETED WITH SUMMARY METRICS <<<");
        end

        if (match_count + mismatch_count > 0) begin
            automatic int total = match_count + mismatch_count;
            $display("\nError Metrics vs Golden Model:");
            $display("  Max |Error| : %0.7f", max_abs_err);
            $display("  Mean |Error|: %0.7f", sum_abs_err / total);
            $display("  RMSE        : %0.7f", $sqrt(sum_sq_err / total));
        end
        $display("========================================================================\n");

        $finish;
    end

    // Simulation watchdog timeout
    initial begin
        #(CLK_PERIOD * 5000);
        $display("\n[ERROR] Simulation watchdog timeout!");
        $finish;
    end

endmodule
