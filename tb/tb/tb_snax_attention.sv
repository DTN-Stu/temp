// =============================================================================
// Module: tb_snax_attention.sv
// Description: Directed, cycle-accurate core-level testbench for one attention
//              query on snax_fused_llm_core:
//
//                Q (1x16) x K^T (16x32) -> scores (32)
//                fused exp-free softmax (2 blocks of 16) -> P (32)
//                P (32) x V (32x16) -> O (16), normalized by l
//
//              Phase 1 (mode 01): two 16-key QK blocks, key mask 0xFFFF then
//              0x00FF, K block 1 has a 2x scale gain so the online max
//              increases and the O/l rescale path is exercised.
//              Phase 2 (mode 10): two P*V blocks, then drain and normalize.
//
//              Checks: QK scores, P packets, l, O and final FP16 against the
//              Python bit-exact golden.  Also reports RTL-vs-fakequant error.
// =============================================================================
`timescale 1ns/1ps
`include "pe_defines.svh"

module tb_snax_attention;

    localparam int NUM_LANES_P = 16;
    localparam int ACC_DEPTH_P = 16;
    localparam int PKT_W       = 72;
    localparam int FRAC_BITS   = 16;
    localparam int CLK_PERIOD  = 10;

    localparam int BLOCK_SIZE  = 16;
    localparam int NUM_BLOCKS  = 2;
    localparam int NUM_KEYS    = NUM_BLOCKS * BLOCK_SIZE;

    // ---------------------------------------------------------------------
    // DUT signals
    // ---------------------------------------------------------------------
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
    logic [NUM_LANES_P-1:0][PKT_W-1:0] wt_in_i;
    logic                            wt_valid_i;
    logic [NUM_LANES_P*16-1:0]       out_vec_fp16_packed_o;
    logic [NUM_LANES_P-1:0][15:0]    out_vec_fp16_lanes_o;
    logic                            out_vec_fp16_valid_o;
    logic [PKT_W-1:0]                out_vec_fp4_packet_o;
    logic                            out_vec_fp4_valid_o;
    logic                            p_fifo_full_o, p_fifo_empty_o;
    logic                            p_fifo_almost_full_o, l_ready_o;

    // ---------------------------------------------------------------------
    // Test vectors
    // ---------------------------------------------------------------------
    logic [PKT_W-1:0] mem_q          [0:0];
    logic [PKT_W-1:0] mem_k          [0:NUM_KEYS-1];
    logic [PKT_W-1:0] mem_v          [0:NUM_KEYS-1];
    logic [15:0]      mem_key_masks  [0:NUM_BLOCKS-1];
    logic [PKT_W-1:0] mem_golden_p   [0:NUM_BLOCKS-1];
    logic [31:0]      mem_scores_q16 [0:NUM_KEYS-1];
    logic [31:0]      mem_golden_l   [0:0];
    logic [31:0]      mem_golden_o   [0:NUM_LANES_P-1];
    logic [15:0]      mem_golden_fp16[0:NUM_LANES_P-1];
    logic [15:0]      mem_ref_fp16   [0:NUM_LANES_P-1];
    logic [15:0]      mem_gauss_fp16 [0:NUM_LANES_P-1];

    // ---------------------------------------------------------------------
    // Score / P / output checks
    // ---------------------------------------------------------------------
    int score_match = 0, score_mismatch = 0, score_count = 0;
    int p_match = 0, p_mismatch = 0, p_count = 0;
    int fp16_match = 0, fp16_mismatch = 0;
    int l_match = 0, l_mismatch = 0;
    int o_match = 0, o_mismatch = 0;

    real rtl_max_abs_err = 0.0;
    real rtl_sum_abs_err = 0.0;
    real rtl_sum_sq_err  = 0.0;
    real rtl_max_rel_err = 0.0;
    real rtl_sum_rel_err = 0.0;
    int  rtl_err_count   = 0;
    int  rtl_rel_count   = 0;
    int  rtl_worst_lane  = 0;
    int  rtl_err_fd      = 0;

    // ---------------------------------------------------------------------
    // DUT
    // ---------------------------------------------------------------------
    snax_fused_llm_core #(
        .NUM_LANES_P (NUM_LANES_P),
        .ACC_DEPTH_P (ACC_DEPTH_P),
        .PKT_W       (PKT_W),
        .FRAC_BITS   (FRAC_BITS)
    ) dut (
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
                val = (1.0 + real'(mant) / 1024.0)
                    * $itor(32'd1 << exp) / 32768.0;
            end
            return sign ? -val : val;
        end
    endfunction

    // ---------------------------------------------------------------------
    // Monitors: QK score comparison and softmax P packet comparison.
    // ---------------------------------------------------------------------
    always @(posedge clk) begin
        if (rst_n && (cfg_mode_i == 2'b01) && dut.raw_drained_valid) begin
            for (int l = 0; l < NUM_LANES_P; l++) begin
                logic signed [31:0] actual_score;
                logic signed [31:0] golden_score;
                actual_score = dut.score_array[l];
                golden_score = signed'(mem_scores_q16[score_count * NUM_LANES_P + l]);
                if (actual_score == golden_score) begin
                    score_match++;
                end else begin
                    score_mismatch++;
                    $display("[%0t ns] QK SCORE MISMATCH block=%0d lane=%0d actual=%0d (0x%08X) golden=%0d (0x%08X)",
                             $time, score_count, l, actual_score,
                             actual_score, golden_score, golden_score);
                end
            end
            $display("[%0t ns] QK block %0d scores captured (%0d lanes)",
                     $time, score_count, NUM_LANES_P);
            score_count++;
        end

        if (rst_n && dut.sm_out_valid) begin
            logic [PKT_W-1:0] actual_p;
            actual_p = {dut.sm_p_blk_out, dut.sm_code_out};
            if (p_count < NUM_BLOCKS) begin
                if (actual_p == mem_golden_p[p_count]) begin
                    p_match++;
                end else begin
                    p_mismatch++;
                    $display("[%0t ns] P PACKET MISMATCH block=%0d actual=0x%018X golden=0x%018X",
                             $time, p_count, actual_p, mem_golden_p[p_count]);
                end
            end
            $display("[%0t ns] P block %0d: p_blk=%0d csum=%0d l_contrib=%0d code0=0x%0X",
                     $time, p_count, $signed(dut.sm_p_blk_out),
                     dut.u_softmax_requantizer.csum_out,
                     $signed(dut.sm_l_contrib_out),
                     dut.sm_code_out[3:0]);
            p_count++;
        end
    end

    // ---------------------------------------------------------------------
    // Main sequence
    // ---------------------------------------------------------------------
    initial begin
        $display("\n========================================================================");
        $display("   TESTBENCH: ATTENTION Q(1x16) x K(32x16)^T -> SOFTMAX -> P(32) x V(32x16) ");
        $display("   Core-level fused FP4 attention, 32 keys, head dim 16, 2 key blocks   ");
        $display("========================================================================");

        $readmemh("sim_attn/hex/q_packet.hex", mem_q);
        $readmemh("sim_attn/hex/k_packets.hex", mem_k);
        $readmemh("sim_attn/hex/v_packets.hex", mem_v);
        $readmemh("sim_attn/hex/key_masks.hex", mem_key_masks);
        $readmemh("sim_attn/hex/golden_p_packets.hex", mem_golden_p);
        $readmemh("sim_attn/hex/golden_scores_q16.hex", mem_scores_q16);
        $readmemh("sim_attn/hex/golden_l_q16.hex", mem_golden_l);
        $readmemh("sim_attn/hex/golden_o_q16.hex", mem_golden_o);
        $readmemh("sim_attn/hex/golden_fp16.hex", mem_golden_fp16);
        $readmemh("sim_attn/hex/golden_ref_fp16.hex", mem_ref_fp16);
        $readmemh("sim_attn/hex/golden_gauss_fp16.hex", mem_gauss_fp16);

        $display("[%0t ns] Loaded Q/K/V, QK-score, P, l/O and FP16 golden files.", $time);

        rst_n                 = 1'b0;
        cfg_mode_i            = 2'b01;
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
        for (int l = 0; l < NUM_LANES_P; l++) wt_in_i[l] = '0;

        wait_cycles(5);
        rst_n = 1'b1;
        wait_cycles(5);

        // =================================================================
        // PHASE 1: QK block 0
        // lane_valid at T, accumulator commit T+1, drain request T+2,
        // raw score T+3 -> softmax input.  P0 appears at T+5.
        // =================================================================
        @(negedge clk);
        cfg_mode_i          = 2'b01;
        cfg_key_mask_i      = mem_key_masks[0];
        cfg_row_start_i     = 1'b0;
        ext_act_broadcast_i = mem_q[0];
        ext_act_valid_i     = 1'b1;
        wt_valid_i          = 1'b1;
        for (int l = 0; l < NUM_LANES_P; l++) wt_in_i[l] = mem_k[l];
        lane_valid_i          = 16'hFFFF;
        lane_acc_idx_i        = '0;
        tile_clear_on_drain_i = 1'b1;
        tile_drain_req_i      = 1'b0;

        @(posedge clk);         // T: QK stage 1, block 0
        @(negedge clk);
        lane_valid_i = '0;

        @(posedge clk);         // T+1: stage 2 commits QK block 0
        @(negedge clk);
        // T+1.5: drain block 0 and start block 1.
        tile_drain_req_i      = 1'b1;
        tile_drain_idx_i      = '0;
        tile_clear_on_drain_i = 1'b1;
        lane_valid_i          = 16'hFFFF;
        lane_acc_idx_i        = '0;
        for (int l = 0; l < NUM_LANES_P; l++) wt_in_i[l] = mem_k[BLOCK_SIZE + l];

        @(posedge clk);         // T+2: drain block 0; stage 1 block 1
        @(negedge clk);
        tile_drain_req_i      = 1'b0;
        tile_clear_on_drain_i = 1'b0;
        lane_valid_i          = '0;

        // T+2.5: row_start and mask for the first softmax block.
        cfg_row_start_i = 1'b1;
        cfg_key_mask_i  = mem_key_masks[0];

        @(posedge clk);         // T+3: raw score block 0 -> softmax input
        @(negedge clk);
        cfg_row_start_i = 1'b0;

        // T+3.5: drain block 1.
        tile_drain_req_i      = 1'b1;
        tile_drain_idx_i      = '0;
        tile_clear_on_drain_i = 1'b1;

        @(posedge clk);         // T+4: drain block 1
        @(negedge clk);
        tile_drain_req_i      = 1'b0;
        tile_clear_on_drain_i = 1'b0;
        cfg_key_mask_i        = mem_key_masks[1];

        @(posedge clk);         // T+5: raw score block 1; P0 valid
        @(negedge clk);

        @(posedge clk);         // T+6: l after block 0
        @(posedge clk);         // T+7: P1 valid; l after block 1 at T+8

        // =================================================================
        // PHASE 2: P*V block 0 and block 1, back to back.
        // l block 1 lands at T+8 together with the first P*V consume;
        // O commit at T+9/T+10, drain T+11, raw O T+12, output T+13.
        // =================================================================
        @(negedge clk);
        cfg_mode_i            = 2'b10;
        cfg_key_mask_i        = '1;
        ext_act_broadcast_i   = '0;
        ext_act_valid_i       = 1'b0;
        for (int l = 0; l < NUM_LANES_P; l++) wt_in_i[l] = mem_v[l];
        lane_valid_i          = 16'hFFFF;
        lane_acc_idx_i        = '0;
        tile_drain_req_i      = 1'b0;
        tile_clear_on_drain_i = 1'b0;

        @(posedge clk);         // T+8: P*V block 0 (P0 popped)
        @(negedge clk);
        for (int l = 0; l < NUM_LANES_P; l++) wt_in_i[l] = mem_v[BLOCK_SIZE + l];

        @(posedge clk);         // T+9: P*V block 1 (P1 popped, O rescale)
        @(negedge clk);
        lane_valid_i = '0;

        @(posedge clk);         // T+10: O stage 2 for block 1
        @(negedge clk);
        tile_drain_req_i      = 1'b1;
        tile_drain_idx_i      = '0;
        tile_clear_on_drain_i = 1'b1;

        @(posedge clk);         // T+11: drain O
        @(negedge clk);
        tile_drain_req_i      = 1'b0;
        tile_clear_on_drain_i = 1'b0;

        @(posedge clk);         // T+12: raw O -> final normalizer
        @(posedge clk);         // T+13: FP16 output valid

        if (!out_vec_fp16_valid_o) begin
            $display("[%0t ns] Waiting for final normalizer output...", $time);
            while (!out_vec_fp16_valid_o) @(posedge clk);
        end

        // =================================================================
        // Final checks
        // =================================================================
        $display("\n------------------------------------------------------------------------");
        $display("[%0t ns] Attention output and final checks", $time);
        $display("------------------------------------------------------------------------");

        rtl_err_fd = $fopen("sim_attn/csv/rtl_vs_fakequant.csv", "w");
        if (rtl_err_fd != 0)
            $fwrite(rtl_err_fd,
                    "lane,actual_fp16,actual_real,ref_fp16,ref_real,abs_err,rel_err\n");

        for (int l = 0; l < NUM_LANES_P; l++) begin
            logic [15:0] actual_hex, golden_hex, ref_hex, gauss_hex;
            real actual_val, golden_val, ref_val;
            real err_val, abs_err, rel_err;

            actual_hex = out_vec_fp16_lanes_o[l];
            golden_hex = mem_golden_fp16[l];
            ref_hex    = mem_ref_fp16[l];
            gauss_hex  = mem_gauss_fp16[l];
            actual_val = fp16_to_real(actual_hex);
            golden_val = fp16_to_real(golden_hex);
            ref_val    = fp16_to_real(ref_hex);

            err_val = actual_val - ref_val;
            abs_err = (err_val < 0.0) ? -err_val : err_val;
            rel_err = 0.0;
            if (ref_val != 0.0) begin
                rel_err = abs_err / ((ref_val < 0.0) ? -ref_val : ref_val);
                rtl_sum_rel_err += rel_err;
                rtl_rel_count++;
                if (rel_err > rtl_max_rel_err) rtl_max_rel_err = rel_err;
            end
            rtl_sum_abs_err += abs_err;
            rtl_sum_sq_err  += err_val * err_val;
            if (abs_err > rtl_max_abs_err) begin
                rtl_max_abs_err = abs_err;
                rtl_worst_lane  = l;
            end
            rtl_err_count++;

            if (rtl_err_fd != 0)
                $fwrite(rtl_err_fd, "%0d,0x%04X,%0.9f,0x%04X,%0.9f,%0.9f,%0.9f\n",
                        l, actual_hex, actual_val, ref_hex, ref_val, abs_err, rel_err);

            if (actual_hex == golden_hex) begin
                fp16_match++;
                $display("  Lane %2d: actual=0x%04X (%9.6f) golden=0x%04X (%9.6f) [MATCH]",
                         l, actual_hex, actual_val, golden_hex, golden_val);
            end else begin
                fp16_mismatch++;
                $display("  Lane %2d: actual=0x%04X (%9.6f) golden=0x%04X (%9.6f) [MISMATCH]",
                         l, actual_hex, actual_val, golden_hex, golden_val);
            end

            $display("           ref fakequant=0x%04X (%9.6f) ref gaussian=0x%04X",
                     ref_hex, ref_val, gauss_hex);
        end

        if (rtl_err_fd != 0) $fclose(rtl_err_fd);

        // l check (internal Q16.16 denominator).
        begin
            logic signed [31:0] actual_l;
            logic signed [31:0] golden_l;
            actual_l = dut.u_l_accumulator.l_total_out;
            golden_l = signed'(mem_golden_l[0]);
            if (actual_l == golden_l) begin
                l_match++;
                $display("  l Q16.16: actual=%0d golden=%0d [MATCH]", actual_l, golden_l);
            end else begin
                l_mismatch++;
                $display("  l Q16.16: actual=%0d golden=%0d [MISMATCH]", actual_l, golden_l);
            end
        end

        // O check (matmul drain word feeding the final normalizer).
        for (int l = 0; l < NUM_LANES_P; l++) begin
            logic signed [31:0] actual_o;
            logic signed [31:0] golden_o;
            actual_o = dut.score_array[l];
            golden_o = signed'(mem_golden_o[l]);
            if (actual_o == golden_o) begin
                o_match++;
            end else begin
                o_mismatch++;
                $display("  O Q16.16 lane %2d: actual=%0d golden=%0d [MISMATCH]",
                         l, actual_o, golden_o);
            end
        end
        $display("  O Q16.16: %0d/%0d lanes match", o_match, NUM_LANES_P);

        wait_cycles(2);
        $display("\n========================================================================");
        $display("                    ATTENTION TEST RESULT                              ");
        $display("========================================================================");
        $display(" QK scores (32)          : %0d match, %0d mismatch, captured %0d",
                 score_match, score_mismatch, score_count);
        $display(" P packets (2)           : %0d match, %0d mismatch, captured %0d",
                 p_match, p_mismatch, p_count);
        $display(" l Q16.16 (1)            : %0d match, %0d mismatch", l_match, l_mismatch);
        $display(" O Q16.16 (16)           : %0d match, %0d mismatch", o_match, o_mismatch);
        $display(" Final FP16 (16)         : %0d match, %0d mismatch",
                 fp16_match, fp16_mismatch);

        if (rtl_err_count > 0) begin
            $display("\nRTL vs fake-quantization reference:");
            $display("  points      : %0d", rtl_err_count);
            $display("  max |err|   : %0.9f (lane %0d)", rtl_max_abs_err, rtl_worst_lane);
            $display("  mean |err|  : %0.9f", rtl_sum_abs_err / rtl_err_count);
            $display("  RMSE        : %0.9f", $sqrt(rtl_sum_sq_err / rtl_err_count));
            $display("  max rel err : %0.9f", rtl_max_rel_err);
            if (rtl_rel_count > 0)
                $display("  mean rel err: %0.9f", rtl_sum_rel_err / rtl_rel_count);
        end

        if (score_match == NUM_KEYS && score_mismatch == 0 &&
            p_match == NUM_BLOCKS && p_mismatch == 0 &&
            l_match == 1 && l_mismatch == 0 &&
            o_match == NUM_LANES_P && o_mismatch == 0 &&
            fp16_match == NUM_LANES_P && fp16_mismatch == 0) begin
            $display("\n >>> [PASS] ATTENTION + FUSED SOFTMAX BIT-EXACT! <<<");
        end else begin
            $display("\n >>> [FAIL] MISMATCHES DETECTED - CHECK TIMING/DATA! <<<");
        end
        $display("========================================================================\n");

        $finish;
    end

    initial begin
        #(CLK_PERIOD * 500);
        $display("\n[ERROR] Simulation timeout reached!");
        $finish;
    end

endmodule
