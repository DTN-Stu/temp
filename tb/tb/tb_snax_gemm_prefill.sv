// =============================================================================
// Module: tb_snax_gemm_prefill.sv
// Description: Dedicated, cycle-accurate, bit-exact Testbench for GEMM W*A
//              specifically targeting the LLM Prefill Phase:
//              - Weight Matrix:     16 (Out channels) x 32 (Reduction dim K)
//              - Activation Matrix: 32 (Reduction dim K) x 16 (Tokens)
//              - Output Matrix:     16 (Out channels) x 16 (Tokens)
//              - Architecture:      Weight-Stationary Dataflow using 16 Accumulator Slots
//              - Automated verification using generated CSV/HEX golden data
// =============================================================================
`timescale 1ns/1ps
`include "pe_defines.svh"

module tb_snax_gemm_prefill;

    localparam int NUM_LANES_P  = 16;
    localparam int ACC_DEPTH_P  = 16;
    localparam int PKT_W        = 72;
    localparam int FRAC_BITS    = 16;
    localparam int CLK_PERIOD   = 10; // 100 MHz clock (10ns)

    localparam int K_DIM        = 256;
    localparam int NUM_TOKENS   = 16;
    localparam int NUM_CHUNKS   = K_DIM / 16; // K_DIM / 16 chunks

    // DUT Signals
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
    logic                            p_fifo_full_o, p_fifo_empty_o, l_ready_o;

    // Memories for HEX test vectors
    logic [PKT_W-1:0] mem_weights     [0:NUM_CHUNKS*NUM_LANES_P - 1];
    logic [PKT_W-1:0] mem_activations [0:NUM_CHUNKS*NUM_TOKENS - 1];
    logic [15:0]      mem_golden_fp16 [0:NUM_TOKENS*NUM_LANES_P - 1];
    logic signed [31:0] mem_ref_q16   [0:NUM_TOKENS*NUM_LANES_P - 1];

    int match_count = 0;
    int mismatch_count = 0;
    int drain_issued = 0;
    int drain_checked = 0;

    // RTL vs fake-quantization error metrics
    real rtl_max_abs_err  = 0.0;
    real rtl_sum_abs_err  = 0.0;
    real rtl_sum_sq_err   = 0.0;
    real rtl_max_rel_err  = 0.0;
    real rtl_sum_rel_err  = 0.0;
    int  rtl_err_count    = 0;
    int  rtl_rel_count    = 0;
    int  rtl_worst_tok    = 0;
    int  rtl_worst_lane   = 0;
    int  rtl_err_fd       = 0;

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
                // Subnormal: mant * 2^-24
                val = real'(mant) / 16777216.0;
            end else begin
                // 2^(exp-15) = (1 << exp) / 2^15; avoid variable-exponent **
                // which crashes Verilator 5.032.
                val = (1.0 + real'(mant) / 1024.0)
                    * $itor(32'd1 << exp) / 32768.0;
            end
            return sign ? -val : val;
        end
    endfunction

    initial begin
        $display("\n========================================================================");
        $display("   TESTBENCH: GEMM W(16x%0d) x A(%0dx16) -> Y(16x16) PREFILL VERIFICATION  ",
                 K_DIM, K_DIM);
        $display("   Using Weight-Stationary Dataflow on snax_fused_llm_core              ");
        $display("========================================================================\n");

        $readmemh("sim_gemm_prefill/hex/weight_packets.hex", mem_weights);
        $readmemh("sim_gemm_prefill/hex/activation_packets.hex", mem_activations);
        $readmemh("sim_gemm_prefill/hex/golden_fp16.hex", mem_golden_fp16);
        $readmemh("sim_gemm_prefill/hex/golden_ref_q16.hex", mem_ref_q16);

        if ($isunknown(mem_ref_q16[0]))
            $display("[WARNING] golden_ref_q16.hex missing; RTL error metrics will be invalid.");

        $display("[%0t ns] Loaded weight/activation/golden HEX files into testbench memory.", $time);

        rst_n                 = 1'b0;
        cfg_mode_i            = 2'b00; // GEMM Mode
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
        $display("[%0t ns] Reset complete. Starting Weight-Stationary GEMM...\n", $time);

        for (int ch = 0; ch < NUM_CHUNKS; ch++) begin
            for (int tok = 0; tok < NUM_TOKENS; tok++) begin
                @(posedge clk);

                // At this edge the previous token (if any) has been captured by
                // stage 1. Swap to the new weight chunk after the edge so the
                // last token of the previous chunk still used the old weights.
                // Nonblocking assignment prevents the swap from reaching the
                // DUT's stage-1 flops at this same edge.
                if (tok == 0) begin
                    $display("------------------------------------------------------------------------");
                    $display("[%0t ns] >>> LOAD WEIGHT CHUNK %0d (K = %0d..%0d) <<<", 
                             $time, ch, ch*16, (ch+1)*16 - 1);
                    $display("------------------------------------------------------------------------");
                    for (int l = 0; l < NUM_LANES_P; l++) begin
                        wt_in_i[l] <= mem_weights[ch * NUM_LANES_P + l];
                    end
                    wt_valid_i <= 1'b1;
                end

                // Keep lane_valid high across chunk boundaries: one token/cycle.
                ext_act_broadcast_i = mem_activations[ch * NUM_TOKENS + tok];
                ext_act_valid_i     = 1'b1;
                lane_valid_i        = 16'hFFFF;
                lane_acc_idx_i      = tok[5:0]; // Accumulate into the slot for this token

                $display("[%0t ns] Chunk %0d | Cycle %2d: Stream Token %2d -> Latch Slot %2d (Weight Reused!)",
                         $time, ch, tok, tok, tok);
            end
        end

        // Capture the final token, then let stage 2 commit it.
        @(posedge clk);
        ext_act_valid_i = 1'b0;
        lane_valid_i    = '0;
        wt_valid_i      = 1'b0;
        wait_cycles(2);

        $display("\n========================================================================");
        $display("[%0t ns] Completed all K=%0d accumulation for 16 Tokens!", $time, K_DIM);
        $display("         Starting DRAIN and BIT-EXACT CHECK...");
        $display("========================================================================\n");

        // Pipelined drain: issue one slot per cycle.  The drain path has II=1
        // (request Td -> word Td+1 -> FP16 valid Td+2), so no inter-slot idle
        // cycles are required.  Outputs arrive in the same token order, one
        // per cycle, two cycles after the corresponding request.
        rtl_err_fd = $fopen("sim_gemm_prefill/csv/rtl_vs_fakequant.csv", "w");
        if (rtl_err_fd != 0)
            $fwrite(rtl_err_fd,
                    "token,lane,actual_fp16,actual_real,ref_q16_hex,ref_real,abs_err,rel_err\n");

        drain_issued  = 0;
        drain_checked = 0;
        while (drain_checked < NUM_TOKENS) begin
            @(posedge clk);

            if (drain_issued < NUM_TOKENS) begin
                tile_drain_req_i      = 1'b1;
                tile_drain_idx_i      = drain_issued[5:0];
                tile_clear_on_drain_i = 1'b1;
                drain_issued++;
            end else begin
                tile_drain_req_i      = 1'b0;
                tile_clear_on_drain_i = 1'b0;
            end

            if (out_vec_fp16_valid_o) begin
                $display("------------------------------------------------------------------------");
                $display("[%0t ns] Result for TOKEN %2d (Slot %2d):", $time,
                         drain_checked, drain_checked);

                for (int l = 0; l < NUM_LANES_P; l++) begin
                    logic [15:0] actual_hex;
                    logic [15:0] golden_hex;
                    logic signed [31:0] ref_q16;
                    real actual_val;
                    real golden_val;
                    real ref_real;
                    real err_val;
                    real abs_err;
                    real rel_err;

                    actual_hex = out_vec_fp16_lanes_o[l];
                    golden_hex = mem_golden_fp16[drain_checked * NUM_LANES_P + l];
                    actual_val = fp16_to_real(actual_hex);
                    golden_val = fp16_to_real(golden_hex);

                    // Error against the exact fake-quant reference Y_fp = W_q @ A_q.
                    ref_q16  = mem_ref_q16[drain_checked * NUM_LANES_P + l];
                    ref_real = $itor(ref_q16) / 65536.0;
                    err_val  = actual_val - ref_real;
                    abs_err  = (err_val < 0.0) ? -err_val : err_val;
                    rel_err  = 0.0;
                    if (ref_real != 0.0) begin
                        rel_err = abs_err / ((ref_real < 0.0) ? -ref_real : ref_real);
                        rtl_sum_rel_err += rel_err;
                        rtl_rel_count++;
                        if (rel_err > rtl_max_rel_err)
                            rtl_max_rel_err = rel_err;
                    end
                    rtl_sum_abs_err += abs_err;
                    rtl_sum_sq_err  += err_val * err_val;
                    if (abs_err > rtl_max_abs_err) begin
                        rtl_max_abs_err = abs_err;
                        rtl_worst_tok   = drain_checked;
                        rtl_worst_lane  = l;
                    end
                    rtl_err_count++;

                    if (rtl_err_fd != 0)
                        $fwrite(rtl_err_fd,
                                "%0d,%0d,0x%04X,%0.9f,0x%08X,%0.9f,%0.9f,%0.9f\n",
                                drain_checked, l, actual_hex, actual_val,
                                ref_q16, ref_real, abs_err, rel_err);

                    if (actual_hex == golden_hex) begin
                        match_count++;
                        $display("  Lane %2d (Out Ch %2d): Actual=0x%04X (~%7.3f) | Golden=0x%04X (~%7.3f) -> [MATCH]", 
                                 l, l, actual_hex, actual_val, golden_hex, golden_val);
                    end else begin
                        mismatch_count++;
                        $display("  Lane %2d (Out Ch %2d): Actual=0x%04X (~%7.3f) | Golden=0x%04X (~%7.3f) -> [MISMATCH!]", 
                                 l, l, actual_hex, actual_val, golden_hex, golden_val);
                    end
                end
                drain_checked++;
            end
        end

        if (rtl_err_fd != 0) $fclose(rtl_err_fd);

        wait_cycles(5);
        $display("\n========================================================================");
        $display("                   PREFILL GEMM TEST RESULT                        ");
        $display("========================================================================");
        $display(" Total checks (16 Tokens x 16 Channels) : %0d", match_count + mismatch_count);
        $display(" BIT-EXACT matches                      : %0d", match_count);
        $display(" Mismatches                             : %0d", mismatch_count);
        $display(" Weight re-use factor                   : 16x");
        if (mismatch_count == 0 && match_count == 256) begin
            $display("\n >>> [PASS] 100%% OF RESULTS MATCH THE CSV & HEX GOLDEN DATA! <<<");
            $display(" >>> WEIGHT-STATIONARY GEMM PREFILL VERIFIED SUCCESSFULLY! <<<");
        end else begin
            $display("\n >>> [WARNING] MISMATCHES DETECTED - CHECK TIMING OR DATA! <<<");
        end

        if (rtl_err_count > 0) begin
            $display("\nRTL vs fake-quant reference (Q16.16):");
            $display("  points      : %0d", rtl_err_count);
            $display("  max |err|   : %0.9f (token %0d, lane %0d)",
                     rtl_max_abs_err, rtl_worst_tok, rtl_worst_lane);
            $display("  mean |err|  : %0.9f", rtl_sum_abs_err / rtl_err_count);
            $display("  RMSE        : %0.9f", $sqrt(rtl_sum_sq_err / rtl_err_count));
            $display("  max rel err : %0.9f", rtl_max_rel_err);
            if (rtl_rel_count > 0)
                $display("  mean rel err: %0.9f", rtl_sum_rel_err / rtl_rel_count);
            else
                $display("  mean rel err: 0.0");
        end
        $display("========================================================================\n");

        $finish;
    end

    initial begin
        #(CLK_PERIOD * 2000);
        $display("\n[ERROR] Simulation timeout reached!");
        $finish;
    end

endmodule
