// =============================================================================
// Module: tb_snax_gemm_tiled.sv
// Description: Tiled GEMM testbench for snax_fused_llm_core, ACC_DEPTH=16.
//
//   W: 32 x 512, A: 512 x 64, Y: 32 x 64
//   Tile: M_TILE=16, N_TILE=16, K_TILE=256 (16 K-blocks)
//   Loop: mt (2) -> nt (4) -> kt (2); 16 (mt,nt,kt) iterations.
//
//   The 16x256 local weight tile is modeled by wt_tile[16][16]; the full W/A
//   are the TCDM/DRAM backing arrays.  K accumulation stays in the 16 latch
//   accumulator slots across both K tiles; drain only after all K=512.
//
//   CSV:
//     sim_gemm_tiled/csv/rtl_vs_fakequant.csv
// =============================================================================
`timescale 1ns/1ps
`include "pe_defines.svh"

module tb_snax_gemm_tiled;

    localparam int NUM_LANES_P = 16;
    localparam int ACC_DEPTH_P = 16;
    localparam int PKT_W       = 72;
    localparam int FRAC_BITS   = 16;
    localparam int CLK_PERIOD  = 10; // 100 MHz

    localparam int M = 32;
    localparam int N = 64;
    localparam int K = 512;

    localparam int M_TILE = 16;
    localparam int N_TILE = 16;
    localparam int K_TILE = 256;
    localparam int K_BLOCK = 16;

    localparam int K_BLOCKS = K / K_BLOCK;       // 32
    localparam int M_TILES  = M / M_TILE;        // 2
    localparam int N_TILES  = N / N_TILE;        // 4
    localparam int K_TILES  = K / K_TILE;        // 2
    localparam int KT_BLOCKS = K_TILE / K_BLOCK; // 16

    // DUT signals
    logic                     clk;
    logic                     rst_n;
    logic [1:0]               cfg_mode_i;
    logic                     cfg_row_start_i;
    logic [NUM_LANES_P-1:0]   cfg_key_mask_i;
    logic                     cfg_enable_slq_i;
    logic                     tile_drain_req_i;
    logic [3:0]               tile_drain_idx_i;
    logic                     tile_clear_on_drain_i;
    logic [NUM_LANES_P-1:0]   lane_valid_i;
    logic [3:0]               lane_acc_idx_i;
    logic [PKT_W-1:0]         ext_act_broadcast_i;
    logic                     ext_act_valid_i;
    logic [NUM_LANES_P-1:0][PKT_W-1:0] wt_in_i;
    logic                     wt_valid_i;
    logic [NUM_LANES_P*16-1:0] out_vec_fp16_packed_o;
    logic [NUM_LANES_P-1:0][15:0]    out_vec_fp16_lanes_o;
    logic                     out_vec_fp16_valid_o;
    logic [PKT_W-1:0]         out_vec_fp4_packet_o;
    logic                     out_vec_fp4_valid_o;
    logic                     p_fifo_full_o, p_fifo_empty_o, l_ready_o;

    // Full matrices in TCDM/DRAM model
    logic [PKT_W-1:0]  mem_w       [0:M*K_BLOCKS-1]; // 1024
    logic [PKT_W-1:0]  mem_a       [0:N*K_BLOCKS-1]; // 2048
    logic [15:0]       mem_golden  [0:M*N-1];        // 2048
    logic signed [31:0] mem_ref_q16[0:M*N-1];        // 2048

    // 16x256 local weight tile (16 lanes x 16 K-blocks)
    logic [PKT_W-1:0]  wt_tile [0:NUM_LANES_P-1][0:KT_BLOCKS-1];

    int match_count = 0;
    int mismatch_count = 0;
    int drain_issued = 0;
    int drain_checked = 0;

    // RTL vs fake-quant error metrics
    real rtl_max_abs_err = 0.0;
    real rtl_sum_abs_err = 0.0;
    real rtl_sum_sq_err  = 0.0;
    real rtl_max_rel_err = 0.0;
    real rtl_sum_rel_err = 0.0;
    int  rtl_err_count   = 0;
    int  rtl_rel_count   = 0;
    int  rtl_worst_m     = 0;
    int  rtl_worst_n     = 0;
    int  rtl_err_fd      = 0;

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
                val = real'(mant) / 16777216.0;
            end else begin
                val = (1.0 + real'(mant) / 1024.0)
                    * $itor(32'd1 << exp) / 32768.0;
            end
            return sign ? -val : val;
        end
    endfunction

    initial begin
        $display("\n========================================================================");
        $display("   TESTBENCH: TILED GEMM W(32x512) x A(512x64) -> Y(32x64)");
        $display("   ACC_DEPTH=16, tile M=16 N=16 K=256, loop M->N->K");
        $display("========================================================================\n");

        $readmemh("sim_gemm_tiled/hex/weight_packets.hex", mem_w);
        $readmemh("sim_gemm_tiled/hex/activation_packets.hex", mem_a);
        $readmemh("sim_gemm_tiled/hex/golden_fp16.hex", mem_golden);
        $readmemh("sim_gemm_tiled/hex/golden_ref_q16.hex", mem_ref_q16);

        if ($isunknown(mem_ref_q16[0]))
            $display("[WARNING] golden_ref_q16.hex missing; error metrics invalid.");

        $display("[%0t ns] Loaded tiled GEMM HEX files.", $time);

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
        for (int l = 0; l < NUM_LANES_P; l++) wt_in_i[l] = '0;

        wait_cycles(5);
        rst_n = 1'b1;
        wait_cycles(5);
        $display("[%0t ns] Reset complete. Starting tiled GEMM.\n", $time);

        rtl_err_fd = $fopen("sim_gemm_tiled/csv/rtl_vs_fakequant.csv", "w");
        if (rtl_err_fd != 0)
            $fwrite(rtl_err_fd,
                    "m,n,actual_fp16,actual_real,ref_q16_hex,ref_real,abs_err,rel_err\n");

        for (int mt = 0; mt < M_TILES; mt++) begin
            for (int nt = 0; nt < N_TILES; nt++) begin
                $display("------------------------------------------------------------------------");
                $display("[%0t ns] OUTPUT TILE mt=%0d nt=%0d (M %0d..%0d, N %0d..%0d)",
                         $time, mt, nt,
                         mt*M_TILE, mt*M_TILE+M_TILE-1,
                         nt*N_TILE, nt*N_TILE+N_TILE-1);
                $display("------------------------------------------------------------------------");

                // K tiles are accumulated into the same 16 slots; no drain here.
                for (int kt = 0; kt < K_TILES; kt++) begin
                    // Load the 16x256 weight tile (16 lanes x 16 K-blocks).
                    for (int lane = 0; lane < NUM_LANES_P; lane++) begin
                        for (int kb = 0; kb < KT_BLOCKS; kb++) begin
                            wt_tile[lane][kb] =
                                mem_w[(mt*M_TILE + lane)*K_BLOCKS +
                                      (kt*KT_BLOCKS + kb)];
                        end
                    end

                    // 16 K-blocks x 16 tokens; one token per cycle.
                    for (int kb = 0; kb < KT_BLOCKS; kb++) begin
                        @(posedge clk);

                        // New weight is captured from the next edge onward;
                        // the last token of the previous K-block used the old one.
                        for (int lane = 0; lane < NUM_LANES_P; lane++) begin
                            wt_in_i[lane] <= wt_tile[lane][kb];
                        end
                        wt_valid_i <= 1'b1;

                        // Token 0 of this K-block.
                        ext_act_broadcast_i =
                            mem_a[(nt*N_TILE + 0)*K_BLOCKS +
                                  (kt*KT_BLOCKS + kb)];
                        ext_act_valid_i = 1'b1;
                        lane_valid_i    = 16'hFFFF;
                        lane_acc_idx_i  = 4'd0;

                        // Tokens 1..15, one per cycle.
                        for (int tok = 1; tok < N_TILE; tok++) begin
                            @(posedge clk);
                            ext_act_broadcast_i =
                                mem_a[(nt*N_TILE + tok)*K_BLOCKS +
                                      (kt*KT_BLOCKS + kb)];
                            ext_act_valid_i = 1'b1;
                            lane_valid_i    = 16'hFFFF;
                            lane_acc_idx_i  = tok[3:0];
                        end
                    end
                end

                // Capture the final token of the final K-block.
                @(posedge clk);
                ext_act_valid_i = 1'b0;
                lane_valid_i    = '0;
                wt_valid_i      = 1'b0;
                wait_cycles(2);

                // Pipelined drain of the 16 slots (N_TILE=ACC_DEPTH=16).
                drain_issued  = 0;
                drain_checked = 0;
                while (drain_checked < N_TILE) begin
                    @(posedge clk);

                    if (drain_issued < N_TILE) begin
                        tile_drain_req_i      = 1'b1;
                        tile_drain_idx_i      = drain_issued[3:0];
                        tile_clear_on_drain_i = 1'b1;
                        drain_issued++;
                    end else begin
                        tile_drain_req_i      = 1'b0;
                        tile_clear_on_drain_i = 1'b0;
                    end

                    if (out_vec_fp16_valid_o) begin
                        for (int l = 0; l < NUM_LANES_P; l++) begin
                            logic [15:0] actual_hex;
                            logic [15:0] golden_hex;
                            logic signed [31:0] ref_q16;
                            int m_global;
                            int n_global;
                            real actual_val;
                            real golden_val;
                            real ref_real;
                            real err_val;
                            real abs_err;
                            real rel_err;

                            actual_hex = out_vec_fp16_lanes_o[l];
                            m_global   = mt*M_TILE + l;
                            n_global   = nt*N_TILE + drain_checked;
                            golden_hex = mem_golden[m_global*N + n_global];
                            actual_val = fp16_to_real(actual_hex);
                            golden_val = fp16_to_real(golden_hex);

                            ref_q16  = mem_ref_q16[m_global*N + n_global];
                            ref_real = $itor(ref_q16) / 65536.0;
                            err_val  = actual_val - ref_real;
                            abs_err  = (err_val < 0.0) ? -err_val : err_val;
                            rel_err  = 0.0;
                            if (ref_real != 0.0) begin
                                rel_err = abs_err /
                                          ((ref_real < 0.0) ? -ref_real : ref_real);
                                rtl_sum_rel_err += rel_err;
                                rtl_rel_count++;
                                if (rel_err > rtl_max_rel_err)
                                    rtl_max_rel_err = rel_err;
                            end
                            rtl_sum_abs_err += abs_err;
                            rtl_sum_sq_err  += err_val * err_val;
                            if (abs_err > rtl_max_abs_err) begin
                                rtl_max_abs_err = abs_err;
                                rtl_worst_m     = m_global;
                                rtl_worst_n     = n_global;
                            end
                            rtl_err_count++;

                            if (rtl_err_fd != 0)
                                $fwrite(rtl_err_fd,
                                        "%0d,%0d,0x%04X,%0.9f,0x%08X,%0.9f,%0.9f,%0.9f\n",
                                        m_global, n_global, actual_hex, actual_val,
                                        ref_q16, ref_real, abs_err, rel_err);

                            if (actual_hex == golden_hex) begin
                                match_count++;
                            end else begin
                                mismatch_count++;
                                $display("  MISMATCH m=%0d n=%0d lane=%0d: Actual=0x%04X (~%7.3f) Golden=0x%04X (~%7.3f)",
                                         m_global, n_global, l,
                                         actual_hex, actual_val,
                                         golden_hex, golden_val);
                            end
                        end
                        drain_checked++;
                    end
                end

                $display("[%0t ns] Tile mt=%0d nt=%0d done: matches=%0d mismatches=%0d",
                         $time, mt, nt, match_count, mismatch_count);
            end
        end

        if (rtl_err_fd != 0) $fclose(rtl_err_fd);

        wait_cycles(5);
        $display("\n========================================================================");
        $display("                   TILED GEMM TEST RESULT");
        $display("========================================================================");
        $display(" Total checks                    : %0d", match_count + mismatch_count);
        $display(" BIT-EXACT matches               : %0d", match_count);
        $display(" Mismatches                      : %0d", mismatch_count);
        if (mismatch_count == 0 && match_count == M*N) begin
            $display("\n >>> [PASS] TILED GEMM BIT-EXACT! <<<");
        end else begin
            $display("\n >>> [WARNING] TILED GEMM MISMATCHES! <<<");
        end

        if (rtl_err_count > 0) begin
            $display("\nRTL vs fake-quant reference (Q16.16):");
            $display("  points      : %0d", rtl_err_count);
            $display("  max |err|   : %0.9f (m %0d, n %0d)",
                     rtl_max_abs_err, rtl_worst_m, rtl_worst_n);
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
        #(CLK_PERIOD * 200000); // 2 ms timeout
        $display("\n[ERROR] Simulation timeout reached!");
        $finish;
    end

endmodule
