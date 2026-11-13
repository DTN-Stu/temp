#!/usr/bin/env python3
"""
generate_tiled_gemm_data.py

Tiled GEMM data generator for snax_fused_llm_core with ACC_DEPTH=16.

Problem:
    W: M x K = 32 x 512
    A: K x N = 512 x 64
    Y: M x N = 32 x 64

Hardware tile:
    M_TILE = 16  (one core pass, 16 lanes)
    N_TILE = 16  (ACC_DEPTH=16 accumulator slots)
    K_TILE = 256 (16 K-blocks; one 16x256 weight tile in local SRAM)
    K_BLOCK = 16 (one FP4 dot-product packet)

Tiling:
    M_TILES = 2, N_TILES = 4, K_TILES = 2  -> 16 (mt,nt,kt) iterations.

Packet order (same as the TB expects):
    weight_packets.hex     : for m in 0..M-1, for kb in 0..K_BLOCKS-1
    activation_packets.hex : for n in 0..N-1, for kb in 0..K_BLOCKS-1
    golden_fp16.hex        : for m in 0..M-1, for n in 0..N-1
    golden_ref_q16.hex     : for m in 0..M-1, for n in 0..N-1

The bit-exact helpers are imported from generate_prefill_gemm_data.py so
the RTL model stays in one place.
"""

import os
import csv
import numpy as np

from generate_prefill_gemm_data import (
    GAUSS_SIGMA,
    GAIN_OCTAVES,
    encode_72b_packet,
    fp16_hex_to_float,
    make_gaussian_matrix,
    quantize_matrix,
    to_q16_hex,
    error_metrics,
    fp4_mult_lut_py,
    csd_scaler_py,
    shifter_py,
    sat_add32,
    drain_norm_fp16_py,
)

# Problem dimensions
M = 32
N = 64
K = 512

# Hardware tile
M_TILE = 16
N_TILE = 16
K_TILE = 256
K_BLOCK = 16

K_BLOCKS = K // K_BLOCK
M_TILES = M // M_TILE
N_TILES = N // N_TILE
K_TILES = K // K_TILE
KT_BLOCKS = K_TILE // K_BLOCK


def main():
    np.random.seed(42)

    out_csv = "sim_gemm_tiled/csv"
    out_hex = "sim_gemm_tiled/hex"
    os.makedirs(out_csv, exist_ok=True)
    os.makedirs(out_hex, exist_ok=True)

    print(f"=== Tiled GEMM W({M}x{K}) x A({K}x{N}) -> Y({M}x{N}) ===")
    print(f"tile: M={M_TILE}, N={N_TILE}, K={K_TILE}; "
          f"tiles: M={M_TILES}, N={N_TILES}, K={K_TILES}")

    # ------------------------------------------------------------------
    # 1. Gaussian inputs, then block-scale FP4 quantization
    # ------------------------------------------------------------------
    W_gauss = make_gaussian_matrix(M, K, GAUSS_SIGMA, GAIN_OCTAVES)
    A_gauss = make_gaussian_matrix(K, N, GAUSS_SIGMA, GAIN_OCTAVES)

    W_codes, W_q, W_scale = quantize_matrix(W_gauss, K_BLOCKS, block_axis=1)
    A_codes, A_q, A_scale = quantize_matrix(A_gauss, K_BLOCKS, block_axis=0)

    Y_q = np.matmul(W_q, A_q)          # exact fake-quant product
    Y_input = np.matmul(W_gauss, A_gauss)

    # ------------------------------------------------------------------
    # 2. Bit-exact hardware model (same as the RTL)
    # ------------------------------------------------------------------
    Psum = np.zeros((M, N), dtype=np.int32)

    for kb in range(K_BLOCKS):
        k_start = kb * K_BLOCK
        k_end = k_start + K_BLOCK

        for n in range(N):
            act_slice = A_codes[k_start:k_end, n]
            scale_a = int(A_scale[kb, n])

            for m in range(M):
                wt_slice = W_codes[m, k_start:k_end]
                scale_w = int(W_scale[m, kb])

                dot = 0
                for el in range(K_BLOCK):
                    dot += fp4_mult_lut_py(act_slice[el], wt_slice[el])

                res_scale = scale_a + scale_w
                q_exp = res_scale // 8
                r_exp = res_scale % 8

                scaled = csd_scaler_py(dot, r_exp)
                aligned = shifter_py(scaled, q_exp, frac_bits=16)
                Psum[m, n] = sat_add32(Psum[m, n], aligned)

    Y_fp16_hex = np.zeros((M, N), dtype=np.uint16)
    Y_fp16_real = np.zeros((M, N), dtype=float)
    for m in range(M):
        for n in range(N):
            h = drain_norm_fp16_py(int(Psum[m, n]), frac_bits=16)
            Y_fp16_hex[m, n] = h
            Y_fp16_real[m, n] = fp16_hex_to_float(h)

    # ------------------------------------------------------------------
    # 3. Error summary
    # ------------------------------------------------------------------
    comparisons = [
        ("model_fp16_vs_fakequant", Y_fp16_real, Y_q),
        ("fakequant_vs_gaussian",   Y_q,        Y_input),
        ("model_fp16_vs_gaussian",  Y_fp16_real, Y_input),
    ]
    with open(f"{out_csv}/error_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["comparison", "max_abs", "mean_abs", "rmse",
                         "max_rel", "mean_rel", "worst_m", "worst_n"])
        for name, actual, ref in comparisons:
            e = error_metrics(actual, ref)
            writer.writerow([
                name,
                f"{e['max_abs']:.9f}",
                f"{e['mean_abs']:.9f}",
                f"{e['rmse']:.9f}",
                f"{e['max_rel']:.9f}",
                f"{e['mean_rel']:.9f}",
                e["worst_lane"],
                e["worst_token"],
            ])

    print("Error summary:")
    for name, actual, ref in comparisons:
        e = error_metrics(actual, ref)
        print(f"  {name}: max_abs={e['max_abs']:.6g}, "
              f"mean_abs={e['mean_abs']:.6g}, rmse={e['rmse']:.6g}, "
              f"worst=(m {e['worst_lane']}, n {e['worst_token']})")

    # ------------------------------------------------------------------
    # 4. Long golden CSV for eye inspection
    # ------------------------------------------------------------------
    with open(f"{out_csv}/golden_output.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["m", "n", "fp16_hex", "fp16_real",
                         "float_ref", "float_ref_input",
                         "quant_error", "psum_error", "fp16_error"])
        for m in range(M):
            for n in range(N):
                h = int(Y_fp16_hex[m, n])
                real = float(Y_fp16_real[m, n])
                ref = float(Y_q[m, n])
                ref_in = float(Y_input[m, n])
                psum = float(Psum[m, n]) / 65536.0
                writer.writerow([
                    m, n, f"0x{h:04X}", f"{real:.9f}",
                    f"{ref:.9f}", f"{ref_in:.9f}",
                    f"{ref - ref_in:.9f}",
                    f"{psum - ref:.9f}",
                    f"{real - ref:.9f}",
                ])

    # ------------------------------------------------------------------
    # 5. Tile plan CSV (what the TB/scheduler should execute)
    # ------------------------------------------------------------------
    with open(f"{out_csv}/tile_plan.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tile_idx", "mt", "nt", "kt",
                         "m_start", "n_start", "k_start",
                         "m_len", "n_len", "k_len",
                         "weight_packets", "activation_packets",
                         "output_points"])
        idx = 0
        for mt in range(M_TILES):
            for nt in range(N_TILES):
                for kt in range(K_TILES):
                    writer.writerow([
                        idx, mt, nt, kt,
                        mt * M_TILE, nt * N_TILE, kt * K_TILE,
                        M_TILE, N_TILE, K_TILE,
                        M_TILE * KT_BLOCKS,
                        N_TILE * KT_BLOCKS,
                        M_TILE * N_TILE,
                    ])
                    idx += 1

    # ------------------------------------------------------------------
    # 6. HEX files for the TB
    # ------------------------------------------------------------------
    # Weight packets: m-major, k-block-minor.
    with open(f"{out_hex}/weight_packets.hex", "w") as f:
        for m in range(M):
            f.write(f"// Out channel {m}\n")
            for kb in range(K_BLOCKS):
                nibbles = W_codes[m, kb * K_BLOCK:(kb + 1) * K_BLOCK]
                pkt = encode_72b_packet(nibbles, int(W_scale[m, kb]))
                f.write(f"{pkt:018X}\n")

    # Activation packets: n-major, k-block-minor.
    with open(f"{out_hex}/activation_packets.hex", "w") as f:
        for n in range(N):
            f.write(f"// Token {n}\n")
            for kb in range(K_BLOCKS):
                nibbles = A_codes[kb * K_BLOCK:(kb + 1) * K_BLOCK, n]
                pkt = encode_72b_packet(nibbles, int(A_scale[kb, n]))
                f.write(f"{pkt:018X}\n")

    # Golden FP16: m-major, n-minor.
    with open(f"{out_hex}/golden_fp16.hex", "w") as f:
        for m in range(M):
            f.write(f"// Out channel {m}\n")
            for n in range(N):
                f.write(f"{Y_fp16_hex[m, n]:04X}\n")

    # Fake-quant exact reference Q16.16: m-major, n-minor.
    with open(f"{out_hex}/golden_ref_q16.hex", "w") as f:
        for m in range(M):
            f.write(f"// Out channel {m} fake-quant Q16.16\n")
            for n in range(N):
                f.write(f"{to_q16_hex(Y_q[m, n]):08X}\n")

    print(f"Packets: weight={M * K_BLOCKS}, activation={N * K_BLOCKS}, "
          f"golden={M * N}")
    print(f"Output dir: {os.path.abspath('sim_gemm_tiled')}")
    print("[SUCCESS] Tiled GEMM data generated.")


if __name__ == "__main__":
    main()
