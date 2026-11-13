#!/usr/bin/env python3
"""
generate_prefill_gemm_data.py
Generator script for SNAX FP4 Fused LLM Core GEMM Prefill Verification
- Matrix W: 16 (Out channels) x K (Reduction dim)
- Matrix A: K (Reduction dim) x 16 (Tokens)
- Matrix Y: 16 (Out channels) x 16 (Tokens)
- W/A are sampled from N(0, GAUSS_SIGMA), then quantized per 16-element block
  to E2M1 with an 8-bit log8 block scale (1/8-octave units), matching the
  block-scale method used by sim/gen_ws_data.py.
- Generates CSV files for human readability and HEX files for SystemVerilog $readmemh.
- CSV views:
    *_matrix_fp.csv         original Gaussian input
    *_matrix_quantized.csv  dequantized block-scaled FP4 (hardware input)
    *_matrix_fp_digital.csv dequantized value + raw 4-bit code
    *_packets_digital.csv   block scale + 72-bit SRAM word + per-element codes
- Also writes packet-level CSV views of the 72-bit SRAM words.
"""

import os
import csv
import math
import struct
import numpy as np

# 8 FP4 Magnitude levels (Sign-Magnitude code)
# Code 0..7 -> Magnitude value
FP4_MAG_MAP = {
    0: 0.0,
    1: 0.5,
    2: 1.0,
    3: 1.5,
    4: 2.0,
    5: 3.0,
    6: 4.0,
    7: 6.0
}
FP4_VALS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

# Midpoints between adjacent E2M1 magnitudes (nearest-level decision).
FP4_MID = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]

# Input distribution for W/A. Pure N(0, sigma); set GAIN_OCTAVES > 0 to
# multiply each 16-element block by 2**U(-g, +g) for extra scale diversity.
GAUSS_SIGMA  = 1.0
GAIN_OCTAVES = 0.0

def quantize_to_fp4(val):
    """Finds closest FP4 sign-magnitude code (4-bit: [3]=sign, [2:0]=mag)."""
    sign = 1 if val < 0 else 0
    abs_v = abs(val)
    # Find closest magnitude
    best_code = 0
    min_diff = abs(abs_v - FP4_VALS[0])
    for code, m in FP4_MAG_MAP.items():
        diff = abs(abs_v - m)
        if diff < min_diff:
            min_diff = diff
            best_code = code
    actual_val = -FP4_MAG_MAP[best_code] if sign else FP4_MAG_MAP[best_code]
    encoded_nibble = (sign << 3) | (best_code & 0x7)
    return encoded_nibble, actual_val

def encode_72b_packet(nibbles_16, scale_exp=0):
    """
    Encodes 16 4-bit nibbles + 8-bit scale exponent into a 72-bit integer.

    Use Python ints explicitly: numpy int64 overflows when nibble 15 is
    shifted by 60 and has bit 3 set, which used to emit negative hex words
    (e.g. -04FE...) instead of the intended 72-bit unsigned packet.
    """
    data_64 = 0
    for i in range(16):
        data_64 |= (int(nibbles_16[i]) & 0xF) << (i * 4)
    data_64 &= (1 << 64) - 1
    packet_72 = ((int(scale_exp) & 0xFF) << 64) | data_64
    return packet_72

def float_to_fp16_hex(val):
    """Converts a float to IEEE 754 half-precision (FP16) 16-bit hex integer."""
    packed = struct.pack('>e', np.float16(val))
    return struct.unpack('>H', packed)[0]

def fp16_hex_to_float(h):
    packed = struct.pack('>H', h)
    return struct.unpack('>e', packed)[0]

def make_gaussian_matrix(rows, cols, sigma=1.0, gain_octaves=0.0):
    """rows x cols i.i.d. N(0, sigma); optional per-16-element block gain."""
    mat = np.zeros((rows, cols), dtype=float)
    for r in range(rows):
        for kb in range(cols // 16):
            gain = 1.0
            if gain_octaves > 0.0:
                gain = 2.0 ** np.random.uniform(-gain_octaves, gain_octaves)
            mat[r, kb * 16:(kb + 1) * 16] = \
                np.random.normal(0.0, sigma, 16) * gain
    return mat

def quantize_block(xs):
    """
    Block-scale FP4 quantizer matching sim/gen_ws_data.py.

    Scale is an 8-bit signed value in 1/8-octave units:
        s = round(8*log2(amax/6)),  S = 2^(s/8)
    value_i = sign_i * E2M1[mag_i] * S
    Returns (codes[16], dequantized_values[16], s).
    """
    amax = float(np.max(np.abs(xs)))
    if amax == 0.0:
        return [0] * 16, [0.0] * 16, 0

    s = int(round(8.0 * math.log2(amax / 6.0)))
    s = max(-128, min(127, s))
    scale = 2.0 ** (s / 8.0)

    codes = []
    vals = []
    for x in xs:
        y = abs(float(x)) / scale
        idx = 0
        for i, mid in enumerate(FP4_MID):
            if y >= mid:
                idx = i + 1
        sign = 1 if x < 0 else 0
        codes.append((sign << 3) | idx)
        vals.append((-1.0 if sign else 1.0) * FP4_VALS[idx] * scale)
    return codes, vals, s

def quantize_matrix(mat, num_chunks, block_axis):
    """
    Quantize every 16-element block of mat.
      block_axis=1: blocks along columns (weights: [lane, K])
      block_axis=0: blocks along rows    (activations: [K, token])
    Returns (codes, dequantized, scales):
      scales is [rows, chunks] for block_axis=1,
                [chunks, cols] for block_axis=0.
    """
    codes = np.zeros_like(mat, dtype=int)
    vals = np.zeros_like(mat, dtype=float)

    if block_axis == 1:
        scales = np.zeros((mat.shape[0], num_chunks), dtype=int)
        for r in range(mat.shape[0]):
            for ch in range(num_chunks):
                seg = mat[r, ch * 16:(ch + 1) * 16]
                c, v, s = quantize_block(seg)
                codes[r, ch * 16:(ch + 1) * 16] = c
                vals[r, ch * 16:(ch + 1) * 16] = v
                scales[r, ch] = s
    else:
        scales = np.zeros((num_chunks, mat.shape[1]), dtype=int)
        for c in range(mat.shape[1]):
            for ch in range(num_chunks):
                seg = mat[ch * 16:(ch + 1) * 16, c]
                cc, vv, s = quantize_block(seg)
                codes[ch * 16:(ch + 1) * 16, c] = cc
                vals[ch * 16:(ch + 1) * 16, c] = vv
                scales[ch, c] = s
    return codes, vals, scales

def write_matrix_csv(path, row_header, row_prefix, col_labels, matrix,
                     fmt=".6f"):
    """Plain real-valued matrix CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([row_header] + col_labels)
        for r in range(matrix.shape[0]):
            writer.writerow([f"{row_prefix}_{r}"] +
                            [f"{matrix[r, c]:{fmt}}"
                             for c in range(matrix.shape[1])])

def fp4_code_fields(code):
    """Decode a 4-bit FP4 sign-magnitude code into (sign, mag_code, mag, real)."""
    sign = (code >> 3) & 0x1
    mag_code = code & 0x7
    mag_value = FP4_MAG_MAP[mag_code]
    real_value = -mag_value if sign else mag_value
    return sign, mag_code, mag_value, real_value

def fp4_cell(real_value, code):
    """Cell text with both the real value and its 4-bit digital code."""
    return f"{real_value:.4f} [0x{code & 0xF:X}]"

def write_matrix_with_codes(path, row_header, row_prefix, col_labels,
                            real_matrix, code_matrix):
    """Matrix CSV where every cell contains 'real [0xcode]'."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([row_header] + col_labels)
        for r in range(real_matrix.shape[0]):
            row = [f"{row_prefix}_{r}"]
            for c in range(real_matrix.shape[1]):
                row.append(fp4_cell(float(real_matrix[r, c]),
                                    int(code_matrix[r, c])))
            writer.writerow(row)

def write_packet_digital_csv(path, kind, code_matrix, num_chunks,
                             rows_per_chunk, row_axis, row_prefix, scales):
    """
    One row per 72-bit packet (the exact SRAM word), with the decoded
    4-bit codes and dequantized real values of all 16 elements.

    row_axis='lane'  -> code_matrix is [lane, K],      scales [lane, chunk]
    row_axis='token' -> code_matrix is [K, token],     scales [chunk, token]
    """
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = [
            "kind", "chunk", "pkt_idx", "row_idx", "row_label",
            "k_start", "k_end", "scale_hex", "scale_value",
            "data_hex", "packet_hex",
        ]
        for i in range(16):
            header += [f"E{i}_code_hex", f"E{i}_code_bin", f"E{i}_real"]
        writer.writerow(header)

        pkt_idx = 0
        for ch in range(num_chunks):
            k_start = ch * 16
            k_end = k_start + 16
            for r in range(rows_per_chunk):
                if row_axis == "lane":
                    nibbles = [int(x) & 0xF
                               for x in code_matrix[r, k_start:k_end]]
                    sc = int(scales[r, ch])
                else:
                    nibbles = [int(x) & 0xF
                               for x in code_matrix[k_start:k_end, r]]
                    sc = int(scales[ch, r])

                data_64 = 0
                for i, nib in enumerate(nibbles):
                    data_64 |= (nib & 0xF) << (i * 4)
                packet_72 = ((sc & 0xFF) << 64) | data_64
                scale_value = 2.0 ** (sc / 8.0)

                row = [
                    kind, ch, pkt_idx, r, f"{row_prefix}_{r}",
                    k_start, k_end,
                    f"0x{sc & 0xFF:02X}", f"{scale_value:.9f}",
                    f"0x{data_64:016X}", f"0x{packet_72:018X}",
                ]
                for nib in nibbles:
                    _, _, _, nominal = fp4_code_fields(nib)
                    row += [f"0x{nib:X}", f"0b{nib:04b}",
                            f"{nominal * scale_value:.9f}"]
                writer.writerow(row)
                pkt_idx += 1

def write_psum_hex_matrix(path, psum_int32):
    """Matrix view of the 32-bit accumulator words in hex."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Out_Channel"] +
                        [f"Token_{c}" for c in range(psum_int32.shape[1])])
        for r in range(psum_int32.shape[0]):
            writer.writerow([f"Lane_{r}"] + [
                f"0x{int(psum_int32[r, c]) & 0xFFFFFFFF:08X}"
                for c in range(psum_int32.shape[1])
            ])

def write_fp16_digital_matrix(path, fp16_hex, fp16_real):
    """Matrix view of FP16 outputs as 'real [0xhex]'."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Out_Channel"] +
                        [f"Token_{c}" for c in range(fp16_hex.shape[1])])
        for r in range(fp16_hex.shape[0]):
            row = [f"Lane_{r}"]
            for c in range(fp16_hex.shape[1]):
                h = int(fp16_hex[r, c])
                row.append(f"{fp16_real[r, c]:.3f} [0x{h:04X}]")
            writer.writerow(row)

def write_golden_output_combined(path, y_float, y_input, psum_int32,
                                 fp16_hex, fp16_real):
    """Long CSV: one row per (lane, token) with all numeric/digital views.

    float_ref       : exact FP4-quantized W @ A (what the hardware sees)
    float_ref_input : exact Gaussian W @ A (pre-quantization)
    quant_error     : float_ref - float_ref_input
    psum_error      : Q16.16 accumulator - float_ref (CSD/shift/saturation)
    fp16_error      : FP16 output - float_ref (FP16 rounding)
    """
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "lane", "token", "hex_index",
            "psum_int32", "psum_hex", "psum_q16_16",
            "fp16_hex", "fp16_sign", "fp16_exp", "fp16_mant_hex",
            "fp16_real", "float_ref", "float_ref_input",
            "psum_error", "quant_error", "fp16_error",
        ])
        for m in range(fp16_hex.shape[1]):
            for lane in range(fp16_hex.shape[0]):
                idx = m * fp16_hex.shape[0] + lane
                psum = int(psum_int32[lane, m])
                h = int(fp16_hex[lane, m])
                psum_q16_16 = psum / 65536.0
                fp16_val = float(fp16_real[lane, m])
                ref = float(y_float[lane, m])
                ref_in = float(y_input[lane, m])
                writer.writerow([
                    lane, m, idx,
                    psum, f"0x{psum & 0xFFFFFFFF:08X}",
                    f"{psum_q16_16:.9f}",
                    f"0x{h:04X}", (h >> 15) & 0x1, (h >> 10) & 0x1F,
                    f"0x{h & 0x3FF:03X}",
                    f"{fp16_val:.9f}", f"{ref:.9f}", f"{ref_in:.9f}",
                    f"{psum_q16_16 - ref:.9f}",
                    f"{ref - ref_in:.9f}",
                    f"{fp16_val - ref:.9f}",
                ])

def to_q16_hex(v):
    """Round a real value to signed Q16.16 and return its 8-digit hex word."""
    q = int(round(float(v) * 65536.0))
    q = max(-2147483648, min(2147483647, q))
    return q & 0xFFFFFFFF

def error_metrics(actual, reference):
    """Max/mean abs, RMSE and relative error for two [lane, token] arrays."""
    a = np.asarray(actual, dtype=float)
    r = np.asarray(reference, dtype=float)
    diff = a - r
    abs_diff = np.abs(diff)

    max_abs = float(np.max(abs_diff))
    mean_abs = float(np.mean(abs_diff))
    rmse = float(np.sqrt(np.mean(diff * diff)))

    mask = np.abs(r) > 1e-12
    if np.any(mask):
        rel = abs_diff[mask] / np.abs(r[mask])
        max_rel = float(np.max(rel))
        mean_rel = float(np.mean(rel))
    else:
        max_rel = 0.0
        mean_rel = 0.0

    worst = np.unravel_index(int(np.argmax(abs_diff)), abs_diff.shape)
    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "rmse": rmse,
        "max_rel": max_rel,
        "mean_rel": mean_rel,
        "worst_lane": int(worst[0]),
        "worst_token": int(worst[1]),
    }

def fp4_mult_lut_py(a_code, b_code):
    """Bit-exact Python mirror of fp4_mult_lut.sv."""
    sign = (a_code >> 3) ^ (b_code >> 3)
    mag_a = a_code & 0x7
    mag_b = b_code & 0x7
    
    # Q1 fixed values for 0..7
    q1_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 6, 6: 8, 7: 12}
    u_prod = q1_map[mag_a] * q1_map[mag_b] # Q2 format
    return -u_prod if sign else u_prod

def csd_scaler_py(prod_in, r):
    """Bit-exact mirror of logscale_frac_scaler."""
    m_r_table = [32, 35, 38, 41, 46, 49, 54, 59]
    return prod_in * m_r_table[r]

def shifter_py(scaled_prod, q, frac_bits=16):
    """Bit-exact mirror of bidirectional_shifter.sv."""
    base_shift = frac_bits - 7
    net_shift = q + base_shift
    if scaled_prod == 0:
        return 0
    if net_shift < 0:
        rshift = min(19, -net_shift)
        val = scaled_prod >> rshift
        return val
    elif net_shift > 15:
        if net_shift >= 26:
            return -2147483648 if scaled_prod < 0 else 2147483647
        wide = (scaled_prod << (net_shift - 16))
        val = wide << 16
        if val > 2147483647: return 2147483647
        if val < -2147483648: return -2147483648
        return val
    else:
        val = scaled_prod << net_shift
        if val > 2147483647: return 2147483647
        if val < -2147483648: return -2147483648
        return val

def sat_add32(a, b):
    res = a + b
    if res > 2147483647: return 2147483647
    if res < -2147483648: return -2147483648
    return res

def lzc32(abs_val):
    if abs_val == 0: return 32
    bin_str = f"{abs_val:032b}"
    for i, c in enumerate(bin_str):
        if c == '1': return i
    return 32

def drain_norm_fp16_py(int32_val, frac_bits=16):
    """Bit-exact mirror of logscale_lazy_drain_norm."""
    if int32_val == 0:
        return 0x0000
    fp_sign = 1 if int32_val < 0 else 0
    abs_y = abs(int32_val) & 0xFFFFFFFF
    lzc = lzc32(abs_y)
    if lzc >= 32:
        return 0x0000
    msb_pos = 31 - lzc
    raw_exp = msb_pos - frac_bits + 15
    norm_shifted = (abs_y << lzc) & 0xFFFFFFFF
    mant_bits = (norm_shifted >> 21) & 0x3FF
    guard = (norm_shifted >> 20) & 1
    round_b = (norm_shifted >> 19) & 1
    sticky = 1 if (norm_shifted & 0x7FFFF) != 0 else 0
    
    round_up = guard & (round_b | sticky | (mant_bits & 1))
    rounded_mant = mant_bits + round_up
    final_exp = raw_exp + 1 if (rounded_mant & 0x400) != 0 else raw_exp
    
    if final_exp <= 0:
        return 0x0000
    elif final_exp >= 31:
        return (fp_sign << 15) | (0x1F << 10) | 0x3FF # Saturation
    else:
        return (fp_sign << 15) | ((final_exp & 0x1F) << 10) | (rounded_mant & 0x3FF)

def main():
    np.random.seed(42)
    K = 256  # K_DIM / 16 chunks
    NUM_LANES = 16
    NUM_TOKENS = 16
    NUM_CHUNKS = K // 16

    os.makedirs("sim_gemm_prefill/csv", exist_ok=True)
    os.makedirs("sim_gemm_prefill/hex", exist_ok=True)

    print(f"=== Generating Data for Prefill GEMM W(16x{K}) x A({K}x16) -> Y(16x16) ===")

    # 1. Real Gaussian inputs, then block-scale FP4 quantization.
    # Weight: 16 out-channels (rows) x K (cols)
    # Activation: K (rows) x 16 tokens (cols)
    W_gauss = make_gaussian_matrix(NUM_LANES, K, GAUSS_SIGMA, GAIN_OCTAVES)
    A_gauss = make_gaussian_matrix(K, NUM_TOKENS, GAUSS_SIGMA, GAIN_OCTAVES)

    # 16-element blocks along K; one 8-bit log8 scale per block.
    W_codes, W_fp, W_scale = quantize_matrix(W_gauss, NUM_CHUNKS, block_axis=1)
    A_codes, A_fp, A_scale = quantize_matrix(A_gauss, NUM_CHUNKS, block_axis=0)

    # Exact real products: quantized (hardware input) and original Gaussian.
    Y_fp = np.matmul(W_fp, A_fp)
    Y_input = np.matmul(W_gauss, A_gauss)

    # 2. Simulate Hardware Execution (Weight-Stationary)
    # Psums table: 16 lanes x 16 tokens
    Psum_int32 = np.zeros((NUM_LANES, NUM_TOKENS), dtype=np.int32)

    # Loop over K chunks
    for ch in range(NUM_CHUNKS):
        k_start = ch * 16
        k_end = k_start + 16

        # In Weight-Stationary: 16 weights are loaded once per chunk
        # For each token m in 0..15:
        for m in range(NUM_TOKENS):
            act_slice = A_codes[k_start:k_end, m]
            scale_a = int(A_scale[ch, m])
            # Each lane computes dot product with its weight slice
            for lane in range(NUM_LANES):
                wt_slice = W_codes[lane, k_start:k_end]
                scale_w = int(W_scale[lane, ch])
                # Dot product 16
                dot_val = 0
                for el in range(16):
                    p = fp4_mult_lut_py(act_slice[el], wt_slice[el])
                    dot_val += p

                # Combined block scale: 8-bit log8 units, 1/8 octave each.
                res_scale = scale_a + scale_w
                q_exp = res_scale // 8
                r_exp = res_scale % 8

                # CSD scaler
                scaled_p = csd_scaler_py(dot_val, r_exp)
                # Shifter to Q16
                aligned_32 = shifter_py(scaled_p, q_exp, frac_bits=16)
                # Accumulate into slot m
                Psum_int32[lane, m] = sat_add32(Psum_int32[lane, m], aligned_32)

    # Convert Psum to FP16
    Y_fp16_hex = np.zeros((NUM_LANES, NUM_TOKENS), dtype=np.uint16)
    Y_fp16_real = np.zeros((NUM_LANES, NUM_TOKENS), dtype=float)
    for lane in range(NUM_LANES):
        for m in range(NUM_TOKENS):
            hex_val = drain_norm_fp16_py(int(Psum_int32[lane, m]), frac_bits=16)
            Y_fp16_hex[lane, m] = hex_val
            Y_fp16_real[lane, m] = fp16_hex_to_float(hex_val)

    # 2b. Error summary: hardware FP16 model vs fake-quant, plus quantization
    # error against the original Gaussian product.
    comparisons = [
        ("model_fp16_vs_fakequant", Y_fp16_real, Y_fp),
        ("fakequant_vs_gaussian",   Y_fp,        Y_input),
        ("model_fp16_vs_gaussian",  Y_fp16_real, Y_input),
    ]
    with open("sim_gemm_prefill/csv/error_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["comparison", "max_abs", "mean_abs", "rmse",
                         "max_rel", "mean_rel", "worst_lane", "worst_token"])
        for name, actual, ref in comparisons:
            m = error_metrics(actual, ref)
            writer.writerow([
                name,
                f"{m['max_abs']:.9f}",
                f"{m['mean_abs']:.9f}",
                f"{m['rmse']:.9f}",
                f"{m['max_rel']:.9f}",
                f"{m['mean_rel']:.9f}",
                m["worst_lane"],
                m["worst_token"],
            ])

    print("Error summary:")
    for name, actual, ref in comparisons:
        m = error_metrics(actual, ref)
        print(f"  {name}: max_abs={m['max_abs']:.6g}, "
              f"mean_abs={m['mean_abs']:.6g}, rmse={m['rmse']:.6g}, "
              f"max_rel={m['max_rel']:.6g}, "
              f"worst=(lane {m['worst_lane']}, token {m['worst_token']})")

    # 3. Export CSV files
    # Original Gaussian inputs (what the model would see without FP4).
    write_matrix_csv(
        "sim_gemm_prefill/csv/weight_matrix_fp.csv",
        "Out_Channel", "Lane", [f"K_{c}" for c in range(K)], W_gauss)
    write_matrix_csv(
        "sim_gemm_prefill/csv/activation_matrix_fp.csv",
        "Reduction_Dim_K", "K", [f"Token_{c}" for c in range(NUM_TOKENS)],
        A_gauss)

    # Dequantized block-scaled FP4 values (what the hardware actually computes).
    write_matrix_csv(
        "sim_gemm_prefill/csv/weight_matrix_quantized.csv",
        "Out_Channel", "Lane", [f"K_{c}" for c in range(K)], W_fp)
    write_matrix_csv(
        "sim_gemm_prefill/csv/activation_matrix_quantized.csv",
        "Reduction_Dim_K", "K", [f"Token_{c}" for c in range(NUM_TOKENS)],
        A_fp)

    # Golden Int32 Psums CSV
    with open("sim_gemm_prefill/csv/golden_psum_int32.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Out_Channel"] + [f"Token_{c}" for c in range(NUM_TOKENS)])
        for r in range(NUM_LANES):
            writer.writerow([f"Lane_{r}"] + [f"{int(Psum_int32[r, c])}" for c in range(NUM_TOKENS)])

    # Golden FP16 Hex CSV
    with open("sim_gemm_prefill/csv/golden_output_fp16_hex.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Out_Channel"] + [f"Token_{c}" for c in range(NUM_TOKENS)])
        for r in range(NUM_LANES):
            writer.writerow([f"Lane_{r}"] + [f"0x{Y_fp16_hex[r, c]:04X}" for c in range(NUM_TOKENS)])

    # Golden Output Float (Derived from FP16) CSV
    with open("sim_gemm_prefill/csv/golden_output_float.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Out_Channel"] + [f"Token_{c}" for c in range(NUM_TOKENS)])
        for r in range(NUM_LANES):
            writer.writerow([f"Lane_{r}"] + [f"{Y_fp16_real[r, c]:.3f}" for c in range(NUM_TOKENS)])

    # 3b. Digital-code views (real value + raw 4-bit code in the same cell)
    write_matrix_with_codes(
        "sim_gemm_prefill/csv/weight_matrix_fp_digital.csv",
        "Out_Channel", "Lane", [f"K_{c}" for c in range(K)],
        W_fp, W_codes)

    write_matrix_with_codes(
        "sim_gemm_prefill/csv/activation_matrix_fp_digital.csv",
        "Reduction_Dim_K", "K", [f"Token_{c}" for c in range(NUM_TOKENS)],
        A_fp, A_codes)

    # 3c. Packet-level views: exactly what the 72-bit SRAM stores + decoded codes
    write_packet_digital_csv(
        "sim_gemm_prefill/csv/weight_packets_digital.csv",
        "weight", W_codes, NUM_CHUNKS, NUM_LANES, "lane", "Lane", W_scale)

    write_packet_digital_csv(
        "sim_gemm_prefill/csv/activation_packets_digital.csv",
        "activation", A_codes, NUM_CHUNKS, NUM_TOKENS, "token", "Token",
        A_scale)

    # 3d. Golden result views: int32 hex matrix + FP16 real/hex matrix + long table
    write_psum_hex_matrix(
        "sim_gemm_prefill/csv/golden_psum_int32_hex.csv",
        Psum_int32)

    write_fp16_digital_matrix(
        "sim_gemm_prefill/csv/golden_output_fp16_digital.csv",
        Y_fp16_hex, Y_fp16_real)

    write_golden_output_combined(
        "sim_gemm_prefill/csv/golden_output_combined.csv",
        Y_fp, Y_input, Psum_int32, Y_fp16_hex, Y_fp16_real)

    # 4. Export HEX files for SystemVerilog $readmemh
    # (a) Weight Packets: For each chunk ch in 0..NUM_CHUNKS-1, for each lane in 0..15: 1 72-bit hex word
    # Total = NUM_CHUNKS * 16 = 32 words
    with open("sim_gemm_prefill/hex/weight_packets.hex", "w") as f:
        for ch in range(NUM_CHUNKS):
            k_start = ch * 16
            k_end = k_start + 16
            f.write(f"// --- Chunk {ch} (K={k_start}..{k_end-1}) ---\n")
            for lane in range(NUM_LANES):
                nibbles = W_codes[lane, k_start:k_end]
                pkt = encode_72b_packet(
                    nibbles, scale_exp=int(W_scale[lane, ch]))
                f.write(f"{pkt:018X}\n")

    # (b) Activation Packets: For each chunk ch, for each token m in 0..15: 1 72-bit hex word
    # Total = NUM_CHUNKS * 16 = 32 words
    with open("sim_gemm_prefill/hex/activation_packets.hex", "w") as f:
        for ch in range(NUM_CHUNKS):
            k_start = ch * 16
            k_end = k_start + 16
            f.write(f"// --- Chunk {ch} (K={k_start}..{k_end-1}) for Tokens 0..15 ---\n")
            for m in range(NUM_TOKENS):
                nibbles = A_codes[k_start:k_end, m]
                pkt = encode_72b_packet(
                    nibbles, scale_exp=int(A_scale[ch, m]))
                f.write(f"{pkt:018X}\n")

    # (c) Golden FP16 Hex (token-major or lane-major):
    # Let's save as: Token 0 (16 lanes), Token 1 (16 lanes) ... Token 15 (16 lanes)
    # Total 256 words
    with open("sim_gemm_prefill/hex/golden_fp16.hex", "w") as f:
        for m in range(NUM_TOKENS):
            f.write(f"// Token {m} (Lanes 0..15)\n")
            for lane in range(NUM_LANES):
                f.write(f"{Y_fp16_hex[lane, m]:04X}\n")

    # (d) Golden Int32 Psums: Total 256 words
    with open("sim_gemm_prefill/hex/golden_int32.hex", "w") as f:
        for m in range(NUM_TOKENS):
            f.write(f"// Token {m} (Lanes 0..15)\n")
            for lane in range(NUM_LANES):
                f.write(f"{int(Psum_int32[lane, m]) & 0xFFFFFFFF:08X}\n")

    # (e) Fake-quant exact reference: Y_fp = W_q @ A_q, Q16.16.
    #     Token-major / lane-minor, same order as golden_fp16.hex.
    with open("sim_gemm_prefill/hex/golden_ref_q16.hex", "w") as f:
        for m in range(NUM_TOKENS):
            f.write(f"// Token {m} (Lanes 0..15) fake-quant W@A Q16.16\n")
            for lane in range(NUM_LANES):
                f.write(f"{to_q16_hex(Y_fp[lane, m]):08X}\n")

    print(f"Gaussian inputs: sigma={GAUSS_SIGMA}, gain_octaves={GAIN_OCTAVES}")
    print(f"Block scale codes (1/8 octave): "
          f"W {int(W_scale.min())}..{int(W_scale.max())}, "
          f"A {int(A_scale.min())}..{int(A_scale.max())}")
    print("[SUCCESS] All CSV and HEX files generated in sim_gemm_prefill/ directory.")

if __name__ == "__main__":
    main()
