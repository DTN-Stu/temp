#!/usr/bin/env python3
"""
generate_snax_llm_top_clean_data.py

Stimulus + golden generator for the full 21-master TOP testbench
(tb/tb_snax_llm_top_clean.sv).

The numbers are produced by the same bit-exact model that drives the core
tiling test (generate_snax_core_flash_tiling_data.py), so the TOP and the
core-level testbenches share identical Q/K/V packets and golden outputs.

Outputs
-------
sim/sim_snax_llm_top_clean/hex/
    q_packets.hex / k_packets.hex / v_packets.hex   input packets
    golden_core_fp16.hex                            1024 golden FP16 values
    tcdm_preload_addr.hex / tcdm_preload_data.hex   DMA preload image
    expected_out_words.hex                          256 expected 72b words
sim/sim_snax_llm_top_clean/csv/
    q_matrix_fp_h{h}.csv, q_matrix_quantized_h{h}.csv, q_matrix_fp_digital_h{h}.csv
    k_matrix_*.csv, v_matrix_*.csv
    golden_attn.csv                                 golden output per lane
    tcdm_preload.csv                                TCDM address map
    expected_out_words.csv                          expected word vs lane values

TCDM map used by snax_llm_sequencer (identical to the core TBs)
    Q   : bank 16, word = h*32 + q*2  + kb
    K   : bank 0..15, word = h*4  + kj*2 + kb
    V   : bank 0..15, word = 16 + h*4 + dvi*2 + kj
    OUT : bank 17, word = 0..  (one 256b FP16 vector = 4 consecutive words)
"""
import os
import csv
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import generate_snax_core_flash_tiling_data as model

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.join(HERE, "..", "sim")

SRC_DIR = os.path.join(SIM, "sim_snax_core_flash")
OUT_DIR = os.path.join(SIM, "sim_snax_llm_top_clean")
HEX = os.path.join(OUT_DIR, "hex")
CSV = os.path.join(OUT_DIR, "csv")

# Problem shape (must match the testbench localparams)
H = 2
N_Q = 16
N_KV = 32
D_K = 32
D_V = 32
M_Q = 8
K_BLOCKS = D_K // 16      # 2
Q_TILES = N_Q // M_Q      # 2
DV_TILES = D_V // 16      # 2
KV_TILES = N_KV // 16     # 2

Q_BANK = 16
OUT_BANK = 17
NUM_VEC = H * Q_TILES * DV_TILES * M_Q      # 64
NUM_WORD = NUM_VEC * 4                      # 256
NUM_FP16 = NUM_WORD * 4                     # 1024

OUT_BASE = (OUT_BANK << 4)                  # 0x110


def tcdm_addr(word, bank):
    """Byte address: bank in [8:4], word in [18:9]."""
    return ((word & 0x3FF) << 9) | ((bank & 0x1F) << 4)


def q_word(h, q, kb):
    return h * (N_Q * K_BLOCKS) + q * K_BLOCKS + kb


def k_word(h, kj, kb):
    return h * (KV_TILES * K_BLOCKS) + kj * K_BLOCKS + kb


def v_word(h, dvi, kj):
    return 16 + h * (DV_TILES * KV_TILES) + dvi * KV_TILES + kj


def read_hex(path, width):
    vals = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                vals.append(int(line, 16) & ((1 << width) - 1))
    return vals


def copy(src, dst):
    shutil.copyfile(src, dst)


def main():
    os.makedirs(HEX, exist_ok=True)
    os.makedirs(CSV, exist_ok=True)

    print("=== Generating SNAX LLM TOP (21 masters) stimulus ===")

    # ------------------------------------------------------------------
    # Run the proven core-level model: it writes sim_snax_core_flash/hex
    # and sim_snax_core_flash/csv
    # ------------------------------------------------------------------
    print("[1/4] Running the core tiling model ...")
    model.main()

    # ------------------------------------------------------------------
    # Input packets + golden
    # ------------------------------------------------------------------
    print("[2/4] Copying packets and golden reference ...")
    for name in ("q_packets.hex", "k_packets.hex", "v_packets.hex",
                 "golden_core_fp16.hex"):
        copy(os.path.join(SRC_DIR, "hex", name), os.path.join(HEX, name))

    q_pkts = read_hex(os.path.join(HEX, "q_packets.hex"), 72)
    k_pkts = read_hex(os.path.join(HEX, "k_packets.hex"), 72)
    v_pkts = read_hex(os.path.join(HEX, "v_packets.hex"), 72)
    golden = read_hex(os.path.join(HEX, "golden_core_fp16.hex"), 16)

    assert len(q_pkts) == H * N_Q * K_BLOCKS, len(q_pkts)
    assert len(k_pkts) == H * KV_TILES * K_BLOCKS * 16, len(k_pkts)
    assert len(v_pkts) == H * DV_TILES * KV_TILES * 16, len(v_pkts)
    assert len(golden) == NUM_FP16, len(golden)

    # ------------------------------------------------------------------
    # TCDM preload image (byte address, 72b data), word granularity
    # ------------------------------------------------------------------
    print("[3/4] Building the TCDM preload image ...")
    preload = []   # (addr, data, description)

    for h in range(H):
        for q in range(N_Q):
            for kb in range(K_BLOCKS):
                idx = h * (N_Q * K_BLOCKS) + q * K_BLOCKS + kb
                preload.append((tcdm_addr(q_word(h, q, kb), Q_BANK),
                                q_pkts[idx],
                                f"Q h={h} q={q} kb={kb}"))

    for h in range(H):
        for kj in range(KV_TILES):
            for kb in range(K_BLOCKS):
                for lane in range(16):
                    idx = ((h * KV_TILES + kj) * K_BLOCKS + kb) * 16 + lane
                    preload.append((tcdm_addr(k_word(h, kj, kb), lane),
                                    k_pkts[idx],
                                    f"K h={h} kj={kj} kb={kb} lane={lane}"))

    for h in range(H):
        for dvi in range(DV_TILES):
            for kj in range(KV_TILES):
                for lane in range(16):
                    idx = ((h * DV_TILES + dvi) * KV_TILES + kj) * 16 + lane
                    preload.append((tcdm_addr(v_word(h, dvi, kj), lane),
                                    v_pkts[idx],
                                    f"V h={h} dvi={dvi} kj={kj} ch={lane}"))

    with open(os.path.join(HEX, "tcdm_preload_addr.hex"), "w") as fa, \
         open(os.path.join(HEX, "tcdm_preload_data.hex"), "w") as fd:
        for addr, data, _ in preload:
            fa.write(f"{addr:08X}\n")
            fd.write(f"{data:018X}\n")

    with open(os.path.join(CSV, "tcdm_preload.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "kind", "addr_hex", "bank", "word",
                    "data_hex", "detail"])
        for i, (addr, data, detail) in enumerate(preload):
            w.writerow([i, detail.split()[0], f"0x{addr:08X}",
                        (addr >> 4) & 0x1F, (addr >> 9) & 0x3FF,
                        f"0x{data:018X}", detail])

    # ------------------------------------------------------------------
    # Expected write-out words (FP16 packing of each 256b vector)
    # ------------------------------------------------------------------
    print("[4/4] Building expected output words ...")
    expected = []
    with open(os.path.join(CSV, "expected_out_words.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["vector", "h", "query_tile", "dv_slice", "m", "word",
                    "lane0_hex", "lane1_hex", "lane2_hex", "lane3_hex",
                    "word_hex"])
        for v in range(NUM_VEC):
            m = v % M_Q
            dvi = (v // M_Q) % DV_TILES
            qi = (v // (M_Q * DV_TILES)) % Q_TILES
            h = v // (M_Q * DV_TILES * Q_TILES)
            q_global = qi * M_Q + m
            for wd in range(4):
                lanes = [golden[(h * N_Q + q_global) * D_V + dvi * 16 + wd * 4 + j]
                         for j in range(4)]
                packed = (lanes[3] << 48) | (lanes[2] << 32) | \
                         (lanes[1] << 16) | lanes[0]
                expected.append(packed)
                w.writerow([v, h, qi, dvi, m, wd,
                            f"0x{lanes[0]:04X}", f"0x{lanes[1]:04X}",
                            f"0x{lanes[2]:04X}", f"0x{lanes[3]:04X}",
                            f"0x{packed:018X}"])

    with open(os.path.join(HEX, "expected_out_words.hex"), "w") as f:
        for word in expected:
            f.write(f"{word:018X}\n")

    assert len(expected) == NUM_WORD

    # ------------------------------------------------------------------
    # Debug CSVs for the matrices and the golden per-lane values
    # ------------------------------------------------------------------
    for h in range(H):
        for stem in ("q_matrix_fp", "q_matrix_quantized", "q_matrix_fp_digital",
                     "k_matrix_fp", "k_matrix_quantized", "k_matrix_fp_digital",
                     "v_matrix_fp", "v_matrix_hw_quantized",
                     "v_matrix_hw_fp_digital"):
            src = os.path.join(SRC_DIR, "csv", f"{stem}_h{h}.csv")
            if os.path.exists(src):
                copy(src, os.path.join(CSV, f"{stem}_h{h}.csv"))

    for name in ("golden_attn.csv", "golden_output_combined.csv",
                 "qk_scores.csv", "p_blocks.csv", "l_state.csv",
                 "v_block_quantized.csv", "q_packets_digital_h0.csv",
                 "q_packets_digital_h1.csv", "k_packets_digital_h0.csv",
                 "k_packets_digital_h1.csv", "v_packets_digital.csv"):
        src = os.path.join(SRC_DIR, "csv", name)
        if os.path.exists(src):
            copy(src, os.path.join(CSV, name))

    print(f"\n[SUCCESS] Wrote {len(preload)} TCDM preload words, "
          f"{NUM_WORD} expected output words, {NUM_FP16} golden FP16 values.")
    print(f"  HEX : {os.path.normpath(HEX)}")
    print(f"  CSV : {os.path.normpath(CSV)}")


if __name__ == "__main__":
    main()
