# SNAX FP4 LLM Subsystem — Phân tích, tối ưu và mô phỏng

Tài liệu này tổng hợp: giải nén + phân tích RTL, các tối ưu hiệu năng đã triển khai,
cách chạy testbench (kèm FST), và TB hệ thống mới (CPU + DMA + GEMM).

Repo: `DTN-Stu/temp` (nhánh `main`).

---

## 1. Giải nén & cấu trúc

Hai archive được giải nén:

```
rtl/01_snax_llm_fp4_subsystem/   34 file .sv/.svh   (RTL)
tb/tb/                           18 TB .sv + 11 script Python + notebook
tb/sim/                          stimulus/golden sinh bởi Python (sinh khi chạy)
```

`01_snax_llm_fp4_subsystem` là NPU cho LLM dùng **FP4 log-scale (E2M1)**, packet
72-bit = 64-bit (16×FP4) + 8-bit log-scale exponent. Làm được **dense GEMM** và
**FlashAttention** (softmax exp-free, online rescale).

Top chính: `snax_llm_top_clean` (21 master). Hai top phụ: `snax_llm_top` (rút gọn,
địa chỉ hard-code) và `snax_llm_top_flash` (bản FlashAttention cũ).

---

## 2. Môi trường & cách chạy

Cài đặt:

```bash
sudo apt-get install -y verilator        # Verilator 5.020
pip install numpy
```

Sinh stimulus/golden + build + chạy, xuất waveform FST:

```bash
cd /workspaces/temp
./run.sh              # cả clean/gemm/llama
./run.sh gemm         # một TB
./run.sh layer        # TB hệ thống CPU+DMA+GEMM (smoke)
./run.sh clean --deep # dump toàn hierarchy
./run.sh gemm --no-trace   # chạy nhanh, không xuất sóng
```

Output:
- `sim_out/build/<t>/` — cây build Verilator
- `sim_out/waves/<tb>.fst` — waveform (mở bằng `gtkwave`)
- `sim_out/logs/<tb>.log` — log mô phỏng

Filelist tự chứa trong `filelists/tb_<t>.f` (RTL + include + TB).

---

## 3. Testbench của `snax_llm_top_clean`

| TB | Chế độ | Kích thước |
|---|---|---|
| `tb_snax_llm_top_clean` | FlashAttention | H=2, N_Q=16, N_KV=32, D_K=D_V=32, M_Q=8 |
| `tb_snax_llm_top_gemm` | Dense GEMM | M=32, N=32, K=64 |
| `tb_snax_llm_top_llama` | Attention shape LLaMA | H=1, N_Q=16, N_KV=64, D_K=D_V=128 |
| `tb_snax_llm_layer_smoke` | **CPU+DMA+GEMM (mới)** | dùng stimulus GEMM |

---

## 4. Tối ưu hiệu năng đã triển khai

### 4.1 FIFO input trong `snax_streamer_write_out`
- Thêm FIFO 256-bit, `FIFO_DEPTH=16`, `core_out_*_ready_o = !full`.
- FP16 giữ head FIFO qua 4 word rồi mới pop → hết bubble bắt tay mỗi vector.
- Kết quả GEMM: **760 → 580** cycle.

### 4.2 Drain GEMM chạy song song compute (overlap)
- Engine drain nền (`gdr_idx_q`) phát `tile_drain_req` slot 0..15 liên tiếp
  (Port B) trong khi tile kế tích lũy (Port A) — latch array 2R1W.
- Compute beat `t` bị gate theo tiến độ drain (`t < gdr_idx`) để đảm bảo slot đã
  clear trước khi ghi đè (an toàn, không tăng `ACC_DEPTH`).
- Kết quả GEMM: **580 → 391** cycle.

### 4.3 Một launch streamer cho cả tile (AGU đi hết K)
- `act/wt_k_blocks = cfg_gemm_k_blocks`, `k_stride = 0x200`; FSM ở lại `RUN`
  giữa các K-block thay vì relaunch.
- Kết quả GEMM: **391 → 367** cycle.

### Tổng kết GEMM
| Bước | Cycles | Trạng thái |
|---|---|---|
| Gốc | 760 | bit-exact |
| + FIFO | 580 | bit-exact |
| + Drain overlap | 391 | bit-exact |
| + Single-launch | **367** | bit-exact |

Attention (`clean`, `llama`) giữ nguyên bit-exact; nút thắt còn lại của attention
là QK bị tính lại theo từng DV-slice (chưa xử lý trong đợt này).

---

## 5. TB hệ thống mới: `tb_snax_llm_layer_smoke`

Dựng **model DMA** và **model CPU** tương tác với TCDM để kiểm chứng plumbing
quanh accelerator:

- **DMA model**: mảng DRAM (72b, index = `tcdm_addr>>4`) + master 0; task
  `dma_block_copy(dram_addr, tcdm_addr, size, dir)` mô phỏng đúng thao tác một
  DMA descriptor thực hiện (copy 16-byte slot DRAM↔TCDM).
- **CPU model**: master 1 (`cpu_mem_*`) đọc/ghi TCDM + CSR (`cpu_csr_*`).
- Luồng: build DRAM → DMA copy DRAM→TCDM → CPU poke/peek → CPU lập trình CSR →
  chạy GEMM qua sequencer → DMA readback → so bit-exact.

Kết quả: **PASS — 1024/1024 FP16 bit-exact**, GEMM 367 cycle, 18/18 check OK.
Đây là nền để mở rộng lên **1 layer prefill LLaMA-style** (RMSNorm/RoPE/SwiGLU/
residual chạy trên CPU model, ghép chuỗi job qua CSR).

### 5.1 Primitives của layer đã có & được kiểm chứng

- **CPU elementwise kernels** (chạy trên FP16 trong TCDM, self-test):
  `cpu_rmsnorm4`, `cpu_silu4`, `cpu_residual4` — 12 lane check PASS.
- **CPU FP4 quantizer** `cpu_quantize16` (FP16→packet 72b, mô phỏng đúng
  `slq_encoder`/`log_quantizer`) — packet bit-exact với golden Python.
- **HW readback-requant job** (opcode 11): read streamer (master20) → **khối
  `log_quantizer` HW** → write-out FP4 (master19). Packet đọc lại **bit-exact**
  `72'h00B391F7E691D5C4A2` → khối lượng tử HW đã được exercise thật.

### 5.2 Sửa RTL để chạy đa job

Khi thêm job requant phát hiện 2 lỗi trong `snax_llm_sequencer.sv`:
- `ST_DONE` không cho phép khởi động job kế tiếp → sửa để `start_i` quay về
  `ST_PREP_DRAIN` (bắt buộc cho chuỗi job của một layer).
- Xung `rd_done` 1 chu kỳ bị lỡ khi write-out còn bận → thêm latch sticky
  `rd_done_seen` rồi thoát `ST_RD_RUN` khi write-out đã xả.

Regression sau sửa: `clean` / `gemm` / `llama` vẫn bit-exact (gemm 367 cyc,
llama 6704 cyc).

---

## 6. Hạn chế & việc tiếp theo

- Attention: cần **cache P** để bỏ QK tính lặp theo DV-slice (cải thiện lớn nhất
  còn lại cho attention, nhất là shape LLaMA D_V=128 → lặp 8×).
- Weight streamer: **prefetch K kế trong lúc replay** để bỏ nốt ~2 cyc/K-block.
- Hoàn thiện **full layer**: thêm kernel CPU (RMSNorm/RoPE/SiLU/residual), định
  nghĩa layout + repack GEMM→attention, và golden toàn layer nếu cần bit-exact.
- CPU/DMA model hiện là **behavioral** (functional), không cycle-accurate.

---

## 7. File thay đổi

RTL:
- `rtl/01_snax_llm_fp4_subsystem/snax_streamer_write_out.sv` — thêm input FIFO.
- `rtl/01_snax_llm_fp4_subsystem/snax_llm_sequencer.sv` — drain overlay + single-launch.

TB/script:
- `tb/tb/tb_snax_llm_top_gemm.sv`, `tb_snax_llm_top_llama.sv` — thêm block dump FST.
- `tb/tb/tb_snax_llm_layer_smoke.sv` — TB hệ thống CPU+DMA+GEMM (mới).
- `run.sh`, `filelists/tb_*.f` — script build/run + FST + target `layer`.
