`ifndef PE_DEFINES_SVH
`define PE_DEFINES_SVH

// Accumulator depth. Synthesis overrides via +define+ACC_DEPTH_64 so the
// collector SRAM becomes 32 words x 256 bits (== fakeram45_256x32).
// Simulation (no define) keeps the default of 4.
`ifdef ACC_DEPTH_128
`define ACC_DEPTH_SYN 128
`elsif ACC_DEPTH_64
`define ACC_DEPTH_SYN 64
`else
`define ACC_DEPTH_SYN 4
`endif

parameter int BLOCK_SIZE       = 16;  // 16 elements per vector block
parameter int FP4_WIDTH        = 4;   // 4-bit FP4 (E2M1)
parameter int EXP_WIDTH        = 8;   // 8-bit Log-Scale Exponent
parameter int DATA_BUS_WIDTH   = BLOCK_SIZE * FP4_WIDTH; // 64 bits
parameter int PACKET_WIDTH     = DATA_BUS_WIDTH + EXP_WIDTH; // 72 bits (64b data + 8b exp)

parameter int NUM_LANES        = 16;  // 16 Vector Lanes in NPU Core
parameter int ACCUMULATOR_DEPTH= `ACC_DEPTH_SYN;  // Default Accumulator Depth (N_tile)
parameter int NUM_BINS         = 8;   // 8 Bins for r in {0..7}
parameter int BIN_WIDTH        = 32;  // 32-bit signed accumulator per bin

// FEBA precision: pre-shift the intra-block dot product by GUARD_BITS before the
// octave barrel shift, so the right shift discards zero padding instead of real
// data (Q2 -> Q(2+GUARD_BITS)). Q_LEFT_MAX caps the positive octave shift so the
// 32-bit datapath cannot overflow.
parameter int GUARD_BITS       = 12;  // pre-shift guard bits
parameter int Q_LEFT_MAX       = 6;   // cap for positive q (overflow guard)
parameter int PROD_INT_WIDTH   = 16;  // 16-bit signed intra-block dot-product result
parameter int UCS_OUT_WIDTH    = 36;  // 36-bit intermediate scaled product in UCS
parameter int FP16_WIDTH       = 16;  // 16-bit FP16 result output
parameter int VEC_RES_WIDTH    = NUM_LANES * FP16_WIDTH; // 256 bits parallel vector output

`endif // PE_DEFINES_SVH
