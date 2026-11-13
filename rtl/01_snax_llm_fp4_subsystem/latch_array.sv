`timescale 1ns/1ps

// =============================================================================
// Cell implementations (two versions of the `icg_cell` / `dlatch` cells):
//
// - Default (macros undefined): behavioral models -> RTL simulation.
// - ASAP7 synthesis: compile with +define+IGC_CELL +define+LATCH_CELL to swap
//   in the real ASAP7 standard cells:
//       icg_cell -> ICGx4_ASAP7_75t_R   (GCLK,ENA,SE,CLK - SE tied to 0)
//       dlatch   -> DHLx2_ASAP7_75t_R   (transparent-high D-latch: Q,D,CLK)
//   ASAP7 (RVT flavor) sources in lib/ASAP7/:
//       verilog: verilog/asap7sc7p5t_SEQ_RVT_TT_220101.v          (sim)
//       liberty: lib/asap7sc7p5t_SEQ_RVT_{TT,SS,FF}_nldm_*.lib     (DC/Genus)
//       lef    : lef/asap7sc7p5t_28_R_1x_220121a.lef               (PnR)
//   Other drive strengths available: DHLx1/x3, ICGx1..x8(+DC); switch VT flavor
//   by replacing the "_R" suffix and linking the L/SLVT views.
// =============================================================================


// ----------------------------------------------------------------------------
// Clock gate: samples `en` while clk==0, drives gclk = clk & en_sampled.
// ----------------------------------------------------------------------------
`ifdef IGC_CELL
// ASAP7 structural implementation (synthesis)
module icg_cell (
    input  clk, 
    input  en,
    output gclk
);
    (* dont_touch = "true" *) ICGx4_ASAP7_75t_R u_icg (
        .GCLK (gclk),
        .ENA  (en),
        .SE   (1'b0),   // scan enable: tie 0 (no scan path on this array)
        .CLK  (clk)
    );
endmodule
`else
// Behavioral implementation (simulation)
module icg_cell (
    input  clk, 
    input  en,
    output gclk
); 
    logic en_q; 
    always@(*) begin 
        if (~clk) en_q = en;
    end

    assign gclk = en_q & clk; 

endmodule
`endif


// ----------------------------------------------------------------------------
// D latch: transparent while clk==1, holds its value when clk==0.
// ----------------------------------------------------------------------------
`ifdef LATCH_CELL
// ASAP7 structural implementation (synthesis)
module dlatch (
    input  clk, 
    input  d, 
    output q
);
    (* dont_touch = "true" *) DHLx2_ASAP7_75t_R u_dhl (
        .Q   (q),
        .D   (d),
        .CLK (clk)
    );
endmodule
`else
// Behavioral implementation (simulation)
module dlatch (
    input  clk, 
    input  d, 
    output logic q
);
    always@(*) begin 
        if (clk) q = d; 
    end 

endmodule
`endif


// ----------------------------------------------------------------------------
// D flip-flop: rising-edge triggered, asynchronous active-low reset.
// Used when `USE_FF = 1` in `latch_array`. The clock input is the ICG-gated
// clock, so an inactive column simply sees no edge and holds its value.
//
// Synthesis note: the asynchronous reset requires a DFF cell that exposes a
// reset pin, e.g. ASAP7 `DFFASRHQx2_ASAP7_75t_R` (Q, D, CLK, RESET_B, SET_B)
// with SET_B tied high. Swap the body below for that cell when targeting a
// standard-cell library; `DFFHQx*` has no reset pin and cannot be used as-is.
// ----------------------------------------------------------------------------
module dff (
    input  logic clk,
    input  logic rst_n,
    input  logic d,
    output logic q
);
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) q <= 1'b0;
        else        q <= d;
    end
endmodule


// =============================================================================
// Module: latch_array
// Description: 2R1W standard-cell Latch Array (DEPTH words x DW bits, binary addr).
//   Word bits are split into groups of ICG_GROUP: one ICG cell per group so a
//   single ICG never drives more than ICG_GROUP latch CLK pins (ASAP7 ICGx4
//   GCLK max_capacitance = 92.16 fF; DHLx2 CLK pin = 0.474 fF => 512 loads =
//   242.9 fF would violate by 2.6x; 32 loads = 15.2 fF with large margin).
//
// Timing contract:
// - Write: command presented in cycle N is registered at posedge N+1 and
//   committed during cycle N+1 (addressed column's latches are transparent
//   while clk == 1 and close at the following negedge). The new value is
//   visible on both read ports starting at posedge N+1.
// - Read: both read ports are purely combinational.
// - Reset: reset_n clears the whole array to 0 through the ICG enable path
//   (requires clk running while reset_n is low). wdata_q resets asynchronously.
//
// Usage constraints:
// - wen / waddr / wdata must be cycle-stable (driven by FFs of the same clock
//   domain): the address decode is sampled by the ICG during clk == 0 and
//   wdata is captured at posedge clk.
// - DEPTH = number of words (power of 2 recommended); AW = $clog2(DEPTH) is
//   the binary address width ([AW-1:0]).
//
// STORAGE IMPLEMENTATION SELECT (`USE_FF`, elaboration-time):
//   USE_FF = 1'b0 : transparent-high D-latch (`DHLx2_ASAP7_75t_R`) + ICG  (default)
//   USE_FF = 1'b1 : rising-edge D flip-flop (`DFFHQx2_ASAP7_75t_R`)   + ICG
//   Both variants are CYCLE-EQUIVALENT from the outside:
//     * a write command presented in cycle N is captured at the posedge that
//       starts cycle N+1 and is observable on rdata1/rdata2 during cycle N+1;
//     * reads are combinational (same cycle);
//     * when raddr == waddr the read returns the OLD word (read-before-write),
//       because the write is only committed at the end of the cycle.
//   The FF variant captures the combinational `wdata` directly (there is no
//   `wdata_q` retiming stage), which is what preserves the same 1-cycle write
//   latency and therefore the same same-cycle RMW behaviour used by
//   `vsq_matmul_engine` (which additionally bypasses the finishing slot through
//   `drain_word_comb`).
//   Note for FPGA/mapping: the FF variant reads a DEPTH:1 mux per bit, so it is
//   intended for simulation, formal and FF-based (non-SRAM) builds. For a
//   distributed-RAM mapping apply `(* ram_style = "distributed" *)` on
//   `q_store` in the target flow.
// =============================================================================
module latch_array #( 
    parameter DEPTH = 16,               // number of words (2^k recommended)
    parameter DW    = 32,               // word width in bits
    parameter ICG_GROUP = 32,           // max cell CLK pins driven by one ICG cell
    parameter bit USE_FF = 1'b0,        // 0: D-latch (DHL), 1: D flip-flop (DFF)
    localparam AW   = $clog2(DEPTH),    // binary address width
    localparam NUM_ICG = (DW + ICG_GROUP - 1) / ICG_GROUP
) (
    input logic                 clk, 
    input logic                 reset_n,

    input logic [AW-1:0]        waddr, 
    input logic [DW-1:0]        wdata, 
    input logic                 wen, 
    input logic                 ren1,
    input logic                 ren2, 
    input logic [AW-1:0]        raddr1, 
    input logic [AW-1:0]        raddr2, 
    output logic [DW-1:0]       rdata1, 
    output logic [DW-1:0]       rdata2
);  

    logic  [DEPTH-1:0][DW-1:0]      q_store;      // shared storage array
    logic  [DEPTH-1:0]              en_col;       // per-word write enable
    logic  [DEPTH-1:0][NUM_ICG-1:0] gclk_col;     // per-word, per-group gated clock
    int    la_i;

    // NOTE: the open-source simulator does not evaluate level-sensitive
    // inferred latches driven by gate-level ICG cells cycle-accurately, so a
    // build with that tool silently diverges from Questa and from silicon.
    // It also dislikes the per-bit cell netlist performance-wise, so that tool
    // gets a plain behavioural array with identical cycle behaviour: the word
    // is committed on the posedge that starts the next cycle, exactly like the
    // latch and DFF variants.
`ifdef VERILATOR
  `ifdef LATCH_ARRAY_FF
    // Opt back into the per-bit cell netlist (useful to cross-check the model).
    localparam bit USE_FF_EFF = 1'b1;
    localparam bit USE_MODEL  = 1'b0;
  `else
    localparam bit USE_FF_EFF = 1'b0;
    localparam bit USE_MODEL  = 1'b1;
  `endif
`else
    localparam bit USE_FF_EFF = USE_FF;
    localparam bit USE_MODEL  = 1'b0;
`endif

    // =========================================================================
    // Storage implementation (elaboration-time select, see header)
    // =========================================================================
    generate
        if (USE_MODEL) begin : gen_model_storage
            // -----------------------------------------------------------------
            // Behavioural array: one wide register per word instead of DW
            // single-bit cells.  Functionally identical to the DFF variant.
            // -----------------------------------------------------------------
            always_ff @(posedge clk or negedge reset_n) begin
                if (!reset_n) begin
                    for (la_i = 0; la_i < DEPTH; la_i++) q_store[la_i] <= 'd0;
                end else if (wen) begin
                    q_store[waddr] <= wdata;
                end
            end

        end else if (USE_FF_EFF) begin : gen_ff_storage
            // -----------------------------------------------------------------
            // Flip-flop implementation
            //   - no `wdata_q` retiming stage: the D input is the combinational
            //     `wdata`, so the word is committed at the posedge that starts
            //     the next cycle (same latency as the latch version).
            //   - reset is asynchronous (inside `dff`), so the address decode
            //     does not need the `~reset_n` term.
            // -----------------------------------------------------------------
            always_comb begin
                for (la_i = 0; la_i < DEPTH; la_i++) begin
                    en_col[la_i] = wen & (waddr == AW'(la_i));
                end
            end

            for (genvar i=0; i < DEPTH; i++) begin: col_gen
                for (genvar g=0; g < NUM_ICG; g++) begin: icg_gen
                    icg_cell u_icg_cell (
                        .clk        (clk),
                        .en         (en_col[i]),
                        .gclk       (gclk_col[i][g])
                    );

                    for (genvar j=0; j<ICG_GROUP; j++) begin: row_gen
                        if ((g*ICG_GROUP + j) < DW) begin: bit_gen
                            dff u_dff (
                                .clk    (gclk_col[i][g]),
                                .rst_n  (reset_n),
                                .d      (wdata[g*ICG_GROUP + j]),
                                .q      (q_store[i][g*ICG_GROUP + j])
                            );
                        end
                    end
                end
            end

        end else begin : gen_latch_storage
            // -----------------------------------------------------------------
            // Transparent-high D-latch implementation (original)
            // -----------------------------------------------------------------
            logic [DW-1:0] wdata_q;
            always_ff @( posedge clk or negedge reset_n) begin
                if (!reset_n) begin
                    wdata_q <=  'd0;
                end
                else begin
                    if (wen) wdata_q <= wdata;
                end
            end

            for (genvar i=0; i < DEPTH; i++) begin: col_gen

                assign en_col[i] = (~reset_n) | (wen & (waddr == i));

                for (genvar g=0; g < NUM_ICG; g++) begin: icg_gen
                    icg_cell u_icg_cell (
                        .clk        (clk),
                        .en         (en_col[i]),
                        .gclk       (gclk_col[i][g])
                    );

                    for (genvar j=0; j<ICG_GROUP; j++) begin: row_gen
                        if ((g*ICG_GROUP + j) < DW) begin: bit_gen
                            dlatch u_dlatch (
                                .clk    (gclk_col[i][g]),
                                .d      (wdata_q[g*ICG_GROUP + j]),
                                .q      (q_store[i][g*ICG_GROUP + j])
                            );
                        end
                    end
                end
            end
        end
    endgenerate


    assign rdata1 = ren1 ? q_store[raddr1] : 'd0 ;
    assign rdata2 = ren2 ? q_store[raddr2] : 'd0 ;

    
endmodule