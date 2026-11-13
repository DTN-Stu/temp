`timescale 1ns/1ps

// =============================================================================
// Module: l_accumulator
// Description: Per-query running denominator accumulator and reciprocal LUT
//              for online FlashAttention.
//
// Each query has its own l_q[q].  row_start / l_valid / resc_* apply to the
// query selected by query_idx.  The final normalizer reads the per-query
// reciprocal array and selects it by the O-drain slot.
// =============================================================================
module l_accumulator #(
    parameter int SCORE_FRAC  = 16,
    parameter int NUM_QUERIES = 1,
    parameter int QW = (NUM_QUERIES > 1) ? $clog2(NUM_QUERIES) : 1
)(
    input  logic                 clk,
    input  logic                 rst_n,

    // Control from softmax_requantizer / FSM
    input  logic [QW-1:0]        query_idx,
    input  logic                 row_start,
    input  logic                 clear,          // reset ALL query denominators
    input  logic                 l_valid,        // Valid block contribution
    input  logic                 resc_valid,     // Running max increased
    input  logic signed [7:0]    resc_q,         // Octave shift (<= 0)
    input  logic        [2:0]    resc_r,         // CSD residual index [0..7]
    input  logic signed [31:0]   l_contrib_in,   // Block contribution in Q16

    // Running accumulated denominator
    output logic signed [31:0]   l_total_out,
    output logic                 l_ready,

    // Selected-query reciprocal (backward-compatible single outputs)
    output logic [15:0]          inv_l_mant,
    output logic [4:0]           inv_l_exp,

    // Per-query reciprocals for the final normalizer
    output logic [NUM_QUERIES-1:0][15:0] inv_l_mant_q,
    output logic [NUM_QUERIES-1:0][4:0]  inv_l_exp_q
);

    int la_q_seq;
    int la_q_comb;

    logic signed [NUM_QUERIES-1:0][31:0] l_q;
    logic signed [31:0] l_curr;
    logic signed [31:0] l_rescaled;
    logic signed [31:0] l_next;

    assign l_curr = l_q[query_idx];

    vsq_rescaler_32b u_l_rescaler (
        .val_in     (l_curr),
        .resc_q     (resc_q),
        .resc_r     (resc_r),
        .scaled_out (l_rescaled)
    );

    function automatic logic signed [31:0] sat_add32(
        input logic signed [31:0] a,
        input logic signed [31:0] b
    );
        logic signed [32:0] s;
        begin
            s = $signed({a[31], a}) + $signed({b[31], b});
            if (s > 33'sd2147483647)       sat_add32 = 32'sh7FFF_FFFF;
            else if (s < -33'sd2147483648) sat_add32 = 32'sh8000_0000;
            else                           sat_add32 = s[31:0];
        end
    endfunction

    always_comb begin
        // row_start has priority: initialise the selected query denominator.
        if (row_start) begin
            l_next = l_contrib_in;
        end else if (!l_valid) begin
            l_next = l_curr;
        end else if (resc_valid) begin
            l_next = sat_add32(l_rescaled, l_contrib_in);
        end else begin
            l_next = sat_add32(l_curr, l_contrib_in);
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (la_q_seq = 0; la_q_seq < NUM_QUERIES; la_q_seq++) l_q[la_q_seq] <= 32'sd0;
            l_ready <= 1'b0;
        end else begin
            // `clear` resets every query's denominator at once (start of a row),
            // so the per-block accumulate below needs no seed path.
            if (clear) begin
                for (la_q_seq = 0; la_q_seq < NUM_QUERIES; la_q_seq++) l_q[la_q_seq] <= 32'sd0;
            end else if (l_valid) begin
                l_q[query_idx] <= l_next;
            end
            l_ready <= l_valid;
        end
    end

    assign l_total_out = l_q[query_idx];

    // =========================================================================
    // Per-query reciprocal generator: LZC + 32-entry LUT
    // =========================================================================
    logic [NUM_QUERIES-1:0][4:0]  lzc;
    logic [NUM_QUERIES-1:0][30:0] l_norm;
    logic [NUM_QUERIES-1:0][4:0]  lut_idx;

    logic [15:0] recip_lut [0:31];
    always_comb begin
        recip_lut[0]  = 16'd32768; recip_lut[1]  = 16'd31775; recip_lut[2]  = 16'd30840; recip_lut[3]  = 16'd29960;
        recip_lut[4]  = 16'd29127; recip_lut[5]  = 16'd28339; recip_lut[6]  = 16'd27592; recip_lut[7]  = 16'd26884;
        recip_lut[8]  = 16'd26214; recip_lut[9]  = 16'd25579; recip_lut[10] = 16'd24976; recip_lut[11] = 16'd24403;
        recip_lut[12] = 16'd23859; recip_lut[13] = 16'd23341; recip_lut[14] = 16'd22847; recip_lut[15] = 16'd22376;
        recip_lut[16] = 16'd21926; recip_lut[17] = 16'd21495; recip_lut[18] = 16'd21083; recip_lut[19] = 16'd20687;
        recip_lut[20] = 16'd20308; recip_lut[21] = 16'd19944; recip_lut[22] = 16'd19594; recip_lut[23] = 16'd19258;
        recip_lut[24] = 16'd18934; recip_lut[25] = 16'd18622; recip_lut[26] = 16'd18322; recip_lut[27] = 16'd18032;
        recip_lut[28] = 16'd17752; recip_lut[29] = 16'd17482; recip_lut[30] = 16'd17220; recip_lut[31] = 16'd16968;
    end

    logic [NUM_QUERIES-1:0][15:0] inv_l_mant_comb;
    logic [NUM_QUERIES-1:0][4:0]  inv_l_exp_comb;

    always_comb begin
        for (la_q_comb = 0; la_q_comb < NUM_QUERIES; la_q_comb++) begin
            unique casez (l_q[la_q_comb][30:0])
                31'b1??????????????????????????????: lzc[la_q_comb] = 5'd0;
                31'b01?????????????????????????????: lzc[la_q_comb] = 5'd1;
                31'b001????????????????????????????: lzc[la_q_comb] = 5'd2;
                31'b0001???????????????????????????: lzc[la_q_comb] = 5'd3;
                31'b00001??????????????????????????: lzc[la_q_comb] = 5'd4;
                31'b000001?????????????????????????: lzc[la_q_comb] = 5'd5;
                31'b0000001????????????????????????: lzc[la_q_comb] = 5'd6;
                31'b00000001???????????????????????: lzc[la_q_comb] = 5'd7;
                31'b000000001??????????????????????: lzc[la_q_comb] = 5'd8;
                31'b0000000001?????????????????????: lzc[la_q_comb] = 5'd9;
                31'b00000000001????????????????????: lzc[la_q_comb] = 5'd10;
                31'b000000000001???????????????????: lzc[la_q_comb] = 5'd11;
                31'b0000000000001??????????????????: lzc[la_q_comb] = 5'd12;
                31'b00000000000001?????????????????: lzc[la_q_comb] = 5'd13;
                31'b000000000000001????????????????: lzc[la_q_comb] = 5'd14;
                31'b0000000000000001???????????????: lzc[la_q_comb] = 5'd15;
                31'b00000000000000001??????????????: lzc[la_q_comb] = 5'd16;
                31'b000000000000000001?????????????: lzc[la_q_comb] = 5'd17;
                31'b0000000000000000001????????????: lzc[la_q_comb] = 5'd18;
                31'b00000000000000000001???????????: lzc[la_q_comb] = 5'd19;
                31'b000000000000000000001??????????: lzc[la_q_comb] = 5'd20;
                31'b0000000000000000000001?????????: lzc[la_q_comb] = 5'd21;
                31'b00000000000000000000001????????: lzc[la_q_comb] = 5'd22;
                31'b000000000000000000000001???????: lzc[la_q_comb] = 5'd23;
                31'b0000000000000000000000001??????: lzc[la_q_comb] = 5'd24;
                31'b00000000000000000000000001?????: lzc[la_q_comb] = 5'd25;
                31'b000000000000000000000000001????: lzc[la_q_comb] = 5'd26;
                31'b0000000000000000000000000001???: lzc[la_q_comb] = 5'd27;
                31'b00000000000000000000000000001??: lzc[la_q_comb] = 5'd28;
                31'b000000000000000000000000000001?: lzc[la_q_comb] = 5'd29;
                default:                             lzc[la_q_comb] = 5'd30;
            endcase
            l_norm[la_q_comb]  = l_q[la_q_comb][30:0] << lzc[la_q_comb];
            lut_idx[la_q_comb] = l_norm[la_q_comb][29:25];
            inv_l_mant_comb[la_q_comb] = (l_q[la_q_comb] <= 32'sd0) ? 16'd0 : recip_lut[lut_idx[la_q_comb]];
            inv_l_exp_comb[la_q_comb]  = lzc[la_q_comb];
        end
    end

    // Sequential output registers: eliminates LZC and LUT mux from the final_normalizer path
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            inv_l_mant_q <= '0;
            inv_l_exp_q  <= '0;
        end else begin
            inv_l_mant_q <= inv_l_mant_comb;
            inv_l_exp_q  <= inv_l_exp_comb;
        end
    end

    assign inv_l_mant = inv_l_mant_q[query_idx];
    assign inv_l_exp  = inv_l_exp_q[query_idx];

endmodule
