`timescale 1ns/1ps

// =============================================================================
// Module: vsq_rescaler_32b
// Description: Dedicated 32-bit CSD Scaler and Arithmetic Shifter for Online
//              Accumulator Rescaling in FlashAttention.
//
// Mathematical Formulation:
//   Output = val_in * 2^(m_old - m_new)
//          = val_in * 2^(resc_q) * (m_r / 32)
//   where:
//     - resc_q <= 0 is the octave exponent
//     - resc_r in [0..7] selects the 3-term CSD multiplier m_r in {32,35,38,41,46,49,54,59}
//     - Zero hardware multipliers: implemented with hardwired shifts + 3:2 CSA + CPA
// =============================================================================
module vsq_rescaler_32b (
    input  logic signed [31:0] val_in,      // 32-bit Q16.16 accumulator operand
    input  logic signed [7:0]  resc_q,      // Octave shift (<= 0)
    input  logic        [2:0]  resc_r,      // Fractional CSD residual index [0..7]
    output logic signed [31:0] scaled_out   // Rescaled 32-bit Q16.16 operand
);

    // Sign-extend 32-bit input to 38 bits to hold up to 64 * val_in
    logic signed [37:0] val_ext;
    assign val_ext = {{6{val_in[31]}}, val_in};

    // Term 0: Select 64*val_in (for r >= 6) or 32*val_in (for r < 6)
    logic [37:0] t0;
    assign t0 = (resc_r[2] & resc_r[1]) ? (val_ext <<< 6) : (val_ext <<< 5);

    // Term 1 & Term 2 magnitudes and subtraction flags
    logic [37:0] t1_mag, t2_mag;
    logic        sub_t1, sub_t2;

    always_comb begin
        unique case (resc_r)
            3'd0: begin t1_mag = 38'd0;         sub_t1 = 1'b0; t2_mag = 38'd0;         sub_t2 = 1'b0; end // 32
            3'd1: begin t1_mag = (val_ext <<< 1); sub_t1 = 1'b0; t2_mag = val_ext;         sub_t2 = 1'b0; end // 32 + 2 + 1 = 35
            3'd2: begin t1_mag = (val_ext <<< 2); sub_t1 = 1'b0; t2_mag = (val_ext <<< 1); sub_t2 = 1'b0; end // 32 + 4 + 2 = 38
            3'd3: begin t1_mag = (val_ext <<< 3); sub_t1 = 1'b0; t2_mag = val_ext;         sub_t2 = 1'b0; end // 32 + 8 + 1 = 41
            3'd4: begin t1_mag = (val_ext <<< 4); sub_t1 = 1'b0; t2_mag = (val_ext <<< 1); sub_t2 = 1'b1; end // 32 + 16 - 2 = 46
            3'd5: begin t1_mag = (val_ext <<< 4); sub_t1 = 1'b0; t2_mag = val_ext;         sub_t2 = 1'b0; end // 32 + 16 + 1 = 49
            3'd6: begin t1_mag = (val_ext <<< 3); sub_t1 = 1'b1; t2_mag = (val_ext <<< 1); sub_t2 = 1'b1; end // 64 - 8 - 2 = 54
            3'd7: begin t1_mag = (val_ext <<< 2); sub_t1 = 1'b1; t2_mag = val_ext;         sub_t2 = 1'b1; end // 64 - 4 - 1 = 59
        endcase
    end

    // 1's complement inversion for subtraction
    logic [37:0] t1, t2;
    assign t1 = sub_t1 ? (~t1_mag) : t1_mag;
    assign t2 = sub_t2 ? (~t2_mag) : t2_mag;

    // Single 38-bit 3:2 Carry-Save Adder
    logic [37:0] csa_sum, csa_cout;
    assign csa_sum  = t0 ^ t1 ^ t2;
    assign csa_cout = (t0 & t1) | (t1 & t2) | (t0 & t2);

    // Left-shifted carry vector with sub_t1 absorbed at bit 0
    logic [37:0] csa_carry_shifted;
    assign csa_carry_shifted = {csa_cout[36:0], sub_t1};

    // 38-bit CPA with sub_t2 absorbed as carry-in
    logic signed [37:0] prod_38;
    assign prod_38 = $signed(csa_sum + csa_carry_shifted + 38'(sub_t2));

    // Net Right Shift Calculation:
    //   m_r is scaled by 32 (5 fractional bits).
    //   resc_q <= 0 represents an octave shift by 2^resc_q.
    //   Total right shift amount = 5 - resc_q >= 5.
    logic [5:0] net_rshift;
    always_comb begin
        if (resc_q < -8'sd26) begin
            net_rshift = 6'd32; // Underflow to 0
        end else begin
            net_rshift = 6'd5 + unsigned'(-resc_q[4:0]);
        end
    end

    // Arithmetic right shift with rounding bit
    logic signed [37:0] shifted_wide;
    assign shifted_wide = (prod_38 + (38'sd1 <<< (net_rshift - 1))) >>> net_rshift;

    // Output assignment (magnitude strictly decreases since 2^dm <= 1, no overflow)
    assign scaled_out = shifted_wide[31:0];

endmodule
