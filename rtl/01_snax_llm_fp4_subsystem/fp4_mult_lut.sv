`timescale 1ns/1ps

module fp4_mult_lut (
    input  logic [3:0]        a,        // [3]=sign, [2:0]=mag code
    input  logic [3:0]        b,        // [3]=sign, [2:0]=mag code
    output logic signed [8:0] prod_out  // 9-bit signed integer product [-144 .. +144] (Q2)
);

    logic [7:0] unsigned_prod;
    logic       sign_prod;

    assign sign_prod = a[3] ^ b[3];

    // Direct 6-bit Combinational Logic LUT (64 entries, zero hardware multipliers)
    always_comb begin
        unique case ({a[2:0], b[2:0]})
            // Multiply by 0.0 (3'b000)
            6'h00, 6'h01, 6'h02, 6'h03, 6'h04, 6'h05, 6'h06, 6'h07,
            6'h08, 6'h10, 6'h18, 6'h20, 6'h28, 6'h30, 6'h38: unsigned_prod = 8'd0;

            // 0.5 (x2 = 1)
            6'h09:       unsigned_prod = 8'd1;   // 1 x 1 = 1
            6'h0A, 6'h11: unsigned_prod = 8'd2;  // 1 x 2 = 2
            6'h0B, 6'h19: unsigned_prod = 8'd3;  // 1 x 3 = 3
            6'h0C, 6'h21: unsigned_prod = 8'd4;  // 1 x 4 = 4
            6'h0D, 6'h29: unsigned_prod = 8'd6;  // 1 x 6 = 6
            6'h0E, 6'h31: unsigned_prod = 8'd8;  // 1 x 8 = 8
            6'h0F, 6'h39: unsigned_prod = 8'd12; // 1 x 12 = 12

            // 1.0 (x2 = 2)
            6'h12:       unsigned_prod = 8'd4;   // 2 x 2 = 4
            6'h13, 6'h1A: unsigned_prod = 8'd6;  // 2 x 3 = 6
            6'h14, 6'h22: unsigned_prod = 8'd8;  // 2 x 4 = 8
            6'h15, 6'h2A: unsigned_prod = 8'd12; // 2 x 6 = 12
            6'h16, 6'h32: unsigned_prod = 8'd16; // 2 x 8 = 16
            6'h17, 6'h3A: unsigned_prod = 8'd24; // 2 x 12 = 24

            // 1.5 (x2 = 3)
            6'h1B:       unsigned_prod = 8'd9;   // 3 x 3 = 9
            6'h1C, 6'h23: unsigned_prod = 8'd12; // 3 x 4 = 12
            6'h1D, 6'h2B: unsigned_prod = 8'd18; // 3 x 6 = 18
            6'h1E, 6'h33: unsigned_prod = 8'd24; // 3 x 8 = 24
            6'h1F, 6'h3B: unsigned_prod = 8'd36; // 3 x 12 = 36

            // 2.0 (x2 = 4)
            6'h24:       unsigned_prod = 8'd16;  // 4 x 4 = 16
            6'h25, 6'h2C: unsigned_prod = 8'd24; // 4 x 6 = 24
            6'h26, 6'h34: unsigned_prod = 8'd32; // 4 x 8 = 32
            6'h27, 6'h3C: unsigned_prod = 8'd48; // 4 x 12 = 48

            // 3.0 (x2 = 6)
            6'h2D:       unsigned_prod = 8'd36;  // 6 x 6 = 36
            6'h2E, 6'h35: unsigned_prod = 8'd48; // 6 x 8 = 48
            6'h2F, 6'h3D: unsigned_prod = 8'd72; // 6 x 12 = 72

            // 4.0 (x2 = 8)
            6'h36:       unsigned_prod = 8'd64;  // 8 x 8 = 64
            6'h37, 6'h3E: unsigned_prod = 8'd96; // 8 x 12 = 96

            // 6.0 (x2 = 12)
            6'h3F:       unsigned_prod = 8'd144; // 12 x 12 = 144
            default:     unsigned_prod = 8'd0;
        endcase
    end

    assign prod_out = sign_prod ? -signed'({1'b0, unsigned_prod})
                                :  signed'({1'b0, unsigned_prod});

endmodule
