`timescale 1ns/1ps

// Standard 4:2 Carry-Save Compressor Bit-Slice
// in0 + in1 + in2 + in3 + cin = sum + 2 * (carry + cout)
module compressor_4to2 (
    input  logic in0,
    input  logic in1,
    input  logic in2,
    input  logic in3,
    input  logic cin,
    output logic sum,
    output logic carry,
    output logic cout
);
    logic xor_012, xor_0123;

    assign xor_012  = in0 ^ in1 ^ in2;
    assign cout     = (in0 & in1) | (in1 & in2) | (in0 & in2);
    assign xor_0123 = xor_012 ^ in3;
    assign sum      = xor_0123 ^ cin;
    assign carry    = (xor_0123 & cin) | (~xor_0123 & in3);

endmodule
