// =============================================================================
// Module: fixed_priority_arb_n
// Description: Fixed-priority per-bank arbiter for the 72-bit interconnect.
//
// Priority:
//   * Every non-host master (act/wt/output streamers) has priority over the
//     host/DMA master.
//   * Among non-host masters, the lowest master index wins (fixed priority).
//     With the current map: act(1) > wt(2..17) > out(18) > host(0).
//
// This implements the intended policy: a CPU/DMA access that collides with a
// stream-engine access to the same SRAM bank waits, while the stream engine
// keeps the bank and cannot stall the matmul core.
//
// NOTE: strict fixed priority can starve the host if a streamer continuously
// requests the same bank.  The host/DMA must tolerate `ready` being low for
// long periods, or the system must use double buffering / disjoint banks.
// =============================================================================
`timescale 1ns/1ps

module fixed_priority_arb_n #(
    parameter int unsigned NUM_REQS    = 19,
    parameter int unsigned HOST_MASTER = 0,
    // Optional extra low-priority masters (DMA + CPU in the clean top).  When
    // left at '0 the mask falls back to the single HOST_MASTER.
    parameter logic [NUM_REQS-1:0] HOST_MASK = '0
)(
    input  logic [NUM_REQS-1:0] req_i,
    output logic [NUM_REQS-1:0] grant_o
);

    logic [NUM_REQS-1:0] low_mask;
    logic [NUM_REQS-1:0] non_host_req;
    logic [NUM_REQS-1:0] non_host_grant;

    function automatic [NUM_REQS-1:0] onehot(input int unsigned idx);
        onehot = '0;
        onehot[idx] = 1'b1;
    endfunction

    assign low_mask      = (HOST_MASK == '0) ? onehot(HOST_MASTER) : HOST_MASK;
    assign non_host_req  = req_i & ~low_mask;

    // Lowest set bit = highest fixed priority among non-host masters.
    assign non_host_grant = non_host_req & ~(non_host_req - 1'b1);

    assign grant_o = (|non_host_req) ? non_host_grant :
                     (req_i & low_mask & ~((req_i & low_mask) - 1'b1));

endmodule
