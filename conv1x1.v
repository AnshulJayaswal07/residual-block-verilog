`timescale 1ns / 1ps

module conv1x1 #(
    parameter BATCH_SIZE   = 1,
    parameter IN_CHANNELS  = 1,
    parameter OUT_CHANNELS = 1,
    parameter IN_HEIGHT    = 4,
    parameter IN_WIDTH     = 4,
    parameter STRIDE       = 1, // Stride is passed down to conv2d
    parameter DATA_WIDTH   = 32
)(
    input clk,
    input rst,
    input start,
    output wire done,

    input  [BATCH_SIZE*IN_CHANNELS*IN_HEIGHT*IN_WIDTH*DATA_WIDTH-1:0] input_tensor_flat,
    input  [OUT_CHANNELS*IN_CHANNELS*1*1*DATA_WIDTH-1:0] weights_flat, // Kernel size is 1x1
    input  [OUT_CHANNELS*DATA_WIDTH-1:0] bias_flat,

    // Calculate output dimensions for a 1x1 convolution with PADDING=0 and given STRIDE
    // OUT_HEIGHT = (IN_HEIGHT - KERNEL_SIZE + 2*PADDING) / STRIDE + 1
    // Here KERNEL_SIZE=1, PADDING=0. So:
    // OUT_HEIGHT = (IN_HEIGHT - 1 + 0) / STRIDE + 1 = (IN_HEIGHT - 1) / STRIDE + 1
    // Similarly for OUT_WIDTH.
    output [BATCH_SIZE*OUT_CHANNELS*
            ((IN_HEIGHT - 1) / STRIDE + 1) * // Calculated Output Height
            ((IN_WIDTH  - 1) / STRIDE + 1) * // Calculated Output Width
            DATA_WIDTH-1:0] output_tensor_flat
);

    // conv1x1 is a wrapper around conv2d with KERNEL_SIZE=1 and PADDING=0
    conv2d #(
        .BATCH_SIZE(BATCH_SIZE),
        .IN_CHANNELS(IN_CHANNELS),
        .OUT_CHANNELS(OUT_CHANNELS),
        .IN_HEIGHT(IN_HEIGHT),
        .IN_WIDTH(IN_WIDTH),
        .KERNEL_SIZE(1), // Fixed to 1 for 1x1 convolution
        .STRIDE(STRIDE), // Passed directly from conv1x1's parameter
        .PADDING(0),     // Fixed to 0 for 1x1 convolution
        .DATA_WIDTH(DATA_WIDTH)
    ) conv_inst (
        .clk(clk),
        .rst(rst),
        .start(start),
        .done(done),
        .input_tensor_flat(input_tensor_flat),
        .weights_flat(weights_flat),
        .bias_flat(bias_flat),
        .output_tensor_flat(output_tensor_flat)
    );

endmodule