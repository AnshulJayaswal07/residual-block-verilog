`timescale 1ns / 1ps

module tb_residual_block;

    parameter DATA_WIDTH = 8;
    parameter IN_CHANNELS = 1;
    parameter OUT_CHANNELS = 1;
    parameter IN_HEIGHT = 4;
    parameter IN_WIDTH = 4;
    parameter KERNEL_SIZE = 2;
    parameter STRIDE = 1;
    parameter USE_PROJECTION_SHORTCUT = 1;

    localparam IN_SIZE = IN_CHANNELS * IN_HEIGHT * IN_WIDTH;
    localparam OUT_HEIGHT = (IN_HEIGHT - KERNEL_SIZE + 2*0) / STRIDE + 1;
    localparam OUT_WIDTH  = (IN_WIDTH  - KERNEL_SIZE + 2*0) / STRIDE + 1;
    localparam OUT_SIZE = OUT_CHANNELS * OUT_HEIGHT * OUT_WIDTH;

    reg clk = 0;
    reg rst = 0;

    reg  signed [DATA_WIDTH-1:0] input_tensor_flat      [0:IN_SIZE-1];
    wire signed [DATA_WIDTH-1:0] output_tensor_flat     [0:OUT_SIZE-1];

    wire [IN_SIZE*DATA_WIDTH-1:0] input_tensor_flat_bus;
    wire [OUT_SIZE*DATA_WIDTH-1:0] output_tensor_flat_bus;

    // Flatten input array to bus
    genvar i;
    generate
        for (i = 0; i < IN_SIZE; i = i + 1) begin : FLATTEN_INPUT
            assign input_tensor_flat_bus[i*DATA_WIDTH +: DATA_WIDTH] = input_tensor_flat[i];
        end
    endgenerate

    // Unflatten output bus to array
    generate
        for (i = 0; i < OUT_SIZE; i = i + 1) begin : UNFLATTEN_OUTPUT
            assign output_tensor_flat[i] = output_tensor_flat_bus[i*DATA_WIDTH +: DATA_WIDTH];
        end
    endgenerate

    // Clock generation
    always #5 clk = ~clk;

    // Instantiate residual_block
    residual_block #(
        .DATA_WIDTH(DATA_WIDTH),
        .IN_CHANNELS(IN_CHANNELS),
        .OUT_CHANNELS(OUT_CHANNELS),
        .IN_HEIGHT(IN_HEIGHT),
        .IN_WIDTH(IN_WIDTH),
        .KERNEL_SIZE(KERNEL_SIZE),
        .STRIDE(STRIDE),
        .USE_PROJECTION_SHORTCUT(USE_PROJECTION_SHORTCUT)
    ) dut (
        .clk(clk),
        .rst(rst),
        .input_tensor(input_tensor_flat_bus),
        .output_tensor(output_tensor_flat_bus)
    );

    integer j;

    initial begin
        $display("Starting residual block testbench...");
        rst = 1;
        #20;
        rst = 0;

        // Initialize input tensor with test values
        for (j = 0; j < IN_SIZE; j = j + 1) begin
            input_tensor_flat[j] = j;
        end

        #100;  // Wait for output to settle

        $display("Input Tensor:");
        for (j = 0; j < IN_SIZE; j = j + 1) begin
            $display("input_tensor_flat[%0d] = %0d", j, input_tensor_flat[j]);
        end

        $display("Output Tensor:");
        for (j = 0; j < OUT_SIZE; j = j + 1) begin
            $display("output_tensor_flat[%0d] = %0d", j, output_tensor_flat[j]);
        end

        $finish;
    end

endmodule
