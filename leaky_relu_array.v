module leaky_relu_array #(
    parameter DATA_WIDTH = 8,
    parameter NUM_ELEMENTS = 16
)(
    input  [NUM_ELEMENTS*DATA_WIDTH-1:0] in_tensor,
    output [NUM_ELEMENTS*DATA_WIDTH-1:0] out_tensor
);

    integer i;
    reg signed [DATA_WIDTH-1:0] x;
    reg signed [DATA_WIDTH-1:0] y;
    reg signed [NUM_ELEMENTS*DATA_WIDTH-1:0] result;

    always @* begin
        for (i = 0; i < NUM_ELEMENTS; i = i + 1) begin
            x = in_tensor[i*DATA_WIDTH +: DATA_WIDTH];
            if (x[DATA_WIDTH-1] == 1'b1) begin
                y = x >>> 7; // multiply by ~0.0078 for Q1.7
            end else begin
                y = x;
            end
            result[i*DATA_WIDTH +: DATA_WIDTH] = y;
        end
    end

    assign out_tensor = result;

endmodule
