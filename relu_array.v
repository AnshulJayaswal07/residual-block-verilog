module relu_array #(
    parameter DATA_WIDTH = 8,
    parameter NUM_ELEMENTS = 16
)(
    input  [NUM_ELEMENTS*DATA_WIDTH-1:0] in_tensor,
    output [NUM_ELEMENTS*DATA_WIDTH-1:0] out_tensor
);

    integer i;
    reg signed [DATA_WIDTH-1:0] x;
    reg signed [DATA_WIDTH-1:0] y;
    reg [NUM_ELEMENTS*DATA_WIDTH-1:0] result;

    always @* begin
        for (i = 0; i < NUM_ELEMENTS; i = i + 1) begin
            x = in_tensor[i*DATA_WIDTH +: DATA_WIDTH];
            y = (x[DATA_WIDTH-1] == 1'b1) ? 0 : x;
            result[i*DATA_WIDTH +: DATA_WIDTH] = y;
        end
    end

    assign out_tensor = result;

endmodule
