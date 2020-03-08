module Register_EX_MEM
(
    clk_i,

    memRead_i,
    memWrite_i,
    memToReg_i,
    regWrite_i,
    aluResult_i,
    rtData_i,
    wbAddr_i,

    memRead_o,
    memWrite_o,
    memToReg_o,
    regWrite_o,
    aluResult_o,
    rtData_o,
    wbAddr_o
);

// Interface
input               clk_i;
input               memRead_i;
input               memWrite_i;
input               memToReg_i;
input               regWrite_i;
input       [31:0]  aluResult_i;
input       [31:0]  rtData_i;
input       [4:0]   wbAddr_i;

output  reg         memRead_o;
output  reg         memWrite_o;
output  reg         memToReg_o;
output  reg         regWrite_o;
output  reg  [31:0] aluResult_o;
output  reg  [31:0] rtData_o;
output  reg  [4:0]  wbAddr_o;

// Calculation
always @ (posedge clk_i ) begin
    if(clk_i)begin
        memRead_o   <= memRead_i;
        memWrite_o <= memWrite_i;
        memToReg_o  <= memToReg_i;
        regWrite_o  <= regWrite_i;
        aluResult_o <= aluResult_i;
        rtData_o   <= rtData_i;
        wbAddr_o    <= wbAddr_i;
    end
end

endmodule