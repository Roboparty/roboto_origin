import torch
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Convert PyTorch .pt model to ONNX format")
    parser.add_argument("--pt_path", type=str, required=True, help="Path to the input .pt model file")
    parser.add_argument("--onnx_path", type=str, required=True, help="Path to save the output ONNX model")
    parser.add_argument(
        "--input_shape",
        type=int,
        nargs="+",
        default=[1, 780],
        help="Input shape for the model, e.g., 1 3 224 224 for batch_size, channels, height, width",
    )
    parser.add_argument("--opset_version", type=int, default=11, 
                       help="ONNX opset version to use")
    return parser.parse_args()

def load_model(pt_path):
    """加载 PyTorch 模型"""
    try:
        model = torch.jit.load(pt_path, map_location=torch.device('cpu'))
        model.eval()  # 设置为评估模式
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load PyTorch model from {pt_path}: {str(e)}")

def convert_to_onnx(model, input_shape, onnx_path, opset_version):
    """将 PyTorch 模型转换为 ONNX 格式"""
    # 创建虚拟输入张量
    dummy_input = torch.randn(*input_shape)
    
    # 导出模型到 ONNX
    torch.onnx.export(
        model,                    # PyTorch 模型
        dummy_input,              # 虚拟输入
        onnx_path,                # 输出 ONNX 文件路径
        export_params=True,       # 导出模型参数
        opset_version=opset_version,  # ONNX 算子集版本
        do_constant_folding=True, # 是否执行常量折叠优化
        input_names=['input'],    # 输入张量名称
        output_names=['output']   # 输出张量名称
    )
    print(f"Model has been successfully converted to ONNX and saved at {onnx_path}")

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 确保输出目录存在
    output_dir = Path(args.onnx_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载 PyTorch 模型
    model = load_model(args.pt_path)
    
    # 转换为 ONNX
    convert_to_onnx(model, args.input_shape, args.onnx_path, args.opset_version)

if __name__ == "__main__":
    main()
