import sys
import torch
import argparse
import torch.onnx
from ann_models import AnnRMSModel

torch_path = f"/home/mrcong/Code/mylab-nilm-files/nilm-physical-features/results/models/VNDALE1/window_1800/5_comb/mlp_['Irms', 'P', 'MeanPF', 'S', 'Q'].pt"
onnx_model_path = f"/home/mrcong/Code/mylab-nilm-files/nilm-physical-features/results/models/VNDALE1/window_1800/5_comb/mlp_['Irms', 'P', 'MeanPF', 'S', 'Q'].onnx"
def convert_pytorch_to_onnx(pytorch_model_path, onnx_model_path, input_size):
    # Load the PyTorch model
    model = AnnRMSModel(input_dim=5, output_dim=128, is_bn=False, dropout=0)
    model.load_state_dict(torch.load(pytorch_model_path))
    model.eval()

    # Create a dummy input tensor with the appropriate size
    dummy_input = torch.randn(1, *input_size)

    # Export the model to ONNX format
    torch.onnx.export(model, dummy_input, onnx_model_path, 
                      export_params=True, 
                      opset_version=10, 
                      do_constant_folding=True, 
                      input_names=['input'], 
                      output_names=['output'])

if __name__ == "__main__":
    convert_pytorch_to_onnx(torch_path, onnx_model_path, (5,))