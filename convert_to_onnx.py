import torch
import os
from app.model import LogisticRegressionModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'app', 'model', 'model.pth')
ONNX_PATH = os.path.join(BASE_DIR, 'app', 'model', 'model.onnx')

def convert_model():
    state = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    model = LogisticRegressionModel(input_dim=state['input_dim'])
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    
    dummy_input = torch.randn(1, state['input_dim'])
    
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"Model converted to ONNX: {ONNX_PATH}")

if __name__ == "__main__":
    convert_model()
