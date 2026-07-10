import os
import torch
import torch.nn as nn

class AudioClassifier(nn.Module):
    def __init__(self):
        super(AudioClassifier, self).__init__()
        # Simulating a small network processing 128 audio feature inputs
        self.network = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3) # 3 outputs: 0=Safe, 1=Toxic, 2=Urgent Threat
        )

    def forward(self, x):
        return self.network(x)

# Instantiate and set to evaluation mode
model = AudioClassifier()
model.eval()

# Create dummy input matching the shape expected by the model (Batch Size 1, 128 Features)
dummy_input = torch.randn(1, 128)

# --- FIXED PATH LOGIC ---
# 1. Get the directory where this script lives (/Users/nandu/echoguard/scripts)
script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Go up one level to the project root (/Users/nandu/echoguard)
project_root = os.path.dirname(script_dir)

# 3. Force an absolute path to the models folder (/Users/nandu/echoguard/models)
output_dir = os.path.join(project_root, "models")
os.makedirs(output_dir, exist_ok=True)

# 4. Construct the final absolute file path
absolute_onnx_path = os.path.join(output_dir, "audio_classifier.onnx")
# ------------------------

# Export to ONNX format using the strict absolute path
torch.onnx.export(
    model, 
    dummy_input, 
    absolute_onnx_path, # <-- Pass the absolute path here
    input_names=['input_audio'], 
    output_names=['predictions'],
    dynamic_axes={'input_audio': {0: 'batch_size'}, 'predictions': {0: 'batch_size'}},
    dynamo=True 
)

print(f"✅ Model successfully exported to: {absolute_onnx_path}")
