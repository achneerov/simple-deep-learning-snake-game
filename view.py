import torch

# Relative path to the model.pth file inside the model directory
model_path = 'model/model.pth'

# Load the model from the specified relative path
model = torch.load(model_path)

# Print the loaded model
print(model)
