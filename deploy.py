import gradio as gr
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
from training import cnn  # ✅ works now after your fix

# Load model
model = cnn()
model.load_state_dict(torch.load("catdog_model.pth", map_location="cpu"))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Predict function
def predict(image):
    # ✅ Convert NumPy array (from Gradio) to PIL image if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'), 'RGB')

    image = transform(image).unsqueeze(0)
    output = model(image)
    pred = torch.argmax(output, dim=1).item()
    return "Dog 🐶" if pred == 1 else "Cat 🐱"

# Gradio UI
gr.Interface(
    fn=predict,
    inputs="image",
    outputs="text",
    title="Cat vs Dog Classifier",
    description="Upload a cat 🐱 or dog 🐶 image and see what the model predicts!"
).launch()
