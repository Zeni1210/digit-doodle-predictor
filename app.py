import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt

# ---------- 1. MODEL DEFINITION ---------- #
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# Load model
model = Net()
model.load_state_dict(torch.load("mnist_model.pt", map_location="cpu"))
model.eval()

# ---------- 2. STREAMLIT UI ---------- #
st.set_page_config(page_title="Digit Recognizer", layout="centered")
st.title("🖌️ Digit Recognizer")
st.markdown("Draw a digit (0–9) below and click **Predict**.")

canvas_result = st_canvas(
    fill_color="white",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Predict"):
    if canvas_result.image_data is None or not np.any(canvas_result.image_data[:, :, :3]):
        st.warning("Please draw a digit first.")
        st.stop()

    # Process the canvas image - MNIST-style preprocessing
    # Extract RGB channels from RGBA
    canvas_rgb = canvas_result.image_data[:, :, :3]
    
    # Convert to grayscale using standard formula
    gray_array = np.dot(canvas_rgb[...,:3], [0.2989, 0.5870, 0.1140])
    
    # Convert to PIL for better processing
    gray_img = Image.fromarray(gray_array.astype(np.uint8), mode='L')
    
    # Find bounding box of the drawing to center it properly
    bbox = gray_img.getbbox()
    if bbox:
        # Crop to bounding box
        cropped = gray_img.crop(bbox)
        
        # Calculate size to fit in 20x20 (leaving 4px border like MNIST)
        w, h = cropped.size
        if w > h:
            new_w = 20
            new_h = int(20 * h / w)
        else:
            new_h = 20
            new_w = int(20 * w / h)
        
        # Resize maintaining aspect ratio
        resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Create 28x28 black image and paste centered
        final_img = Image.new('L', (28, 28), 0)
        paste_x = (28 - new_w) // 2
        paste_y = (28 - new_h) // 2
        final_img.paste(resized, (paste_x, paste_y))
        
        img_array = np.array(final_img) / 255.0
    else:
        # Fallback if no drawing detected
        img_array = np.zeros((28, 28))
    
    # Convert to tensor
    img_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(1)

    st.success(f"**Prediction: {pred.item()}** • Confidence: {conf.item()*100:.2f}%")
    
    # Show all probabilities
    with st.expander("View all probabilities"):
        prob_data = []
        for i, prob in enumerate(probs[0]):
            prob_data.append(f"Digit {i}: {prob.item()*100:.2f}%")
        for item in prob_data:
            st.write(item)

    # Display images using matplotlib to avoid PIL/JPEG issues
    col1, col2 = st.columns(2)
    
    with col1:
        fig1, ax1 = plt.subplots(figsize=(2, 2))
        ax1.imshow(img_array, cmap='gray')
        ax1.set_title("Processed (28×28)")
        ax1.axis('off')
        st.pyplot(fig1)
        plt.close(fig1)
    
    with col2:
        fig2, ax2 = plt.subplots(figsize=(2, 2))
        ax2.imshow(gray_array, cmap='gray')
        ax2.set_title("Original Drawing")
        ax2.axis('off')
        st.pyplot(fig2)
        plt.close(fig2)