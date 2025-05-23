"""
Streamlit MNIST Digit Recognizer App

This application allows users to draw handwritten digits (0-9) on a canvas,
uses a trained PyTorch model to predict the digit and confidence,
and stores user feedback (true label) aalong with image data in a PostgreSQL database.
Recent prediction history is diplayed at the bottom.
"""

from datetime import datetime, timezone
import os
import io

import numpy as np
from PIL import Image, ImageOps
import pandas as pd
import torch
import torch.nn as nn
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from sqlalchemy import  (
    create_engine,
    MetaData,
    Table, Column,
    Integer, Float, LargeBinary, TIMESTAMP
)

# ---------- 1. MODEl DEFINITION ---------- #
class Net(nn.Module):
    """
    Simple fully-connected neural network for MNIST digit classification.
    Architecture:
    - Flatten 28x28 input to 784
    - Fully-connected layer (784 -> 128) neurons and ReLU activation
    - Fully-connected layer (128 -> 10 classes)
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # Hidden layer
        self.relu = nn.ReLU()               # Non-lnear activation
        self.fc2 = nn.Linear(128, 10)       # Output layer (10 digits)

    def forward(self, x):
        x = x.view(-1, 28 * 28)                 # Flatten input tensor to (batch_size, 784)
        x = self.relu(self.fc1(x))              # Pass through hidden layer with ReLU
        return self.fc2(x)                      # Return raw scores (logits) for each class

# Instantiate the model and load pre-trained weights
model = Net() 
# Load weights onto CPU to ensure compatibility
model.load_state_dict(torch.load("mnist_model.pt", map_location="cpu"))
model.eval()            # Switch model to evaluation mode

# ---------- 2. DATABASE SETUP---------- #

#Retrieve database credentials from environment variables (with deafults)
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASS = os.environ.get("DB_PASSWORD", "postgres")
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "mnistdb")

# Construct SQLAlchemy database URI
DB_URI = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
# Create SQLAlchemy engine and metadata object
engine   = create_engine(DB_URI, echo=False)
metadata = MetaData()

# Define "digit_feedback" table schema
digit_feedback = Table(
    "digit_feedback", metadata,
    Column("id",         Integer, primary_key=True),  # Auto-incrementing ID
    Column("prediction", Integer),                    # The model's predicted digit
    Column("confidence", Float),                      # Model's confidence score
    Column("true_label", Integer),                    # User-provided correct label
    Column("image_data", LargeBinary),                # Stored PNG bytes of the drawn image
    Column("created_at", TIMESTAMP, default=datetime.now(timezone.utc))  # Timestamp of record
)
# Create table if it does not exist
metadata.create_all(engine)        

# ---------- 3. STREAMLIT UI LAYOUT ---------- #
st.title("Digit Recognizer")
st.markdown("Draw a digit (0–9) below and click Predict.")

# Create a drawable canvas for user sketch digits
canvas_result = st_canvas(
    fill_color   = "white",       # Color inside the brush stroke
    stroke_width = 15,            # Thickness of the brush
    stroke_color ="white",        # Brush color
    background_color = "black",   # Canvas background color
    height = 280, width = 280,    # Canvas size
    drawing_mode = "freedraw",    # Allow freehand drawing
    key = "canvas",             # Unique key to reset canvas
    update_streamlit=True         # Send canvas updates back to Streamlit
)

# ---------- 4. PREDICTION PROCESS ---------- #
if st.button("Predict"):
    # Check if user has drawn anything
    if canvas_result.image_data is None or not np.any(canvas_result.image_data[:, :, :3]):
        st.warning("Please draw a digit first.")
        st.stop()

    # Convert canvas RGBA array to PIL grayscale image
    pil_img = Image.fromarray(canvas_result.image_data.astype("uint8")).convert("L")
    # Resize to 28x28 and invert colors (white digits on black background)
    pil_img = ImageOps.invert(pil_img.resize((28, 28)))
    # Normalize pixel values to [0, 1] and add batch and channel dimensions
    img_tensor = torch.tensor(np.array(pil_img) / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # Run inference without gradient conputation
    with torch.no_grad():
        # Convert logits to probabilities
        probs = torch.softmax(model(img_tensor), dim=1) 
        # Get highest probability and index
        conf, pred = probs.max(1)

    # Store prediction, confidence, and tensor in session state
    st.session_state["pred"] = int(pred)
    st.session_state["conf"] = float(conf)
    st.session_state["img"] = img_tensor

    # Display prediction result to user
    st.success(f"Prediction: {pred.item()}  ·  Confidence: {conf.item()*100:.2f}%")

# ---------- 5. SUBMIT USER FEEDBACK ---------- #
# If we have a prediction in session, prompt user for the true label
if "pred" in st.session_state:
    true_label = st.number_input("True Label (0–9)", 0, 9, step=1)

    if st.button("Submit"):
        # Convert processed image back into PNG bytes
        buf = io.BytesIO()
        img_np = st.session_state["img"].squeeze().numpy()
        Image.fromarray((img_np * 255).astype(np.uint8)).save(buf, format="PNG")

        # Insert prediction and feedback into database
        with engine.begin() as conn:
            conn.execute(digit_feedback.insert().values(
                prediction  = st.session_state["pred"],
                confidence  = st.session_state["conf"],
                true_label  = true_label,
                image_data  = buf.getvalue()
            ))

        st.success("Saved to database.")

        # Clear prediction-related session state
        for k in ("pred", "conf", "img"):
            st.session_state.pop(k, None)

# ---------- 6. DISPLAY HISTORY OF RECENT PREDICTIONS---------- #
with engine.connect() as conn:
    result = conn.execute(
        digit_feedback.select().order_by(digit_feedback.c.created_at.desc()).limit(10)
    )
    rows = result.fetchall()

if rows:
    # Convert records to a pandas DataFrame for display
    df = pd.DataFrame.from_records(rows, columns=result.keys())
    st.markdown("Recent Predictions")
    # Show only timestamp, prediction, and true label
    st.dataframe(df[["created_at", "prediction", "true_label"]])
else:
    st.info("No prediction history yet.")
