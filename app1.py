import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import os
import shutil
import numpy as np
import cv2
import time
from torchvision import transforms
from transformers import TimesformerForVideoClassification
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# --- Configuration ---
class Config:
    MODEL_NAME = "facebook/timesformer-base-finetuned-k400"
    NUM_FRAMES = 8
    FRAME_EXTRACTION_METHOD = "uniform"
    FRAME_EXTRACTION_RATE = 5
    MAX_FRAMES_TO_EXTRACT = 100
    INPUT_SIZE = 224
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TEMP_DIR = "temp_frames"
    MODEL_PATH = "vqa_model15ep.pth"  # Adjust if needed

# --- Model Definition ---
class VideoQualityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = TimesformerForVideoClassification.from_pretrained(
            Config.MODEL_NAME,
            num_frames=Config.NUM_FRAMES,
            ignore_mismatched_sizes=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        outputs = self.base_model.timesformer(pixel_values=x)
        cls_token = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_token).squeeze(), cls_token

# --- Image Transform ---
def get_transforms():
    return transforms.Compose([
        transforms.Resize((Config.INPUT_SIZE, Config.INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# --- Frame Extraction ---
def extract_frames(video_path, output_dir, method="uniform", rate=5, max_frames=100):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps else 0

    if method == "all":
        frame_indices = list(range(min(total_frames, max_frames)))
    elif method == "uniform":
        step = max(1, total_frames // min(max_frames, total_frames))
        frame_indices = list(range(0, total_frames, step))[:max_frames]
    elif method == "random":
        frame_indices = sorted(np.random.choice(total_frames, min(max_frames, total_frames), replace=False))
    else:
        frame_indices = []

    current_index = 0
    executor = ThreadPoolExecutor()
    for i in frame_indices:
        if method == "uniform":
            while current_index < i:
                cap.read()
                current_index += 1
            ret, frame = cap.read()
            current_index += 1
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (Config.INPUT_SIZE, Config.INPUT_SIZE))
            frame_path = os.path.join(output_dir, f"frame_{i:06d}.jpg")
            executor.submit(Image.fromarray(frame).save, frame_path)

    cap.release()
    executor.shutdown(wait=True)
    return sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.jpg')])

# --- Prediction Function ---
@torch.no_grad()
def predict_video_quality(video_path, model):
    if os.path.exists(Config.TEMP_DIR):
        shutil.rmtree(Config.TEMP_DIR)
    os.makedirs(Config.TEMP_DIR, exist_ok=True)

    frame_files = extract_frames(
        video_path,
        Config.TEMP_DIR,
        method=Config.FRAME_EXTRACTION_METHOD,
        rate=Config.FRAME_EXTRACTION_RATE,
        max_frames=Config.NUM_FRAMES
    )

    transform = get_transforms()
    frames = [transform(Image.open(f)) for f in frame_files]
    frames = torch.stack(frames).unsqueeze(0).to(Config.DEVICE)

    model.eval()
    score, _ = model(frames)
    scaled_score = max(0, min(score.item(), 5))

    shutil.rmtree(Config.TEMP_DIR)
    return scaled_score

# --- Load Model ---
@st.cache_resource
def load_model():
    model = VideoQualityModel()
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE))
    model.to(Config.DEVICE)
    return model

# --- Streamlit UI ---
st.set_page_config(page_title="🎥 Video Quality Predictor", layout="centered")
st.title("🎥 Transformer-Based Video Quality Prediction")
st.markdown("Upload a video to get its **predicted quality score (0-5)** using a TimeSformer-based deep learning model.")

uploaded_file = st.file_uploader("Upload MP4 video", type=["mp4"])

if uploaded_file:
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.read())

    st.video("temp_video.mp4")

    with st.spinner("🔍 Analyzing video..."):
        model = load_model()
        score = predict_video_quality("temp_video.mp4", model)

    st.success(f"✅ Predicted Quality Score: **{score:.2f}**")
