# app.py
import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import json
import time

# --- Konfigurasi ---
NGROK_BASE_URL = "https://Ihortis.simosachi.cloud"
MODEL_PATH = "best_model(1).pth"
SOLUSI_PATH = "solusi_penyakit_tanaman_cabai_tomat.json"
REFRESH_INTERVAL = 10  # detik
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

st.set_page_config(page_title="AI Image Processor (PyTorch)", layout="centered")
st.title("🌱 AI Image Processor (PyTorch)")
st.markdown("---")

# --- Load Model & Solusi ---
@st.cache_resource
def load_torch_model():
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(idx_to_class)

    try:
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    except Exception:
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, idx_to_class

@st.cache_resource
def load_solusi(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["Penyakit"]: item for item in data}

model, idx2cls = load_torch_model()
solusi_dict = load_solusi(SOLUSI_PATH)

# --- Preprocessing ---
infer_tfms = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.15)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

def preprocess_image(image_bytes):
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    return infer_tfms(img).unsqueeze(0)

# --- API Calls ---
def get_unprocessed_image():
    try:
        resp = requests.get(f"{NGROK_BASE_URL}/api/ai/unprocessed")
        if resp.status_code == 200:
            return resp.json()
    except:
        return None
    return None

def submit_ai_result(id_ai, hasil_ai):
    try:
        data = {"hasil_ai": hasil_ai}
        requests.post(f"{NGROK_BASE_URL}/api/ai/submit_result/{id_ai}", json=data)
    except:
        pass

# --- Prediksi ---
def predict_with_model(model, tensor_img):
    with torch.no_grad():
        logits = model(tensor_img)
        probs = F.softmax(logits, dim=1)[0]
        conf, pred_class = torch.max(probs, 0)

    cls = idx2cls[int(pred_class)]
    prob = conf.item()
    info = solusi_dict.get(cls, {})
    penyebab = info.get("Penyebab", "-")
    gejala = info.get("Gejala", "-")
    solusi = info.get("Solusi", "-")

    result_text = (
        f"Klasifikasi: {cls} ({prob*100:.2f}%)\n\n"
        f"Penyebab: {penyebab}\n"
        f"Gejala: {gejala}\n"
        f"Solusi:\n{solusi}"
    )
    return cls, result_text

# --- Loop Proses Otomatis ---
placeholder = st.empty()
with placeholder.container():
    while True:
        data = get_unprocessed_image()
        if data:
            try:
                img_resp = requests.get(data["image_url"])
                if img_resp.status_code == 200:
                    tensor_img = preprocess_image(img_resp.content)
                    cls, result = predict_with_model(model, tensor_img)
                    submit_ai_result(data["id_ai"], result)
                    st.image(Image.open(BytesIO(img_resp.content)), caption=f"Gambar ID {data['id_ai']}")
                    st.text_area("Prediksi AI:", value=result, height=200)
                    time.sleep(REFRESH_INTERVAL)
                    st.rerun()
            except Exception as e:
                st.error(f"Gagal proses: {e}")
        else:
            st.info("Tidak ada gambar yang belum diproses.")
            time.sleep(REFRESH_INTERVAL)
            st.rerun()
