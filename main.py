import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import json

# ==============================
# 1. Load Model PyTorch
# ==============================
@st.cache_resource
def load_torch_model():
    ckpt = torch.load("best_model(1).pth", map_location="cpu")
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

model, idx2cls = load_torch_model()

# ==============================
# 2. Preprocessing
# ==============================
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

infer_tfms = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.15)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# ==============================
# 3. Load Solusi Penyakit
# ==============================
with open("solusi_penyakit_tanaman_cabai_tomat.json") as f:
    solusi_data = json.load(f)
solusi = {item["Penyakit"]: item for item in solusi_data}

# ==============================
# 4. OOD Detection Function
# ==============================
def compute_energy(logits: torch.Tensor) -> float:
    return -torch.logsumexp(logits, dim=1).item()

# ==============================
# 5. Streamlit UI (Versi Clean)
# ==============================
st.title("🌱 AI Deteksi Penyakit Tanaman Cabai (BETA)")

uploaded = st.file_uploader("📤 Upload gambar daun cabai...", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Gambar Upload", use_column_width=True)

    # Preprocessing
    x = infer_tfms(img).unsqueeze(0)

    # Prediksi
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]
        conf, pred_class = torch.max(probs, 0)

    cls = idx2cls[int(pred_class)]

    # Logika cek unknown
    if cls.lower() == "unknown":
        st.error("🚫 Gambar yang diupload **bukan daun cabai**. Silakan upload ulang dengan foto daun cabai yang jelas.")
    else:
        st.success(f"✅ Prediksi utama: **{cls}** ({conf.item():.2%})")

        # Probabilitas semua kelas
        st.subheader("📊 Probabilitas Tiap Kelas")
        prob_dict = {idx2cls[i]: float(probs[i]) for i in range(len(probs))}
        st.bar_chart(prob_dict)

        # Info penyakit & solusi
        info = solusi.get(cls)
        if info:
            st.subheader("📝 Informasi Penyakit & Solusi")
            st.write(f"**Penyebab:** {info['Penyebab']}")
            st.write(f"**Gejala:** {info['Gejala']}")
            st.write(f"**Solusi:** {info['Solusi']}")
        else:
            st.warning("⚠️ Solusi tidak tersedia untuk penyakit ini.")
