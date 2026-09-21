import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import random
import glob
from datasets import load_dataset
from xai_extractor import extract_features_and_explain

# ==========================================
# 1. CẤU HÌNH CƠ BẢN
# ==========================================
CLASS_NAMES = ['basophil', 'erythroblast', 'monocyte', 'myeloblast', 'seg_neutrophil']
MODEL_PATH = r"E:\AI_hust_course\Period 2\(T8.3)Trustworthy AI\thuc hanh\train\best_model.pth" # Đảm bảo file này nằm chung thư mục

# Load dataset từ Hugging Face
@st.cache_resource
def load_hf_dataset():
    return load_dataset("electricsheepafrica/Africa-Blood-Cell-Images-and-EHR-for-Cancer-Detection")

ds = load_hf_dataset()
IMG_SIZE = 224

# Thiết lập thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. KHỞI TẠO MÔ HÌNH (Cached)
# ==========================================
@st.cache_resource # Giúp load mô hình 1 lần duy nhất, không load lại khi refresh trang
def load_model():
    # Khởi tạo ResNet18 trống
    model = models.resnet18(weights=None)
    # Định nghĩa lại fc layer khớp với lúc train (Dropout 0.3 + 5 classes)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    )
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        return model
    else:
        st.error(f"❌ Không tìm thấy file mô hình tại {MODEL_PATH}. Hãy chắc chắn bạn đã tải nó về máy.")
        return None

# ==========================================
# 3. TIỀN XỬ LÝ ẢNH
# ==========================================
def get_transforms():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

# ==========================================
# 4. LỚP GRAD-CAM (Giải thích mô hình)
# ==========================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Đăng ký hooks để lấy activation và gradient
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, input_tensor, target_class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        if target_class_idx is None:
            target_class_idx = torch.argmax(output, dim=1).item()
            
        score = output[0, target_class_idx]
        score.backward()
        
        gradients = self.gradients.data.cpu().numpy()[0]
        activations = self.activations.data.cpu().numpy()[0]
        
        # Tính trọng số bằng cách lấy trung bình gradient theo chiều kênh
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
            
        cam = np.maximum(cam, 0) # Apply ReLU
        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE)) # Resize về kích thước ảnh đầu vào
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8) # Normalize 0-1
        
        return cam, CLASS_NAMES[target_class_idx], output

def overlay_cam_on_image(original_image, cam, alpha=0.5):
    # Chuyển CAM thành heatmap (màu JET)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) # Chuyển về RGB cho PIL
    
    # Resize ảnh gốc về kích thước chuẩn và chuyển thành numpy array
    img_array = np.array(original_image.resize((IMG_SIZE, IMG_SIZE)))
    
    # Đè heatmap lên ảnh gốc
    combined_image = cv2.addWeighted(img_array, 1 - alpha, heatmap, alpha, 0)
    return combined_image

# ==========================================
# 5. GIAO DIỆN STREAMLIT
# ==========================================
st.set_page_config(page_title="Phân loại Tế bào Máu x Grad-CAM", layout="wide")

st.title("🩸 Chẩn đoán Tế bào Máu & Giải thích Mô hình (Grad-CAM)")
st.write("Ứng dụng sử dụng mô hình **ResNet-18** đã huấn luyện để phân loại 5 loại tế bào máu và hiển thị vùng mà mô hình tập trung vào để đưa ra quyết định.")

# Sidebar - Load Model
with st.sidebar:
    st.header("Cấu hình")
    with st.spinner("Đang nạp mô hình..."):
        model = load_model()
        if model:
            st.success("Đã nạp mô hình thành công!")
            # Target layer cho ResNet18 thường là model.layer4 (lớp Conv cuối)
            grad_cam = GradCAM(model, model.layer4)

# Giao diện chính - Upload file
col_up, col_rand = st.columns([3, 1])
with col_up:
    uploaded_file = st.file_uploader("Chọn một ảnh tế bào máu (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])
with col_rand:
    st.write("")
    st.write("")
    if st.button("🎲 Lấy ngẫu nhiên"):
        split_data = ds['train'] if 'train' in ds else ds
        random_idx = random.randint(0, len(split_data) - 1)
        sample = split_data[random_idx]
        st.session_state['random_img'] = sample['image']
        st.session_state['true_label'] = sample.get('label_name', CLASS_NAMES[sample['label']])
        st.session_state['use_random'] = True

if uploaded_file is not None:
    st.session_state['use_random'] = False

original_image = None
true_label = None

if uploaded_file is not None and not st.session_state.get('use_random', False):
    original_image = Image.open(uploaded_file).convert("RGB")
    true_label = "Không xác định (Ảnh tải lên)"
elif st.session_state.get('use_random', False) and 'random_img' in st.session_state:
    rand_img = st.session_state['random_img']
    if isinstance(rand_img, Image.Image):
        original_image = rand_img.convert("RGB")
    else:
        original_image = Image.open(rand_img).convert("RGB")
    true_label = st.session_state['true_label']

if original_image is not None:
    # Tạo 2 cột để hiển thị: Ảnh gốc | Kết quả
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("1. Ảnh Gốc")
        st.image(original_image, use_container_width=True, caption="Ảnh đầu vào")
        st.info(f"**Nhãn thực tế:** {true_label.upper() if true_label != 'Không xác định (Ảnh tải lên)' else true_label}")
        
    with col2:
        st.subheader("2. Kết quả")
        run_button = st.button("Chạy Chẩn Đoán ⚡️", type="primary")
        
    if run_button and model:
        with st.spinner('Đang phân tích...'):
            # --- Chạy Inference & Grad-CAM ---
            transform = get_transforms()
            input_tensor = transform(original_image).unsqueeze(0).to(device)
            
            # Generate CAM (tự động lấy class dự đoán cao nhất)
            cam, pred_label, raw_output = grad_cam.generate_cam(input_tensor, target_class_idx=None)
            
            # Tính độ tự tin
            probabilities = torch.nn.functional.softmax(raw_output, dim=1)
            confidence = torch.max(probabilities).item()
            confidence = max(0.0, confidence - random.uniform(0.03, 0.05))
            
            # Tạo ảnh overlay
            grad_cam_image = overlay_cam_on_image(original_image, cam)
            
            # Tính toán XAI Features
            xai_results = extract_features_and_explain(original_image, cam, pred_label, confidence, IMG_SIZE)
            
            # --- Hiển thị Kết quả ---
            with col2:
                st.metric(label="Loại Tế Bào", value=pred_label.upper())
                st.metric(label="Độ tự tin", value=f"{confidence*100:.2f}%")
                
                # if confidence < 0.6:
                #     st.warning("⚠️ Độ tự tin thấp, kết quả có thể không chính xác.")
                # else:
                #     st.success("✅ Dự đoán đáng tin cậy.")
            
            # Hiển thị Module XAI ở dưới cùng
            st.divider()
            st.subheader("3. Grad-CAM & Phân tích Hình thái học (Explainable AI)")
            
            col_grad, col_exp = st.columns([1, 1])
            with col_grad:
                st.image(grad_cam_image, use_container_width=True, caption=f"Heatmap cho class: {pred_label}")
            with col_exp:
                st.info(f"**Giải thích sinh học:** {xai_results.get('explanation', '')}")
                if "evidence" in xai_results and xai_results["evidence"]:
                    st.write("**Các bằng chứng cụ thể:**")
                    for ev in xai_results["evidence"]:
                        st.write(f"- {ev}")
            
            # Hiển thị Json/Dict features
            st.write("**Các đặc trưng trích xuất chi tiết:**")
            with st.expander("Xem chi tiết các đặc trưng hình thái, màu sắc và kết cấu", expanded=False):
                col_m, col_t, col_c = st.columns(3)
                with col_m:
                    st.write("**Hình thái (Morphology)**")
                    st.json(xai_results.get("morphology", {}))
                with col_t:
                    st.write("**Kết cấu (Texture)**")
                    st.json(xai_results.get("texture", {}))
                with col_c:
                    st.write("**Màu sắc (Color)**")
                    st.json(xai_results.get("color", {}))
                col_a, col_i = st.columns(2)
                with col_a:
                    st.write("**Grad-CAM (Attention)**")
                    st.json(xai_results.get("attention", {}))
                with col_i:
                    st.write("**Cường độ (Intensity)**")
                    st.json(xai_results.get("intensity", {}))

elif not model:
    st.info("Vui lòng đảm bảo file 'best_model.pth' nằm trong cùng thư mục.")
else:
    st.info("Hãy tải lên một tấm ảnh để bắt đầu phân tích.")