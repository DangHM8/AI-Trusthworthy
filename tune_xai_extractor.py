import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects
import sys
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# =====================================================
# THAM SỐ CÓ THỂ TINH CHỈNH (TUNE HERE)
# =====================================================
CELL_MIN_SIZE = 300
NUCLEUS_MORPH_KERNEL_SIZE = 3
IMG_SIZE = 224
MODEL_PATH = r"E:\AI_hust_course\Period 2\(T8.3)Trustworthy AI\thuc hanh\train\best_model.pth"
CLASS_NAMES = ['basophil', 'erythroblast', 'monocyte', 'myeloblast', 'seg_neutrophil']

# =====================================================
# MÔ HÌNH VÀ GRAD-CAM
# =====================================================
def get_transforms():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, input_tensor, target_class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if target_class_idx is None:
            target_class_idx = torch.argmax(output, dim=1).item()
        score = output[0, target_class_idx]
        score.backward()
        gradients = self.gradients.data.cpu().numpy()[0]
        activations = self.activations.data.cpu().numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, CLASS_NAMES[target_class_idx]

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    )
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        return model, device
    return None, device

# =====================================================
# PHÂN VÙNG (SEGMENTATION)
# =====================================================
def segment_cell(gray):
    thresh = threshold_otsu(gray)
    mask = gray < thresh
    mask = remove_small_objects(mask, min_size=CELL_MIN_SIZE)
    return mask

def segment_nucleus(saturation_img):
    _, mask = cv2.threshold(saturation_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((NUCLEUS_MORPH_KERNEL_SIZE, NUCLEUS_MORPH_KERNEL_SIZE), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask > 0

def largest_region(mask):
    lbl = label(mask)
    props = regionprops(lbl)
    if len(props) == 0:
        return None
    return max(props, key=lambda p: p.area)

def process_and_visualize(image_path):
    print(f"Đang xử lý ảnh: {image_path}")
    if not os.path.exists(image_path):
        print("Lỗi: Không tìm thấy file ảnh!")
        return

    img = cv2.imread(image_path)
    if img is None:
        print("Lỗi: Không thể đọc ảnh bằng OpenCV!")
        return
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    # 1. Tạo Grad-CAM để lấy vùng chú ý (ROI) trước tiên
    model, device = load_model()
    cam = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    pred_class = "Unknown"
    
    if model:
        grad_cam = GradCAM(model, model.layer4)
        transform = get_transforms()
        pil_img = Image.fromarray(img_rgb)
        input_tensor = transform(pil_img).unsqueeze(0).to(device)
        cam, pred_class = grad_cam.generate_cam(input_tensor)
        print(f"Mô hình dự đoán: {pred_class}")
        
    # 2. Tạo ảnh fade (cho mục đích trực quan hóa, không dùng cho phân vùng)
    cam_3d = np.expand_dims(cam, axis=-1)
    faded_img = (img_resized.astype(np.float32) * cam_3d + 255 * (1 - cam_3d)).astype(np.uint8)
    
    # Lấy kênh Red (gray_red) TỪ ẢNH GỐC để tìm cell
    gray_red = img_resized[:, :, 1]
    
    # Lấy kênh Saturation TỪ ẢNH GỐC để tìm nucleus
    img_hsv = cv2.cvtColor(img_resized, cv2.COLOR_RGB2HSV)
    saturation = img_hsv[:, :, 1]
    
    # 3. Phân vùng ban đầu (từ ảnh gốc)
    cell_mask_raw = segment_cell(gray_red)
    raw_nucleus_mask = segment_nucleus(saturation)
    
    # --- ĐOẠN LỌC ĐỂ CHỈ GIỮ LẠI TẾ BÀO MỤC TIÊU ---
    # Tìm các thành phần liên thông từ mặt nạ thô
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cell_mask_raw.astype(np.uint8))
    
    cell_mask = np.zeros_like(cell_mask_raw)
    best_label = 0
    max_cam_score = -1

    # Lướt qua từng object tìm được để xem cái nào trùng với Grad-CAM nhất
    for i in range(1, num_labels):  # Bỏ qua nhãn 0 là nền
        component_mask = (labels == i)
        # Tính giá trị Grad-CAM trung bình bên trong object này
        mean_cam_val = np.mean(cam[component_mask])
        
        if mean_cam_val > max_cam_score:
            max_cam_score = mean_cam_val
            best_label = i

    # Cập nhật lại cell_mask và nucleus_mask chuẩn chỉ cho 1 cell duy nhất
    if best_label > 0:
        cell_mask = (labels == best_label)
        nucleus_mask = raw_nucleus_mask & cell_mask
    else:
        cell_mask = cell_mask_raw
        nucleus_mask = raw_nucleus_mask & cell_mask
    # ------------------------------------------------
    
    # 3. Trực quan hóa
    fig, axs = plt.subplots(2, 4, figsize=(18, 9))
    fig.canvas.manager.set_window_title('Tuning XAI Extractor & Grad-CAM Fade')
    axs = axs.flatten()
    
    axs[0].imshow(img_resized)
    axs[0].set_title(f"Ảnh gốc ({IMG_SIZE}x{IMG_SIZE})")
    axs[0].axis('off')
    
    axs[1].imshow(saturation, cmap='gray')
    axs[1].set_title("Ảnh HSV (Kênh Saturation)")
    axs[1].axis('off')
    
    axs[2].imshow(gray_red, cmap='gray')
    axs[2].set_title("Ảnh Gray Green (Kênh G)")
    axs[2].axis('off')
    
    axs[3].imshow(nucleus_mask, cmap='gray')
    axs[3].set_title("Nucleus Mask (Nhân tế bào)")
    axs[3].axis('off')
    
    axs[4].imshow(cell_mask, cmap='gray')
    axs[4].set_title("Cell Mask (Tế bào)")
    axs[4].axis('off')
    
    # Heatmap hiển thị màu cho Grad-CAM
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    axs[5].imshow(heatmap)
    axs[5].set_title(f"Grad-CAM Heatmap\n(Dự đoán: {pred_class})")
    axs[5].axis('off')
    
    # Overlay hiển thị viền trên ảnh gốc
    overlay = img_resized.copy()
    cell_contours, _ = cv2.findContours(cell_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, cell_contours, -1, (0, 255, 0), 2)
    nucleus_contours, _ = cv2.findContours(nucleus_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, nucleus_contours, -1, (255, 0, 0), 2)
    
    axs[6].imshow(overlay)
    axs[6].set_title("Overlay Contours (Ảnh gốc)")
    axs[6].axis('off')
    
    # Ẩn khung hình cuối cùng
    axs[7].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import random
    import glob
    dataset_dir = r"E:\AI_hust_course\Period 2\(T8.3)Trustworthy AI\thuc hanh\datasets\images_full"
    all_images = glob.glob(os.path.join(dataset_dir, "*", "*.jpg"))
    default_image = random.choice(all_images) if all_images else r"E:\AI_hust_course\Period 2\(T8.3)Trustworthy AI\thuc hanh\datasets\images_full\erythroblast\img_4884.jpg"
    image_path = sys.argv[1] if len(sys.argv) > 1 else default_image
    process_and_visualize(image_path)
