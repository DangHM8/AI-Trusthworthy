import cv2
import numpy as np
import torch
from torchvision import models, transforms
from skimage.measure import label, regionprops
from skimage.feature import graycomatrix, graycoprops
import sys
sys.stdout.reconfigure(encoding='utf-8')

# =====================================================================
# 1. BỘ TRÍCH XUẤT MASK GRAD-CAM
# =====================================================================
class GradCAMExtractor:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_mask(self, input_tensor, threshold=0.4):
        self.model.zero_grad()
        output = self.model(input_tensor)
        class_idx = output.argmax(dim=1).item()
        
        output[0, class_idx].backward()
        grads = self.gradients.cpu().data.numpy()[0]
        acts = self.activations.cpu().data.numpy()[0]
        
        weights = np.mean(grads, axis=(1, 2))
        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i]

        cam = np.maximum(cam, 0)
        if np.max(cam) != 0:
            cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
            
        return (cam > threshold).astype(np.uint8), class_idx

# =====================================================================
# 2. PIPELINE TÍNH TOÁN ĐẶC TRƯNG & TẠO PROMPT
# =====================================================================
def generate_chatbot_prompt(image_path):
    # --- Bước 1: Đọc và tiền xử lý ảnh ---
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Lỗi: Không tìm thấy hoặc không đọc được ảnh tại đường dẫn: {image_path}")
        return
        
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w, _ = img_bgr.shape

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img_rgb).unsqueeze(0)

    # --- Bước 2: Chạy ResNet50 + Grad-CAM ---
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.eval()
    
    cam_extractor = GradCAMExtractor(model, model.layer4[-1])
    binary_mask, pred_class = cam_extractor.generate_mask(input_tensor, threshold=0.4)
    binary_mask = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # --- Bước 3: Tính đặc trưng hình thái (scikit-image) ---
    labeled_mask = label(binary_mask)
    props = regionprops(labeled_mask)
    
    if len(props) == 0:
        print("Grad-CAM không tìm thấy vùng kích hoạt thích hợp. Đang thử lấy toàn bộ vùng trung tâm...")
        # Tạo mask giả lập vùng tâm nếu không kích hoạt được mạng
        binary_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(binary_mask, (w//2, h//2), min(h, w)//4, 1, -1)
        props = regionprops(label(binary_mask))
        
    main_region = max(props, key=lambda x: x.area)

    area = main_region.area
    eccentricity = main_region.eccentricity
    solidity = main_region.solidity
    perimeter = main_region.perimeter
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

    # --- Bước 4: Tính đặc trưng kết cấu (GLCM) ---
    minr, minc, maxr, maxc = main_region.bbox
    cropped_gray = img_gray[minr:maxr, minc:maxc]
    
    if cropped_gray.size == 0:
        cropped_gray = img_gray

    glcm = graycomatrix(cropped_gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                        levels=256, symmetric=True, normed=True)
    
    contrast = np.mean(graycoprops(glcm, 'contrast'))
    homogeneity = np.mean(graycoprops(glcm, 'homogeneity'))
    energy = np.mean(graycoprops(glcm, 'energy'))

    # --- Bước 5: Đóng gói và in ra đoạn Prompt cho Chatbot ---
    prompt = f"""Tôi có dữ liệu trích xuất từ một ảnh tế bào máu (thuộc một trong năm loại: basophil, erythroblast, monocyte, myeloblast, seg_neutrophil) bằng hệ thống kết hợp Deep Learning và Computer Vision (ResNet + Grad-CAM + scikit-image + GLCM).

Dưới đây là các chỉ số định lượng thu được từ vùng đặc trưng:
- Lớp dự đoán sơ bộ (ResNet Class ID): {pred_class}
- Area (Diện tích vùng kích hoạt): {area} pixels
- Circularity (Độ tròn): {circularity:.4f}
- Eccentricity (Độ lệch tâm/độ thuôn dài): {eccentricity:.4f}
- Solidity (Độ đặc/độ nguyên vẹn của ranh giới): {solidity:.4f}
- GLCM Contrast (Độ tương phản bề mặt): {contrast:.4f}
- GLCM Homogeneity (Độ đồng nhất bề mặt): {homogeneity:.4f}
- GLCM Energy (Năng lượng cấu trúc mức xám): {energy:.4f}

Hãy đóng vai trò là một chuyên gia Huyết học và Chuyên gia Giải thích AI (XAI). Dựa vào các chỉ số hình thái và kết cấu định lượng ở trên, hãy biện luận chi tiết xem tế bào này chính xác là loại nào trong 5 loại kể trên. Giải thích rõ ràng mối quan hệ giữa các con số toán học này với đặc điểm hình thái sinh học thực tế của tế bào đó (ví dụ: tại sao độ tròn lại như vậy, kết cấu thô hay mịn tương ứng với thành phần nào trong tế bào)."""

    print("\n" + "="*25 + " COPY ĐOẠN DƯỚI ĐÂY VÀO CHATBOT " + "="*25)
    print(prompt)
    print("="*80 + "\n")

# =====================================================================
# THỰC THI CHƯƠNG TRÌNH
# =====================================================================
if __name__ == "__main__":
    # Thay đổi đường dẫn đến file ảnh tế bào máu bạn muốn kiểm tra ở đây
    file_anh = r"E:\AI_hust_course\Period 2\(T8.3)Trustworthy AI\thuc hanh\datasets\images_full\erythroblast\img_59.jpg"
    generate_chatbot_prompt(file_anh)