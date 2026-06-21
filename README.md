# Multimodal Cancer Detection Training

Tài liệu này tổng hợp thông tin về quá trình huấn luyện mô hình phát hiện ung thư từ dữ liệu đa phương thức (Multimodal).

## 1. Dữ liệu (Dataset)

- **Nguồn**: `electricsheepafrica/Africa-Blood-Cell-Images-and-EHR-for-Cancer-Detection` (Hugging Face).
- **Thành phần**:
  - **Ảnh (Images)**: Hình ảnh tế bào máu (224x224).
  - **EHR (Electronic Health Records)**: 10 chỉ số xét nghiệm (Tuổi, WBC, Hemoglobin, Platelets, Neutrophils%, ...).
- **Phân chia dữ liệu**:
  - Tập Train: 4000 mẫu.
  - Tập Valid: 500 mẫu.
  - Tập Test: 500 mẫu.

## 2. Tiền xử lý dữ liệu

- **Ảnh**:
  - Resize về kích thước `224x224`.
  - Augmentation: Lật ngang ngẫu nhiên, xoay ngẫu nhiên (20 độ).
  - Chuẩn hóa theo ImageNet mean & std.
- **EHR**: Sử dụng `StandardScaler` để chuẩn hóa các chỉ số về phân phối chuẩn (fit trên tập Train).

## 3. Kiến trúc Mô hình (Architecture)

Mô hình `MultimodalModel` kết hợp hai nhánh:

- **Nhánh Thị giác (Vision Branch)**: Sử dụng **ResNet18** (pre-trained) làm đặc trưng ảnh, output là vector 128 chiều.
- **Nhánh EHR (EHR Branch)**: Mạng MLP đơn giản (Linear -> ReLU) biến đổi 10 đặc trưng EHR thành vector 32 chiều.
- **Bộ phân loại (Classifier)**: Nối (concatenate) hai vector đặc trưng và đi qua các lớp Fully Connected để dự đoán 5 lớp kết quả.

## 4. Quá trình Huấn luyện

- **Loss Function**: `CrossEntropyLoss`.
- **Optimizer**: `Adam` (learning rate = 0.0001).
- **Epochs**: 10 epochs.
- **Kết quả lưu trữ**:
  - Mô hình tốt nhất: `best_trustworthy_model.pth` (lưu cả state_dict và scaler).
  - Lịch sử huấn luyện: `train_history.json`.
  - Đồ thị loss/acc: `learning_curves.png`.

## 5. Giải thích mô hình (XAI)

Sử dụng kỹ thuật **Grad-CAM** để tạo Heatmap trên ảnh tế bào máu, giúp xác định các vùng quan trọng mà mô hình tập trung vào để đưa ra quyết định chẩn đoán.

---

_Ghi chú: File notebook thực hiện quá trình này nằm tại `only_image_no_maskv2.ipynb`._
