import numpy as np
import cv2

from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
from skimage.feature import graycomatrix, graycoprops
from skimage.morphology import remove_small_objects


# =====================================================
# SEGMENTATION
# =====================================================

def segment_cell(gray):

    thresh = threshold_otsu(gray)

    mask = gray < thresh

    mask = remove_small_objects(
        mask,
        min_size=300
    )

    return mask


def segment_nucleus(saturation_img):

    _, mask = cv2.threshold(
        saturation_img,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = np.ones((3, 3), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return mask > 0


# =====================================================
# GEOMETRY
# =====================================================

def largest_region(mask):

    lbl = label(mask)

    props = regionprops(lbl)

    if len(props) == 0:
        return None

    return max(props, key=lambda p: p.area)


def circularity(area, perimeter):

    if perimeter <= 0:
        return 0

    return (4 * np.pi * area) / (perimeter ** 2)


# =====================================================
# FEATURE EXTRACTION
# =====================================================

def extract_features_and_explain(
        original_image,
        cam,
        pred_label,
        confidence,
        img_size=224
):

    img = np.array(
        original_image.resize(
            (img_size, img_size)
        )
    )

    if img.shape[-1] == 4:
        img = img[:, :, :3]

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2GRAY
    )
    
    channel_g = img[:, :, 1]
    
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    saturation = img_hsv[:, :, 1]

    # ==========================================
    # Segmentation
    # ==========================================

    cell_mask_raw = segment_cell(channel_g)
    raw_nucleus_mask = segment_nucleus(saturation)

    # Lọc tế bào dựa trên điểm số Grad-CAM
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cell_mask_raw.astype(np.uint8))
    
    cell_mask = np.zeros_like(cell_mask_raw)
    best_label = 0
    max_cam_score = -1

    for i in range(1, num_labels):
        component_mask = (labels == i)
        mean_cam_val = np.mean(cam[component_mask])
        if mean_cam_val > max_cam_score:
            max_cam_score = mean_cam_val
            best_label = i

    if best_label > 0:
        cell_mask = (labels == best_label)
        nucleus_mask = raw_nucleus_mask & cell_mask
    else:
        cell_mask = cell_mask_raw
        nucleus_mask = raw_nucleus_mask & cell_mask

    cell_region = largest_region(cell_mask)

    nuc_region = largest_region(
        nucleus_mask
    )

    # ==========================================
    # Cell Features
    # ==========================================

    if cell_region:

        cell_area = float(
            cell_region.area
        )

        cell_perimeter = float(
            cell_region.perimeter
        )

        cell_solidity = float(
            cell_region.solidity
        )

        cell_eccentricity = float(
            cell_region.eccentricity
        )

        cell_circularity = float(
            circularity(
                cell_area,
                cell_perimeter
            )
        )

    else:

        cell_area = 0
        cell_perimeter = 0
        cell_solidity = 0
        cell_eccentricity = 0
        cell_circularity = 0

    # ==========================================
    # Nucleus Features
    # ==========================================

    if nuc_region:

        nucleus_area = float(
            nuc_region.area
        )

        nucleus_perimeter = float(
            nuc_region.perimeter
        )

        nucleus_circularity = float(
            circularity(
                nucleus_area,
                nucleus_perimeter
            )
        )

    else:

        nucleus_area = 0
        nucleus_perimeter = 0
        nucleus_circularity = 0

    nucleus_ratio = (
        min(1.0, nucleus_area / cell_area)
        if cell_area > 0 else 0
    )

    # ==========================================
    # Intensity Features
    # ==========================================

    roi_pixels = gray[cell_mask]

    if len(roi_pixels) > 0:

        mean_intensity = float(
            np.mean(roi_pixels)
        )

        std_intensity = float(
            np.std(roi_pixels)
        )

    else:

        mean_intensity = 0
        std_intensity = 0

    # ==========================================
    # RGB Features
    # ==========================================

    roi_rgb = img[cell_mask]

    if len(roi_rgb) > 0:

        mean_rgb = np.mean(
            roi_rgb,
            axis=0
        )

        std_rgb = np.std(
            roi_rgb,
            axis=0
        )

    else:

        mean_rgb = [0, 0, 0]
        std_rgb = [0, 0, 0]

    # ==========================================
    # Texture Features
    # ==========================================

    if cell_region:

        minr, minc, maxr, maxc = (
            cell_region.bbox
        )

        crop_gray = gray[
            minr:maxr,
            minc:maxc
        ].copy()

        crop_mask = cell_mask[
            minr:maxr,
            minc:maxc
        ]

        crop_gray[
            ~crop_mask
        ] = 0

        glcm = graycomatrix(
            crop_gray,
            distances=[1],
            angles=[0],
            levels=256,
            symmetric=True,
            normed=True
        )

        contrast = float(
            graycoprops(
                glcm,
                'contrast'
            )[0, 0]
        )

        correlation = float(
            graycoprops(
                glcm,
                'correlation'
            )[0, 0]
        )

        homogeneity = float(
            graycoprops(
                glcm,
                'homogeneity'
            )[0, 0]
        )

        energy = float(
            graycoprops(
                glcm,
                'energy'
            )[0, 0]
        )

    else:

        contrast = 0
        correlation = 0
        homogeneity = 0
        energy = 0

    # ==========================================
    # GradCAM Features
    # ==========================================

    cam_mask = cam > 0.5

    attention_area = int(
        np.sum(cam_mask)
    )

    attention_coverage = float(
        attention_area /
        (cell_area + 1e-6)
    )

    attention_peak = float(
        np.max(cam)
    )

# ==========================================
    # Evidence (Phân tích biện giải dựa trên Nhãn Dự Đoán)
    # ==========================================
    evidence = []

    # --- 1. BIỆN GIẢI HÌNH THÁI DỰA TRÊN TỪNG LOẠI TẾ BÀO ---
    if pred_label == 'seg_neutrophil':
        evidence.append("[Đặc trưng dòng Bạch cầu đoạn trung tính (seg_neutrophil)]:")
        if nucleus_ratio > 0 and nucleus_circularity < 0.6:
            evidence.append(f" - Nhân tế bào phân thùy rõ rệt với độ tròn thấp ({nucleus_circularity:.2f}), phù hợp hình thái thắt đoạn đặc trưng.")
        else:
            evidence.append(f" - Cảnh báo: Nhân có độ tròn khá cao ({nucleus_circularity:.2f}), chưa thể hiện rõ cấu trúc chia đoạn.")
        if contrast > 100 or energy < 0.15:
            evidence.append(f" - Độ tương phản GLCM cao ({contrast:.2f}) và năng lượng kết cấu thấp ({energy:.3f}), minh chứng cho sự xuất hiện của các hạt mịn dày đặc trong bào tương.")

    elif pred_label == 'basophil':
        evidence.append("[Đặc trưng dòng Bạch cầu ưa kiềm (basophil)]:")
        if contrast > 130 or homogeneity < 0.35:
            evidence.append(f" - Kết cấu có tương phản GLCM cực cao ({contrast:.2f}) và độ đồng nhất thấp ({homogeneity:.2f}).")
        if std_intensity > 35: # Ngưỡng mẫu, có thể tinh chỉnh dựa trên thực tế dữ liệu của bạn
            evidence.append(f" - Độ lệch chuẩn độ sáng rất lớn ({std_intensity:.2f}), phản ánh bề mặt bị nhiễu loạn mạnh bởi các hạt kiềm tính thô to che lấp nhân.")

    elif pred_label == 'monocyte':
        evidence.append("[Đặc trưng dòng Bạch cầu Monocyte]:")
        if cell_area > 4000:
            evidence.append(f" - Diện tích tế bào lớn ({cell_area:.0f} px), phù hợp với kích thước đại thực bào đầu dòng.")
        if cell_solidity < 0.92: # Viền nhấp nhô do chân giả
            evidence.append(f" - Độ đầy đặn viền thấp ({cell_solidity:.2f}) và độ thuôn dài ({cell_eccentricity:.2f}) rõ rệt, thể hiện màng tế bào dạng giả túc (amoeboid) không đều.")
        if nucleus_ratio > 0 and nucleus_circularity < 0.58:
            evidence.append(f" - Nhân gập góc, nếp gấp sâu với độ tròn thấp ({nucleus_circularity:.2f}), hỗ trợ dạng nhân hình hạt đậu hoặc móng ngựa.")

    elif pred_label == 'myeloblast':
        evidence.append("[Đặc trưng dòng Nguyên tủy bào - Tế bào non (myeloblast)]:")
        if nucleus_ratio > 0.65:
            evidence.append(f" - Tỷ lệ nhân/tế bào (N/C) rất lớn ({nucleus_ratio:.2f}), nhân khổng lồ chiếm gần trọn tế bào - đặc trưng cốt lõi của dạng Blast.")
        if homogeneity > 0.4 or energy > 0.2:
            evidence.append(f" - Năng lượng kết cấu ({energy:.3f}) và độ đồng nhất ({homogeneity:.2f}) cao, khẳng định chất nhiễm sắc mịn màng, bào tương thuần nhất và chưa biệt hóa tạo hạt.")
        if cell_solidity > 0.95:
            evidence.append(f" - Viền tế bào rất lồi và căng đều (Solidity: {cell_solidity:.2f}), phù hợp với hình thái tròn nguyên bản của tế bào non đầu dòng.")

    elif pred_label == 'erythroblast':
        evidence.append("[Đặc trưng dòng Hồng cầu có nhân (erythroblast)]:")
        if nucleus_ratio > 0 and nucleus_circularity > 0.8:
            evidence.append(f" - Nhân tế bào có độ tròn tiệm cận tuyệt đối ({nucleus_circularity:.2f}) với cấu trúc đặc vô định hình.")
        if cell_circularity > 0.8 and cell_eccentricity < 0.4:
            evidence.append(f" - Toàn bộ cấu trúc cell có độ tròn cao ({cell_circularity:.2f}) và độ lệch tâm rất thấp ({cell_eccentricity:.2f}), khẳng định dạng hình cầu trơn láng.")

    # --- 2. ĐÁNH GIÁ VÙNG CHÚ Ý CỦA MÔ HÌNH (GRAD-CAM) ---
    evidence.append("[Phân tích vùng chú ý Grad-CAM]:")
    if attention_coverage > 0.6:
        evidence.append(f" - Bản đồ nhiệt bao phủ diện rộng ({attention_coverage*100:.1f}%), chứng tỏ mô hình học được đặc trưng tổng thể tổng hòa từ cả nhân và bào tương.")
    elif 0.3 <= attention_coverage <= 0.6:
        evidence.append(f" - Bản đồ nhiệt tập trung cục bộ ({attention_coverage*100:.1f}%), mô hình đang dồn trọng số vào vùng biên nhân hoặc vùng hạt đặc hiệu.")
    else:
        evidence.append(f" - Mô hình chỉ tập trung vào một chấm nhỏ đặc dị ({attention_coverage*100:.1f}%), cần đề phòng trường hợp mô hình bị nhiễu do dị vật hoặc bắt sai vùng.")

    explanation = (
        f"Dựa trên các đặc trưng tính toán được, mô hình đưa ra dự đoán {pred_label} "
        f"với độ tin cậy {confidence*100:.1f}%. "
        f"Các bằng chứng hình thái học hỗ trợ bao gồm: " + " ".join(evidence)
    )

    return {

        "prediction": pred_label,

        "confidence": float(confidence),

        "morphology": {

            "Cell Area": cell_area,
            "Cell Perimeter": cell_perimeter,
            "Cell Circularity": cell_circularity,
            "Cell Solidity": cell_solidity,
            "Cell Eccentricity": cell_eccentricity,

            "Nucleus Area": nucleus_area,
            "Nucleus Perimeter": nucleus_perimeter,
            "Nucleus Circularity": nucleus_circularity,
            "Nucleus-to-Cell Ratio": nucleus_ratio
        },

        "texture": {

            "GLCM Contrast": contrast,
            "GLCM Correlation": correlation,
            "GLCM Homogeneity": homogeneity,
            "GLCM Energy": energy
        },

        "color": {

            "Mean R": float(mean_rgb[0]),
            "Mean G": float(mean_rgb[1]),
            "Mean B": float(mean_rgb[2]),

            "Std R": float(std_rgb[0]),
            "Std G": float(std_rgb[1]),
            "Std B": float(std_rgb[2])
        },

        "attention": {

            "Attention Area": attention_area,
            "Attention Coverage": attention_coverage,
            "Attention Peak": attention_peak
        },

        "intensity": {

            "Mean Intensity": mean_intensity,
            "Std Intensity": std_intensity
        },

        "evidence": evidence,

        "explanation": explanation
    }

