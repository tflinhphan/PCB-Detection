import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import pandas as pd


def region_growing(img_gray, seed_y, seed_x, threshold=10):
    """
    Region Growing để tách từng linh kiện
    """
    h, w = img_gray.shape
    visited = np.zeros((h, w), dtype=bool)
    mask = np.zeros((h, w), dtype=np.uint8)

    # 8 hướng lân cận
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    queue = deque([(seed_y, seed_x)])
    visited[seed_y, seed_x] = True
    mask[seed_y, seed_x] = 255
    seed_value = int(img_gray[seed_y, seed_x])

    while queue:
        y, x = queue.popleft()

        for dy, dx in directions:
            ny, nx = y + dy, x + dx

            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                visited[ny, nx] = True
                neighbor_value = int(img_gray[ny, nx])

                if abs(neighbor_value - seed_value) <= threshold:
                    mask[ny, nx] = 255
                    queue.append((ny, nx))

    return mask


def apply_region_growing(img_gray, seeds_mask):
    """
    Áp dụng Region Growing cho tất cả các seed points
    """
    regions = []

    # Tìm tất cả các seed points
    contours, _ = cv2.findContours(seeds_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        M = cv2.moments(cnt)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            # Region growing từ seed point
            region_mask = region_growing(img_gray, cy, cx, threshold=8) #sửa thre

            # Kiểm tra diện tích hợp lệ
            area = cv2.countNonZero(region_mask)
            if 200 < area < 50000: #thay đổi phù hợp kích thước
                regions.append(region_mask)

    return regions


# ========== 3. TRÍCH XUẤT ĐẶC TRƯNG ==========
def extract_features(mask, img_gray):
    """
    Trích xuất đặc trưng từ region mask
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    cnt = max(contours, key=cv2.contourArea)

    # Đặc trưng hình học
    area = cv2.contourArea(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    perimeter = cv2.arcLength(cnt, True)

    # Tính các tỷ lệ
    rect_area = w * h
    solidity = area / rect_area if rect_area > 0 else 0
    aspect_ratio = w / h if h > 0 else 0
    circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0

    # Đặc trưng màu (độ sáng trung bình)
    mean_intensity = cv2.mean(img_gray, mask=mask)[0]

    return {
        'contour': cnt,
        'area': area,
        'bbox': (x, y, w, h),
        'solidity': solidity,
        'aspect_ratio': aspect_ratio,
        'circularity': circularity,
        'mean_intensity': mean_intensity,
        'perimeter': perimeter
    }


# ========== 4. K-MEANS CLUSTERING ==========
def kmeans_detect_defects(features_list):
    """
    Sử dụng K-means để phát hiện linh kiện sai lệch
    """
    if len(features_list) < 3:
        return [0] * len(features_list), None

    # Chuẩn bị feature vectors
    feature_vectors = []
    for f in features_list:
        feature_vectors.append([
            f['area'],
            f['solidity'],
            f['aspect_ratio'],
            f['mean_intensity']
        ])

    features_array = np.array(feature_vectors, dtype=np.float32)

    # Chuẩn hóa features
    min_vals = features_array.min(axis=0)
    max_vals = features_array.max(axis=0)
    normalized = (features_array - min_vals) / (max_vals - min_vals + 1e-10)

    # K-means với K=2 (nhóm OK và nhóm DEFECT)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(
        normalized, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )

    return labels.flatten(), centers


# ========== 5. SPLIT & MERGE (BỔ SUNG) ==========
def split_merge_analysis(features_list, labels):
    """
    Split & Merge để tinh chỉnh phân loại
    """
    # Phân tích cluster statistics
    cluster_0 = [f for i, f in enumerate(features_list) if labels[i] == 0]
    cluster_1 = [f for i, f in enumerate(features_list) if labels[i] == 1]

    if not cluster_0 or not cluster_1:
        return labels

    # Tính độ lệch chuẩn của mỗi cluster
    def get_cluster_stats(cluster):
        areas = [f['area'] for f in cluster]
        solidities = [f['solidity'] for f in cluster]
        return {
            'mean_area': np.mean(areas),
            'std_area': np.std(areas),
            'mean_solidity': np.mean(solidities),
            'std_solidity': np.std(solidities)
        }

    stats_0 = get_cluster_stats(cluster_0)
    stats_1 = get_cluster_stats(cluster_1)

    # Cluster có độ lệch chuẩn thấp hơn → OK
    # Cluster có độ lệch chuẩn cao hơn → DEFECT (không đồng đều)
    ok_label = 0 if stats_0['std_area'] < stats_1['std_area'] else 1

    return ok_label, stats_0, stats_1


# ========== 6. PHÂN LOẠI DEFECT ==========
def classify_defects(features_list, labels, ok_label, img_gray):
    """
    Phân loại chi tiết: CORRECT, MISSING, MISALIGNED
    """
    results = []

    # Tính giá trị trung bình của cluster OK
    ok_features = [f for i, f in enumerate(features_list) if labels[i] == ok_label]

    if ok_features:
        avg_area = np.mean([f['area'] for f in ok_features])
        avg_intensity = np.mean([f['mean_intensity'] for f in ok_features])
    else:
        avg_area = np.mean([f['area'] for f in features_list])
        avg_intensity = np.mean([f['mean_intensity'] for f in features_list])

    for i, f in enumerate(features_list):
        if labels[i] == ok_label:
            status = "CORRECT"
            status_vn = "Đúng"
            color = (0, 255, 0)  # Xanh lá
        else:
            # Phân tích chi tiết lỗi
            area_diff = abs(f['area'] - avg_area) / avg_area
            intensity_diff = abs(f['mean_intensity'] - avg_intensity)

            if f['area'] < avg_area * 0.5:
                status = "MISSING"
                status_vn = "Thiếu"
                color = (255, 0, 0)  # Đỏ
            elif area_diff > 0.3 or f['solidity'] < 0.5:
                status = "MISALIGNED"
                status_vn = "Lệch"
                color = (0, 165, 255)  # Cam
            else:
                status = "DEFECT"
                status_vn = "Lỗi"
                color = (255, 0, 0)  # Đỏ

        results.append({
            'id': i + 1,
            'status': status,
            'status_vn': status_vn,
            'color': color,
            'area': f['area'],
            'bbox': f['bbox'],
            'solidity': f['solidity'],
            'mean_intensity': f['mean_intensity']
        })

    return results


# ========== 7. MAIN PROCESSING PIPELINE ==========
def pcb_defect_detection(img_path, show_steps=True):
    """
    Pipeline chính cho phát hiện lỗi PCB
    """
    print("=" * 60)
    print("PCB DEFECT DETECTION SYSTEM")
    print("=" * 60)

    # Đọc ảnh
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Lỗi: Không đọc được ảnh từ {img_path}")
        return None

    print(f"✓ Đã load ảnh: {img.shape}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Hiển thị ảnh gốc
    if show_steps:
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Ảnh PCB Gốc", fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 8))
        plt.imshow(gray, cmap='gray')
        plt.title("Ảnh Grayscale", fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    # ===== BƯỚC 1: PHÂN VÙNG =====
    print("\n[1/5] Phân vùng linh kiện (Adaptive Threshold + Edge Detection)...")

    # Làm mờ
    blurred = cv2.GaussianBlur(gray, (7, 7), 0) # sửa ksize
    if show_steps:
        plt.figure(figsize=(10, 8))
        plt.imshow(blurred, cmap='gray')
        plt.title("Bước 1.1: Gaussian Blur", fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    # Adaptive Threshold
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 25 # block C Sửa
    )
    if show_steps:
        plt.figure(figsize=(10, 8))
        plt.imshow(thresh, cmap='gray')
        plt.title("Bước 1.2: Adaptive Threshold", fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    # Edge Detection
    edges = cv2.Canny(blurred, 50, 150) #Sửa
    if show_steps:
        plt.figure(figsize=(10, 8))
        plt.imshow(edges, cmap='gray')
        plt.title("Bước 1.3: Canny Edge Detection", fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    # Kết hợp
    combined = cv2.bitwise_or(thresh, edges)
    combined = cv2.medianBlur(combined, 9) # Sửa
    if show_steps:
        plt.figure(figsize=(10, 8))
        plt.imshow(combined, cmap='gray')
        plt.title("Bước 1.4: Kết hợp Threshold + Edges", fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    # Morphology Close
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) # Sửa ksize
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close, iterations=2) # Sửa iter
    if show_steps:
        plt.figure(figsize=(10, 8))
        plt.imshow(closed, cmap='gray')
        plt.title("Bước 1.5: Morphology Closing", fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    # Morphology Open
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    segmented_mask = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open, iterations=1)
    if show_steps:
        plt.figure(figsize=(10, 8))
        plt.imshow(segmented_mask, cmap='gray')
        plt.title("Bước 1.6: Morphology Opening (Mask cuối)", fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    # ===== BƯỚC 2: TÌM SEED POINTS =====
    print("[2/5] Tìm seed points cho Region Growing...")
    eroded = cv2.erode(segmented_mask, np.ones((5, 5), np.uint8), iterations=1) # Sửa shape iter

    if show_steps:
        plt.figure(figsize=(10, 8))
        plt.imshow(eroded, cmap='gray')
        plt.title("Bước 2: Seed Points (Erosion)", fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    # ===== BƯỚC 3: REGION GROWING =====
    print("[3/5] Áp dụng Region Growing...")
    regions = apply_region_growing(gray, eroded)
    print(f"✓ Phát hiện {len(regions)} regions")

    # Hiển thị các regions
    if show_steps and len(regions) > 0:
        # Tạo ảnh kết hợp tất cả regions
        all_regions = np.zeros_like(gray)
        for region in regions:
            all_regions = cv2.bitwise_or(all_regions, region)

        plt.figure(figsize=(10, 8))
        plt.imshow(all_regions, cmap='gray')
        plt.title(f"Bước 3: Region Growing ({len(regions)} regions)",
                  fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        # Hiển thị một vài regions riêng lẻ (tối đa 6)
        num_display = min(6, len(regions))
        if num_display > 0:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()

            for i in range(num_display):
                axes[i].imshow(regions[i], cmap='gray')
                axes[i].set_title(f"Region {i + 1}", fontsize=12)
                axes[i].axis('off')

            # Ẩn các subplot thừa
            for i in range(num_display, 6):
                axes[i].axis('off')

            plt.suptitle("Các Regions Riêng Lẻ (Mẫu)", fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()

    # ===== BƯỚC 4: TRÍCH XUẤT ĐẶC TRƯNG =====
    print("[4/5] Trích xuất đặc trưng...")
    features_list = []
    for region_mask in regions:
        features = extract_features(region_mask, gray)
        if features:
            features_list.append(features)

    print(f"✓ Trích xuất được {len(features_list)} linh kiện hợp lệ")

    # Hiển thị các contours đã trích xuất
    if show_steps and len(features_list) > 0:
        contour_img = img.copy()
        for i, f in enumerate(features_list):
            cv2.drawContours(contour_img, [f['contour']], -1, (0, 255, 0), 2)
            x, y, w, h = f['bbox']
            cv2.putText(contour_img, f"#{i + 1}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        plt.figure(figsize=(12, 10))
        plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Bước 4: Contours đã trích xuất ({len(features_list)} linh kiện)",
                  fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        # Hiển thị biểu đồ phân bố đặc trưng
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        areas = [f['area'] for f in features_list]
        solidities = [f['solidity'] for f in features_list]
        aspect_ratios = [f['aspect_ratio'] for f in features_list]
        intensities = [f['mean_intensity'] for f in features_list]

        axes[0, 0].hist(areas, bins=20, color='blue', alpha=0.7)
        axes[0, 0].set_title('Phân bố Diện tích', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Area')
        axes[0, 0].set_ylabel('Số lượng')

        axes[0, 1].hist(solidities, bins=20, color='green', alpha=0.7)
        axes[0, 1].set_title('Phân bố Solidity', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Solidity')
        axes[0, 1].set_ylabel('Số lượng')

        axes[1, 0].hist(aspect_ratios, bins=20, color='orange', alpha=0.7)
        axes[1, 0].set_title('Phân bố Aspect Ratio', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Aspect Ratio')
        axes[1, 0].set_ylabel('Số lượng')

        axes[1, 1].hist(intensities, bins=20, color='red', alpha=0.7)
        axes[1, 1].set_title('Phân bố Độ sáng trung bình', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Mean Intensity')
        axes[1, 1].set_ylabel('Số lượng')

        plt.suptitle('Phân Bố Các Đặc Trưng', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    if len(features_list) < 2:
        print("❌ Không đủ linh kiện để phân tích!")
        return None

    # ===== BƯỚC 5: K-MEANS + PHÂN LOẠI =====
    print("[5/5] Phát hiện lỗi bằng K-means...")
    labels, centers = kmeans_detect_defects(features_list)
    ok_label, stats_0, stats_1 = split_merge_analysis(features_list, labels)

    # Hiển thị kết quả clustering
    if show_steps:
        # Vẽ scatter plot 2D của features
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        areas = [f['area'] for f in features_list]
        solidities = [f['solidity'] for f in features_list]

        colors = ['green' if l == ok_label else 'red' for l in labels]

        # Plot 1: Area vs Solidity
        axes[0].scatter(areas, solidities, c=colors, s=100, alpha=0.6, edgecolors='black')
        axes[0].set_xlabel('Area', fontsize=12)
        axes[0].set_ylabel('Solidity', fontsize=12)
        axes[0].set_title('K-means Clustering: Area vs Solidity', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Thêm legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', label='OK (Cluster)'),
            Patch(facecolor='red', label='Defect (Cluster)')
        ]
        axes[0].legend(handles=legend_elements, loc='best')

        # Plot 2: Aspect Ratio vs Mean Intensity
        aspect_ratios = [f['aspect_ratio'] for f in features_list]
        intensities = [f['mean_intensity'] for f in features_list]

        axes[1].scatter(aspect_ratios, intensities, c=colors, s=100, alpha=0.6, edgecolors='black')
        axes[1].set_xlabel('Aspect Ratio', fontsize=12)
        axes[1].set_ylabel('Mean Intensity', fontsize=12)
        axes[1].set_title('K-means Clustering: Aspect Ratio vs Intensity', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(handles=legend_elements, loc='best')

        plt.suptitle('Bước 5: Kết Quả K-means Clustering', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    # Phân loại chi tiết
    results = classify_defects(features_list, labels, ok_label, gray)

    # ===== VẼ KẾT QUẢ =====
    output_img = img.copy()

    for r in results:
        x, y, w, h = r['bbox']
        color = r['color']

        # Vẽ bounding box
        cv2.rectangle(output_img, (x, y), (x + w, y + h), color, 2)

        # Vẽ label
        label = f"#{r['id']}: {r['status_vn']}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(label, font, 0.5, 2)[0]

        # Nền cho text
        cv2.rectangle(output_img, (x, y - text_size[1] - 8),
                      (x + text_size[0] + 4, y), color, -1)
        cv2.putText(output_img, label, (x + 2, y - 5),
                    font, 0.5, (255, 255, 255), 2)

    # Hiển thị kết quả cuối cùng
    if show_steps:
        plt.figure(figsize=(14, 10))
        plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))

        count_correct = sum(1 for r in results if r['status'] == 'CORRECT')
        count_missing = sum(1 for r in results if r['status'] == 'MISSING')
        count_misaligned = sum(1 for r in results if r['status'] == 'MISALIGNED')

        title = f"KẾT QUẢ CUỐI CÙNG\n"
        title += f"Tổng: {len(results)} | ✓Đúng: {count_correct} | ✗Thiếu: {count_missing} | ⚠Lệch: {count_misaligned}"
        plt.title(title, fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    # ===== THỐNG KÊ =====
    print("\n" + "=" * 60)
    print("KẾT QUẢ PHÂN TÍCH")
    print("=" * 60)

    count_correct = sum(1 for r in results if r['status'] == 'CORRECT')
    count_missing = sum(1 for r in results if r['status'] == 'MISSING')
    count_misaligned = sum(1 for r in results if r['status'] == 'MISALIGNED')
    count_other = len(results) - count_correct - count_missing - count_misaligned

    print(f"Tổng số linh kiện: {len(results)}")
    print(f"  ✓ Đúng (CORRECT):     {count_correct} ({100 * count_correct / len(results):.1f}%)")
    print(f"  ✗ Thiếu (MISSING):    {count_missing} ({100 * count_missing / len(results):.1f}%)")
    print(f"  ⚠ Lệch (MISALIGNED):  {count_misaligned} ({100 * count_misaligned / len(results):.1f}%)")
    if count_other > 0:
        print(f"  ? Lỗi khác:           {count_other} ({100 * count_other / len(results):.1f}%)")

    # ===== TẠO BẢNG KẾT QUẢ =====
    df = pd.DataFrame([{
        'ID': r['id'],
        'Trạng thái': r['status_vn'],
        'Diện tích': f"{r['area']:.0f}",
        'Solidity': f"{r['solidity']:.3f}",
        'Độ sáng TB': f"{r['mean_intensity']:.1f}"
    } for r in results])

    print("\n" + "=" * 60)
    print("BẢNG CHI TIẾT")
    print("=" * 60)
    print(df.to_string(index=False))

    # ===== VISUALIZATION =====
    if show_steps:
        plt.figure(figsize=(18, 6))

        # Ảnh gốc
        plt.subplot(1, 4, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("1. Ảnh PCB Gốc", fontsize=12, fontweight='bold')
        plt.axis('off')

        # Phân vùng
        plt.subplot(1, 4, 2)
        plt.imshow(segmented_mask, cmap='gray')
        plt.title("2. Phân vùng\n(Adaptive + Edge)", fontsize=12, fontweight='bold')
        plt.axis('off')

        # Seed points
        plt.subplot(1, 4, 3)
        plt.imshow(eroded, cmap='gray')
        plt.title("3. Seed Points\n(Region Growing)", fontsize=12, fontweight='bold')
        plt.axis('off')

        # Kết quả
        plt.subplot(1, 4, 4)
        plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
        title = f"4. Kết Quả Phát Hiện Lỗi\n✓{count_correct} | ✗{count_missing} | ⚠{count_misaligned}"
        plt.title(title, fontsize=12, fontweight='bold')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig('pcb_result.png', dpi=150, bbox_inches='tight')
        print(f"\n✓ Đã lưu kết quả vào 'pcb_result.png'")
        plt.show()

    # Lưu bảng kết quả
    df.to_csv('pcb_defect_report.csv', index=False, encoding='utf-8-sig')
    print(f"✓ Đã lưu bảng kết quả vào 'pcb_defect_report.csv'")

    return output_img, results, df


# ========== CHẠY CHƯƠNG TRÌNH ==========
if __name__ == "__main__":
    img_path = "../data/wrong/incorrect-7.jpg"

    result = pcb_defect_detection(img_path, show_steps=True)

    if result:
        print("\n" + "=" * 60)
        print("✓ HOÀN THÀNH!")
        print("=" * 60)