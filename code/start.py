import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import deque


def region_growing_manual(img_gray, seeds, threshold=5):
    h, w = img_gray.shape
    visited = np.zeros_like(img_gray, dtype=np.uint8)
    out_mask = np.zeros_like(img_gray, dtype=np.uint8)
    
    # 8 hướng lân cận
    dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    
    q = deque()
    
    for sy, sx in seeds:
        q.append((sy, sx))
        visited[sy, sx] = 1
        out_mask[sy, sx] = 255
        
    while q:
        y, x = q.popleft()
        curr_val = int(img_gray[y, x])
        
        for dy, dx in dirs:
            ny, nx = y + dy, x + dx
            
            if 0 <= ny < h and 0 <= nx < w and visited[ny, nx] == 0:
                neighbor_val = int(img_gray[ny, nx])
                
                if abs(neighbor_val - curr_val) <= threshold:
                    visited[ny, nx] = 1
                    out_mask[ny, nx] = 255
                    q.append((ny, nx))
    return out_mask

# --- HÀM 2: K-MEANS CLUSTERING ---
def apply_kmeans(features, K=2):
    Z = np.float32(features)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    return label.flatten(), center

# --- CHƯƠNG TRÌNH CHÍNH ---

current_folder = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(current_folder, '..', 'data', 'wrong', 'incorrect-2.jpg')
img_path = os.path.normpath(img_path)

img = cv2.imread(img_path)

if img is None:
    print("Lỗi đọc ảnh")
else:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # ---------------------------------------------------------
    # BƯỚC 1: TÁCH NỀN (ADAPTIVE THRESHOLD + EDGE DETECTION)
    # ---------------------------------------------------------
    
    # 1.1 Adaptive Threshold
    thresh_adp = cv2.adaptiveThreshold(gray, 255, 
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 
                                      31, 15) 
    
    # 1.2 Edge Detection
    edges = cv2.Canny(gray, 50, 150)
    
    # 1.3 Kết hợp (OR): Lấy cả vùng đặc và biên sắc nét
    combined = cv2.bitwise_or(thresh_adp, edges)

    # 1.4 Morphology: Cắt mạch điện, giữ linh kiện
    kernel_close = np.ones((7, 7), np.uint8) 
    # Closing để hàn các vết nứt trên linh kiện
    mask_closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    
    # Opening để cắt đứt các dây mạch mảnh nối với linh kiện
    kernel_open = np.ones((3, 3), np.uint8)
    mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel_open, iterations=2)

    # ---------------------------------------------------------
    # BƯỚC 2: REGION GROWING (THAY THẾ COMPONENT LABELING)
    # ---------------------------------------------------------
    
    # Tìm Seeds bằng cách co nhỏ vùng (Erosion)
    sure_fg = cv2.erode(mask_clean, np.ones((7,7), np.uint8), iterations=3)
    num_seeds, labels_im, stats, centroids = cv2.connectedComponentsWithStats(sure_fg, connectivity=8)
    
    seeds_list = []
    print("--- ĐANG CHẠY REGION GROWING ---")
    
    # Lọc seeds nhiễu
    for i in range(1, num_seeds):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 50: # Chỉ lấy seeds đủ lớn
            cy, cx = int(centroids[i][1]), int(centroids[i][0])
            seeds_list.append((cy, cx))

    # Hàm Region Growing
    # Threshold=40: Chấp nhận chênh lệch màu để loang hết linh kiện
    mask_grown = region_growing_manual(gray, seeds_list, threshold=15)
    
    # Làm gọn kết quả sau khi growing
    mask_final = cv2.morphologyEx(mask_grown, cv2.MORPH_CLOSE, kernel_close)

    # ---------------------------------------------------------
    # BƯỚC 3: PHÁT HIỆN LỖI BẰNG K-MEANS
    # ---------------------------------------------------------
    
    contours, _ = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_img = img.copy()
    
    feature_data = [] # Chứa [Solidity, Aspect Ratio]
    valid_contours = [] # Chứa contours tương ứng
    
    print(f"Tìm thấy {len(contours)} đối tượng tiềm năng.")

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 400: continue # Lọc nhiễu
        
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Tính đặc trưng 1: Độ đặc (Solidity)
        rect_area = w * h
        solidity = float(area) / rect_area
        
        # Tính đặc trưng 2: Tỷ lệ khung hình (Aspect Ratio)
        aspect_ratio = float(w) / h
        
        feature_data.append([solidity, aspect_ratio])
        valid_contours.append(cnt)

    # Chỉ chạy K-means nếu có đủ dữ liệu
    if len(feature_data) > 1:
        # Gom làm 2 nhóm: 1 nhóm OK, 1 nhóm Lỗi/Lệch
        labels, centers = apply_kmeans(feature_data, K=2)
        
        # Phân tích xem nhóm nào là OK, nhóm nào là Lỗi
        # Giả định: Nhóm OK thường có Độ đặc (Solidity) trung bình cao hơn (vuông vức hơn)
        mean_solidity_0 = np.mean([f[0] for i, f in enumerate(feature_data) if labels[i] == 0])
        mean_solidity_1 = np.mean([f[0] for i, f in enumerate(feature_data) if labels[i] == 1])
        
        ok_label = 0 if mean_solidity_0 > mean_solidity_1 else 1
        
        count_ok = 0
        
        for i, cnt in enumerate(valid_contours):
            cluster_id = labels[i]
            x, y, w, h = cv2.boundingRect(cnt)
            solidity = feature_data[i][0]
            
            if cluster_id == ok_label:
                color = (0, 255, 0) # XANH LÁ - OK
                status = "OK"
                count_ok += 1
            else:
                color = (0, 0, 255) # ĐỎ - LỆCH/LỖI
                status = "ERROR" # Do K-means phát hiện khác biệt
            
            # Vẽ hộp
            cv2.rectangle(output_img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(output_img, status, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            print(f"Linh kiện: Solidity={solidity:.2f} -> Cluster {cluster_id} ({status})")
            
        print(f"Số linh kiện OK: {count_ok}/{len(valid_contours)}")

    # ---------------------------------------------------------
    # HIỂN THỊ KẾT QUẢ
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 6))
    
    # Bên trái: Mask sạch (đầu vào cho Region Growing)
    plt.subplot(1, 2, 1)
    plt.imshow(mask_clean, cmap='gray')
    plt.title("Mask (Adaptive + Canny + Morph)")
    plt.axis('off')
    
    # Bên phải: Kết quả cuối cùng
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
    plt.title("Kết quả (K-means Detection)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()