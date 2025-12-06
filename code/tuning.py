import cv2
import numpy as np
import os
from collections import deque

# --- 1. CẤU HÌNH ---
INPUT_FOLDER = 'data/wrong'
OUTPUT_FOLDER = 'results_tuning_v2'  # Lưu vào folder mới cho đỡ lẫn

# Các tham số để Grid Search
BLOCK_SIZES = [35, 61, 91]       # Kích thước vùng Adaptive
CONSTANTS_C = [5, 10, 15]        # Lọc nhiễu nền
GROW_THRESHOLDS = [20, 40, 60]   # Ngưỡng loang vùng

# --- 2. HÀM REGION GROWING THỦ CÔNG ---
def region_growing_manual(gray, seeds, tau=5):
    H, W = gray.shape
    visited = np.zeros_like(gray, dtype=np.uint8)
    out = np.zeros_like(gray, dtype=np.uint8)
    dirs = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    
    q = deque()
    for sy, sx in seeds:
        q.append((sy, sx))
        visited[sy, sx] = 1
        out[sy, sx] = 255
        
    while q:
        y, x = q.popleft()
        curr_val = int(gray[y, x])
        for dy, dx in dirs:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and visited[ny, nx] == 0:
                diff = abs(int(gray[ny, nx]) - curr_val)
                if diff <= tau:
                    visited[ny, nx] = 1
                    out[ny, nx] = 255
                    q.append((ny, nx))
    return out

# --- 3. HÀM XỬ LÝ 1 TRƯỜNG HỢP ---
def process_single_case(img_path, filename, block_size, C, tau):
    img = cv2.imread(img_path)
    if img is None: return

    # 1. Tiền xử lý
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (7, 7), 0) # Blur mạnh hơn chút

    # 2. Adaptive Threshold
    thresh = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, block_size, C)

    # 3. Morphology (Cắt mạch & Hàn linh kiện)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask_solid = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2) # Hàn lỗ
    mask_clean = cv2.morphologyEx(mask_solid, cv2.MORPH_OPEN, kernel, iterations=2) # Cắt dây

    # 4. Tìm Seeds
    sure_fg = cv2.erode(mask_clean, kernel, iterations=2) # Co nhỏ để tìm nhân
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(sure_fg, connectivity=8)
    
    seeds = []
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] > 50:
            cx, cy = int(centroids[i][0]), int(centroids[i][1])
            seeds.append((cy, cx))

    # 5. Region Growing
    mask_grown = np.zeros_like(gray)
    if len(seeds) > 0:
        mask_grown = region_growing_manual(gray_blur, seeds, tau=tau)
        # Làm mịn mask kết quả
        mask_grown = cv2.morphologyEx(mask_grown, cv2.MORPH_CLOSE, kernel, iterations=2)
        
    # 6. Vẽ kết quả lên ảnh màu
    contours, _ = cv2.findContours(mask_grown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result_img = img.copy()
    
    count_obj = 0
    for cnt in contours:
        if cv2.contourArea(cnt) > 300: # Lọc nhiễu
            x, y, w, h = cv2.boundingRect(cnt)
            # Vẽ khung xanh
            cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            count_obj += 1
    
    # 7. TẠO ẢNH GHÉP (MÀU | MASK)
    # Chuyển mask sang 3 kênh màu để ghép được với ảnh gốc
    mask_bgr = cv2.cvtColor(mask_grown, cv2.COLOR_GRAY2BGR)
    
    # Vẽ chữ lên ảnh màu
    info_text = f"Block:{block_size} | C:{C} | Tau:{tau}"
    cv2.rectangle(result_img, (0, 0), (400, 30), (0, 0, 0), -1)
    cv2.putText(result_img, info_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Vẽ chữ lên ảnh mask
    cv2.putText(mask_bgr, "Mask (Trang den)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Ghép ngang 2 ảnh
    combined_img = np.hstack((result_img, mask_bgr))

    # Lưu ảnh
    save_name = f"{filename[:-4]}_B{block_size}_C{C}_T{tau}.jpg"
    save_path = os.path.join(OUTPUT_FOLDER, save_name)
    cv2.imwrite(save_path, combined_img)
    print(f"--> Đã lưu: {save_name}")

# --- MAIN LOOP ---
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(('.jpg', '.png'))]
    print(f"Đang xử lý {len(files)} ảnh...")

    for f in files:
        path = os.path.join(INPUT_FOLDER, f)
        for b in BLOCK_SIZES:
            for c in CONSTANTS_C:
                for t in GROW_THRESHOLDS:
                    try:
                        process_single_case(path, f, b, c, t)
                    except Exception as e:
                        print(f"Lỗi: {e}")
    
    print("\nHOÀN TẤT! Mở folder 'results_tuning_v2' để xem ảnh so sánh.")