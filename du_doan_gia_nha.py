import pandas as pd
import numpy as np
import re
import json
import random
import time

# ==============================================================================
# === PHẦN A: CÁC HÀM XỬ LÝ DỮ LIỆU
# ==============================================================================

def extract_district(address_str):
    '''Trích xuất Quận/Huyện từ chuỗi địa chỉ'''
    if not isinstance(address_str, str):
        return 'Không rõ'
    districts = [
        '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
        'Gò Vấp', 'Tân Bình', 'Tân Phú', 'Bình Thạnh', 'Phú Nhuận',
        'Thủ Đức', 'Bình Tân', 'Bình Chánh', 'Hóc Môn', 'Củ Chi', 'Nhà Bè', 'Cần Giờ'
    ]
    sorted_districts = sorted(districts, key=len, reverse=True)
    for district in sorted_districts:
        # r'\\b' là cú pháp đúng để regex \b hoạt động
        if re.search(r'(Quận|Huyện)?\s*' + re.escape(district) + r'\\b', address_str, re.IGNORECASE):
            if district.isdigit(): return f'Quận {district}'
            return district
    try:
        parts = address_str.split(',')
        if len(parts) >= 3:
            potential_district = parts[-2].strip()
            for d in districts:
                if d.lower() in potential_district.lower():
                    if d.isdigit(): return f'Quận {d}'
                    return d
            return potential_district
    except Exception: pass
    return 'Không rõ'

#
# --- [SỬA LỖI] ---
# Hàm parse_price được viết lại để xử lý lỗi "could not convert string to float: '.'"
#
def parse_price(v):
    '''Chuyển đổi chuỗi giá (vd: "2.5 Tỷ") thành số (2.5)'''
    if not isinstance(v, str):
        return np.nan
    v = v.strip().replace(",", ".")
    
    # Regex mới: tìm một hoặc nhiều chữ số, theo sau có thể là
    # (dấu chấm và 0+ chữ số)
    # HOẶC là (một dấu chấm và 1+ chữ số)
    # Điều này tránh việc bắt một dấu "." đơn lẻ.
    matches = re.findall(r"(\d+\.?\d*|\.\d+)", v)
    
    # Nếu tìm thấy một chuỗi số hợp lệ
    if matches:
        try:
            number_val = float(matches[0])
            if "Tỷ" in v:
                return number_val
            if "Triệu" in v:
                return number_val / 1000
            # Nếu không có "Tỷ" hay "Triệu", giả sử nó là Tỷ
            return number_val 
        except ValueError:
            # Bị lỗi nếu regex vẫn sai (vd: "1.2.3")
            return np.nan
    
    # Nếu regex không tìm thấy số nào
    return np.nan
# --- [KẾT THÚC SỬA LỖI] ---
#

def load_and_prepare_data(input_file="moso_api_data.csv"):
    '''
    Hàm này thực hiện toàn bộ quá trình xử lý dữ liệu và trả về
    dữ liệu sẵn sàng cho mô hình VÀ các thông tin cần thiết
    cho việc dự đoán (danh sách cột, danh sách quận...).
    '''
    print(f"Đang tải dữ liệu thô từ: {input_file}")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy tệp '{input_file}'.")
        return None, None, None, None, None

    print("Đang làm sạch dữ liệu cơ bản (giá, số phòng)...")
    # Áp dụng hàm parse_price đã sửa lỗi
    df["price_billion"] = df["Giá"].apply(parse_price)
    
    # Bỏ qua các dòng có giá NaN ngay từ đầu
    df = df.dropna(subset=["price_billion"])
    
    df["Phòng ngủ"] = pd.to_numeric(df["Phòng ngủ"], errors="coerce")
    df["Phòng tắm"] = pd.to_numeric(df["Phòng tắm"], errors="coerce")

    # Điền giá trị thiếu (NaN) bằng median
    for col in ["Diện tích sử dụng", "Diện tích đất", "Phòng ngủ", "Phòng tắm"]:
        if col in df.columns:
            median_val = df[col].median()
            if pd.isna(median_val): median_val = 0
            df[col] = df[col].fillna(median_val)

    print("Đang xử lý outliers...")
    price_q01 = df['price_billion'].quantile(0.01)
    price_q99 = df['price_billion'].quantile(0.99)
    area_q99 = df['Diện tích sử dụng'].quantile(0.99)
    
    df_clean = df[
        (df['price_billion'] >= price_q01) &
        (df['price_billion'] <= price_q99) &
        (df['Diện tích sử dụng'] <= area_q99) &
        (df['Diện tích sử dụng'] > 0)
    ].copy()
    
    print(f"Dữ liệu sau khi loại bỏ outliers: {len(df_clean)} dòng.")
    print("Đang xử lý đặc trưng (trích xuất quận, mã hóa)...")

    df_clean['khu_vuc'] = df_clean['Địa chỉ'].apply(extract_district)
    
    # Lấy danh sách các giá trị duy nhất TRƯỚC KHI mã hóa
    # (Để hiển thị cho người dùng khi dự đoán)
    unique_districts = sorted([d for d in df_clean['khu_vuc'].unique() if d])
    unique_legal = sorted([l for l in df_clean['Giấy tờ pháp lý'].unique() if l and pd.notna(l)])

    numeric_features = ['Diện tích sử dụng', 'Diện tích đất', 'Phòng ngủ', 'Phòng tắm']
    categorical_features = ['khu_vuc', 'Giấy tờ pháp lý','Loại nhà']
    
    df_encoded = pd.get_dummies(df_clean, columns=categorical_features, drop_first=True, dtype=int)
    
    encoded_feature_cols = [col for col in df_encoded.columns if any(col.startswith(cat + '_') for cat in categorical_features)]
    features_to_use = numeric_features + encoded_feature_cols
    
    X = df_encoded[features_to_use]
    y = df_encoded['price_billion']

    X = X.fillna(0)
    y = y.dropna()
    X = X.loc[y.index]
    
    # Chuyển sang numpy arrays để tăng tốc độ cho thuật toán tự code
    X_np = X.values
    y_np = y.values
    
    print(f"Tập dữ liệu cuối cùng có {X_np.shape[0]} mẫu và {X_np.shape[1]} đặc trưng.")
    
    # Trả về MỌI THỨ cần thiết
    return X_np, y_np, features_to_use, unique_districts, unique_legal

# ==============================================================================
# === PHẦN B: THUẬT TOÁN TỰ CODE (RANDOM FOREST)
# ==============================================================================

class Node:
    '''Đại diện cho một nút trong cây (có thể là lá hoặc nút chia).'''
    def __init__(self, value=None, feature_index=None, threshold=None, left=None, right=None):
        self.value = value  # Giá trị dự đoán (nếu là lá)
        self.feature_index = feature_index # Chỉ số của đặc trưng dùng để chia
        self.threshold = threshold # Ngưỡng chia
        self.left = left   # Nút con (nếu <= threshold)
        self.right = right # Nút con (nếu > threshold)

class MyDecisionTreeRegressor:
    '''Thuật toán Cây Quyết định Hồi quy tự code.'''
    def __init__(self, max_depth=20, min_samples_leaf=3, max_features_ratio=0.8):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features_ratio = max_features_ratio
        self.root = None

    def fit(self, X, y):
        self.n_features_ = X.shape[1]
        self.root = self._build_tree(X, y, depth=0)

    def _calculate_mse(self, y):
        '''Tính Mean Squared Error (Phương sai).'''
        if len(y) == 0:
            return 0
        mean = np.mean(y)
        return np.mean((y - mean) ** 2)

    def _find_best_split(self, X, y):
        '''Tìm đặc trưng và ngưỡng chia tốt nhất để giảm MSE.'''
        n_samples, n_features = X.shape
        if n_samples <= self.min_samples_leaf:
            return None, None

        parent_mse = self._calculate_mse(y)
        best_gain = -1.0
        best_feature_idx = None
        best_threshold = None

        # Chọn ngẫu nhiên một tập con các đặc trưng (phong cách RF)
        n_features_subset = int(n_features * self.max_features_ratio)
        if n_features_subset == 0: n_features_subset = 1
        
        feature_indices = np.random.choice(n_features, n_features_subset, replace=False)

        for feat_idx in feature_indices:
            feature_values = X[:, feat_idx]
            unique_thresholds = np.unique(feature_values)
            
            if len(unique_thresholds) <= 1:
                continue

            # Duyệt qua các ngưỡng
            for threshold in unique_thresholds:
                left_indices = np.where(X[:, feat_idx] <= threshold)[0]
                right_indices = np.where(X[:, feat_idx] > threshold)[0]

                # Bỏ qua nếu việc chia tạo ra một lá quá nhỏ
                if len(left_indices) < self.min_samples_leaf or len(right_indices) < self.min_samples_leaf:
                    continue

                y_left, y_right = y[left_indices], y[right_indices]
                
                mse_left = self._calculate_mse(y_left)
                mse_right = self._calculate_mse(y_right)
                
                weight_left = len(y_left) / n_samples
                weight_right = len(y_right) / n_samples
                
                # Tính MSE có trọng số của các nút con
                weighted_child_mse = (weight_left * mse_left) + (weight_right * mse_right)
                # Lợi ích = MSE cha - MSE con
                gain = parent_mse - weighted_child_mse
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature_idx = feat_idx
                    best_threshold = threshold
        
        return best_feature_idx, best_threshold

    def _build_tree(self, X, y, depth):
        '''Hàm đệ quy để xây dựng cây.'''
        n_samples = X.shape[0]
        # Giá trị của lá là trung bình của các mẫu
        leaf_value = np.mean(y)

        # Điều kiện dừng
        if (depth >= self.max_depth or
            n_samples < self.min_samples_leaf or
            len(np.unique(y)) == 1):
            return Node(value=leaf_value)

        best_feature_idx, best_threshold = self._find_best_split(X, y)

        # Dừng nếu không tìm thấy cách chia tốt
        if best_feature_idx is None:
            return Node(value=leaf_value)

        # Chia dữ liệu
        left_indices = np.where(X[:, best_feature_idx] <= best_threshold)[0]
        right_indices = np.where(X[:, best_feature_idx] > best_threshold)[0]

        # Dừng nếu việc chia không hợp lệ
        if len(left_indices) == 0 or len(right_indices) == 0:
             return Node(value=leaf_value)

        # Đệ quy xây dựng cây con
        left_child = self._build_tree(X[left_indices, :], y[left_indices], depth + 1)
        right_child = self._build_tree(X[right_indices, :], y[right_indices], depth + 1)

        return Node(feature_index=best_feature_idx, threshold=best_threshold, left=left_child, right=right_child)

    def predict(self, X):
        '''Dự đoán cho một mảng X (2D).'''
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        '''Duyệt cây để dự đoán cho 1 mẫu x (1D).'''
        if node.value is not None:
            return node.value # Đã đến lá

        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

class MyRandomForestRegressor:
    '''Thuật toán Rừng Ngẫu nhiên Hồi quy tự code.'''
    def __init__(self, n_estimators=20, max_depth=10, min_samples_leaf=5, max_features_ratio=0.8, bootstrap_ratio=0.8):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features_ratio = max_features_ratio
        self.bootstrap_ratio = bootstrap_ratio
        self.trees = []
        print(f"Khởi tạo Random Forest (Tự code) với {n_estimators} cây.")
        print("CẢNH BÁO: Quá trình huấn luyện sẽ CHẬM vì code bằng Python thuần.")

    def _create_bootstrap_sample(self, X, y):
        '''Tạo mẫu bootstrap (lấy mẫu có hoàn lại).'''
        n_samples = X.shape[0]
        sample_size = int(n_samples * self.bootstrap_ratio)
        indices = np.random.choice(n_samples, size=sample_size, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        '''Huấn luyện (tạo) N cây.'''
        self.trees = []
        start_time = time.time()
        for i in range(self.n_estimators):
            print(f"  Đang huấn luyện cây {i+1}/{self.n_estimators}...")
            
            X_sample, y_sample = self._create_bootstrap_sample(X, y)
            
            tree = MyDecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features_ratio=self.max_features_ratio
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            
        end_time = time.time()
        print(f"Huấn luyện hoàn tất {self.n_estimators} cây trong {end_time - start_time:.2f} giây.")

    def predict(self, X):
        '''Dự đoán bằng cách lấy trung bình dự đoán của tất cả các cây.'''
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        # Lấy trung bình theo cột (axis=0)
        forest_predictions = np.mean(tree_predictions, axis=0)
        return forest_predictions

# ==============================================================================
# === PHẦN C: HÀM DỰ ĐOÁN TƯƠNG TÁC
# ==============================================================================

def get_user_input_and_predict(model, all_feature_columns, unique_districts, unique_legal):
    print("\n--- NHẬP THÔNG TIN NHÀ CẦN DỰ ĐOÁN ---")
    
    try:
        # 1. Thu thập thông tin số
        dien_tich_sd = float(input("  - Diện tích sử dụng (m2): "))
        dien_tich_dat = float(input("  - Diện tích đất (m2) (nhập 0 nếu là căn hộ): "))
        phong_ngu = int(input("  - Số phòng ngủ: "))
        phong_tam = int(input("  - Số phòng tắm: "))

        # 2. Thu thập thông tin phân loại (Khu vực)
        print("\n  --- Chọn Khu vực (Quận/Huyện) ---")
        for i, district in enumerate(unique_districts):
            print(f"    {i+1}: {district}")
        district_choice_idx = int(input("  - Nhập số thứ tự Quận/Huyện: ")) - 1
        if not 0 <= district_choice_idx < len(unique_districts):
            print("Lựa chọn không hợp lệ.")
            return
        user_district = unique_districts[district_choice_idx]

        # 3. Thu thập thông tin phân loại (Giấy tờ)
        print("\n  --- Chọn Giấy tờ pháp lý ---")
        for i, legal in enumerate(unique_legal):
            print(f"    {i+1}: {legal}")
        legal_choice_idx = int(input("  - Nhập số thứ tự Giấy tờ: ")) - 1
        if not 0 <= legal_choice_idx < len(unique_legal):
            print("Lựa chọn không hợp lệ.")
            return
        user_legal = unique_legal[legal_choice_idx]
        
        # 4. Định dạng dữ liệu đầu vào (QUAN TRỌNG)
        # Tạo một 'dòng' rỗng (Series) với tất cả các cột đặc trưng, điền 0
        input_data = pd.Series(0, index=all_feature_columns, dtype=float)
        
        # Điền các giá trị số
        input_data['Diện tích sử dụng'] = dien_tich_sd
        input_data['Diện tích đất'] = dien_tich_dat
        input_data['Phòng ngủ'] = phong_ngu
        input_data['Phòng tắm'] = phong_tam
        
        # Điền các giá trị one-hot
        # (Lưu ý: 'drop_first=True' có nghĩa là nếu người dùng
        # chọn giá trị đầu tiên trong danh sách (đã bị drop),
        # thì tất cả các cột khác của nhóm đó sẽ là 0)
        
        # Xử lý khu vực
        district_col_name = f"khu_vuc_{user_district}"
        if district_col_name in input_data.index:
            input_data[district_col_name] = 1
            
        # Xử lý giấy tờ
        legal_col_name = f"Giấy tờ pháp lý_{user_legal}"
        if legal_col_name in input_data.index:
            input_data[legal_col_name] = 1
            
        # 5. Chuyển sang NumPy và Dự đoán
        # Cần reshape(1, -1) vì mô hình mong đợi một mảng 2D (batch)
        input_np = input_data.values.reshape(1, -1)
        
        prediction = model.predict(input_np)
        
        print("\n--- KẾT QUẢ DỰ ĐOÁN ---")
        print(f"Giá nhà dự đoán: {prediction[0]:.2f} Tỷ đồng")

    except ValueError:
        print("Lỗi: Vui lòng nhập đúng định dạng (số).")
    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")

# ==============================================================================
# === PHẦN D: HÀM CHÍNH ĐỂ CHẠY
# ==============================================================================

def main():
    # 1. Tải và xử lý toàn bộ dữ liệu
    # Lấy X, y và các metadata cần cho dự đoán
    X_np, y_np, feature_cols, district_list, legal_list = load_and_prepare_data()
    
    if X_np is None:
        return

    # 2. Huấn luyện mô hình
    # (Sử dụng tham số nhẹ vì code-tay rất chậm)
    rf_model_scratch = MyRandomForestRegressor(
        n_estimators=20,       # 20 cây (nhiều hơn sẽ rất lâu)
        max_depth=10,          # Giới hạn độ sâu
        min_samples_leaf=5     # Số mẫu tối thiểu ở mỗi lá
    )
    
    rf_model_scratch.fit(X_np, y_np)
    
    # 3. Bắt đầu vòng lặp dự đoán
    while True:
        get_user_input_and_predict(
            rf_model_scratch, 
            feature_cols, 
            district_list, 
            legal_list
        )
        
        if input("\nBạn có muốn dự đoán tiếp không? (y/n): ").lower() == 'n':
            break

if __name__ == "__main__":
    main()