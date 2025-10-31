
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# --- B1. ĐỌC DỮ LIỆU CSV ---
df = pd.read_csv("moso_api_data.csv")

# --- B2. LÀM SẠCH DỮ LIỆU ---
# Chuẩn hóa cột giá sang tỷ đồng
def parse_price(v):
    if not isinstance(v, str):
        return np.nan
    v = v.strip().replace(",", ".")
    if "Tỷ" in v:
        return float(re.findall(r"[\d.]+", v)[0])
    if "Triệu" in v:
        return float(re.findall(r"[\d.]+", v)[0]) / 1000
    try:
        return float(v)
    except:
        return np.nan

df["price_billion"] = df["Giá"].apply(parse_price)
df["price_million"] = df["price_billion"] * 1000

# Làm sạch phòng ngủ / tắm
df["Phòng ngủ"] = pd.to_numeric(df["Phòng ngủ"], errors="coerce")
df["Phòng tắm"] = pd.to_numeric(df["Phòng tắm"], errors="coerce")

# Điền giá trị thiếu bằng median
for col in ["Diện tích sử dụng", "Diện tích đất", "Phòng ngủ", "Phòng tắm"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

# --- B3. XỬ LÝ NGOẠI LAI ---
def remove_outliers(series):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return series.between(lower, upper)

mask = (
    remove_outliers(df["price_billion"]) &
    remove_outliers(df["Diện tích sử dụng"])
)
df_clean = df[mask].copy()

# --- B4. CHUẨN HÓA DỮ LIỆU ---
num_cols = ["price_million", "Diện tích sử dụng", "Diện tích đất", "Phòng ngủ", "Phòng tắm"]
scaler = StandardScaler()
df_scaled = df_clean.copy()
df_scaled[num_cols] = scaler.fit_transform(df_clean[num_cols])

df_scaled.to_csv("moso_api_data_clean_scaled.csv", index=False, encoding="utf-8-sig")

# --- B5. TRÍCH XUẤT QUẬN TỪ ĐỊA CHỈ ---
def extract_district(address):
    if not isinstance(address, str):
        return "UNKNOWN"
    parts = [p.strip() for p in address.split(",") if p.strip()]
    for p in reversed(parts[:-1]):
        m = re.search(r"(Q\.?\s*\d+|Quận\s*\d+)", p, flags=re.IGNORECASE)
        if m:
            return m.group().upper().replace("QUẬN", "Q").replace(" ", "")
    if len(parts) >= 2:
        return parts[-2]
    return "UNKNOWN"

df_scaled["district"] = df_scaled["Địa chỉ"].apply(extract_district)

# --- B6. TỔNG HỢP THEO QUẬN ---
agg = df_scaled.groupby("district").agg(
    count=("URL", "count"),
    mean_price=("price_billion", "mean"),
    median_price=("price_billion", "median"),
    mean_area=("Diện tích sử dụng", "mean")
).reset_index().sort_values("count", ascending=False)
agg.to_csv("agg_by_district.csv", index=False, encoding="utf-8-sig")

# --- B7. BIỂU ĐỒ TRỰC QUAN ---
plt.rcParams["figure.figsize"] = (10,6)

# 7.1 Histogram giá
plt.figure()
plt.hist(df_clean["price_billion"], bins=40, color="orange", edgecolor="black")
plt.title("Phân bố giá BĐS (tỷ đồng)")
plt.xlabel("Giá (tỷ)")
plt.ylabel("Số lượng tin")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

# 7.2 Scatter: diện tích vs giá
plt.figure()
plt.scatter(df_clean["Diện tích sử dụng"], df_clean["price_billion"], s=10, alpha=0.6)
plt.title("Quan hệ giữa diện tích và giá")
plt.xlabel("Diện tích sử dụng (m2)")
plt.ylabel("Giá (tỷ)")
plt.grid(True)
plt.show()

# 7.3 Boxplot: giá theo số phòng ngủ
beds = sorted(df_clean["Phòng ngủ"].unique())
data_by_bed = [df_clean.loc[df_clean["Phòng ngủ"] == b, "price_billion"] for b in beds]
plt.figure()
plt.boxplot(data_by_bed, labels=beds, showfliers=False)
plt.title("Phân bố giá theo số phòng ngủ")
plt.xlabel("Số phòng ngủ")
plt.ylabel("Giá (tỷ)")
plt.show()

# 7.4 Bar chart: top quận nhiều tin nhất
top_districts = agg.head(10)
plt.figure()
plt.bar(top_districts["district"], top_districts["count"], color="orange")
plt.title("Top 10 quận có nhiều tin đăng nhất")
plt.xlabel("Quận")
plt.ylabel("Số lượng tin")
plt.grid(axis='y', linestyle="--", alpha=0.5)
plt.show()

# --- B8. NHẬN XÉT ---
print("=== NHẬN XÉT NHANH ===")
print(f"Tổng số bản ghi sau làm sạch: {len(df_clean)}")
print("Các quận có nhiều tin nhất:")
print(top_districts[["district", "count", "mean_price"]])
print("\nQuan sát: Giá trung bình (tỷ) tăng dần theo quận trung tâm, diện tích trung bình nhỏ dần.")
print("Các file đã lưu: moso_api_data_clean_scaled.csv, agg_by_district.csv")