import streamlit as st
import pandas as pd
import joblib
import datetime as dt

# ==== 1. Load model & object ====
rf_model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")
segment_map = joblib.load("segment_map.pkl")
features_vn = joblib.load("features_vn.pkl")

Ngay_Hien_Tai = dt.date(2025, 12, 30)  # mốc snapshot như khi train

# ==== 2. Hàm build feature từ dữ liệu giao dịch gốc ====
def build_customer_features(data, snapshot):
    rfm = data.groupby('ID_KhachHang').agg({
        'Ngay_Mua': lambda x: (snapshot - x.max().date()).days,
        'ID_DonHang': 'count',
        'Tong_Chi_Tieu': 'sum',
        'TG_Truy_Cap': 'mean',
        'So_Trang_Xem': 'mean'
    }).reset_index()

    rfm.columns = [
        'ID_KhachHang',
        'Do_Moi',
        'Tan_Suat',
        'Tong_Chi_Tieu',
        'TG_Truy_Cap_TB',
        'So_Trang_Xem_TB'
    ]
    return rfm

new_names = {
    'Order_ID': 'ID_DonHang',
    'Customer_ID': 'ID_KhachHang',
    'Date': 'Ngay_Mua',
    'Age': 'Tuoi',
    'Gender': 'Gioi_Tinh',
    'City': 'Thanh_Pho',
    'Product_Category': 'Loai_San_Pham',
    'Unit_Price': 'Don_Gia',
    'Quantity': 'So_Luong',
    'Discount_Amount': 'Tien_Giam_Gia',
    'Total_Amount': 'Tong_Chi_Tieu',
    'Session_Duration_Minutes': 'TG_Truy_Cap',
    'Pages_Viewed': 'So_Trang_Xem',
    'Customer_Rating': 'Danh_Gia',
    'Delivery_Time_Days': 'Thoi_Gian_Giao_Hang',
    'Payment_Method': 'Phuong_Thuc_Thanh_Toan',
    'Device_Type': 'Loai_Thiet_Bi',
    'Is_Returning_Customer': 'Khach_Hang_Quay_Lai'
}

# ==== 3. Cấu hình giao diện chung ====
st.set_page_config(
    page_title="Phân khúc khách hàng",
    page_icon="🛒",
    layout="wide"
)

st.title("Ứng dụng dự đoán phân khúc khách hàng")
st.markdown(
    "Giúp doanh nghiệp nhận diện **VIP**, khách **tiềm năng** và khách **vãng lai** "
    "từ dữ liệu giao dịch thương mại điện tử."
)

# Sidebar chọn chế độ
mode = st.sidebar.radio(
    "Chọn chế độ",
    ["📂 Phân khúc từ file CSV", "👤 Dự đoán 1 khách mới"]
)

# ==== 4A. CHẾ ĐỘ 1: Upload file & phân khúc hàng loạt ====
if mode.startswith("📂"):
    st.subheader("📂 Phân khúc khách hàng từ file giao dịch")

    file = st.file_uploader("Chọn file CSV (cấu trúc giống datasetV2.csv)", type=["csv"])

    if file is not None:
        df_new = pd.read_csv(file)
        df_new = df_new.rename(columns=new_names)
        df_new['Ngay_Mua'] = pd.to_datetime(df_new['Ngay_Mua'])

        st.markdown("**Dữ liệu gốc (5 dòng đầu):**")
        st.dataframe(df_new.head())

        rfm_new = build_customer_features(df_new, Ngay_Hien_Tai)
        rfm_new[features_vn] = rfm_new[features_vn].fillna(rfm_new[features_vn].median())

        st.markdown("**Đặc trưng hành vi đã xây dựng (5 dòng đầu):**")
        st.dataframe(rfm_new.head())

        all_features = features_vn
        selected_features = st.multiselect(
            "Chọn các đặc trưng dùng cho mô hình",
            options=all_features,
            default=all_features
        )

        if len(selected_features) == 0:
            st.warning("Hãy chọn ít nhất 1 đặc trưng để dự đoán.")
        else:
            X_new_scaled = scaler.transform(rfm_new[selected_features])
            cum_pred = rf_model.predict(X_new_scaled)
            rfm_new['Cum'] = cum_pred
            rfm_new['Phan_Khuc'] = rfm_new['Cum'].map(segment_map)

            st.markdown("**Kết quả phân khúc (10 khách đầu):**")
            st.dataframe(rfm_new.head(10))

            col1, col2 = st.columns(2)
            with col1:
                st.write("Số lượng khách theo phân khúc:")
                st.dataframe(rfm_new['Phan_Khuc'].value_counts())

            csv_out = rfm_new.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                "⬇️ Tải file kết quả phân khúc",
                csv_out,
                "ket_qua_phan_khuc_moi.csv",
                "text/csv"
            )

# ==== 4B. CHẾ ĐỘ 2: Nhập tay 1 khách mới ====
else:
    st.subheader("👤 Dự đoán phân khúc cho 1 khách hàng mới")

    col_left, col_right = st.columns(2)

    with col_left:
        ngay_mua_gan_nhat = st.date_input(
            "Ngày mua gần nhất của khách",
            value=Ngay_Hien_Tai
        )
        tan_suat = st.number_input("Tần suất (số đơn hàng)", min_value=0, value=2)
        tong_chi_tieu = st.number_input("Tổng chi tiêu", min_value=0.0, value=250.0, step=50.0)

    with col_right:
        tg_tb = st.number_input("Thời gian truy cập TB (phút)", min_value=0.0, value=15.0)
        so_trang_tb = st.number_input("Số trang xem TB", min_value=0.0, value=8.0)

    if st.button("🔍 Dự đoán phân khúc"):
        do_moi = (Ngay_Hien_Tai - ngay_mua_gan_nhat).days

        data = pd.DataFrame(
            [[do_moi, tan_suat, tong_chi_tieu, tg_tb, so_trang_tb]],
            columns=features_vn
        )
        data_scaled = scaler.transform(data)
        cum = rf_model.predict(data_scaled)[0]
        phan_khuc = segment_map[cum]

        st.success(f"Khách hàng thuộc phân khúc: **{phan_khuc}**")
        st.markdown(
            f"- Độ mới: {do_moi} ngày\n"
            f"- Tần suất: {tan_suat} đơn\n"
            f"- Tổng chi tiêu: {tong_chi_tieu}\n"
            f"- TG truy cập TB: {tg_tb} phút\n"
            f"- Số trang xem TB: {so_trang_tb}"
        )
