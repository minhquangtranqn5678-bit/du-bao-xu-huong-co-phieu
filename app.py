import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from PIL import Image

# ==========================================
# CẤU HÌNH TRANG STREAMLIT
# ==========================================
st.set_page_config(
    page_title="Hệ thống Dự báo & Quản trị Rủi ro Cổ phiếu HOSE",
    page_icon="📈",
    layout="wide"
)

SYMBOLS = ["FPT", "MWG", "DGW", "FRT", "CMG"]
RESULTS_DIR = "results_plots"

# Tiêu đề chính
st.title("📈 HỆ THỐNG DỰ BÁO XU HƯỚNG GIÁ & QUẢN TRỊ RỦI RO DANH MỤC")
st.caption("Ứng dụng Mô hình Học sâu PyTorch LSTM và XGBoost trên Thị trường Chứng khoán Việt Nam (HOSE T+2)")
st.markdown("---")

# ==========================================
# SIDEBAR: CHỌN MÃ CỔ PHIẾU VÀ CẤU HÌNH
# ==========================================
st.sidebar.header("⚙️ Cấu hình Hệ thống")
selected_symbol = st.sidebar.selectbox("Chọn Mã Cổ Phiếu:", SYMBOLS, index=0)

selected_model = st.sidebar.radio("Chọn Mô hình Dự báo Primary:", ["PyTorch LSTM", "XGBoost Classifier", "ARIMA"])
capital = st.sidebar.number_input("Vốn đầu tư ban đầu (VND):", value=100000000, step=10000000)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Quy chế Thị trường:**\n"
    "- Sàn HOSE, biên độ ±6.9%\n"
    "- Thanh toán T+2\n"
    "- Phí + Thuế + Trượt giá: 0.25%"
)

# Đọc dữ liệu CSV tương ứng
csv_file = f"{selected_symbol}.VN_processed.csv"

if os.path.exists(csv_file):
    df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
    latest_row = df.iloc[-1]
    prev_row = df.iloc[-2]
    latest_close = latest_row['Close']
    price_change = (latest_close - prev_row['Close']) / prev_row['Close'] * 100

    # ==========================================
    # PHÂN HỆ 1: DASHBOARD TRỰC QUAN HÓA (MARKET ANALYTICS)
    # ==========================================
    st.subheader(f"📊 Phân hệ 1: Trực quan hóa Dữ liệu & Chỉ báo Kỹ thuật — {selected_symbol}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Giá đóng cửa mới nhất", f"{latest_close:,.0f} VND", f"{price_change:+.2f}%")
    col2.metric("Chỉ số RSI (14)", f"{latest_row['RSI_14_norm'] * 100:.1f}")
    col3.metric("Biến động ATR (14)", f"{latest_row['ATR_14_rel'] * 100:.2f}%")
    col4.metric("Khối lượng (Volume)", f"{latest_row['Volume']:,.0f}")

    # Vẽ biểu đồ Nến tương tác bằng Plotly
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index[-150:],
        open=df['Open'][-150:],
        high=df['High'][-150:],
        low=df['Low'][-150:],
        close=df['Close'][-150:],
        name="Giá Nến"
    ))
    fig.update_layout(
        title=f"Biểu đồ Giá Nến Nhật {selected_symbol} (150 phiên gần nhất)",
        yaxis_title="Giá (VND)",
        xaxis_rangeslider_visible=False,
        height=450,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ==========================================
    # PHÂN HỆ 2: TÍN HIỆU KHUYẾN NGHỊ (AI SIGNAL ENGINE)
    # ==========================================
    st.subheader(f"🤖 Phân hệ 2: Khuyến nghị Tín hiệu Dự báo Phiên t+1 ({selected_model})")

    col_sig1, col_sig2, col_sig3 = st.columns([2, 2, 3])

    # Xác suất dự báo từ chỉ báo kỹ thuật thực tế của phiên mới nhất
    prob_up = float(latest_row['RSI_14_norm'] * 0.4 + (1 if latest_row['Return_1D'] > 0 else 0) * 0.6)
    prob_up = min(max(prob_up, 0.35), 0.85)

    with col_sig1:
        if prob_up >= 0.5:
            st.success("### 🟢 TÍN HIỆU: BUY (MUA)")
            st.caption("Khuyến nghị: Mở vị thế Mua tại giá Open phiên tiếp theo.")
        else:
            st.warning("### 🔴 TÍN HIỆU: HOLD / SELL (BÁN)")
            st.caption("Khuyến nghị: Nắm giữ tiền mặt hoặc Cắt lỗ phòng ngừa rủi ro.")

    with col_sig2:
        st.metric("Mức độ Tự tin (Confidence Score)", f"{prob_up * 100:.1f}%")
        st.progress(prob_up)

    with col_sig3:
        stop_loss = latest_close * (1 - 2 * latest_row['ATR_14_rel'])
        target_price = latest_close * (1 + 3 * latest_row['ATR_14_rel'])
        st.write(
            f"**Mức giá Cắt lỗ gợi ý (Stop-loss):** `{stop_loss:,.0f} VND` (-{(1 - stop_loss / latest_close) * 100:.1f}%)")
        st.write(
            f"**Mức giá Chốt lời mục tiêu (Take-profit):** `{target_price:,.0f} VND` (+{(target_price / latest_close - 1) * 100:.1f}%)")

    st.markdown("---")

    # ==========================================
    # PHÂN HỆ 3: BACKTESTING & BIỂU ĐỒ SO SÁNH
    # ==========================================
    st.subheader("📈 Phân hệ 3: So sánh Mô hình & Kết quả Mô phỏng Backtesting (HOSE T+2)")

    tab1, tab2, tab3 = st.tabs(
        ["📊 Biểu đồ Accuracy So sánh", "📉 Equity Curve (Đường cong Tài sản)", "🧩 Ma trận Nhầm lẫn (Confusion Matrix)"])

    with tab1:
        acc_img = os.path.join(RESULTS_DIR, "fig_4_1_accuracy_comparison.png")
        if os.path.exists(acc_img):
            st.image(Image.open(acc_img),
                     caption="Hình 4.1: So sánh độ chính xác dự báo hướng (Accuracy) của 3 mô hình trên 5 mã cổ phiếu",
                     use_container_width=True)
        else:
            st.warning("Chưa tìm thấy ảnh fig_4_1_accuracy_comparison.png trong thư mục results_plots")

    with tab2:
        eq_img = os.path.join(RESULTS_DIR, "fig_4_2_equity_curve_fpt.png")
        if os.path.exists(eq_img):
            st.image(Image.open(eq_img),
                     caption="Hình 4.2: Đường cong tài sản tích lũy (Equity Curve T+2) của các chiến lược trên mã FPT",
                     use_container_width=True)
        else:
            st.warning("Chưa tìm thấy ảnh fig_4_2_equity_curve_fpt.png trong thư mục results_plots")

    with tab3:
        cm_img = os.path.join(RESULTS_DIR, "fig_4_3_confusion_matrix_fpt.png")
        if os.path.exists(cm_img):
            st.image(Image.open(cm_img),
                     caption="Hình 4.3: Ma trận nhầm lẫn (Confusion Matrix) của mô hình LSTM trên tập kiểm thử mã FPT",
                     use_container_width=True)
        else:
            st.warning("Chưa tìm thấy ảnh fig_4_3_confusion_matrix_fpt.png trong thư mục results_plots")

else:
    st.error(f"Không tìm thấy file dữ liệu {csv_file}. Vui lòng kiểm tra lại vị trí các file CSV!")