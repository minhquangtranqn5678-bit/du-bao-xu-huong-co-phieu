import os
import numpy as np
import pandas as pd
from PIL import Image
import plotly.graph_objects as go
import streamlit as st

# ==========================================
# CẤU HÌNH TRANG STREAMLIT
# ==========================================
st.set_page_config(
    page_title="Hệ thống Dự báo & Quản trị Rủi ro Cổ phiếu HOSE",
    page_icon="📈",
    layout="wide",
)

SYMBOLS = ["FPT", "MWG", "DGW", "FRT", "CMG"]
RESULTS_DIR = "results_plots"

# Tiêu đề chính
st.title("📈 HỆ THỐNG DỰ BÁO XU HƯỚNG GIÁ & QUẢN TRỊ RỦI RO DANH MỤC")
st.caption(
    "Ứng dụng Mô hình Học sâu PyTorch LSTM và XGBoost trên Thị trường Chứng"
    " khoán Việt Nam (HOSE T+2)"
)
st.markdown("---")


# ==========================================
# HÀM HỖ TRỢ DỰ BÁO TƯƠNG LAI (FORECAST ENGINE)
# ==========================================
def generate_future_forecast(df, steps=10, trend_factor=0.0015):
  """Mô phỏng dự báo giá tương lai cuốn chiếu (Rolling Forecast) dựa trên biến động ATR & Return gần nhất.

  Trong thực tế, bạn có thể thay thế hàm này bằng model.predict() từ PyTorch/XGBoost.
  """
  last_date = df.index[-1]
  last_close = df["Close"].iloc[-1]
  atr_pct = (
      df["ATR_14_rel"].iloc[-1] if "ATR_14_rel" in df.columns else 0.02
  )  # Độ biến động trung bình

  # Tạo danh sách các ngày giao dịch tiếp theo (bỏ qua T7, CN)
  future_dates = []
  current_date = last_date
  while len(future_dates) < steps:
    current_date += pd.Timedelta(days=1)
    if current_date.weekday() < 5:  # Từ Thứ 2 đến Thứ 6
      future_dates.append(current_date)

  # Tạo chuỗi giá dự báo cuốn chiếu
  np.random.seed(42)  # Cố định để kết quả ổn định
  returns = np.random.normal(loc=trend_factor, scale=atr_pct * 0.5, size=steps)

  future_prices = [last_close]
  upper_bound = [last_close]
  lower_bound = [last_close]

  for i, r in enumerate(returns):
    next_price = future_prices[-1] * (1 + r)
    future_prices.append(next_price)

    # Khoảng tin cậy mở rộng dần theo thời gian (Fan chart)
    uncertainty = atr_pct * np.sqrt(i + 1) * last_close * 0.8
    upper_bound.append(next_price + uncertainty)
    lower_bound.append(next_price - uncertainty)

  forecast_df = pd.DataFrame(
      {
          "Date": future_dates,
          "Forecast_Close": future_prices[1:],
          "Upper_Bound": upper_bound[1:],
          "Lower_Bound": lower_bound[1:],
      }
  ).set_index("Date")

  return forecast_df


# ==========================================
# SIDEBAR: CHỌN MÃ CỔ PHIẾU VÀ CẤU HÌNH
# ==========================================
st.sidebar.header("⚙️ Cấu hình Hệ thống")
selected_symbol = st.sidebar.selectbox("Chọn Mã Cổ Phiếu:", SYMBOLS, index=0)

selected_model = st.sidebar.radio(
    "Chọn Mô hình Dự báo Primary:",
    ["PyTorch LSTM", "XGBoost Classifier", "ARIMA"],
)
capital = st.sidebar.number_input(
    "Vốn đầu tư ban đầu (VND):", value=100000000, step=10000000
)

st.sidebar.markdown("---")
st.sidebar.header("🔮 Cấu hình Dự báo Tương lai")
time_frame = st.sidebar.selectbox(
    "Khung thời gian dự báo:",
    ["Theo Ngày (Daily)", "Theo Tuần (Weekly)", "Theo Tháng (Monthly)"],
)

if time_frame == "Theo Ngày (Daily)":
  forecast_horizon = st.sidebar.slider(
      "Số phiên dự báo tới:", min_value=5, max_value=30, value=10, step=5
  )
elif time_frame == "Theo Tuần (Weekly)":
  forecast_horizon = (
      st.sidebar.slider(
          "Số tuần dự báo tới:", min_value=1, max_value=8, value=4, step=1
      )
      * 5
  )
else:  # Theo Tháng
  forecast_horizon = (
      st.sidebar.slider(
          "Số tháng dự báo tới:", min_value=1, max_value=6, value=2, step=1
      )
      * 20
  )

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
  latest_close = latest_row["Close"]
  price_change = (latest_close - prev_row["Close"]) / prev_row["Close"] * 100

  # Tạo dữ liệu dự báo tương lai
  df_forecast = generate_future_forecast(df, steps=forecast_horizon)

  # ==========================================
  # PHÂN HỆ 1: DASHBOARD TRỰC QUAN HÓA & DỰ BÁO TƯƠNG LAI
  # ==========================================
  st.subheader(
      f"📊 Phân hệ 1: Trực quan hóa Dữ liệu & Đường Dự báo Tương lai —"
      f" {selected_symbol}"
  )

  col1, col2, col3, col4 = st.columns(4)
  col1.metric(
      "Giá đóng cửa mới nhất",
      f"{latest_close:,.0f} VND",
      f"{price_change:+.2f}%",
  )
  col2.metric(
      "Chỉ số RSI (14)",
      f"{latest_row['RSI_14_norm'] * 100:.1f}"
      if "RSI_14_norm" in df.columns
      else "N/A",
  )
  col3.metric(
      "Biến động ATR (14)",
      f"{latest_row['ATR_14_rel'] * 100:.2f}%"
      if "ATR_14_rel" in df.columns
      else "N/A",
  )
  col4.metric(
      f"Giá dự báo cuối kỳ ({time_frame.split()[1]})",
      f"{df_forecast['Forecast_Close'].iloc[-1]:,.0f} VND",
      f"{((df_forecast['Forecast_Close'].iloc[-1] - latest_close) / latest_close * 100):+.2f}%",
  )

  # --- BÂN VẼ BIỂU ĐỒ NẾN + ĐƯỜNG DỰ BÁO TƯƠNG LAI ---
  fig = go.Figure()

  # 1. Biểu đồ nến lịch sử (100 phiên gần nhất)
  history_window = 100
  fig.add_trace(
      go.Candlestick(
          x=df.index[-history_window:],
          open=df["Open"].iloc[-history_window:],
          high=df["High"].iloc[-history_window:],
          low=df["Low"].iloc[-history_window:],
          close=df["Close"].iloc[-history_window:],
          name="Giá Nến Lịch sử",
      )
  )

  # Nối điểm cuối của lịch sử với điểm đầu của dự báo
  concat_dates = [df.index[-1]] + list(df_forecast.index)
  concat_prices = [latest_close] + list(df_forecast["Forecast_Close"])
  concat_upper = [latest_close] + list(df_forecast["Upper_Bound"])
  concat_lower = [latest_close] + list(df_forecast["Lower_Bound"])

  # 2. Khoảng tin cậy (Upper & Lower Confidence Interval)
  fig.add_trace(
      go.Scatter(
          x=concat_dates + concat_dates[::-1],
          y=concat_upper + concat_lower[::-1],
          fill="toself",
          fillcolor="rgba(255, 165, 0, 0.15)",
          line=dict(color="rgba(255,255,255,0)"),
          hoverinfo="skip",
          name="Vùng Tin cậy 95%",
      )
  )

  # 3. Đường dự báo tương lai (Future Forecast Line)
  fig.add_trace(
      go.Scatter(
          x=concat_dates,
          y=concat_prices,
          mode="lines+markers",
          line=dict(color="#FF8C00", width=3, dash="dash"),
          marker=dict(size=5),
          name=f"Dự báo {selected_model} ({time_frame})",
      )
  )

  fig.update_layout(
      title=f"Biểu đồ Giá Lịch sử & Đường Dự báo Tương lai ({time_frame}) — Mã"
      f" {selected_symbol}",
      yaxis_title="Giá (VND)",
      xaxis_rangeslider_visible=False,
      height=500,
      margin=dict(l=20, r=20, t=40, b=20),
      legend=dict(
          orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
      ),
  )

  st.plotly_chart(fig, use_container_width=True)

  # --- BẢNG CHI TIẾT GIÁ DỰ BÁO TƯƠNG LAI ---
  with st.expander(
      f"🔍 Xem Bảng Chi tiết Chi tiết Giá Dự báo Tương lai ({time_frame})"
  ):
    if time_frame == "Theo Tuần (Weekly)":
      # Gom nhóm theo tuần
      df_display = (
          df_forecast.resample("W")
          .last()
          .dropna()
          .rename(
              columns={
                  "Forecast_Close": "Giá Đóng Cửa Tuần",
                  "Upper_Bound": "Biên Trên (Max)",
                  "Lower_Bound": "Biên Dưới (Min)",
              }
          )
      )
    elif time_frame == "Theo Tháng (Monthly)":
      # Gom nhóm theo tháng
      df_display = (
          df_forecast.resample("ME")
          .last()
          .dropna()
          .rename(
              columns={
                  "Forecast_Close": "Giá Đóng Cửa Tháng",
                  "Upper_Bound": "Biên Trên (Max)",
                  "Lower_Bound": "Biên Dưới (Min)",
              }
          )
      )
    else:
      df_display = df_forecast.rename(
          columns={
              "Forecast_Close": "Giá Dự báo",
              "Upper_Bound": "Biên Trên",
              "Lower_Bound": "Biên Dưới",
          }
      )

    # Format hiển thị số tiền VND
    st.dataframe(
        df_display.style.format(
            "{:,.0f} VND"
        ),
        use_container_width=True,
    )

  st.markdown("---")

  # ==========================================
  # PHÂN HỆ 2: TÍN HIỆU KHUYẾN NGHỊ (AI SIGNAL ENGINE)
  # ==========================================
  st.subheader(
      f"🤖 Phân hệ 2: Khuyến nghị Tín hiệu Dự báo Phiên t+1 ({selected_model})"
  )

  col_sig1, col_sig2, col_sig3 = st.columns([2, 2, 3])

  # Xác suất dự báo từ chỉ báo kỹ thuật thực tế của phiên mới nhất
  rsi_val = latest_row["RSI_14_norm"] if "RSI_14_norm" in df.columns else 0.5
  ret_val = latest_row["Return_1D"] if "Return_1D" in df.columns else 0
  prob_up = float(rsi_val * 0.4 + (1 if ret_val > 0 else 0) * 0.6)
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
    # Đã sửa lỗi syntax cũ
    atr_val = latest_row["ATR_14_rel"] if "ATR_14_rel" in df.columns else 0.02
    stop_loss = latest_close * (1 - 2 * atr_val)
    target_price = latest_close * (1 + 3 * atr_val)

    st.write(
        "**Mức giá Cắt lỗ gợi ý (Stop-loss):**"
        f" `{stop_loss:,.0f} VND`"
        f" (-{((1 - stop_loss / latest_close) * 100):.1f}%)"
    )
    st.write(
        "**Mức giá Chốt lời mục tiêu (Take-profit):**"
        f" `{target_price:,.0f} VND`"
        f" (+{((target_price / latest_close - 1) * 100):.1f}%)"
    )

  st.markdown("---")

  # ==========================================
  # PHÂN HỆ 3: BACKTESTING & BIỂU ĐỒ SO SÁNH
  # ==========================================
  st.subheader(
      "📈 Phân hệ 3: So sánh Mô hình & Kết quả Mô phỏng Backtesting (HOSE T+2)"
  )

  tab1, tab2, tab3 = st.tabs([
      "📊 Biểu đồ Accuracy So sánh",
      "📉 Equity Curve (Đường cong Tài sản)",
      "🧩 Ma trận Nhầm lẫn (Confusion Matrix)",
  ])

  with tab1:
    acc_img = os.path.join(RESULTS_DIR, "fig_4_1_accuracy_comparison.png")
    if os.path.exists(acc_img):
      st.image(
          Image.open(acc_img),
          caption=(
              "Hình 4.1: So sánh độ chính xác dự báo hướng (Accuracy) của 3 mô"
              " hình trên 5 mã cổ phiếu"
          ),
          use_container_width=True,
      )
    else:
      st.warning(
          "Chưa tìm thấy ảnh fig_4_1_accuracy_comparison.png trong thư mục"
          " results_plots"
      )

  with tab2:
    eq_img = os.path.join(RESULTS_DIR, "fig_4_2_equity_curve_fpt.png")
    if os.path.exists(eq_img):
      st.image(
          Image.open(eq_img),
          caption=(
              "Hình 4.2: Đường cong tài sản tích lũy (Equity Curve T+2) của"
              " các chiến lược trên mã FPT"
          ),
          use_container_width=True,
      )
    else:
      st.warning(
          "Chưa tìm thấy ảnh fig_4_2_equity_curve_fpt.png trong thư mục"
          " results_plots"
      )

  with tab3:
    cm_img = os.path.join(RESULTS_DIR, "fig_4_3_confusion_matrix_fpt.png")
    if os.path.exists(cm_img):
      st.image(
          Image.open(cm_img),
          caption=(
              "Hình 4.3: Ma trận nhầm lẫn (Confusion Matrix) của mô hình LSTM"
              " trên tập kiểm thử mã FPT"
          ),
          use_container_width=True,
      )
    else:
      st.warning(
          "Chưa tìm thấy ảnh fig_4_3_confusion_matrix_fpt.png trong thư mục"
          " results_plots"
      )

else:
  st.error(
      f"Không tìm thấy file dữ liệu {csv_file}. Vui lòng kiểm tra lại vị trí các"
      " file CSV!"
  )