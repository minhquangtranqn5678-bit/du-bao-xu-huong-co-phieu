import os
import time
import requests
import numpy as np
import pandas as pd

# Danh sách mã chứng khoán HOSE
TICKERS = ['FPT', 'MWG', 'DGW', 'FRT', 'CMG']
START_DATE = '2018-01-01'
END_DATE = '2026-09-30'


def fetch_vietnam_stock_data(symbol, start_date, end_date):
    """
    Tải dữ liệu lịch sử giá cổ phiếu trực tiếp từ API chứng khoán Việt Nam.
    """
    print(f"--- Đang tải dữ liệu cho mã: {symbol} ---")
    start_ts = int(pd.Timestamp(start_date).timestamp())
    end_ts = int(pd.Timestamp(end_date).timestamp())

    url = f"https://dchart-api.vndirect.com.vn/dchart/history?resolution=D&symbol={symbol}&from={start_ts}&to={end_ts}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    response = requests.get(url, headers=headers, timeout=15)
    data = response.json()

    if data.get('s') != 'ok':
        raise ValueError(f"Không thể lấy dữ liệu cho mã {symbol}")

    df = pd.DataFrame({
        'Date': pd.to_datetime(data['t'], unit='s'),
        'Open': data['o'],
        'High': data['h'],
        'Low': data['l'],
        'Close': data['c'],
        'Volume': data['v']
    })

    df.set_index('Date', inplace=True)
    df = df[df['Volume'] > 0].copy()
    return df


def build_18_features(df):
    """
    Tính toán hệ thống 18 đặc trưng tương đối theo Bảng 1 & Bảng 8 trong báo cáo.
    """
    data = df.copy()
    C = data['Close']
    H = data['High']
    L = data['Low']
    V = data['Volume']

    # 1-5. SMA & EMA Tương đối
    data['SMA_20_rel'] = C / C.rolling(20).mean() - 1.0
    data['SMA_50_rel'] = C / C.rolling(50).mean() - 1.0
    data['SMA_200_rel'] = C / C.rolling(200).mean() - 1.0

    ema12 = C.ewm(span=12, adjust=False).mean()
    ema26 = C.ewm(span=26, adjust=False).mean()
    data['EMA_12_rel'] = C / ema12 - 1.0
    data['EMA_26_rel'] = C / ema26 - 1.0

    # 6-7. MACD & Signal Tương đối
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    data['MACD_rel'] = macd / C
    data['MACD_Signal_rel'] = macd_signal / C

    # 8. RSI_14 Chuẩn hóa
    delta = C.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    data['RSI_14_norm'] = (100.0 - (100.0 / (1.0 + rs))) / 100.0

    # 9. Stochastic %K_14 Chuẩn hóa
    low_14 = L.rolling(14).min()
    high_14 = H.rolling(14).max()
    data['STOCH_K_norm'] = ((C - low_14) / (high_14 - low_14 + 1e-9))

    # 10-12. Bollinger Bands Tương đối & Bandwidth
    sma20 = C.rolling(20).mean()
    std20 = C.rolling(20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    data['BB_Upper_rel'] = bb_upper / C - 1.0
    data['BB_Lower_rel'] = bb_lower / C - 1.0
    data['BB_Bandwidth'] = (bb_upper - bb_lower) / (sma20 + 1e-9)

    # 13. ATR_14 Tương đối
    tr1 = H - L
    tr2 = (H - C.shift(1)).abs()
    tr3 = (L - C.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    data['ATR_14_rel'] = atr14 / C

    # 14. OBV Flow 20 phiên
    obv_change = np.sign(C.diff()).fillna(0) * V
    obv = obv_change.cumsum()
    data['OBV_Flow_20'] = (obv - obv.shift(20)) / (V.rolling(20).sum() + 1e-9)

    # 15. VROC_14
    data['VROC_14'] = np.log(V / (V.shift(14) + 1e-9))

    # 16-18. Lợi suất trễ Logarit
    data['Return_1D'] = np.log(C / C.shift(1))
    data['Return_5D'] = np.log(C / C.shift(5))
    data['Return_10D'] = np.log(C / C.shift(10))

    # Nhãn Target: y_{t+1} = 1 nếu Close_{t+1} > Close_t
    data['Target'] = (C.shift(-1) > C).astype(int)

    # Loại bỏ 200 phiên đầu thiếu SMA200 và phiên cuối thiếu Target
    data = data.iloc[200:-1].copy()
    return data


if __name__ == "__main__":
    for ticker in TICKERS:
        try:
            df_raw = fetch_vietnam_stock_data(ticker, START_DATE, END_DATE)
            df_featured = build_18_features(df_raw)

            # Lưu file dưới dạng TICKER.VN_processed.csv để khớp với file 2_model_and_evaluation.py
            output_filename = f"{ticker}.VN_processed.csv"
            df_featured.to_csv(output_filename)
            print(f"-> Đã tạo thành công: {output_filename} ({len(df_featured)} phiên giao dịch)\n")
            time.sleep(1)  # Nghỉ 1s giữa các request
        except Exception as e:
            print(f"Xảy ra lỗi khi xử lý mã {ticker}: {e}")