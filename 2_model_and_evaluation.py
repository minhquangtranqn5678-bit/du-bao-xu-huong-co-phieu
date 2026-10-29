import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBClassifier

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. CẤU HÌNH HỆ THỐNG & ĐẶC TRƯNG CHUẨN (BẢNG 1 & BẢNG 8)
# ==========================================
FEATURE_COLS = [
    'SMA_20_rel', 'SMA_50_rel', 'SMA_200_rel', 'EMA_12_rel', 'EMA_26_rel',
    'MACD_rel', 'MACD_Signal_rel', 'RSI_14_norm', 'STOCH_K_norm',
    'BB_Upper_rel', 'BB_Lower_rel', 'BB_Bandwidth', 'ATR_14_rel',
    'OBV_Flow_20', 'VROC_14', 'Return_1D', 'Return_5D', 'Return_10D'
]

SYMBOLS = ["FPT", "MWG", "DGW", "FRT", "CMG"]
LOOKBACK = 10  # Cửa sổ W = 10 phiên theo Mục 3.3.4
TRAIN_RATIO = 0.8

# Thư mục lưu đồ thị
OS_DIR = "results_plots"
os.makedirs(OS_DIR, exist_ok=True)

# Cấu hình đồ thị
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


# ==========================================
# 2. KIẾN TRÚC MẠNG LSTM CHUẨN MỤC 3.3.4 (PyTorch)
# ==========================================
class LSTMPredictor(nn.Module):
    def __init__(self, input_dim):
        super(LSTMPredictor, self).__init__()
        # Lớp 1: LSTM(64) -> Dropout(0.2)
        self.lstm1 = nn.LSTM(input_dim, 64, batch_first=True)
        self.drop1 = nn.Dropout(0.2)
        # Lớp 2: LSTM(32) -> Dropout(0.2)
        self.lstm2 = nn.LSTM(64, 32, batch_first=True)
        self.drop2 = nn.Dropout(0.2)
        # Lớp Dense(16, ReLU) -> Dense(1, Sigmoid)
        self.fc1 = nn.Linear(32, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.drop1(out)
        out, _ = self.lstm2(out)
        out = self.drop2(out[:, -1, :])  # Lấy hidden state phiên cuối
        out = self.relu(self.fc1(out))
        out = self.fc2(out)  # Xuất Logits
        return out


def create_sequences(data, window=10):
    seqs = []
    for i in range(len(data) - window + 1):
        seqs.append(data[i:i + window])
    return np.array(seqs)


# ==========================================
# 3. ENGINE BACKTESTING CHUẨN HOSE T+2
# ==========================================
def run_backtest_hose(df_test, signals, fee=0.001, tax=0.001, slippage=0.0005, initial_capital=1.0):
    """
    Mô phỏng giao dịch thực tế tuân thủ quy tắc sàn HOSE:
    - Khớp lệnh tại giá Open_{t+1}.
    - Ràng buộc thanh toán T+2.
    - Phí + Thuế + Trượt giá chuẩn.
    - Biên độ trần/sàn (+-6.9%).
    """
    cash = initial_capital
    holdings = 0.0
    buy_session = -999
    equity_curve = []

    closes = df_test['Close'].values
    opens = df_test['Open'].values

    for t in range(len(df_test) - 1):
        current_close = closes[t]
        next_open = opens[t + 1]
        sig = signals[t]

        price_change = (next_open - current_close) / current_close
        is_ceiling = price_change >= 0.069
        is_floor = price_change <= -0.069

        # Mua khi có tín hiệu
        if sig == 1 and holdings == 0 and not is_ceiling:
            buy_price = next_open * (1 + slippage)
            invest_amount = cash * (1 - fee)
            holdings = invest_amount / buy_price
            cash = 0.0
            buy_session = t + 1

        # Bán khi ngắt tín hiệu & Thỏa T+2
        elif sig == 0 and holdings > 0 and (t + 1 >= buy_session + 2) and not is_floor:
            sell_price = next_open * (1 - slippage)
            gross_proceeds = holdings * sell_price
            net_proceeds = gross_proceeds * (1 - fee - tax)
            cash = net_proceeds
            holdings = 0.0

        current_equity = cash + holdings * current_close
        equity_curve.append(current_equity)

    current_equity = cash + holdings * closes[-1]
    equity_curve.append(current_equity)

    equity_series = pd.Series(equity_curve, index=df_test.index)
    cum_return = (equity_series.iloc[-1] - initial_capital) / initial_capital

    # Max Drawdown
    roll_max = equity_series.cummax()
    drawdown = (equity_series - roll_max) / (roll_max + 1e-9)
    mdd = drawdown.min()

    # Sharpe Ratio với Rf = 4%/năm
    daily_returns = equity_series.pct_change().dropna()
    rf_daily = 0.04 / 252
    excess_returns = daily_returns - rf_daily
    sharpe = np.sqrt(252) * (excess_returns.mean() / (excess_returns.std() + 1e-9)) if excess_returns.std() > 0 else 0.0

    return cum_return, mdd, sharpe, equity_series


# ==========================================
# 4. QUY TRÌNH THỰC NGHIỆM ĐÁNH GIÁ MÔ HÌNH
# ==========================================
def evaluate_symbol(sym_name):
    file_path = f"{sym_name}.VN_processed.csv"
    if not os.path.exists(file_path):
        print(f"Không tìm thấy file: {file_path}")
        return None

    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    split_idx = int(len(df) * TRAIN_RATIO)

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    # Baseline Accuracy
    majority_class = train_df['Target'].mode()[0]
    baseline_acc = (test_df['Target'] == majority_class).mean()

    # MinMax Scaling CHUẨN: Fit trên Train, Transform sang Test
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(train_df[FEATURE_COLS])
    X_test_scaled = scaler.transform(test_df[FEATURE_COLS])

    y_train = train_df['Target'].values
    y_test = test_df['Target'].values

    # --- A. BASELINE MODEL: ARIMA Rolling Forecast ---
    print(f"[{sym_name}] Đang chạy ARIMA Rolling Forecast...")
    arima_preds = []
    close_prices = df['Close'].values

    for t in range(len(test_df)):
        try:
            history = close_prices[: split_idx + t]
            model_arima = ARIMA(history[-100:], order=(1, 1, 1)).fit()
            pred_val = model_arima.forecast()[0]
            prev_price = history[-1]
            arima_preds.append(1 if pred_val > prev_price else 0)
        except:
            arima_preds.append(0)
    arima_preds = np.array(arima_preds)

    # --- B. MACHINE LEARNING: XGBOOST (TimeSeriesSplit) ---
    print(f"[{sym_name}] Đang tinh chỉnh XGBoost...")
    tscv = TimeSeriesSplit(n_splits=3)
    best_score = float('inf')
    best_params = {'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 100}

    for md in [3, 5]:
        for lr in [0.03, 0.05]:
            for n_est in [100, 300]:
                losses = []
                for tr_i, val_i in tscv.split(X_train_scaled):
                    X_tr, X_va = X_train_scaled[tr_i], X_train_scaled[val_i]
                    y_tr, y_va = y_train[tr_i], y_train[val_i]

                    pos_w = (len(y_tr) - sum(y_tr)) / (sum(y_tr) + 1e-9)
                    xgb = XGBClassifier(max_depth=md, learning_rate=lr, n_estimators=n_est,
                                        scale_pos_weight=pos_w, random_state=42, eval_metric='logloss')
                    xgb.fit(X_tr, y_tr)
                    probs = xgb.predict_proba(X_va)[:, 1]
                    losses.append(-np.mean(y_va * np.log(probs + 1e-9) + (1 - y_va) * np.log(1 - probs + 1e-9)))
                if np.mean(losses) < best_score:
                    best_score = np.mean(losses)
                    best_params = {'max_depth': md, 'learning_rate': lr, 'n_estimators': n_est}

    pos_w_full = (len(y_train) - sum(y_train)) / (sum(y_train) + 1e-9)
    xgb_final = XGBClassifier(**best_params, scale_pos_weight=pos_w_full, random_state=42, eval_metric='logloss')
    xgb_final.fit(X_train_scaled, y_train)

    xgb_probs = xgb_final.predict_proba(X_test_scaled)[:, 1]
    xgb_preds = (xgb_probs >= 0.5).astype(int)

    # --- C. DEEP LEARNING: LSTM ---
    print(f"[{sym_name}] Đang huấn luyện PyTorch LSTM...")
    # Lấy bù LOOKBACK-1 mẫu cuối Train để tập Test LSTM đồng bộ hoàn toàn số phiên
    X_full_scaled = np.vstack([X_train_scaled[-(LOOKBACK - 1):], X_test_scaled])

    X_train_seq = create_sequences(X_train_scaled, window=LOOKBACK)
    y_train_seq = y_train[LOOKBACK - 1:]
    X_test_seq = create_sequences(X_full_scaled, window=LOOKBACK)

    train_dataset = TensorDataset(
        torch.tensor(X_train_seq, dtype=torch.float32),
        torch.tensor(y_train_seq, dtype=torch.float32).unsqueeze(1)
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    num_pos = np.sum(y_train_seq == 1)
    num_neg = np.sum(y_train_seq == 0)
    pos_weight = torch.tensor([num_neg / (num_pos + 1e-5)], dtype=torch.float32)

    model = LSTMPredictor(input_dim=len(FEATURE_COLS))
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(40):
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test_seq, dtype=torch.float32)).flatten()
        lstm_probs = torch.sigmoid(logits).numpy()

    lstm_preds = (lstm_probs >= 0.5).astype(int)

    # --- D. TÍNH CHỈ SỐ HỌC MÁY (ML METRICS) ---
    ml_metrics = {
        "Symbol": sym_name,
        "Baseline_Acc": baseline_acc,
        "ARIMA_Acc": accuracy_score(y_test, arima_preds),
        "ARIMA_Prec": precision_score(y_test, arima_preds, zero_division=0),
        "ARIMA_Rec": recall_score(y_test, arima_preds, zero_division=0),
        "ARIMA_F1": f1_score(y_test, arima_preds, zero_division=0),
        "XGB_Acc": accuracy_score(y_test, xgb_preds),
        "XGB_Prec": precision_score(y_test, xgb_preds, zero_division=0),
        "XGB_Rec": recall_score(y_test, xgb_preds, zero_division=0),
        "XGB_F1": f1_score(y_test, xgb_preds, zero_division=0),
        "XGB_AUC": roc_auc_score(y_test, xgb_probs),
        "LSTM_Acc": accuracy_score(y_test, lstm_preds),
        "LSTM_Prec": precision_score(y_test, lstm_preds, zero_division=0),
        "LSTM_Rec": recall_score(y_test, lstm_preds, zero_division=0),
        "LSTM_F1": f1_score(y_test, lstm_preds, zero_division=0),
        "LSTM_AUC": roc_auc_score(y_test, lstm_probs),
    }

    # --- E. BACKTESTING CHUẨN HOSE T+2 ---
    bh_cr, bh_mdd, bh_sharpe, bh_eq = run_backtest_hose(test_df, np.ones(len(test_df)))
    arima_cr, arima_mdd, arima_sharpe, arima_eq = run_backtest_hose(test_df, arima_preds)
    xgb_cr, xgb_mdd, xgb_sharpe, xgb_eq = run_backtest_hose(test_df, xgb_preds)
    lstm_cr, lstm_mdd, lstm_sharpe, lstm_eq = run_backtest_hose(test_df, lstm_preds)

    backtest_metrics = {
        "Symbol": sym_name,
        "BNH_TCR": f"{bh_cr * 100:.2f}%", "BNH_Sharpe": f"{bh_sharpe:.2f}", "BNH_MDD": f"{bh_mdd * 100:.2f}%",
        "ARIMA_TCR": f"{arima_cr * 100:.2f}%", "ARIMA_Sharpe": f"{arima_sharpe:.2f}",
        "ARIMA_MDD": f"{arima_mdd * 100:.2f}%",
        "XGB_TCR": f"{xgb_cr * 100:.2f}%", "XGB_Sharpe": f"{xgb_sharpe:.2f}", "XGB_MDD": f"{xgb_mdd * 100:.2f}%",
        "LSTM_TCR": f"{lstm_cr * 100:.2f}%", "LSTM_Sharpe": f"{lstm_sharpe:.2f}", "LSTM_MDD": f"{lstm_mdd * 100:.2f}%"
    }

    equity_curves = {
        "Dates": test_df.index,
        "Buy & Hold": bh_eq,
        "ARIMA Strategy": arima_eq,
        "XGBoost Strategy": xgb_eq,
        "LSTM Strategy": lstm_eq
    }

    raw_preds = {
        "y_true": y_test,
        "lstm_preds": lstm_preds,
        "xgb_preds": xgb_preds
    }

    return ml_metrics, backtest_metrics, equity_curves, raw_preds


# ==========================================
# 5. THỰC THI & LƯU BÁO CÁO / BIỂU ĐỒ
# ==========================================
if __name__ == "__main__":
    all_ml, all_bt = [], []
    equity_dict, raw_preds_dict = {}, {}

    print(">>> BẮT ĐẦU HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH...")
    for sym in SYMBOLS:
        res = evaluate_symbol(sym)
        if res is not None:
            ml_m, bt_m, eq_c, raw_p = res
            all_ml.append(ml_m)
            all_bt.append(bt_m)
            equity_dict[sym] = eq_c
            raw_preds_dict[sym] = raw_p

    df_ml_all = pd.DataFrame(all_ml)
    df_bt_all = pd.DataFrame(all_bt)

    print("\n=================== BẢNG 4.1: CHỈ SỐ HỌC MÁY (ML METRICS) ===================")
    print(df_ml_all.round(4).to_string(index=False))

    print("\n=================== BẢNG 4.2: CHỈ SỐ BACKTESTING TÀI CHÍNH (HOSE T+2) ===================")
    print(df_bt_all.to_string(index=False))

    # 1. Biểu đồ Accuracy So sánh 3 Mô hình
    plt.figure(figsize=(10, 5))
    x = np.arange(len(df_ml_all["Symbol"]))
    width = 0.2
    plt.bar(x - width, df_ml_all["ARIMA_Acc"], width, label="ARIMA", color="#7f7f7f")
    plt.bar(x, df_ml_all["XGB_Acc"], width, label="XGBoost", color="#1f77b4")
    plt.bar(x + width, df_ml_all["LSTM_Acc"], width, label="LSTM", color="#2ca02c")
    plt.xlabel("Mã cổ phiếu")
    plt.ylabel("Accuracy")
    plt.title("Hình 4.1: So sánh độ chính xác dự báo hướng (Accuracy) của 3 mô hình")
    plt.xticks(x, df_ml_all["Symbol"])
    plt.ylim(0.35, 0.75)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OS_DIR, "fig_4_1_accuracy_comparison.png"), dpi=300)
    plt.close()

    # 2. Biểu đồ Equity Curve cho FPT
    if "FPT" in equity_dict:
        eq = equity_dict["FPT"]
        plt.figure(figsize=(10, 5))
        plt.plot(eq["Dates"], eq["Buy & Hold"], label="Buy & Hold", color="black", linestyle="--")
        plt.plot(eq["Dates"], eq["ARIMA Strategy"], label="ARIMA Strategy", color="blue")
        plt.plot(eq["Dates"], eq["XGBoost Strategy"], label="XGBoost Strategy", color="orange")
        plt.plot(eq["Dates"], eq["LSTM Strategy"], label="LSTM Strategy", color="green")
        plt.title("Hình 4.2: Đường cong tài sản tích lũy (Equity Curve T+2) - FPT")
        plt.ylabel("Giá trị danh mục (Gốc = 1.0)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OS_DIR, "fig_4_2_equity_curve_fpt.png"), dpi=300)
        plt.close()

    # 3. Ma trận nhầm lẫn Confusion Matrix cho FPT
    if "FPT" in raw_preds_dict:
        cm = confusion_matrix(raw_preds_dict["FPT"]["y_true"], raw_preds_dict["FPT"]["lstm_preds"])
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Không tăng (0)", "Tăng (1)"],
                    yticklabels=["Không tăng (0)", "Tăng (1)"])
        plt.title("Hình 4.3: Ma trận nhầm lẫn (Confusion Matrix) LSTM - FPT")
        plt.xlabel("Dự báo")
        plt.ylabel("Thực tế")
        plt.tight_layout()
        plt.savefig(os.path.join(OS_DIR, "fig_4_3_confusion_matrix_fpt.png"), dpi=300)
        plt.close()

    print(f"\n>>> ĐÃ HOÀN TẤT! Toàn bộ biểu đồ đã được lưu vào thư mục: {OS_DIR}/")