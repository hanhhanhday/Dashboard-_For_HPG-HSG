import streamlit as st
import pandas as pd
import numpy as np
import joblib, re
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
from underthesea import word_tokenize


# Forecast function implementations
def create_features(data):
    """
    Create technical indicators and statistical features
    """
    # Price movements
    data['Price_Range'] = data['high'] - data['low']
    data['Daily_Return'] = data['close'].pct_change()
    
    # Moving averages
    data['MA_5'] = data['close'].rolling(window=5).mean()
    data['MA_10'] = data['close'].rolling(window=10).mean()
    data['MA_20'] = data['close'].rolling(window=20).mean()
    
    # Momentum indicators
    data['Momentum_5'] = data['close'].diff(5)
    data['Momentum_10'] = data['close'].diff(10)
    
    # Volatility
    data['Volatility_5'] = data['close'].rolling(window=5).std()
    data['Volatility_10'] = data['close'].rolling(window=10).std()
    
    # Relative Strength Index (RSI)
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Drop NA values from feature creation
    data = data.dropna()
    
    return data

def normalize_data(data):
    """
    Normalize features using MinMaxScaler
    """
    features = ['open', 'high', 'low', 'close', 'volume', 'Price_Range', 'Daily_Return',
                'MA_5', 'MA_10', 'MA_20', 'Momentum_5', 'Momentum_10', 'Volatility_5',
                'Volatility_10', 'RSI']
    
    # Initialize scalers
    close_scaler = MinMaxScaler(feature_range=(0, 1))
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Scale close price separately (target)
    data['Close_scaled'] = close_scaler.fit_transform(data[['close']])
    
    # Scale other features
    data[features] = feature_scaler.fit_transform(data[features])
    
    return data, close_scaler, feature_scaler

def create_sequences(data, time_steps, target='Close_scaled'):
    X, y = [], []
    features = ['open', 'high', 'low', 'Close_scaled', 'volume', 'Price_Range', 'Daily_Return',
                'MA_5', 'MA_10', 'MA_20', 'Momentum_5', 'Momentum_10', 'Volatility_5',
                'Volatility_10', 'RSI']
    for i in range(len(data) - time_steps):
        X.append(data.iloc[i : i + time_steps][features].values)
        y.append(data.iloc[i + time_steps][target]) 
    return np.array(X), np.array(y)

def forecast_next_days_meta(meta_model, lstm_model, xgb_model, df, time_steps, n_days):
    # 1) Scale & features
    data_feat = create_features(df)
    scaled_data, scaler_target, scaler_feat = normalize_data(data_feat)
    arr = scaled_data.values
    N, total_feat = arr.shape

    # 2) Window for LSTM
    n_feat_lstm = lstm_model.input_shape[2]
    window = arr[-time_steps:, :n_feat_lstm].reshape(1, time_steps, n_feat_lstm)

    preds_meta_scaled = []
    for _ in range(n_days):
        # Base predictions (scaled)
        p_lstm = lstm_model.predict(window)[0,0]
        flat   = window.reshape(1, time_steps * n_feat_lstm)
        p_xgb  = xgb_model.predict(flat)[0]

        # residual và diff 
        last_scaled = window[0, -1, 0]
        res_lstm = last_scaled - p_lstm
        res_xgb  = last_scaled - p_xgb
        diff     = p_lstm - p_xgb
        abs_diff = abs(diff)

        # create 6 feature
        X_meta = np.array([[p_lstm, p_xgb, res_lstm, res_xgb, diff, abs_diff]])

        # Meta prediction (scaled)
        p_meta = meta_model.predict(X_meta)[0]
        preds_meta_scaled.append(p_meta)

        # Shift window and update
        window = np.roll(window, -1, axis=1)
        if n_feat_lstm > 1:
            window[0, -1, 1:] = window[0, -2, 1:]
        window[0, -1, 0] = p_meta

    # Un‑scale
    return scaler_target.inverse_transform(
        np.array(preds_meta_scaled).reshape(-1,1)
    ).ravel()

def print_forecasts_with_real_dates_meta(company, results_meta_model, stocks, forecast_func, time_steps, n_days):
    records = []
    print(f"\nDự đoán {n_days} ngày tiếp theo cho công ty {company}:")
    
    # 1) Initial DataFrame and index
    df = stocks[company]
    idx = df.index
    
    # 2) Determine freq of index ('B'  là business day, 'D' la daily)
    #    Nếu infer_freq trả về None, fallback về 'B'
    freq = pd.infer_freq(idx)
    if freq is None:
        freq = 'B'
    
    # 3) Take last date from index
    last_date = idx[-1]
    
    # 4) Create n_days next days with freq
    #    create date_range 1 steps longer, then remove first element
    all_dates = pd.date_range(start=last_date, periods=n_days + 1, freq=freq)
    next_dates = all_dates[1:]
    
    # 5) Take array scaled_close used in training process
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_close = scaler.fit_transform(df[['close']].values).ravel()
    
    # 6) Predict next n_days
    preds_meta = forecast_func(
        meta_model = results_meta_model[company]['meta_model'],
        lstm_model  = results_LSTM[company]['model'],
        xgb_model   = results_xgb[company]['model'],
        df          = df,
        time_steps = 180,
        n_days     = 5
    )
    # Lấy lịch sử giá để tính min/max 30 nearest days
    history_prices = df['close'].values[-30:]
    signals = []
    for i, p in enumerate(preds_meta):
        window = list(history_prices) + list(preds_meta[:i+1])
        if len(window) > 30:
            window = window[-30:]
        if p <= min(window):
            sig = 'BUY'
        elif p >= max(window):
            sig = 'SELL'
        else:
            sig = 'HOLD'
        signals.append(sig)

    # tạo record
    for date, price, sig in zip(next_dates, preds_meta, signals):
        records.append({
            'Công ty': company,
            'Ngày': date,
            'Giá dự đoán': price,
            'Đề xuất': sig
        })

    return pd.DataFrame(records)

# Hàm làm sạch text
def clean_text(text):
    # Thay Covid-19 thành Covid
    text = text.replace('Covid-19', 'Covid')
    
    # Thay thế tên tổ chức, người, địa điểm, mã ck... bằng 'name' / 'loc' / 'percent'
    # Ở đây ta sẽ đơn giản: thay ticker (tenma) và tên công ty (tenct)
    for ticker in tenma:
        text = re.sub(rf'\b{re.escape(ticker)}\b', 'name', text)
    for company in tenct:
        text = re.sub(rf'\b{re.escape(company)}\b', 'name', text, flags=re.IGNORECASE)
    
    # Thay số % → 'percent'
    text = re.sub(r'\d+\.?\d*%', 'percent', text)
    # Thay dates patterns → 'date'
    text = re.sub(r'\b(?:quý|Quý)\s*[1-4]\b', 'date', text)
    text = re.sub(r'\b(?:tháng|Tháng)\s*\d{1,2}\b', 'date', text)
    text = re.sub(r'\bnăm\s*\d{4}\b', 'date', text)
    text = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', 'date', text)
    
    # Thay số thuần túy → 'number'
    text = re.sub(r'\b\d+(\.\d+)?\b', 'number', text)
    
    # Loại bỏ ký tự đặc biệt
    text = re.sub(r'[\"“”\[\]\(\)\.,:;]', ' ', text)
    text = re.sub(r'[-–]', ' ', text)
    
    # tokenization và lowercase
    tokens = word_tokenize(text, format="text")
    return tokens.lower()


# ---- Load Data ----
@st.cache_data
def load_data():
    stocks = {}
    for ticker in ['HPG','HSG']:
        path = f'data/{ticker}_price.csv'
        df = pd.read_csv(path, parse_dates=['time'], index_col='time')
        stocks[ticker] = df
    return stocks
@st.cache_data
def load_news():
    # Load danh sách công ty và mã
    macp = pd.read_excel(r'C:\Users\user\Documents\python\Samsung\Data\Macp.xlsx').dropna()
    tenct = macp['Tên Công ty'].tolist()
    for i in range(len(tenct)):
        tenct[i] = str(tenct[i]).lower()
    tenma = macp['Mã '].tolist()
    return tenct, tenma

stocks = load_data()
tenct, tenma = load_news()

# ---- Load Models ----
@st.cache_resource
def load_models():
    lstm = joblib.load(r"C:\Users\user\Documents\python\Samsung\results_LSTM_full.pkl")
    xgb = joblib.load(r"C:\Users\user\Documents\python\Samsung\results_xgb_full.pkl")
    meta = joblib.load(r"C:\Users\user\Documents\python\Samsung\results_meta_model_full.pkl")
    # sentiment pipeline
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    sent_model = AutoModelForSequenceClassification.from_pretrained("phobert-sentiment-stock")
    pipe = TextClassificationPipeline(model=sent_model, tokenizer=tokenizer, framework="pt", return_all_scores=False)
    return lstm, xgb, meta, pipe

results_LSTM, results_xgb, results_meta_model, sentiment_pipe  = load_models()


# ---- Prepare Performance DataFrame ----
LSTM_perf = pd.DataFrame({
    comp: {'MAE': res['MAE'], 'MSE': res['MSE'], 'R2': res['R2']} 
    for comp, res in results_LSTM.items()
}).T.assign(Model='LSTM')
XGB_perf = pd.DataFrame({
    comp: {'MAE': res['MAE'], 'MSE': res['MSE'], 'R2': res['R2']} 
    for comp, res in results_xgb.items()
}).T.assign(Model='XGBoost')
Meta_perf = pd.DataFrame({
    comp: {'MAE': res['MAE'], 'MSE': res['MSE'], 'R2': res['R2']} 
    for comp, res in results_meta_model.items()
}).T.assign(Model='Meta')
records = []
for model_name, df in [
    ('LSTM',    LSTM_perf),
    ('XGBoost', XGB_perf),
    ('Meta',    Meta_perf)
]:
    for company in df.index:
        actual = (results_LSTM if model_name=='LSTM' else
                  results_xgb  if model_name=='XGBoost' else
                  results_meta_model)[company]['Actual']
        mean_actual = actual.mean()

        mae = df.loc[company, 'MAE']
        mse = df.loc[company, 'MSE']

        records.append({
            'Model':    model_name,
            'Company':  company,
            'MAE':      mae,
            'MAE_pct':  f"{mae/mean_actual*100:.2f}%",
            'MSE':      mse,
            'MSE_pct':  f"{mse/mean_actual*100:.2f}%",
            'R2':       f"{df.loc[company, 'R2'] * 100:.2f}%",
        })

perf_df = pd.DataFrame(records, columns=[
    'Model','Company','MAE','MAE_pct','MSE','MSE_pct','R2'
])

# ---- Streamlit UI ----
st.set_page_config(layout="wide")
st.title("📊 Stock Forecast & News Sentiment")
st.markdown(
    """ 
    Made by :rainbow[SmallData Team]
    
    Thank you Mr. Quang and SamSung
    """
)
# Sidebar inputs
ticker = st.sidebar.selectbox("1️⃣ Chọn mã cổ phiếu:", list(stocks.keys()))
n_days = st.sidebar.slider("2️⃣ Số ngày dự báo:", 1, 180, 5)
news = st.sidebar.text_area("3️⃣ Dán tin tức để phân tích sentiment:", height=180)


# Buttons
forecast_btn = st.sidebar.button("🚀 Dự báo giá")
sentiment_btn = st.sidebar.button("📰 Phân tích sentiment")
show_perf = st.sidebar.checkbox("📈 Hiển thị hiệu năng mô hình", value=True)
if st.sidebar.button("Let it snow!"):
    st.snow()
    
if forecast_btn:
    df_forecast = print_forecasts_with_real_dates_meta(
        company    = ticker,    
        results_meta_model = results_meta_model,
        forecast_func = forecast_next_days_meta,
        stocks      = stocks,
        time_steps  = 180,
        n_days      = 5
    )
    st.session_state['df_forecast'] = df_forecast # Save for later use
    
if sentiment_btn:
    if news.strip():
        with st.spinner("Đang phân tích sentiment..."):
            news_cleaned = clean_text(news)
            res = sentiment_pipe(news_cleaned)[0]
        label, score = res['label'], res['score']
        if label == 'LABEL_0':
            text = 'Negative'
        elif label == 'LABEL_2':
            text = 'Positive'
        else:
            text = 'Neutral'
        st.session_state['sentiment'] = (text, score)
    else:
        st.warning("Nhập văn bản để phân tích")
        
# Main layout
col1, col2 = st.columns((3,2))

# Performance table
if show_perf:
    with st.expander("Hiệu Năng Mô Hình", expanded=True):
        st.dataframe(perf_df, use_container_width=True)

# Forecast section
with col1:
    st.subheader(f"🔮 Dự báo giá cho {ticker}")
    st.line_chart(stocks[ticker]['close'])
    if 'df_forecast' in st.session_state:
        tmp_df = st.session_state['df_forecast'] 
        st.write('Kết quả dự báo:')
        st.dataframe(tmp_df)
       
    else:
        st.info("Nhấn nút '🚀 Dự báo giá' để xem kết quả dự báo.")
        
# Sentiment section
with col2:
    st.subheader("📰 Phân tích tâm lý thị trường")
    if 'sentiment' in st.session_state:
        text, score = st.session_state['sentiment']
        color = 'green' if text.lower().startswith('positive') else 'red' if text.lower().startswith('negative') else 'gray'
        emoji = '😊' if text.lower().startswith('positive') else '😞' if text.lower().startswith('negative') else '😐'
        st.markdown(f"<h3 style='color:{color};'>{emoji} {text} — {score:.2%}</h3>", unsafe_allow_html=True)
    else:
        st.info("Nhấn nút '📰 Phân tích sentiment' để xem kết quả.")