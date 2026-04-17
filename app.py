import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import Ridge
import plotly.express as px
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Energy Forecast Pro", layout="wide", page_icon="⚡")

def prepare_features(df_raw, target_cols, t_kyiv, t_odesa):
    df = df_raw.copy()
    # Очищення назв колонок від пробілів
    df.columns = [str(col).strip() for col in df.columns]
    
    df['Дата'] = pd.to_datetime(df['Дата'], dayfirst=True, format='mixed', errors='coerce')
    df = df.dropna(subset=['Дата']).sort_values('Дата').reset_index(drop=True)
    
    df['День_тижня'] = df['Дата'].dt.dayofweek
    df['Вихідний'] = df['День_тижня'].isin([5, 6]).astype(int)
    
    for col in target_cols:
        if col in df.columns:
            for lag in [1, 7]:
                df[f'{col}_Лаг_{lag}'] = df[col].shift(lag)
    
    df['Темп_Київ'] = t_kyiv
    df['Темп_Одеса'] = t_odesa
    return df.ffill().reset_index(drop=True)

with st.sidebar:
    st.title("⚙️ Керування")
    uploaded_file = st.file_uploader("1. Завантажте Excel", type=["xls", "xlsx"])
    
    st.subheader("2. Параметри від Gemini")
    target_date = st.date_input("Дата прогнозу", value=pd.to_datetime("2026-04-18"))
    supply = st.slider("Supply Index", 1, 10, 7)
    renewables = st.slider("Renewable (Сонце)", 1, 10, 9)
    wind = st.slider("Wind (Вітер)", 1, 10, 10)
    pressure = st.slider("Price Pressure", -5, 5, -2)
    
    st.subheader("3. Погода")
    t_k = st.number_input("Темп. Київ", value=12)
    t_o = st.number_input("Темп. Одеса", value=18)

if uploaded_file:
    try:
        raw_data = pd.read_excel(uploaded_file)
        # Очищення назв колонок
        raw_data.columns = [str(col).strip() for col in raw_data.columns]
        
        # Спроба знайти правильну колонку для WAP (Середньозваженої)
        possible_wap_names = ['Середньозважена ціна, грн/МВт.год', 'Середньозважена ціна, грн/МВт·год', 'WAP']
        actual_wap_col = next((c for c in possible_wap_names if c in raw_data.columns), None)

        if not actual_wap_col:
            st.error(f"❌ Не знайдено колонку 'Середньозважена ціна'. Доступні колонки: {list(raw_data.columns)}")
            st.stop()

        цілі = {
            'OffPeak': 'OffPeak, грн/МВт.год',
            'Peak': 'Peak, грн/МВт.год',
            'Base': 'Base, грн/МВт.год',
            'WAP': actual_wap_col
        }
        
        if st.button("🚀 РОЗРАХУВАТИ"):
            with st.spinner('Аналізуємо...'):
                target_ts = pd.Timestamp(target_date)
                if target_ts not in pd.to_datetime(raw_data['Дата'], dayfirst=True).values:
                    future_row = pd.DataFrame({'Дата': [target_ts]})
                    for col in цілі.values(): future_row[col] = np.nan
                    raw_data = pd.concat([raw_data, future_row], ignore_index=True)
                
                df_fin = prepare_features(raw_data, list(цілі.values()), t_k, t_o)
                
                df_fin.loc[df_fin['Дата'] == target_ts, 'supply_index'] = supply
                df_fin.loc[df_fin['Дата'] == target_ts, 'renewable_impact'] = renewables
                df_fin.loc[df_fin['Дата'] == target_ts, 'wind_impact'] = wind
                df_fin.loc[df_fin['Дата'] == target_ts, 'price_pressure'] = pressure
                
                llm_cols = ['supply_index', 'renewable_impact', 'wind_impact', 'price_pressure']
                df_fin[llm_cols] = df_fin[llm_cols].fillna(5)
                
                results = {}
                train_data = df_fin[df_fin[цілі['WAP']].notna()]
                predict_row = df_fin[df_fin['Дата'] == target_ts]
                factors = llm_cols + ['Темп_Київ', 'Темп_Одеса']
                
                for label, col_name in цілі.items():
                    if col_name not in train_data.columns:
                        results[label] = 0
                        continue
                        
                    feats = [c for c in train_data.columns if c not in list(цілі.values()) + ['Дата'] + factors]
                    m_xgb = xgb.XGBRegressor(n_estimators=100).fit(train_data[feats], train_data[col_name])
                    
                    X_meta_train = pd.DataFrame({f'XGB_{label}': m_xgb.predict(train_data[feats])})
                    for f in factors: X_meta_train[f] = train_data[f].values
                    
                    meta = Ridge().fit(X_meta_train, train_data[col_name])
                    
                    p_xgb = m_xgb.predict(predict_row[feats])
                    X_meta_pred = pd.DataFrame({f'XGB_{label}': p_xgb})
                    for f in factors: X_meta_pred[f] = predict_row[f].values
                    results[label] = meta.predict(X_meta_pred)[0]
                
                st.balloons()
                st.markdown(f"<h1 style='text-align: center; color: #FF4B4B;'>🚀 ОЧІКУВАНА ЦІНА: {results['WAP']:.2f} грн</h1>", unsafe_allow_html=True)
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Ніч (OffPeak)", f"{results['OffPeak']:.2f}")
                c2.metric("День (Peak)", f"{results['Peak']:.2f}")
                c3.metric("База (Base)", f"{results['Base']:.2f}")

    except Exception as e:
        st.error(f"Помилка: {e}")
