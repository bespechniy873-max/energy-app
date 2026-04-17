import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import Ridge
import warnings

warnings.filterwarnings('ignore')
st.set_page_config(page_title="Energy Forecast Pro", layout="wide", page_icon="⚡")

def prepare_features(df_raw, target_cols, t_k, t_o):
    df = df_raw.copy()
    df.columns = [str(col).strip() for col in df.columns]
    df['Дата'] = pd.to_datetime(df['Дата'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Дата']).sort_values('Дата').reset_index(drop=True)
    
    df['День_тижня'] = df['Дата'].dt.dayofweek
    df['Вихідний'] = df['День_тижня'].isin([5, 6]).astype(int)
    
    for col in target_cols:
        if col in df.columns:
            df[f'{col}_Lag1'] = df[col].shift(1)
            df[f'{col}_Lag1'] = df[f'{col}_Lag1'].bfill()
            
    df['Темп_Київ'] = t_k
    df['Темп_Одеса'] = t_o
    return df

with st.sidebar:
    st.title("⚙️ Керування")
    uploaded_file = st.file_uploader("1. Завантажте Excel", type=["xls", "xlsx"])
    target_date = st.date_input("Дата прогнозу", value=pd.to_datetime("2026-04-18"))
    supply = st.slider("Supply Index", 1, 10, 7)
    renewables = st.slider("Renewable", 1, 10, 9)
    wind = st.slider("Wind", 1, 10, 10)
    pressure = st.slider("Price Pressure", -5, 5, -2)
    t_k = st.number_input("Темп. Київ", value=12)
    t_o = st.number_input("Темп. Одеса", value=18)

if uploaded_file:
    try:
        raw_data = pd.read_excel(uploaded_file)
        raw_data.columns = [str(col).strip() for col in raw_data.columns]
        
        цілі_назви = {
            'OffPeak': 'OffPeak, грн/МВт.год',
            'Peak': 'Peak, грн/МВт.год',
            'Base': 'Base, грн/МВт.год',
            'WAP': next((c for c in raw_data.columns if 'Середньозважена' in c or 'WAP' in c), None)
        }

        if st.button("🚀 РОЗРАХУВАТИ"):
            target_ts = pd.Timestamp(target_date)
            if target_ts not in pd.to_datetime(raw_data['Дата'], dayfirst=True).values:
                future_row = pd.DataFrame({'Дата': [target_ts]})
                raw_data = pd.concat([raw_data, future_row], ignore_index=True)

            df_fin = prepare_features(raw_data, list(цілі_назви.values()), t_k, t_o)
            
            mask = df_fin['Дата'] == target_ts
            df_fin.loc[mask, ['supply_index', 'renewable_impact', 'wind_impact', 'price_pressure']] = [supply, renewables, wind, pressure]
            df_fin[['supply_index', 'renewable_impact', 'wind_impact', 'price_pressure']] = df_fin[['supply_index', 'renewable_impact', 'wind_impact', 'price_pressure']].fillna(5)

            train_data = df_fin[df_fin[цілі_назви['Base']].notna()]
            predict_row = df_fin[df_fin['Дата'] == target_ts]

            if len(train_data) < 2:
                st.error("❌ Замало даних для навчання.")
            else:
                st.balloons()
                results = {}
                factors = ['supply_index', 'renewable_impact', 'wind_impact', 'price_pressure', 'Темп_Київ', 'Темп_Одеса']
                
                for label, col_name in цілі_назви.items():
                    if not col_name: continue
                    feats = ['День_тижня', 'Вихідний'] + factors
                    model = xgb.XGBRegressor(n_estimators=50, learning_rate=0.1)
                    model.fit(train_data[feats], train_data[col_name])
                    
                    base_pred = model.predict(predict_row[feats])[0]
                    # Фінальний результат з урахуванням твого тиску
                    results[label] = base_pred * (1 + (pressure * 0.015))

                st.markdown(f"<h1 style='text-align: center; color: #FF4B4B;'>🚀 ОЧІКУВАНА ЦІНА: {results['WAP']:.2f} грн</h1>", unsafe_allow_html=True)
                c1, c2, c3 = st.columns(3)
                c1.metric("Ніч", f"{results['OffPeak']:.2f}")
                c2.metric("День", f"{results['Peak']:.2f}")
                c3.metric("База", f"{results['Base']:.2f}")
                
    except Exception as e:
        st.error(f"Помилка: {e}")
