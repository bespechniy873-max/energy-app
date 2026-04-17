import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import Ridge
import warnings

# Налаштування
warnings.filterwarnings('ignore')
st.set_page_config(page_title="Energy Forecast Pro", layout="wide", page_icon="⚡")

# 1. Функція підготовки даних
def prepare_features(df_raw, target_cols, t_k, t_o):
    df = df_raw.copy()
    df.columns = [str(col).strip() for col in df.columns]
    df['Дата'] = pd.to_datetime(df['Дата'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Дата']).sort_values('Дата').reset_index(drop=True)
    
    # Створюємо базові ознаки
    df['День_тижня'] = df['Дата'].dt.dayofweek
    df['Вихідний'] = df['День_тижня'].isin([5, 6]).astype(int)
    
    # Створюємо лаги (тільки 1 день назад для стабільності)
    for col in target_cols:
        if col in df.columns:
            df[f'{col}_Lag1'] = df[col].shift(1).ffill().bfill()
            
    df['Темп_Київ'] = t_k
    df['Темп_Одеса'] = t_o
    return df

# --- ІНТЕРФЕЙС ---
with st.sidebar:
    st.title("⚙️ Керування")
    uploaded_file = st.file_uploader("1. Завантажте Excel", type=["xls", "xlsx"], key="file_uv")
    
    st.subheader("2. Параметри від Gemini")
    target_date = st.date_input("Дата прогнозу", value=pd.to_datetime("2026-04-18"))
    
    supply = st.slider("Supply Index", 1, 10, 7)
    renewables = st.slider("Renewable", 1, 10, 9)
    wind = st.slider("Wind", 1, 10, 10)
    pressure = st.slider("Price Pressure", -5, 5, -2)
    
    st.subheader("3. Погода")
    t_k = st.number_input("Київ (°C)", value=12)
    t_o = st.number_input("Одеса (°C)", value=18)
    
    if st.button("♻️ Очистити кеш"):
        st.cache_data.clear()
        st.rerun()

# --- ЛОГІКА ПРОГНОЗУ ---
if uploaded_file:
    try:
        # Читаємо файл
        raw_data = pd.read_excel(uploaded_file)
        raw_data.columns = [str(col).strip() for col in raw_data.columns]
        
        # Визначаємо цільові колонки
        цілі = {
            'OffPeak': 'OffPeak, грн/МВт.год',
            'Peak': 'Peak, грн/МВт.год',
            'Base': 'Base, грн/МВт.год',
            'WAP': next((c for c in raw_data.columns if 'Середньозважена' in c or 'WAP' in c), None)
        }

        if st.button("🚀 РОЗРАХУВАТИ ПРОГНОЗ"):
            with st.spinner('Модель працює...'):
                target_ts = pd.Timestamp(target_date)
                
                # Додаємо рядок для прогнозу
                if target_ts not in pd.to_datetime(raw_data['Дата'], dayfirst=True).values:
                    future_row = pd.DataFrame({'Дата': [target_ts]})
                    raw_data = pd.concat([raw_data, future_row], ignore_index=True)

                df_fin = prepare_features(raw_data, list(цілі.values()), t_k, t_o)
                
                # Заповнюємо фактори для дати прогнозу
                mask = df_fin['Дата'] == target_ts
                factors_list = ['supply_index', 'renewable_impact', 'wind_impact', 'price_pressure']
                df_fin.loc[mask, factors_list] = [supply, renewables, wind, pressure]
                df_fin[factors_list] = df_fin[factors_list].fillna(5)

                # Дані для навчання та прогнозу
                train = df_fin[df_fin[цілі['Base']].notna()].tail(60) # Беремо останні 60 днів для актуальності
                predict_row = df_fin[df_fin['Дата'] == target_ts]

                if len(train) < 2:
                    st.error("Замало історичних даних у файлі!")
                else:
                    results = {}
                    features = ['День_тижня', 'Вихідний', 'Темп_Київ', 'Темп_Одеса'] + factors_list
                    
                    for label, col_name in цілі.items():
                        if not col_name: continue
                        
                        # Модель
                        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.08, max_depth=3)
                        model.fit(train[features], train[col_name])
                        
                        # Прогноз + коригування тиску
                        base_val = model.predict(predict_row[features])[0]
                        results[label] = base_val * (1 + (pressure * 0.012))

                    # Вивід результатів
                    st.balloons()
                    st.markdown(f"<h1 style='text-align: center; color: #FF4B4B;'>🚀 ПРОГНОЗ WAP: {results['WAP']:.2f} грн</h1>", unsafe_allow_html=True)
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Ніч (OffPeak)", f"{results['OffPeak']:.2f}")
                    c2.metric("День (Peak)", f"{results['Peak']:.2f}")
                    c3.metric("База (Base)", f"{results['Base']:.2f}")
                    
                    st.info(f"Прогноз сформовано для {target_date.strftime('%d.%m.%Y')}. На основі останніх {len(train)} днів історії.")

    except Exception as e:
        st.error(f"Помилка: {e}")
else:
    st.info("👈 Завантажте свій Excel-файл з історією цін, щоб почати.")
