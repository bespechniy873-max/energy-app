import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import Ridge
import plotly.express as px
import warnings

warnings.filterwarnings('ignore')

# Налаштування сторінки
st.set_page_config(page_title="Energy Forecast Pro", layout="wide", page_icon="⚡")

# --- ФУНКЦІЇ ОБРОБКИ (Твоя математика) ---
def prepare_features(df_raw, target_cols, t_kyiv, t_odesa):
    df = df_raw.copy()
    df['Дата'] = pd.to_datetime(df['Дата'], dayfirst=True, format='mixed', errors='coerce')
    df = df.dropna(subset=['Дата']).sort_values('Дата').reset_index(drop=True)
    
    df['День_тижня'] = df['Дата'].dt.dayofweek
    df['Вихідний'] = df['День_тижня'].isin([5, 6]).astype(int)
    
    for col in target_cols:
        if col in df.columns:
            for lag in [1, 7]:
                df[f'{col}_Лаг_{lag}'] = df[col].shift(lag)
    
    # Використовуємо температуру з інтерфейсу
    df['Темп_Київ'] = t_kyiv
    df['Темп_Одеса'] = t_odesa
    return df.ffill().reset_index(drop=True)

# --- ІНТЕРФЕЙС (SIDEBAR) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2731/2731636.png", width=100)
    st.title("Керування")
    
    uploaded_file = st.file_uploader("1. Завантажте Excel", type="xls")
    
    st.subheader("2. Параметри від Gemini")
    target_date = st.date_input("Дата прогнозу", value=pd.Timestamp("2026-04-18"))
    
    supply = st.slider("Supply Index (Генерація)", 1, 10, 7)
    renewables = st.slider("Renewable (Сонце)", 1, 10, 9)
    wind = st.slider("Wind (Вітер)", 1, 10, 10)
    pressure = st.slider("Price Pressure (Тиск)", -5, 5, -2)
    
    st.subheader("3. Погода")
    t_k = st.number_input("Темп. Київ (°C)", value=12)
    t_o = st.number_input("Темп. Одеса (°C)", value=18)

# --- ГОЛОВНИЙ ЕКРАН ---
if uploaded_file:
    try:
        raw_data = pd.read_excel(uploaded_file)
        raw_data.columns = [str(col).strip() for col in raw_data.columns]
        
        цілі = {
            'OffPeak': 'OffPeak, грн/МВт.год',
            'Peak': 'Peak, грн/МВт.год',
            'Base': 'Base, грн/МВт.год',
            'WAP': 'Середньозважена ціна, грн/МВт.год'
        }
        
        if st.button("🚀 ЗАПУСТИТИ ГІБРИДНИЙ РОЗРАХУНОК"):
            with st.spinner('Модель аналізує історію та новини...'):
                
                # Додаємо рядок прогнозу
                if pd.Timestamp(target_date) not in pd.to_datetime(raw_data['Дата'], dayfirst=True).values:
                    future_row = pd.DataFrame({'Дата': [pd.Timestamp(target_date)]})
                    for col in цілі.values(): future_row[col] = np.nan
                    raw_data = pd.concat([raw_data, future_row], ignore_index=True)
                
                # Підготовка
                df_fin = prepare_features(raw_data, list(цілі.values()), t_k, t_o)
                
                # Додаємо дані від Gemini (з повзунків)
                df_fin.loc[df_fin['Дата'] == pd.Timestamp(target_date), 'supply_index'] = supply
                df_fin.loc[df_fin['Дата'] == pd.Timestamp(target_date), 'renewable_impact'] = renewables
                df_fin.loc[df_fin['Дата'] == pd.Timestamp(target_date), 'wind_impact'] = wind
                df_fin.loc[df_fin['Дата'] == pd.Timestamp(target_date), 'price_pressure'] = pressure
                
                # Заповнюємо минуле середніми значеннями, щоб модель не падала
                llm_cols = ['supply_index', 'renewable_impact', 'wind_impact', 'price_pressure']
                df_fin[llm_cols] = df_fin[llm_cols].fillna(5)
                
                # Навчання та Прогноз
                results = {}
                train_data = df_fin[df_fin[цілі['WAP']].notna()]
                predict_row = df_fin[df_fin['Дата'] == pd.Timestamp(target_date)]
                
                factors = llm_cols + ['Темп_Київ', 'Темп_Одеса']
                
                for label, col_name in цілі.items():
                    feats = [c for c in train_data.columns if c not in list(цілі.values()) + ['Дата'] + factors]
                    
                    m_xgb = xgb.XGBRegressor(n_estimators=100).fit(train_data[feats], train_data[col_name])
                    
                    X_meta_train = pd.DataFrame({f'XGB_{label}': m_xgb.predict(train_data[feats])})
                    for f in factors: X_meta_train[f] = train_data[f].values
                    
                    meta = Ridge().fit(X_meta_train, train_data[col_name])
                    
                    p_xgb = m_xgb.predict(predict_row[feats])
                    X_meta_pred = pd.DataFrame({f'XGB_{label}': p_xgb})
                    for f in factors: X_meta_pred[f] = predict_row[f].values
                    
                    results[label] = meta.predict(X_meta_pred)[0]
                
                # --- ВІЗУАЛІЗАЦІЯ РЕЗУЛЬТАТІВ ---
                st.success(f"Прогноз на {target_date.strftime('%d.%m.%Y')} готовий!")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("OffPeak (Ніч)", f"{results['OffPeak']:.2f} грн")
                c2.metric("Peak (День)", f"{results['Peak']:.2f} грн")
                c3.metric("Base (База)", f"{results['Base']:.2f} грн")
                
                st.divider()
                st.balloons()
                st.markdown(f"<h1 style='text-align: center; color: #FF4B4B;'>🚀 СЕРЕДНЬОЗВАЖЕНА: {results['WAP']:.2f} грн/МВт.год</h1>", unsafe_allow_value=True)
                
                # Графік
                plot_df = pd.DataFrame({
                    'Тип': ['Ніч (OffPeak)', 'День (Peak)', 'База', 'Середньозважена'],
                    'Ціна': [results['OffPeak'], results['Peak'], results['Base'], results['WAP']]
                })
                fig = px.bar(plot_df, x='Тип', y='Ціна', color='Тип', text_auto='.2f', title="Порівняння індексів")
                st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Помилка обробки файлу: {e}")
else:
    st.info("👈 Завантажте файл indexes_04.2026.xls у бічній панелі, щоб почати.")
