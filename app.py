import os
import re
import time
import tempfile
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from google import genai
from google.genai import types
from google.genai.errors import ClientError
from pydantic import BaseModel, Field

st.set_page_config(
    page_title="Doc & Market AI Platform",
    page_icon="🌿",
    layout="wide"
)

# --- Получение общего (бесплатного) ключа из Secrets/окружения ---
DEFAULT_SHARED_KEY = ""
if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
    DEFAULT_SHARED_KEY = st.secrets["GEMINI_API_KEY"]
elif "GEMINI_API_KEY" in os.environ:
    DEFAULT_SHARED_KEY = os.environ["GEMINI_API_KEY"]

# --- Боковая панель: Тема, Доступ и Модели ---
with st.sidebar:
    st.markdown("### 🌿 Doc & Market AI")
    dark_mode = st.toggle("🌙 Темная тема (Dark Mode)", value=False)
    st.markdown("---")

    st.subheader("🔑 Доступ и модели")
    user_api_key = st.text_input(
        "Ваш личный Gemini API Key:",
        value="",
        type="password",
        help="Оставьте пустым для работы на базовой бесплатной модели. Вставьте свой ключ для доступа ко всем моделям."
    )

    if user_api_key.strip():
        active_api_key = user_api_key.strip()
        st.success("✅ Личный ключ (BYOK): доступны все модели")
        available_models = [
            "gemini-3.5-flash-lite",
            "gemini-3.8-flash",
            "gemini-4-argon"
        ]
    else:
        active_api_key = DEFAULT_SHARED_KEY
        if active_api_key:
            st.info("🆓 Базовый режим: активна бесплатная модель.")
        else:
            st.warning("⚠️ Вставьте свой бесплатный API-ключ из Google AI Studio (см. вкладку «Гайд»).")
        available_models = ["gemini-3.5-flash-lite"]

    model_choice = st.selectbox("Выбор модели:", available_models, index=0)
    st.markdown("---")
    st.caption("💡 При ошибке лимитов (429) используйте `gemini-3.5-flash-lite` без веб-поиска.")

# --- Контрастные палитры для Светлой и Темной тем ---
if not dark_mode:
    THEME = {
        "bg": "#F4F7F4",
        "card_bg": "#FFFFFF",
        "input_bg": "#FFFFFF",
        "sidebar_bg": "#ECF2EC",
        "text": "#122118",
        "subtext": "#52665A",
        "border": "#D5E2D7",
        "hero_bg": "#112418",
        "hero_text": "#FFFFFF",
        "accent": "#2E9E5B",
        "accent_hover": "#24824A",
        "accent_soft": "#E2F5E9",
        "tab_bar_bg": "#E4EDE5",
        "tab_btn_bg": "#FFFFFF",
        "tab_shadow": "0 4px 12px rgba(17, 36, 24, 0.08)",
        "donut_colors": ["#112418", "#2E9E5B", "#6BCB77", "#A3E4A8", "#C8E6C9", "#DCE8DE"],
        "plotly_template": "plotly_white"
    }
else:
    THEME = {
        "bg": "#0B0E0D",
        "card_bg": "#141A18",
        "input_bg": "#1B2320",
        "sidebar_bg": "#101513",
        "text": "#F0F6F2",
        "subtext": "#9BB0A3",
        "border": "#2A3832",
        "hero_bg": "#193827",
        "hero_text": "#F0F6F2",
        "accent": "#3DD679",
        "accent_hover": "#5CE692",
        "accent_soft": "#193826",
        "tab_bar_bg": "#141A18",
        "tab_btn_bg": "#212B27",
        "tab_shadow": "0 6px 16px rgba(0, 0, 0, 0.45)",
        "donut_colors": ["#3DD679", "#269954", "#6EE79E", "#1B5E37", "#8FA89B", "#2D3B35"],
        "plotly_template": "plotly_dark"
    }

# --- Полный CSS: контрастный Dark Mode + Физические кнопки вкладок + Анимация перелистывания ---
st.markdown(f"""
<style>
    /* Анимация плавного перелистывания контента вкладок */
    @keyframes pageSlideFade {{
        0% {{
            opacity: 0;
            transform: translateY(14px) scale(0.992);
        }}
        100% {{
            opacity: 1;
            transform: translateY(0) scale(1);
        }}
    }}

    /* Глобальный фон и контрастный текст */
    .stApp {{
        background-color: {THEME['bg']} !important;
        color: {THEME['text']} !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        transition: background-color 0.35s ease, color 0.35s ease;
    }}
    .stApp p, .stApp span, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4 {{
        color: {THEME['text']};
    }}
    section[data-testid="stSidebar"] {{
        background-color: {THEME['sidebar_bg']} !important;
        border-right: 1px solid {THEME['border']};
    }}
    header[data-testid="stHeader"] {{
        background: transparent !important;
    }}

    /* --- Исправление всех полей ввода, селектов и загрузчика файлов в Dark/Light Mode --- */
    div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] > div,
    div[data-baseweb="select"] > div {{
        background-color: {THEME['input_bg']} !important;
        border: 1px solid {THEME['border']} !important;
        border-radius: 12px !important;
        color: {THEME['text']} !important;
    }}
    input, textarea {{
        color: {THEME['text']} !important;
        -webkit-text-fill-color: {THEME['text']} !important;
    }}
    div[data-baseweb="select"] span {{
        color: {THEME['text']} !important;
    }}
    /* Выпадающее меню списка моделей */
    ul[role="listbox"] {{
        background-color: {THEME['card_bg']} !important;
        border: 1px solid {THEME['border']} !important;
    }}
    ul[role="listbox"] li {{
        color: {THEME['text']} !important;
    }}
    /* Зона перетаскивания файлов (File Uploader) */
    section[data-testid="stFileUploaderDropzone"] {{
        background-color: {THEME['card_bg']} !important;
        border: 2px dashed {THEME['border']} !important;
        border-radius: 16px !important;
        padding: 20px !important;
        transition: border-color 0.25s ease;
    }}
    section[data-testid="stFileUploaderDropzone"]:hover {{
        border-color: {THEME['accent']} !important;
    }}
    section[data-testid="stFileUploaderDropzone"] button {{
        background-color: {THEME['tab_btn_bg']} !important;
        color: {THEME['text']} !important;
        border: 1px solid {THEME['border']} !important;
        border-radius: 10px !important;
    }}

    /* --- ФИЗИЧЕСКИЕ КНОПКИ-ВКЛАДКИ (Tactile Segmented Tabs) --- */
    div[data-baseweb="tab-list"] {{
        background-color: {THEME['tab_bar_bg']};
        padding: 6px;
        border-radius: 16px;
        border: 1px solid {THEME['border']};
        gap: 8px;
        margin-bottom: 18px;
        display: inline-flex;
        width: 100%;
    }}
    /* Скрываем стандартную тонкую полоску снизу */
    div[data-baseweb="tab-highlight"] {{
        display: none !important;
    }}
    div[data-baseweb="tab-border"] {{
        display: none !important;
    }}
    /* Сама кнопка вкладки */
    button[data-baseweb="tab"] {{
        flex: 1;
        background-color: transparent !important;
        border: 1px solid transparent !important;
        border-radius: 12px !important;
        padding: 12px 18px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        color: {THEME['subtext']} !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
        cursor: pointer;
    }}
    button[data-baseweb="tab"]:hover {{
        color: {THEME['text']} !important;
        background-color: rgba(46, 158, 91, 0.08) !important;
    }}
    /* Активная физическая кнопка-вкладка */
    button[data-baseweb="tab"][aria-selected="true"] {{
        background-color: {THEME['tab_btn_bg']} !important;
        color: {THEME['text']} !important;
        border: 1px solid {THEME['border']} !important;
        box-shadow: {THEME['tab_shadow']}, inset 0 1px 0 rgba(255, 255, 255, 0.08) !important;
        transform: translateY(-1px);
    }}
    button[data-baseweb="tab"]:active {{
        transform: scale(0.98) !important;
    }}
    /* Анимация перелистывания при открытии любой вкладки */
    div[data-baseweb="tab-panel"] {{
        animation: pageSlideFade 0.42s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }}

    /* --- Bento-карточки KPI --- */
    .kpi-card {{
        background-color: {THEME['card_bg']};
        border: 1px solid {THEME['border']};
        border-radius: 18px;
        padding: 20px 22px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06);
        height: 100%;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }}
    .kpi-card:hover {{
        transform: translateY(-2px);
        border-color: {THEME['accent']};
    }}
    .kpi-card-hero {{
        background-color: {THEME['hero_bg']};
        border: 1px solid {THEME['accent']};
        border-radius: 18px;
        padding: 20px 22px;
        box-shadow: 0 8px 24px rgba(17, 36, 24, 0.25);
        height: 100%;
    }}
    .kpi-label {{
        font-size: 13px;
        font-weight: 500;
        color: {THEME['subtext']} !important;
        margin-bottom: 8px;
    }}
    .kpi-card-hero .kpi-label {{
        color: #A6C4B0 !important;
    }}
    .kpi-value {{
        font-size: 28px;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: {THEME['text']} !important;
        margin-bottom: 10px;
    }}
    .kpi-card-hero .kpi-value {{
        color: #FFFFFF !important;
    }}
    .kpi-badge {{
        display: inline-block;
        background-color: {THEME['accent_soft']};
        color: {THEME['accent']} !important;
        font-size: 12px;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 999px;
    }}
    .kpi-card-hero .kpi-badge {{
        background-color: rgba(61, 214, 121, 0.2);
        color: #6EE79E !important;
    }}

    /* Карточка AI Smart Insight */
    .insight-box {{
        background-color: {THEME['hero_bg']};
        border-radius: 18px;
        padding: 22px 24px;
        margin-bottom: 16px;
        border: 1px solid {THEME['accent']};
    }}
    .insight-title {{
        font-size: 14px;
        font-weight: 700;
        color: #6EE79E !important;
        margin-bottom: 8px;
    }}
    .insight-text {{
        font-size: 15px;
        line-height: 1.55;
        color: #F0F7F2 !important;
    }}

    /* Контейнер ответа Агента */
    .bento-panel {{
        background-color: {THEME['card_bg']};
        border: 1px solid {THEME['border']};
        border-radius: 18px;
        padding: 22px 26px;
        margin-top: 12px;
        color: {THEME['text']} !important;
        line-height: 1.6;
    }}

    /* Физические кнопки действий (Primary & Download) */
    .stButton > button[kind="primary"],
    .stDownloadButton > button {{
        background-color: {THEME['accent']} !important;
        color: #0B1911 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 14px rgba(46, 158, 91, 0.3) !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }}
    .stButton > button[kind="primary"]:hover,
    .stDownloadButton > button:hover {{
        background-color: {THEME['accent_hover']} !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(46, 158, 91, 0.45) !important;
    }}
    .stButton > button[kind="primary"]:active,
    .stDownloadButton > button:active {{
        transform: scale(0.97) !important;
    }}
    .stButton > button[kind="primary"] p,
    .stDownloadButton > button p {{
        color: #0B1911 !important;
        font-weight: 700 !important;
    }}

    /* Таблицы (во вкладке Гайд) */
    table {{
        background-color: {THEME['card_bg']} !important;
        color: {THEME['text']} !important;
        border-radius: 12px !important;
        overflow: hidden;
    }}
    th {{
        background-color: {THEME['tab_bar_bg']} !important;
        color: {THEME['text']} !important;
    }}
    td {{
        color: {THEME['text']} !important;
        border-color: {THEME['border']} !important;
    }}
</style>
""", unsafe_allow_html=True)

# --- Заголовок и физические кнопки-вкладки ---
st.title("🌿 Doc & Market AI Platform")
st.caption("Интеллектуальный анализ документов, расчеты и визуальная аналитика")

tab_agent, tab_dashboard, tab_guide = st.tabs([
    "🛠️ Агент (Вопросы и Расчеты)",
    "📊 Дашборд (KPI и Диаграммы)",
    "📖 Гайд и Ссылки"
])

# ---
