import os
import time
import tempfile
import pandas as pd
import streamlit as st
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

st.set_page_config(
    page_title="Doc & Market AI Platform",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Doc & Market AI Platform")
st.caption("Анализ документов, Function Calling, веб-поиск, интерактивные графики и экспорт")

# Автоподхват ключа
saved_key = ""
if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
    saved_key = st.secrets["GEMINI_API_KEY"]
elif "GEMINI_API_KEY" in os.environ:
    saved_key = os.environ["GEMINI_API_KEY"]

with st.sidebar:
    st.header("⚙️ Настройки")
    api_key = st.text_input("Gemini API Key:", value=saved_key, type="password")
    model_choice = st.selectbox("Модель:", ["gemini-3.8-flash", "gemini-3.8-pro"])
    
    st.markdown("---")
    mode = st.radio(
        "Выберите режим работы:",
        [
            "🛠️ Агент: Инструменты, Поиск и Вопросы",
            "📊 Дашборд: Графики, KPI и Выгрузка"
        ]
    )

# --- Пользовательские функции для Function Calling ---
def calculate_growth_rate(old_value: float, new_value: float) -> str:
    """Вычисляет точный темп прироста в процентах между старым и новым значением."""
    if old_value == 0:
        return "Ошибка: базовое значение равно нулю"
    growth = ((new_value - old_value) / abs(old_value)) * 100
    return f"Прирост: {growth:+.2f}% (с {old_value} до {new_value})"

def calculate_ratio(part: float, total: float) -> str:
    """Вычисляет точную долю части от общего объема в процентах (маржинальность или долю рынка)."""
    if total == 0:
        return "Ошибка: общее значение равно нулю"
    ratio = (part / total) * 100
    return f"Доля: {ratio:.2f}% ({part} от {total})"

# --- Схема данных Pydantic для режима Дашборда ---
class MetricPoint(BaseModel):
    year: str = Field(description="Год или период (например, 2023, 2024)")
    value: float = Field(description="Числовое значение показателя")

class MetricSeries(BaseModel):
    metric_name: str = Field(description="Название метрики для графика")
    data_points: list[MetricPoint] = Field(description="Точки динамики по годам")

class ExecutiveSummary(BaseModel):
    company_name: str = Field(description="Название компании")
    period: str = Field(description="Период отчета")
    key_metrics_cards: list[dict[str, str]] = Field(description="3-4 главные метрики для плашек")
    chart_series: list[MetricSeries] = Field(description="Данные для линейных графиков")
    strategic_findings: list[str] = Field(description="5 стратегических выводов")
    marketing_and_clients: list[str] = Field(description="Выводы по маркетингу и клиентам")
    tech_and_ai: list[str] = Field(description="Выводы по IT и AI")
    growth_hypotheses: list[str] = Field(description="3 гипотезы по развитию бизнеса")

# --- Загрузка документа ---
uploaded_file = st.file_uploader("Загрузите PDF-документ:", type=["pdf"])

# ==========================================
# РЕЖИМ 1: АГЕНТ (TOOLS + GOOGLE SEARCH)
# ==========================================
if mode == "🛠️ Агент: Инструменты, Поиск и Вопросы":
    st.subheader("🛠️ Вопрос к документу с внешними инструментами")
    user_task = st.text_area(
        "Ваш вопрос или аналитическое задание:",
        value="Найди ключевые финансовые показатели за последние 2 года, рассчитай их точный прирост и сопоставь с актуальными данными рынка через веб-поиск."
    )

    if st.button("🚀 Запустить агента", type="primary"):
        if not api_key:
            st.error("Введите API-ключ!")
        elif not uploaded_file:
            st.warning("Загрузите PDF-файл.")
        else:
            status = st.empty()
            file_bytes = uploaded_file.getvalue()
            file_size_mb = len(file_bytes) / (1024 * 1024)

            try:
                client = genai.Client(api_key=api_key)
                config = types.GenerateContentConfig(
                    tools=[calculate_growth_rate, calculate_ratio, {"google_search": {}}],
                    temperature=0.2
                )

                if file_size_mb < 20:
                    status.info(f"⏳ Отправка PDF ({file_size_mb:.1f} МБ) напрямую...")
                    pdf_part = types.Part.from_bytes(data=file_bytes, mime_type="application/pdf")
                    contents = [pdf_part, user_task]
                else:
                    status.info(f"⏳ Загрузка большого файла ({file_size_mb:.1f} МБ) через Files API...")
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(file_bytes)
                        tmp_path = tmp.name

                    try:
                        uploaded_file_gemini = client.files.upload(file=tmp_path)
                        attempts = 0
                        while uploaded_file_gemini.state.name == "PROCESSING" and attempts < 30:
                            time.sleep(2)
                            attempts += 1
                            uploaded_file_gemini = client.files.get(name=uploaded_file_gemini.name)
                            status.info(f"⏳ Обработка в Google ({attempts * 2} сек)...")
                        contents = [uploaded_file_gemini, user_task]
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)

                status.info("⏳ Gemini выполняет расчеты и поиск в сети...")
                t0 = time.time()
                response = client.models.generate_content(
                    model=model_choice,
                    contents=contents,
                    config=config
                )
                status.empty()
                st.success(f"Готово за {time.time() - t0:.1f} сек.!")
                st.markdown(response.text)

            except Exception as e:
                status.empty()
                st.error(f"Ошибка: {e}")

# ==========================================
# РЕЖИМ 2: ДАШБОРД (ГРАФИКИ, KPI, СКАЧИВАНИЕ)
# ==========================================
else:
    st.subheader("📊 Генерация наглядного дашборда и графиков")
    
    if st.button("🚀 Сформировать дашборд", type="primary"):
        if not api_key:
            st.error("Введите API-ключ!")
        elif not uploaded_file:
            st.warning("Загрузите PDF-файл.")
        else:
            status = st.empty()
            file_bytes = uploaded_file.getvalue()
            file_size_mb = len(file_bytes) / (1024 * 1024)

            try:
                client = genai.Client(api_key=api_key)
                config = types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ExecutiveSummary,
                    temperature=0.1
                )
                prompt = (
                    "Проанализируй документ. Извлеки ключевые показатели, динамику по годам для графиков, "
                    "выводы по финансам, маркетингу и IT. Заполни схему строго по полям."
                )

                if file_size_mb < 20:
                    status.info(f"⏳ Отправка PDF ({file_size_mb:.1f} МБ)...")
                    pdf_part = types.Part.from_bytes(data=file_bytes, mime_type="application/pdf")
                    contents = [pdf_part, prompt]
                else:
                    status.info(f"⏳ Загрузка большого файла ({file_size_mb:.1f} МБ)...")
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(file_bytes)
                        tmp_path = tmp.name

                    try:
                        uploaded_file_gemini = client.files.upload(file=tmp_path)
                        attempts = 0
                        while uploaded_file_gemini.state.name == "PROCESSING" and attempts < 30:
                            time.sleep(2)
                            attempts += 1
                            uploaded_file_gemini = client.files.get(name=uploaded_file_gemini.name)
                            status.info(f"⏳ Обработка в Google ({attempts * 2} сек)...")
                        contents = [uploaded_file_gemini, prompt]
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)

                status.info("⏳ Генерация дашборда...")
                t0 = time.time()
                response = client.models.generate_content(
                    model=model_choice,
                    contents=contents,
                    config=config
                )
                status.empty()

                data = ExecutiveSummary.model_validate_json(response.text)
                st.success(f"Готово за {time.time() - t0:.1f} сек.!")
                st.header(f"📊 {data.company_name} — {data.period}")

                # Карточки KPI
                if data.key_metrics_cards:
                    cols = st.columns(len(data.key_metrics_cards))
                    for i, card in enumerate(data.key_metrics_cards):
                        with cols[i]:
                            st.metric(
                                label=card.get("label", "Метрика"),
                                value=card.get("value", "-"),
                                delta=card.get("delta", None)
                            )
                st.markdown("---")

                # Графики
                if data.chart_series:
                    st.subheader("📈 Динамика показателей")
                    for s in data.chart_series:
                        if len(s.data_points) > 1:
                            df = pd.DataFrame([{"Период": p.year, s.metric_name: p.value} for p in s.data_points]).set_index("Период")
                            st.write(f"**{s.metric_name}**")
                            st.line_chart(df)

                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("🎯 Главные выводы")
                    for f in data.strategic_findings:
                        st.markdown(f"- {f}")
                    st.subheader("💡 Точки роста")
                    for h in data.growth_hypotheses:
                        st.markdown(f"- {h}")

                with col2:
                    st.subheader("👥 Маркетинг и клиенты")
                    for m in data.marketing_and_clients:
                        st.markdown(f"- {m}")
                    st.subheader("💻 IT и AI")
                    for t in data.tech_and_ai:
                        st.markdown(f"- {t}")

                # Скачивание
                report_md = f"# Отчет: {data.company_name} ({data.period})\n\n"
                for f in data.strategic_findings:
                    report_md += f"- {f}\n"
                st.markdown("---")
                st.download_button("📥 Скачать отчет (.md)", report_md, f"report_{data.company_name}.md")

            except Exception as e:
                status.empty()
                st.error(f"Ошибка: {e}")