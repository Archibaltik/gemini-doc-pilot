import os
import re
import time
import tempfile
import pandas as pd
import streamlit as st
from google import genai
from google.genai import types
from google.genai.errors import ClientError
from pydantic import BaseModel, Field

st.set_page_config(
    page_title="Doc & Market AI Platform",
    page_icon="⚡",
    layout="wide"
)

# --- Получение общего (бесплатного) ключа из Secrets/окружения ---
DEFAULT_SHARED_KEY = ""
if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
    DEFAULT_SHARED_KEY = st.secrets["GEMINI_API_KEY"]
elif "GEMINI_API_KEY" in os.environ:
    DEFAULT_SHARED_KEY = os.environ["GEMINI_API_KEY"]

# --- Боковая панель: Настройки доступа и выбор моделей (Freemium + BYOK) ---
with st.sidebar:
    st.header("🔑 Доступ и модели")
    user_api_key = st.text_input(
        "Ваш личный Gemini API Key (необязательно):",
        value="",
        type="password",
        help="Оставьте пустым для работы на базовом бесплатном тарифе. Вставьте свой ключ, чтобы разблокировать мощные модели."
    )

    if user_api_key.strip():
        active_api_key = user_api_key.strip()
        is_byok = True
        st.success("✅ Активен личный ключ (BYOK): доступны все модели")
        available_models = [
            "gemini-3.5-flash-lite",
            "gemini-3.8-flash",
            "gemini-4-argon"
        ]
    else:
        active_api_key = DEFAULT_SHARED_KEY
        is_byok = False
        if active_api_key:
            st.info("🆓 Базовый режим: используется бесплатная модель. Вставьте свой ключ выше для доступа к старшим моделям.")
        else:
            st.warning("⚠️ Общий ключ не настроен. Вставьте свой бесплатный API-ключ из Google AI Studio (см. вкладку «Гайд»).")
        available_models = ["gemini-3.5-flash-lite"]

    model_choice = st.selectbox(
        "Выбор модели:",
        available_models,
        index=0
    )

    st.markdown("---")
    st.caption("💡 Подсказка: если возникает ошибка лимитов (429), используйте модель `gemini-3.5-flash-lite` или отключите веб-поиск.")

# --- Заголовок и главная навигация по вкладкам ---
st.title("⚡ Doc & Market AI Platform")
st.caption("Анализ документов, Function Calling, интерактивные дашборды и база знаний")

tab_agent, tab_dashboard, tab_guide = st.tabs([
    "🛠️ Агент (Вопросы и Расчеты)",
    "📊 Дашборд (KPI и Графики)",
    "📖 Гайд и Ссылки"
])

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

# --- Исправленная схема Pydantic (без dict / additionalProperties) ---
class KPICard(BaseModel):
    label: str = Field(description="Короткое название показателя (например, Выручка)")
    value: str = Field(description="Значение с единицами измерения (например, 1.5 трлн руб.)")
    delta: str = Field(description="Изменение к прошлому периоду (например, +14.2%)")

class MetricPoint(BaseModel):
    year: str = Field(description="Год или период (например, 2023, 2024)")
    value: float = Field(description="Числовое значение показателя")

class MetricSeries(BaseModel):
    metric_name: str = Field(description="Название метрики для графика")
    data_points: list[MetricPoint] = Field(description="Точки динамики по годам")

class ExecutiveSummary(BaseModel):
    company_name: str = Field(description="Название компании или тема отчета")
    period: str = Field(description="Период отчета")
    key_metrics_cards: list[KPICard] = Field(description="3-4 главные метрики для плашек")
    chart_series: list[MetricSeries] = Field(description="1-2 показателя с историческими точками для графиков")
    strategic_findings: list[str] = Field(description="5 ключевых стратегических выводов")
    marketing_and_clients: list[str] = Field(description="3-4 вывода по маркетингу и клиентам")
    tech_and_ai: list[str] = Field(description="3-4 вывода по IT и AI")
    growth_hypotheses: list[str] = Field(description="3 конкретные гипотезы по развитию бизнеса")

# --- Функция отправки запроса с автоповтором (Retry) при ошибке 429 ---
def generate_with_retry(client, model_name, contents, config, status_placeholder, max_retries=3):
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config
            )
        except ClientError as e:
            if e.code == 429 and attempt < max_retries - 1:
                # Пытаемся извлечь время ожидания из сообщения Google, иначе ждем 15 * (attempt + 1) сек
                match = re.search(r"retry in (\d+(\.\d+)?)s", str(e), re.IGNORECASE)
                wait_sec = float(match.group(1)) + 1.0 if match else 15.0 * (attempt + 1)
                wait_sec = min(wait_sec, 65.0)
                status_placeholder.warning(
                    f"⏳ Превышен минутный лимит API (429). Автоматическая пауза {wait_sec:.0f} сек. "
                    f"(попытка {attempt + 2} из {max_retries})..."
                )
                time.sleep(wait_sec)
                status_placeholder.info("⏳ Повторная отправка запроса в Gemini...")
            else:
                raise

# --- Вспомогательная функция подготовки файла ---
def prepare_file_contents(client, uploaded_file, prompt_text, status_placeholder):
    file_bytes = uploaded_file.getvalue()
    file_size_mb = len(file_bytes) / (1024 * 1024)

    if file_size_mb < 20:
        status_placeholder.info(f"⏳ Передача PDF ({file_size_mb:.1f} МБ) напрямую...")
        pdf_part = types.Part.from_bytes(data=file_bytes, mime_type="application/pdf")
        return [pdf_part, prompt_text]
    else:
        status_placeholder.info(f"⏳ Загрузка крупного файла ({file_size_mb:.1f} МБ) через Files API...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            uploaded_gemini = client.files.upload(file=tmp_path)
            attempts = 0
            while uploaded_gemini.state.name == "PROCESSING" and attempts < 30:
                time.sleep(2)
                attempts += 1
                uploaded_gemini = client.files.get(name=uploaded_gemini.name)
                status_placeholder.info(f"⏳ Распознавание документа в Google ({attempts * 2} сек)...")

            if uploaded_gemini.state.name == "FAILED":
                raise ValueError("Сервер Google не смог обработать PDF-файл.")

            return [uploaded_gemini, prompt_text]
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

# ==========================================
# ВКЛАДКА 1: АГЕНТ (TOOLS + ОПЦИОНАЛЬНЫЙ ПОИСК)
# ==========================================
with tab_agent:
    st.subheader("🛠️ Вопрос к документу с инструментами")
    uploaded_file_agent = st.file_uploader("Загрузите PDF-документ:", type=["pdf"], key="pdf_agent")
    
    col_opt1, col_opt2 = st.columns([2, 1])
    with col_opt1:
        use_math_tools = st.checkbox("Подключить калькуляторы формул (прирост и доли)", value=True)
    with col_opt2:
        use_google_search = st.checkbox(
            "Подключить Google Search (веб-поиск)",
            value=False,
            help="На бесплатном тарифе Google Search часто вызывает ошибку лимита 429. Включайте при необходимости."
        )

    user_task = st.text_area(
        "Задание для аналитика:",
        value="Найди ключевые финансовые показатели за последние 2 года, рассчитай их точный темп роста и сформулируй 3 главных вывода."
    )

    if st.button("🚀 Запустить анализ", type="primary", key="btn_agent"):
        if not active_api_key:
            st.error("Не указан API-ключ! Вставьте свой ключ в боковой панели слева (инструкция во вкладке «Гайд»).")
        elif not uploaded_file_agent:
            st.warning("Загрузите PDF-файл.")
        else:
            status = st.empty()
            try:
                client = genai.Client(api_key=active_api_key)
                
                active_tools = []
                if use_math_tools:
                    active_tools.extend([calculate_growth_rate, calculate_ratio])
                if use_google_search:
                    active_tools.append({"google_search": {}})

                config = types.GenerateContentConfig(
                    tools=active_tools if active_tools else None,
                    temperature=0.2
                )

                contents = prepare_file_contents(client, uploaded_file_agent, user_task, status)
                status.info(f"⏳ Модель `{model_choice}` анализирует документ...")
                t0 = time.time()

                response = generate_with_retry(client, model_choice, contents, config, status)
                status.empty()

                st.success(f"Готово за {time.time() - t0:.1f} сек. (модель: `{model_choice}`)")
                st.markdown(response.text)

            except Exception as e:
                status.empty()
                st.error(f"Ошибка: {e}")

# ==========================================
# ВКЛАДКА 2: ДАШБОРД (KPI, ГРАФИКИ, СКАЧИВАНИЕ)
# ==========================================
with tab_dashboard:
    st.subheader("📊 Интерактивный дашборд по документу")
    uploaded_file_dash = st.file_uploader("Загрузите PDF-документ для дашборда:", type=["pdf"], key="pdf_dash")

    if st.button("🚀 Сформировать дашборд", type="primary", key="btn_dash"):
        if not active_api_key:
            st.error("Не указан API-ключ! Вставьте свой ключ в боковой панели слева (инструкция во вкладке «Гайд»).")
        elif not uploaded_file_dash:
            st.warning("Загрузите PDF-файл.")
        else:
            status = st.empty()
            try:
                client = genai.Client(api_key=active_api_key)
                config = types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ExecutiveSummary,
                    temperature=0.1
                )
                prompt = (
                    "Проанализируй документ. Извлеки 3-4 ключевые метрики для KPI-карточек, исторические точки "
                    "по годам для графиков, а также выводы по стратегии, маркетингу и IT. Заполни схему строго по полям."
                )

                contents = prepare_file_contents(client, uploaded_file_dash, prompt, status)
                status.info(f"⏳ Модель `{model_choice}` собирает структурированный дашборд...")
                t0 = time.time()

                response = generate_with_retry(client, model_choice, contents, config, status)
                status.empty()

                data = ExecutiveSummary.model_validate_json(response.text)
                st.success(f"Дашборд сформирован за {time.time() - t0:.1f} сек.!")
                st.header(f"📊 {data.company_name} — {data.period}")

                # Карточки KPI
                if data.key_metrics_cards:
                    cols = st.columns(len(data.key_metrics_cards))
                    for i, card in enumerate(data.key_metrics_cards):
                        with cols[i]:
                            st.metric(
                                label=card.label,
                                value=card.value,
                                delta=card.delta if card.delta else None
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

                # Скачивание отчета
                report_md = f"# Отчет: {data.company_name} ({data.period})\n\n"
                report_md += "## Ключевые показатели\n"
                for c in data.key_metrics_cards:
                    report_md += f"- **{c.label}**: {c.value} ({c.delta})\n"
                report_md += "\n## Главные выводы\n" + "\n".join([f"- {f}" for f in data.strategic_findings])
                report_md += "\n\n## Маркетинг и клиенты\n" + "\n".join([f"- {m}" for m in data.marketing_and_clients])
                report_md += "\n\n## IT и AI\n" + "\n".join([f"- {t}" for t in data.tech_and_ai])
                report_md += "\n\n## Гипотезы роста\n" + "\n".join([f"- {h}" for h in data.growth_hypotheses])

                st.markdown("---")
                st.download_button(
                    label="📥 Скачать готовый отчет (.md)",
                    data=report_md,
                    file_name=f"report_{data.company_name}.md",
                    mime="text/markdown"
                )

            except Exception as e:
                status.empty()
                st.error(f"Ошибка: {e}")

# ==========================================
# ВКЛАДКА 3: ГАЙД И ПОЛЕЗНЫЕ ССЫЛКИ
# ==========================================
with tab_guide:
    st.header("📖 Как пользоваться платформой и настроить доступ")
    
    st.markdown("""
    ### 1. Бесплатный режим и личный API-ключ (BYOK)
    * **Базовый бесплатный режим:** по умолчанию приложение использует легкую модель **`gemini-3.5-flash-lite`**. Она быстро справляется с большинством документов и имеет наиболее широкие бесплатные квоты.
    * **Зачем создавать свой ключ?** На общем бесплатном тарифе действуют строгие ограничения на количество запросов в минуту (RPM) и объем текста (TPM). Если вы загружаете большие отчеты или хотите использовать флагманские модели, создайте свой бесплатный ключ — это займет 15 секунд.
    * **Безопасность:** ваш ключ нигде не сохраняется на сервере и используется только в рамках текущей сессии браузера.
    """)

    st.markdown("""
    ### 2. Как получить бесплатный Gemini API Key за 3 шага
    1. Перейдите в **[Google AI Studio (раздел API Keys)](https://aistudio.google.com/app/apikey)** и войдите через обычный Google-аккаунт.
    2. Нажмите синюю кнопку **«Create API key»** (Создать API-ключ) и выберите проект (или создайте новый в один клик).
    3. Скопируйте ключ (начинается на `AIzaSy...`) и вставьте его в поле **«Ваш личный Gemini API Key»** в левой боковой панели этого сайта. После этого вам откроется выбор всех моделей!
    """)

    st.markdown("---")
    st.subheader("🧠 Сравнение доступных моделей")
    
    models_info = pd.DataFrame([
        {
            "Модель": "gemini-3.5-flash-lite",
            "Доступ": "Бесплатно по умолчанию",
            "Скорость и лимиты": "Максимальная скорость, самые широкие бесплатные лимиты",
            "Для каких задач": "Быстрый разбор статей, презентаций, выжимка фактов и построение дашбордов"
        },
        {
            "Модель": "gemini-3.8-flash",
            "Доступ": "По личному API-ключу",
            "Скорость и лимиты": "Высокая скорость, продвинутая логика",
            "Для каких задач": "Сложные многостраничные отчеты, вызов инструментов (Function Calling) и аналитика"
        },
        {
            "Модель": "gemini-4-argon",
            "Доступ": "По личному API-ключу",
            "Скорость и лимиты": "Глубокое рассуждение (Reasoning), повышенный расход квоты",
            "Для каких задач": "Глубокий финансовый аудит, поиск скрытых противоречий и сложные стратегии"
        }
    ])
    st.table(models_info)

    st.markdown("---")
    st.subheader("🔗 Полезные ссылки и документация")
    st.markdown("""
    * 🔑 **[Создать / управлять API-ключами (Google AI Studio)](https://aistudio.google.com/app/apikey)** — здесь создается ваш бесплатный ключ.
    * 📊 **[Мониторинг квот и лимитов (Usage & Rate Limits)](https://aistudio.google.com/usage)** — здесь можно посмотреть, сколько бесплатных запросов в минуту и в день доступно вашему аккаунту.
    * 📚 **[Документация по лимитам тарифов Gemini API](https://aistudio.google.com/docs/rate-limits)** — официальная таблица ограничений Free Tier и Pay-as-you-go.
    * 🤖 **[Каталог актуальных моделей Gemini](https://aistudio.google.com/docs/models)** — описание возможностей всех версий моделей.
    """)
