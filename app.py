import io
import os
import re
import time
import tempfile
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from docx import Document
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

# --- Боковая панель: Доступ и Модели ---
with st.sidebar:
    st.markdown("### 🌿 Doc & Market AI")
    st.caption("Для смены темы (Светлая / Темная) нажмите `⋮` в правом верхнем углу → Settings.")
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

# --- Безопасный CSS для Bento-карточек ---
st.markdown("""
<style>
    .kpi-card {
        background-color: #FFFFFF;
        border: 1px solid #E0EAE1;
        border-radius: 16px;
        padding: 18px 20px;
        box-shadow: 0 4px 12px rgba(17, 36, 24, 0.04);
        height: 100%;
    }
    .kpi-card-hero {
        background-color: #112418;
        color: #FFFFFF;
        border-radius: 16px;
        padding: 18px 20px;
        box-shadow: 0 6px 18px rgba(17, 36, 24, 0.14);
        height: 100%;
    }
    .kpi-label {
        font-size: 13px;
        font-weight: 500;
        color: #5E7366;
        margin-bottom: 6px;
    }
    .kpi-card-hero .kpi-label {
        color: #A6C4B0;
    }
    .kpi-value {
        font-size: 26px;
        font-weight: 700;
        color: #122118;
        margin-bottom: 8px;
    }
    .kpi-card-hero .kpi-value {
        color: #FFFFFF;
    }
    .kpi-badge {
        display: inline-block;
        background-color: #E2F5E9;
        color: #2E9E5B;
        font-size: 12px;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 999px;
    }
    .kpi-card-hero .kpi-badge {
        background-color: rgba(61, 214, 121, 0.2);
        color: #6EE79E;
    }
    .insight-box {
        background-color: #112418;
        color: #FFFFFF;
        border-radius: 16px;
        padding: 20px 22px;
        margin-bottom: 16px;
    }
    .insight-title {
        font-size: 14px;
        font-weight: 600;
        color: #6EE79E;
        margin-bottom: 6px;
    }
    .insight-text {
        font-size: 15px;
        line-height: 1.5;
        color: #F0F7F2;
    }
</style>
""", unsafe_allow_html=True)

# --- Заголовок и вкладки (добавлена вкладка Сравнения) ---
st.title("🌿 Doc & Market AI Platform")
st.caption("Интеллектуальный анализ документов, сравнение отчетов, расчеты и экспорт в Word / Excel")

tab_agent, tab_dashboard, tab_compare, tab_guide = st.tabs([
    "🛠️ Агент (Вопросы и Расчеты)",
    "📊 Дашборд (KPI и Диаграммы)",
    "⚖️ Сравнение 2 отчетов",
    "📖 Гайд и Ссылки"
])

# --- Инструменты для Function Calling ---
def calculate_growth_rate(old_value: float, new_value: float) -> str:
    """Вычисляет точный темп прироста в процентах между старым и новым значением."""
    if old_value == 0:
        return "Ошибка: базовое значение равно нулю"
    growth = ((new_value - old_value) / abs(old_value)) * 100
    return f"Прирост: {growth:+.2f}% (с {old_value} до {new_value})"

def calculate_ratio(part: float, total: float) -> str:
    """Вычисляет точную долю части от общего объема в процентах."""
    if total == 0:
        return "Ошибка: общее значение равно нулю"
    ratio = (part / total) * 100
    return f"Доля: {ratio:.2f}% ({part} от {total})"

# --- Pydantic-схема для Дашборда ---
class KPICard(BaseModel):
    label: str = Field(description="Название показателя (например, Выручка)")
    value: str = Field(description="Значение с единицами (например, 1.5 трлн ₽)")
    delta: str = Field(description="Изменение к прошлому периоду (например, +14.2%)")

class MetricPoint(BaseModel):
    year: str = Field(description="Год или период (например, 2023, 2024)")
    value: float = Field(description="Числовое значение показателя")

class MetricSeries(BaseModel):
    metric_name: str = Field(description="Название метрики для графика динамики")
    data_points: list[MetricPoint] = Field(description="Точки динамики по годам")

class BreakdownItem(BaseModel):
    category: str = Field(description="Название сегмента, направления бизнеса или статьи расходов")
    amount: float = Field(description="Числовое значение или доля в процентах")

class ExecutiveSummary(BaseModel):
    company_name: str = Field(description="Название компании или тема отчета")
    period: str = Field(description="Период отчета")
    ai_smart_insight: str = Field(description="Самый главный, концентрированный инсайт по всему документу (2-3 предложения)")
    key_metrics_cards: list[KPICard] = Field(description="Ровно 4 главные метрики для верхних KPI-карточек")
    structure_breakdown: list[BreakdownItem] = Field(description="4-6 категорий для кольцевой диаграммы")
    breakdown_title: str = Field(description="Заголовок для кольцевой диаграммы")
    chart_series: list[MetricSeries] = Field(description="1-2 показателя с историческими точками для графиков динамики")
    strategic_findings: list[str] = Field(description="4-5 ключевых стратегических выводов")
    marketing_and_clients: list[str] = Field(description="3-4 вывода по маркетингу и клиентам")
    tech_and_ai: list[str] = Field(description="3-4 вывода по IT и AI")
    growth_hypotheses: list[str] = Field(description="3 конкретные гипотезы по развитию бизнеса")

# --- Генераторы файлов Word (.docx) и Excel (.xlsx) ---
def build_docx_report(data: ExecutiveSummary) -> bytes:
    doc = Document()
    doc.add_heading(f"Аналитический отчет: {data.company_name} ({data.period})", level=0)

    p_insight = doc.add_paragraph()
    p_insight.add_run("AI Smart Insight: ").bold = True
    p_insight.add_run(data.ai_smart_insight)

    doc.add_heading("1. Ключевые показатели (KPI)", level=1)
    if data.key_metrics_cards:
        table = doc.add_table(rows=1, cols=3)
        table.style = "Table Grid"
        hdr_cells = table.rows[0].cells
        hdr_cells[0].text = "Показатель"
        hdr_cells[1].text = "Значение"
        hdr_cells[2].text = "Динамика"
        for c in data.key_metrics_cards:
            row_cells = table.add_row().cells
            row_cells[0].text = c.label
            row_cells[1].text = c.value
            row_cells[2].text = c.delta or "-"

    sections = [
        ("2. Стратегические выводы", data.strategic_findings),
        ("3. Маркетинг и клиенты", data.marketing_and_clients),
        ("4. Технологии и AI", data.tech_and_ai),
        ("5. Гипотезы роста", data.growth_hypotheses),
    ]
    for title, items in sections:
        doc.add_heading(title, level=1)
        for item in items:
            doc.add_paragraph(item, style="List Bullet")

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()

def build_xlsx_report(data: ExecutiveSummary) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        # Лист 1: KPI
        df_kpi = pd.DataFrame([
            {"Показатель": c.label, "Значение": c.value, "Динамика": c.delta}
            for c in data.key_metrics_cards
        ])
        df_kpi.to_excel(writer, sheet_name="KPI_Показатели", index=False)

        # Лист 2: Структура (для диаграмм)
        if data.structure_breakdown:
            df_struct = pd.DataFrame([
                {"Категория": b.category, "Значение": b.amount}
                for b in data.structure_breakdown
            ])
            df_struct.to_excel(writer, sheet_name="Структура", index=False)

        # Лист 3: Динамика по годам
        if data.chart_series:
            rows_dyn = []
            for s in data.chart_series:
                for p in s.data_points:
                    rows_dyn.append({"Метрика": s.metric_name, "Период": p.year, "Значение": p.value})
            if rows_dyn:
                pd.DataFrame(rows_dyn).to_excel(writer, sheet_name="Динамика", index=False)

        # Лист 4: Текстовые выводы
        rows_text = [{"Раздел": "AI Smart Insight", "Тезис": data.ai_smart_insight}]
        for f in data.strategic_findings:
            rows_text.append({"Раздел": "Стратегия", "Тезис": f})
        for m in data.marketing_and_clients:
            rows_text.append({"Раздел": "Маркетинг и клиенты", "Тезис": m})
        for t in data.tech_and_ai:
            rows_text.append({"Раздел": "IT и AI", "Тезис": t})
        for h in data.growth_hypotheses:
            rows_text.append({"Раздел": "Гипотезы роста", "Тезис": h})
        pd.DataFrame(rows_text).to_excel(writer, sheet_name="Выводы_и_Гипотезы", index=False)

    return buf.getvalue()

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
                match = re.search(r"retry in (\d+(\.\d+)?)s", str(e), re.IGNORECASE)
                wait_sec = float(match.group(1)) + 1.0 if match else 15.0 * (attempt + 1)
                wait_sec = min(wait_sec, 65.0)
                status_placeholder.warning(
                    f"⏳ Пауза {wait_sec:.0f} сек. из-за минутного лимита API (попытка {attempt + 2}/{max_retries})..."
                )
                time.sleep(wait_sec)
            else:
                raise

def prepare_single_file_part(client, uploaded_file, label_prefix, status_placeholder):
    file_bytes = uploaded_file.getvalue()
    file_size_mb = len(file_bytes) / (1024 * 1024)

    if file_size_mb < 20:
        status_placeholder.info(f"⏳ Передача {label_prefix} ({file_size_mb:.1f} МБ)...")
        return types.Part.from_bytes(data=file_bytes, mime_type="application/pdf")
    else:
        status_placeholder.info(f"⏳ Загрузка {label_prefix} ({file_size_mb:.1f} МБ) через Files API...")
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
            if uploaded_gemini.state.name == "FAILED":
                raise ValueError(f"Ошибка обработки {label_prefix} сервером Google.")
            return uploaded_gemini
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

# ==========================================
# ВКЛАДКА 1: АГЕНТ
# ==========================================
with tab_agent:
    st.subheader("🛠️ Вопрос к документу с расчетами")
    uploaded_file_agent = st.file_uploader("Загрузите PDF-документ:", type=["pdf"], key="pdf_agent")

    col_opt1, col_opt2 = st.columns([2, 1])
    with col_opt1:
        use_math_tools = st.checkbox("Подключить калькуляторы формул (прирост и доли)", value=True)
    with col_opt2:
        use_google_search = st.checkbox("Подключить Google Search (веб-поиск)", value=False)

    user_task = st.text_area(
        "Задание для аналитика:",
        value="Найди ключевые финансовые показатели за последние 2 года, рассчитай их точный темп роста и сформулируй 3 главных вывода."
    )

    if st.button("🚀 Запустить анализ", type="primary", key="btn_agent"):
        if not active_api_key:
            st.error("Укажите API-ключ в левой панели (инструкция во вкладке «Гайд»).")
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
                file_part = prepare_single_file_part(client, uploaded_file_agent, "PDF", status)
                status.info(f"⏳ Модель `{model_choice}` анализирует документ...")
                t0 = time.time()

                response = generate_with_retry(client, model_choice, [file_part, user_task], config, status)
                status.empty()

                st.success(f"Готово за {time.time() - t0:.1f} сек.")
                st.markdown(response.text)

            except Exception as e:
                status.empty()
                st.error(f"Ошибка: {e}")

# ==========================================
# ВКЛАДКА 2: ДАШБОРД И ЭКСПОРТ (DOCX / XLSX / MD)
# ==========================================
with tab_dashboard:
    st.subheader("📊 Аналитический дашборд и экспорт отчетов")
    uploaded_file_dash = st.file_uploader("Загрузите PDF-документ для визуализации:", type=["pdf"], key="pdf_dash")

    if st.button("🚀 Сформировать дашборд", type="primary", key="btn_dash"):
        if not active_api_key:
            st.error("Укажите API-ключ в левой панели.")
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
                    "Проанализируй документ. Извлеки ровно 4 ключевые метрики для KPI-карточек (самую важную поставь первой), "
                    "4-6 категорий для кольцевой диаграммы структуры показателей, исторические точки для графиков динамики, "
                    "главный AI-инсайт и выводы по направлениям. Заполни схему строго по полям."
                )

                file_part = prepare_single_file_part(client, uploaded_file_dash, "PDF", status)
                status.info(f"⏳ Модель `{model_choice}` собирает дашборд...")
                t0 = time.time()

                response = generate_with_retry(client, model_choice, [file_part, prompt], config, status)
                status.empty()

                data = ExecutiveSummary.model_validate_json(response.text)
                st.success(f"Дашборд готов за {time.time() - t0:.1f} сек.")
                st.markdown(f"## {data.company_name} — {data.period}")

                if data.key_metrics_cards:
                    cols = st.columns(len(data.key_metrics_cards))
                    for i, card in enumerate(data.key_metrics_cards):
                        card_class = "kpi-card-hero" if i == 0 else "kpi-card"
                        delta_html = f'<span class="kpi-badge">{card.delta}</span>' if card.delta else ""
                        with cols[i]:
                            st.markdown(f"""
                            <div class="{card_class}">
                                <div class="kpi-label">{card.label}</div>
                                <div class="kpi-value">{card.value}</div>
                                {delta_html}
                            </div>
                            """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                col_donut, col_right = st.columns([1.15, 1])
                donut_colors = ["#112418", "#2E9E5B", "#6BCB77", "#A3E4A8", "#C8E6C9", "#DCE8DE"]

                with col_donut:
                    if data.structure_breakdown:
                        st.markdown(f"#### {data.breakdown_title}")
                        labels = [item.category for item in data.structure_breakdown]
                        values = [item.amount for item in data.structure_breakdown]

                        fig_donut = go.Figure(data=[go.Pie(
                            labels=labels,
                            values=values,
                            hole=0.62,
                            marker=dict(colors=donut_colors, line=dict(color="#FFFFFF", width=2)),
                            textinfo="percent",
                            hovertemplate="<b>%{label}</b><br>Значение: %{value}<br>Доля: %{percent}<extra></extra>"
                        )])
                        fig_donut.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            margin=dict(t=10, b=10, l=10, r=10),
                            height=320,
                            legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02)
                        )
                        st.plotly_chart(fig_donut, use_container_width=True)

                with col_right:
                    st.markdown(f"""
                    <div class="insight-box">
                        <div class="insight-title">✨ AI Smart Insight</div>
                        <div class="insight-text">{data.ai_smart_insight}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    if data.chart_series:
                        s = data.chart_series[0]
                        if len(s.data_points) > 1:
                            years = [p.year for p in s.data_points]
                            vals = [p.value for p in s.data_points]
                            fig_line = go.Figure(data=[go.Scatter(
                                x=years,
                                y=vals,
                                mode="lines+markers",
                                name=s.metric_name,
                                line=dict(color="#2E9E5B", width=3, shape="spline"),
                                marker=dict(size=8, color="#2E9E5B"),
                                fill="tozeroy",
                                fillcolor="rgba(46, 158, 91, 0.12)"
                            )])
                            fig_line.update_layout(
                                title=dict(text=s.metric_name, font=dict(size=14)),
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                margin=dict(t=35, b=15, l=15, r=15),
                                height=210
                            )
                            st.plotly_chart(fig_line, use_container_width=True)

                st.markdown("---")

                c_left, c_right = st.columns(2)
                with c_left:
                    st.markdown("#### 🎯 Стратегические выводы")
                    for f in data.strategic_findings:
                        st.markdown(f"- {f}")
                    st.markdown("#### 💡 Гипотезы роста")
                    for h in data.growth_hypotheses:
                        st.markdown(f"- {h}")

                with c_right:
                    st.markdown("#### 👥 Маркетинг и клиенты")
                    for m in data.marketing_and_clients:
                        st.markdown(f"- {m}")
                    st.markdown("#### 💻 Технологии и AI")
                    for t in data.tech_and_ai:
                        st.markdown(f"- {t}")

                # --- Блок экспорта в 3 форматах (Word, Excel, Markdown) ---
                st.markdown("---")
                st.subheader("📥 Экспорт готового отчета")

                docx_bytes = build_docx_report(data)
                xlsx_bytes = build_xlsx_report(data)

                report_md = f"# Отчет: {data.company_name} ({data.period})\n\n"
                report_md += f"> **AI Smart Insight:** {data.ai_smart_insight}\n\n"
                report_md += "## Ключевые показатели\n"
                for c in data.key_metrics_cards:
                    report_md += f"- **{c.label}**: {c.value} ({c.delta})\n"
                report_md += "\n## Стратегические выводы\n" + "\n".join([f"- {f}" for f in data.strategic_findings])
                report_md += "\n\n## Маркетинг и клиенты\n" + "\n".join([f"- {m}" for m in data.marketing_and_clients])
                report_md += "\n\n## IT и AI\n" + "\n".join([f"- {t}" for t in data.tech_and_ai])
                report_md += "\n\n## Гипотезы роста\n" + "\n".join([f"- {h}" for h in data.growth_hypotheses])

                dl_col1, dl_col2, dl_col3 = st.columns(3)
                with dl_col1:
                    st.download_button(
                        label="📘 Скачать Word (.docx)",
                        data=docx_bytes,
                        file_name=f"Report_{data.company_name}.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )
                with dl_col2:
                    st.download_button(
                        label="📗 Скачать Excel (.xlsx)",
                        data=xlsx_bytes,
                        file_name=f"Data_{data.company_name}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                with dl_col3:
                    st.download_button(
                        label="📄 Скачать Markdown (.md)",
                        data=report_md,
                        file_name=f"Report_{data.company_name}.md",
                        mime="text/markdown"
                    )

            except Exception as e:
                status.empty()
                st.error(f"Ошибка: {e}")

# ==========================================
# ВКЛАДКА 3: СРАВНЕНИЕ ДВУХ ОТЧЕТОВ (GAP-АНАЛИЗ)
# ==========================================
with tab_compare:
    st.subheader("⚖️ Сравнительный анализ двух документов")
    st.caption("Загрузите два отчета (например, 2024 vs 2025 год или двух конкурентов) для автоматического сопоставления.")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        doc_1 = st.file_uploader("📄 Документ №1 (Базовый / Конкурент А):", type=["pdf"], key="cmp_doc1")
    with col_f2:
        doc_2 = st.file_uploader("📄 Документ №2 (Новый / Конкурент Б):", type=["pdf"], key="cmp_doc2")

    compare_goal = st.text_area(
        "Критерии сравнения:",
        value="Проведи глубокий сравнительный анализ двух документов:\n1. Составь сводную таблицу ключевых метрик (Документ 1 vs Документ 2 vs Изменение/Разница).\n2. Выдели 4 главных стратегических сдвига или отличия в бизнес-модели.\n3. Определи сильные и слабые стороны каждого отчета и дай итоговое резюме."
    )

    if st.button("⚖️ Сравнить документы", type="primary", key="btn_compare"):
        if not active_api_key:
            st.error("Укажите API-ключ в левой панели.")
        elif not doc_1 or not doc_2:
            st.warning("Загрузите оба PDF-документа для сравнения.")
        else:
            status = st.empty()
            try:
                client = genai.Client(api_key=active_api_key)
                config = types.GenerateContentConfig(
                    tools=[calculate_growth_rate, calculate_ratio],
                    temperature=0.2
                )
                part1 = prepare_single_file_part(client, doc_1, "Документ №1", status)
                part2 = prepare_single_file_part(client, doc_2, "Документ №2", status)

                prompt_cmp = [
                    "Первый документ (Базовый):", part1,
                    "Второй документ (Сравниваемый):", part2,
                    compare_goal
                ]

                status.info(f"⏳ Модель `{model_choice}` сопоставляет оба документа и считает дельты...")
                t0 = time.time()

                response = generate_with_retry(client, model_choice, prompt_cmp, config, status)
                status.empty()

                st.success(f"Сравнение завершено за {time.time() - t0:.1f} сек.")
                st.markdown(response.text)

            except Exception as e:
                status.empty()
                st.error(f"Ошибка при сравнении: {e}")

# ==========================================
# ВКЛАДКА 4: ГАЙД И ПОЛЕЗНЫЕ ССЫЛКИ
# ==========================================
with tab_guide:
    st.header("📖 Как пользоваться платформой и настроить доступ")

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("""
        ### 🆓 1. Бесплатный режим и личный ключ (BYOK)
        * **Базовый режим:** по умолчанию приложение использует легкую модель **`gemini-3.5-flash-lite`**. Она быстро справляется с разбором документов и не требует настроек.
        * **Зачем нужен свой ключ?** На общем бесплатном тарифе действуют строгие лимиты на число запросов в минуту (RPM) и объем текста (TPM). Вставьте свой бесплатный ключ, чтобы снять общие ограничения и открыть доступ к старшим моделям.
        * **Приватность:** ваш API-ключ нигде не сохраняется и живет только внутри текущей сессии браузера.
        """)

    with col_g2:
        st.markdown("""
        ### 🔑 2. Как получить Gemini API Key за 15 секунд
        1. Откройте **[Google AI Studio (API Keys)](https://aistudio.google.com/app/apikey)** и войдите через обычный Google-аккаунт.
        2. Нажмите синюю кнопку **«Create API key»** и выберите проект (или создайте новый в один клик).
        3. Скопируйте ключ (начинается на `AIzaSy...`) и вставьте его в поле **«Ваш личный Gemini API Key»** в левой панели[cite: 3].
        4. После ввода ключа автоматически разблокируются модели `gemini-3.8-flash` и `gemini-4-argon`[cite: 3].
        """)

    st.markdown("---")
    st.subheader("🧠 Сравнение доступных моделей")
    models_info = pd.DataFrame([
        {
            "Модель": "gemini-3.5-flash-lite",
            "Доступ": "Бесплатно по умолчанию",
            "Особенности": "Максимальная скорость, самые широкие бесплатные квоты",
            "Для каких задач": "Быстрый разбор статей, презентаций, выжимка фактов и построение дашбордов"
        },
        {
            "Модель": "gemini-3.8-flash",
            "Доступ": "По личному API-ключу",
            "Особенности": "Высокая скорость, продвинутая логика и работа с инструментами",
            "Для каких задач": "Сложные многостраничные отчеты, сравнение двух документов и точные расчеты"
        },
        {
            "Модель": "gemini-4-argon",
            "Доступ": "По личному API-ключу",
            "Особенности": "Глубокое рассуждение (Reasoning), повышенный расход квоты",
            "Для каких задач": "Глубокий финансовый аудит, поиск скрытых противоречий и сложные стратегии"
        }
    ])
    st.table(models_info)

    st.markdown("---")
    st.subheader("🔗 Полезные ресурсы и документация")
    st.markdown("""
    * 🔑 **[Создать / управлять API-ключами (Google AI Studio)](https://aistudio.google.com/app/apikey)** — генерация вашего бесплатного ключа.
    * 📊 **[Мониторинг квот и лимитов (Usage & Rate Limits)](https://aistudio.google.com/usage)** — проверка доступных лимитов запросов в минуту и в день.
    * 📚 **[Документация по лимитам тарифов Gemini API](https://aistudio.google.com/docs/rate-limits)** — официальная таблица ограничений Free Tier и платного тарифа[cite: 3].
    * 🤖 **[Каталог моделей Gemini](https://aistudio.google.com/docs/models)** — подробное описание возможностей актуальных моделей.
    """)
