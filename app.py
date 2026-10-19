import io
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="AgroCognitivo", page_icon="🌱", layout="wide")

TEXT = {
    "Español": {
        "title": "AgroCognitivo",
        "subtitle": "Sistema inteligente de monitoreo y análisis de un invernadero de tomates",
        "demo": "Prototipo académico con datos simulados. No está conectado a sensores reales.",
        "settings": "Configuración",
        "language": "Idioma del reporte",
        "rows": "Cantidad de registros simulados",
        "seed": "Semilla de simulación",
        "upload": "Cargar archivo CSV propio (opcional)",
        "upload_help": "El CSV debe incluir: fecha, temperatura_c, humedad_relativa_pct, humedad_suelo_pct, ph.",
        "use_demo": "Usar datos simulados",
        "overview": "Resumen del invernadero",
        "records": "Registros analizados",
        "alerts": "Registros con alertas",
        "temp_avg": "Temperatura media (°C)",
        "ph_avg": "pH medio",
        "data": "Datos y análisis exploratorio",
        "tab1": "Tendencias",
        "tab2": "Distribuciones",
        "tab3": "Tabla de datos",
        "trend_title": "Evolución de las variables",
        "dist_title": "Distribución de las mediciones",
        "variable": "Variable para visualizar",
        "diagnosis": "Diagnóstico cognitivo",
        "diagnosis_desc": "El diagnóstico combina reglas agronómicas configurables con detección estadística de anomalías (Isolation Forest). Es una herramienta educativa, no sustituye a un agrónomo.",
        "status": "Estado general",
        "normal": "Sin alertas según los umbrales configurados",
        "attention": "Requiere atención",
        "rule_alerts": "Alertas por reglas",
        "ml_alerts": "Anomalías estadísticas",
        "recommendations": "Recomendaciones",
        "no_recs": "No se detectaron condiciones fuera de los umbrales configurados.",
        "report": "Reporte bilingüe / descargable",
        "download_csv": "Descargar datos analizados (CSV)",
        "download_report": "Descargar reporte (TXT)",
        "ph_note": "El intervalo de pH usado como referencia (5.5–6.5) es orientativo para tomate y debe ajustarse al sustrato, etapa del cultivo y método de medición.",
        "temp": "Temperatura (°C)",
        "humidity": "Humedad relativa (%)",
        "soil": "Humedad del suelo (%)",
        "ph": "pH",
        "date": "Fecha y hora",
        "rule_status": "Resultado de reglas",
        "anomaly": "Anomalía ML",
        "yes": "Sí",
        "no": "No",
        "generated": "Reporte generado",
        "spanish": "ESPAÑOL",
        "english": "ENGLISH",
        "summary": "Resumen",
        "recs_en": "Recommendations",
        "report_title": "AGROCOGNITIVO — REPORTE DEL INVERNADERO",
        "sample": "Origen de datos: simulación académica",
        "analyzed": "Registros analizados",
        "average": "Promedio",
        "min": "Mínimo",
        "max": "Máximo",
        "diagnosis_en": "Greenhouse status",
        "attention_en": "Attention required",
        "normal_en": "No alerts under the configured thresholds",
        "rule_alerts_en": "Rule-based alerts",
        "ml_alerts_en": "Statistical anomalies",
        "disclaimer": "Nota: este prototipo no controla equipos físicos ni ofrece diagnóstico agronómico profesional."
    },
    "English": {
        "title": "AgroCognitivo",
        "subtitle": "Intelligent monitoring and analysis system for a tomato greenhouse",
        "demo": "Academic prototype using simulated data. It is not connected to real sensors.",
        "settings": "Settings",
        "language": "Report language",
        "rows": "Number of simulated records",
        "seed": "Simulation seed",
        "upload": "Upload your own CSV (optional)",
        "upload_help": "CSV must include: fecha, temperatura_c, humedad_relativa_pct, humedad_suelo_pct, ph.",
        "use_demo": "Use simulated data",
        "overview": "Greenhouse overview",
        "records": "Records analyzed",
        "alerts": "Records with alerts",
        "temp_avg": "Average temperature (°C)",
        "ph_avg": "Average pH",
        "data": "Data and exploratory analysis",
        "tab1": "Trends",
        "tab2": "Distributions",
        "tab3": "Data table",
        "trend_title": "Variable trends over time",
        "dist_title": "Measurement distributions",
        "variable": "Variable to display",
        "diagnosis": "Cognitive diagnosis",
        "diagnosis_desc": "Diagnosis combines configurable agronomic rules with statistical anomaly detection (Isolation Forest). This is an educational tool, not a replacement for an agronomist.",
        "status": "Overall status",
        "normal": "No alerts under the configured thresholds",
        "attention": "Attention required",
        "rule_alerts": "Rule-based alerts",
        "ml_alerts": "Statistical anomalies",
        "recommendations": "Recommendations",
        "no_recs": "No conditions outside the configured thresholds were detected.",
        "report": "Bilingual / downloadable report",
        "download_csv": "Download analyzed data (CSV)",
        "download_report": "Download report (TXT)",
        "ph_note": "The reference pH interval (5.5–6.5) is a guideline for tomatoes and should be adjusted for substrate, crop stage and measurement method.",
        "temp": "Temperature (°C)",
        "humidity": "Relative humidity (%)",
        "soil": "Soil moisture (%)",
        "ph": "pH",
        "date": "Date and time",
        "rule_status": "Rule result",
        "anomaly": "ML anomaly",
        "yes": "Yes",
        "no": "No",
        "generated": "Report generated",
        "spanish": "ESPAÑOL",
        "english": "ENGLISH",
        "summary": "Summary",
        "recs_en": "Recommendations",
        "report_title": "AGROCOGNITIVO — GREENHOUSE REPORT",
        "sample": "Data source: academic simulation",
        "analyzed": "Records analyzed",
        "average": "Average",
        "min": "Minimum",
        "max": "Maximum",
        "diagnosis_en": "Greenhouse status",
        "attention_en": "Attention required",
        "normal_en": "No alerts under the configured thresholds",
        "rule_alerts_en": "Rule-based alerts",
        "ml_alerts_en": "Statistical anomalies",
        "disclaimer": "Note: this prototype does not control physical equipment or provide professional agronomic diagnosis."
    }
}

st.sidebar.header("🌿 " + "AgroCognitivo")
lang = st.sidebar.selectbox("Idioma / Language", ["Español", "English"])
t = TEXT[lang]
st.sidebar.subheader(t["settings"])
n = st.sidebar.slider(t["rows"], 50, 1000, 300, 50)
seed = st.sidebar.number_input(t["seed"], min_value=1, max_value=99999, value=42, step=1)
uploaded = st.sidebar.file_uploader(t["upload"], type=["csv"], help=t["upload_help"])

def make_demo_data(n_rows: int, random_seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    dates = pd.date_range(end=pd.Timestamp.now().floor("h"), periods=n_rows, freq="h")
    temp = rng.normal(25, 3.2, n_rows)
    humidity = rng.normal(68, 9, n_rows)
    soil = rng.normal(55, 12, n_rows)
    ph = rng.normal(6.1, 0.45, n_rows)
    # Inject a small number of synthetic events so the dashboard demonstrates alerts.
    count = max(2, n_rows // 25)
    idx = rng.choice(n_rows, size=count, replace=False)
    for j, i in enumerate(idx):
        if j % 4 == 0:
            temp[i] = rng.uniform(33, 38)
        elif j % 4 == 1:
            humidity[i] = rng.uniform(88, 98)
        elif j % 4 == 2:
            soil[i] = rng.uniform(15, 25)
        else:
            ph[i] = rng.choice([rng.uniform(4.5, 5.2), rng.uniform(6.8, 7.5)])
    return pd.DataFrame({
        "fecha": dates,
        "temperatura_c": np.round(temp, 2),
        "humedad_relativa_pct": np.round(np.clip(humidity, 15, 100), 2),
        "humedad_suelo_pct": np.round(np.clip(soil, 1, 100), 2),
        "ph": np.round(np.clip(ph, 3.5, 9.5), 2),
    })

REQUIRED = ["fecha", "temperatura_c", "humedad_relativa_pct", "humedad_suelo_pct", "ph"]
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            st.error("Faltan columnas requeridas: " + ", ".join(missing))
            st.stop()
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
        for c in REQUIRED[1:]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=REQUIRED).copy()
        if df.empty:
            st.error("El CSV no contiene filas válidas después de limpiar los datos.")
            st.stop()
        data_source = "CSV del usuario"
    except Exception as exc:
        st.error(f"No se pudo leer el CSV: {exc}")
        st.stop()
else:
    df = make_demo_data(int(n), int(seed))
    data_source = "Simulación académica"

# Explicit, configurable reference intervals for this educational prototype.
df["alertas_reglas"] = (
    (df["temperatura_c"] < 18) | (df["temperatura_c"] > 30) |
    (df["humedad_relativa_pct"] < 45) | (df["humedad_relativa_pct"] > 85) |
    (df["humedad_suelo_pct"] < 30) | (df["humedad_suelo_pct"] > 80) |
    (df["ph"] < 5.5) | (df["ph"] > 6.5)
)

features = ["temperatura_c", "humedad_relativa_pct", "humedad_suelo_pct", "ph"]
if len(df) >= 20:
    model = IsolationForest(contamination=0.06, random_state=42)
    df["anomalia_ml"] = model.fit_predict(df[features]) == -1
else:
    df["anomalia_ml"] = False

def recommendations(row):
    result = []
    if row["temperatura_c"] < 18:
        result.append("Temperatura baja: revisar ventilación y calefacción." if lang == "Español" else "Low temperature: check ventilation and heating.")
    elif row["temperatura_c"] > 30:
        result.append("Temperatura alta: aumentar ventilación y evaluar sombreado." if lang == "Español" else "High temperature: increase ventilation and consider shading.")
    if row["humedad_relativa_pct"] > 85:
        result.append("Humedad ambiental alta: mejorar ventilación y vigilar enfermedades fúngicas." if lang == "Español" else "High relative humidity: improve ventilation and monitor fungal disease.")
    elif row["humedad_relativa_pct"] < 45:
        result.append("Humedad ambiental baja: revisar riego y condiciones de ventilación." if lang == "Español" else "Low relative humidity: review irrigation and ventilation.")
    if row["humedad_suelo_pct"] < 30:
        result.append("Humedad del suelo baja: comprobar el riego y la humedad en la zona radicular." if lang == "Español" else "Low soil moisture: check irrigation and root-zone moisture.")
    elif row["humedad_suelo_pct"] > 80:
        result.append("Humedad del suelo alta: revisar drenaje y evitar saturación." if lang == "Español" else "High soil moisture: check drainage and avoid saturation.")
    if row["ph"] < 5.5 or row["ph"] > 6.5:
        result.append("pH fuera del intervalo orientativo 5.5–6.5: verificar la calibración y repetir la medición." if lang == "Español" else "pH outside the guideline range 5.5–6.5: verify calibration and repeat the measurement.")
    return result

df["recomendaciones"] = df.apply(recommendations, axis=1)
df["estado"] = np.where(df["alertas_reglas"], t["attention"], t["normal"])

st.title("🌱 " + t["title"])
st.subheader(t["subtitle"])
st.info(t["demo"] + f"  Fuente actual: {data_source}." if lang == "Español" else t["demo"] + f"  Current source: {data_source}.")
st.caption(t["ph_note"])

st.header(t["overview"])
c1, c2, c3, c4 = st.columns(4)
c1.metric(t["records"], f"{len(df):,}")
c2.metric(t["alerts"], f"{int(df['alertas_reglas'].sum()):,}")
c3.metric(t["temp_avg"], f"{df['temperatura_c'].mean():.1f}")
c4.metric(t["ph_avg"], f"{df['ph'].mean():.2f}")

st.header(t["data"])
tab1, tab2, tab3 = st.tabs([t["tab1"], t["tab2"], t["tab3"]])
labels = {
    "temperatura_c": t["temp"], "humedad_relativa_pct": t["humidity"],
    "humedad_suelo_pct": t["soil"], "ph": t["ph"]
}
with tab1:
    variable = st.selectbox(t["variable"], list(labels.keys()), format_func=lambda x: labels[x])
    fig = px.line(df, x="fecha", y=variable, title=f"{t['trend_title']}: {labels[variable]}")
    fig.update_layout(xaxis_title=t["date"], yaxis_title=labels[variable])
    st.plotly_chart(fig, use_container_width=True)
with tab2:
    variable2 = st.selectbox(t["variable"], list(labels.keys()), format_func=lambda x: labels[x], key="dist")
    fig2 = px.histogram(df, x=variable2, nbins=25, title=f"{t['dist_title']}: {labels[variable2]}")
    fig2.update_layout(xaxis_title=labels[variable2], yaxis_title="Frecuencia" if lang == "Español" else "Frequency")
    st.plotly_chart(fig2, use_container_width=True)
with tab3:
    show_cols = ["fecha"] + features + ["estado", "anomalia_ml"]
    table = df[show_cols].copy()
    table = table.rename(columns={
        "fecha": t["date"], "temperatura_c": t["temp"],
        "humedad_relativa_pct": t["humidity"], "humedad_suelo_pct": t["soil"],
        "ph": t["ph"], "estado": t["rule_status"], "anomalia_ml": t["anomaly"]
    })
    st.dataframe(table, use_container_width=True, hide_index=True)

st.header(t["diagnosis"])
st.write(t["diagnosis_desc"])
rule_count = int(df["alertas_reglas"].sum())
ml_count = int(df["anomalia_ml"].sum())
d1, d2 = st.columns(2)
d1.metric(t["rule_alerts"], rule_count)
d2.metric(t["ml_alerts"], ml_count)
if rule_count:
    st.warning(t["attention"])
else:
    st.success(t["normal"])

latest = df.iloc[-1]
st.subheader(t["recommendations"])
latest_recs = latest["recomendaciones"]
if latest_recs:
    for rec in latest_recs:
        st.write("- " + rec)
else:
    st.write(t["no_recs"])

st.header(t["report"])
def build_report():
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    status_es = "Requiere atención" if rule_count else "Sin alertas según los umbrales configurados"
    status_en = "Attention required" if rule_count else "No alerts under the configured thresholds"
    lines = [
        "AGROCOGNITIVO — REPORTE DEL INVERNADERO / GREENHOUSE REPORT",
        f"Fecha / Date: {now}",
        f"Origen de datos / Data source: {data_source}",
        "AVISO / NOTICE: Datos simulados para fines académicos cuando no se carga un CSV propio.",
        "",
        "ESPAÑOL",
        f"Registros analizados: {len(df)}",
        f"Temperatura media: {df['temperatura_c'].mean():.2f} °C",
        f"Humedad relativa media: {df['humedad_relativa_pct'].mean():.2f} %",
        f"Humedad del suelo media: {df['humedad_suelo_pct'].mean():.2f} %",
        f"pH medio: {df['ph'].mean():.2f}",
        f"Registros con alertas por reglas: {rule_count}",
        f"Anomalías detectadas por Isolation Forest: {ml_count}",
        f"Estado general: {status_es}",
        "Recomendaciones para la última lectura:",
        *(["- " + x for x in latest_recs] if latest_recs else ["- No se detectaron condiciones fuera de los umbrales configurados."]),
        "",
        "ENGLISH",
        f"Records analyzed: {len(df)}",
        f"Average temperature: {df['temperatura_c'].mean():.2f} °C",
        f"Average relative humidity: {df['humedad_relativa_pct'].mean():.2f} %",
        f"Average soil moisture: {df['humedad_suelo_pct'].mean():.2f} %",
        f"Average pH: {df['ph'].mean():.2f}",
        f"Records with rule-based alerts: {rule_count}",
        f"Anomalies detected by Isolation Forest: {ml_count}",
        f"Overall status: {status_en}",
        "Guideline: tomato pH reference interval used here is 5.5–6.5; verify against the actual growing system.",
        "DISCLAIMER: This academic prototype does not control physical equipment or replace professional agronomic advice."
    ]
    return "\n".join(lines)

report = build_report()
st.download_button(t["download_report"], data=report.encode("utf-8"), file_name="reporte_agrocognitivo.txt", mime="text/plain")
csv_buffer = io.StringIO()
df.drop(columns=["recomendaciones"]).to_csv(csv_buffer, index=False)
st.download_button(t["download_csv"], data=csv_buffer.getvalue().encode("utf-8"), file_name="datos_agrocognitivo_analizados.csv", mime="text/csv")

with st.expander("Detalles técnicos / Technical details"):
    st.markdown("""
    **Arquitectura / Architecture**
    - **Adquisición / Input:** CSV del usuario o conjunto de datos sintéticos.
    - **Preparación / Preparation:** conversión de tipos, eliminación de filas incompletas en el CSV.
    - **Sistema basado en reglas / Rule-based system:** umbrales configurados para temperatura, humedad relativa, humedad del suelo y pH.
    - **Aprendizaje automático / Machine learning:** Isolation Forest no supervisado para señalar observaciones estadísticamente atípicas.
    - **Presentación / Output:** panel interactivo, gráficas y reporte bilingüe descargable.

    **Limitaciones / Limitations:** los datos simulados no representan una medición real. Una anomalía estadística no equivale necesariamente a un problema agronómico. Los umbrales son orientativos y deben validarse con especialistas y con el sistema de cultivo real.
    """)
