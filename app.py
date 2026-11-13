"""MamaIA · Apoyo al tamizaje de cáncer de mama con aprendizaje automático supervisado.

Ejecutar:  streamlit run app.py
Herramienta educativa y de apoyo. No sustituye el diagnóstico de un profesional de la salud.
"""
import inspect
import io
import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sklearn
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from config import (ALIAS_ETIQUETA, COLOR_BENIGNO, COLOR_MALIGNO, GRUPOS, GRUPOS_UI, RAIZ, VARIABLES,
                    etiqueta_binaria, limpiar, mapear_columnas)

st.set_page_config(page_title="MamaIA · Apoyo al tamizaje de cáncer de mama", page_icon="🎗️",
                   layout="wide", initial_sidebar_state="expanded")


def ancho(funcion) -> dict:
    """Compatibilidad entre versiones de Streamlit (width="stretch" vs use_container_width)."""
    if "width" in inspect.signature(funcion).parameters:
        return {"width": "stretch"}
    return {"use_container_width": True}


# ---------------------------------------------------------------------------
# Estilos
# ---------------------------------------------------------------------------
st.markdown("""
<style>
.block-container{max-width:1250px;padding-top:1.2rem;padding-bottom:3rem}
#MainMenu, footer{visibility:hidden}
.hero{background:linear-gradient(120deg,#9d174d 0%,#be185d 45%,#6d28d9 100%);color:#fff;
      padding:26px 32px 22px;border-radius:20px;margin-bottom:14px;box-shadow:0 8px 24px rgba(109,40,217,.18)}
.hero-top{display:flex;align-items:center;gap:16px}
.hero .logo{font-size:2.8rem;line-height:1}
.hero h1{margin:0;font-size:2.1rem;color:#fff;letter-spacing:-.5px}
.hero p{margin:2px 0 0;opacity:.93;font-size:1.02rem}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:16px}
.chip{background:rgba(255,255,255,.17);border:1px solid rgba(255,255,255,.28);padding:5px 13px;
      border-radius:999px;font-size:.85rem}
.aviso{border-left:4px solid #f59e0b;background:#fffbeb;color:#1f2937;padding:11px 15px;
       border-radius:8px;font-size:.88rem;margin:8px 0}
.veredicto{border-radius:14px;padding:16px 20px;border:2px solid;margin:6px 0 10px}
.veredicto h3{margin:0 0 4px;font-size:1.2rem}
.veredicto p{margin:0;font-size:.95rem}
.veredicto.alto{border-color:#be185d;background:#fff1f2;color:#881337}
.veredicto.medio{border-color:#d97706;background:#fffbeb;color:#78350f}
.veredicto.bajo{border-color:#16a34a;background:#f0fdf4;color:#14532d}
.tarjeta{background:#faf5ff;border:1px solid #e9d5ff;border-radius:12px;padding:12px 16px;color:#1f2937}
.pequeno{font-size:.85rem;color:#6b7280}
div[data-testid="stMetric"]{background:#faf5ff;border:1px solid #f0e6ff;padding:10px 14px;border-radius:12px}
button[data-baseweb="tab"]{font-size:1rem;padding:10px 14px}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Carga de modelos y datos (si faltan o son de otra versión de scikit-learn, se reentrena sola)
# ---------------------------------------------------------------------------
def _leer_artefactos():
    try:
        meta = json.loads((RAIZ / "metricas.json").read_text(encoding="utf-8"))
        if meta.get("version", {}).get("sklearn") != sklearn.__version__:
            return None
        modelos = joblib.load(RAIZ / "modelos.pkl")
        esperados = {"KNN", "SVM", "Random Forest"}
        # Evita reutilizar artefactos antiguos que todavía contienen siete algoritmos.
        if set(modelos) != esperados or set(meta.get("modelos", {})) != esperados:
            return None
        return (modelos, meta, pd.read_csv(RAIZ / "datos.csv"))
    except Exception:
        return None


@st.cache_resource(show_spinner="Preparando los modelos (solo la primera vez, ~1 minuto)…")
def cargar():
    artefactos = _leer_artefactos()
    if artefactos is None:
        import entrenar
        entrenar.entrenar(verbose=False)
        artefactos = _leer_artefactos()
    if artefactos is None:
        raise RuntimeError("No se pudieron generar los modelos. Ejecuta `python entrenar.py` y revisa los mensajes.")
    return artefactos


try:
    MODELOS, META, DF = cargar()
except Exception as error:  # noqa: BLE001
    st.error(f"No se pudo cargar el sistema. {error}")
    st.stop()

F = META["features"]
M = META["modelos"]
NM = list(M)
G = META["ganador"]
NOMBRE = {f: VARIABLES[f]["nombre"] for f in F}
MED = {f: META["rangos"][f][2] for f in F}
Y_TEST = np.array(META["y_test"])
MED_B = DF[DF["maligno"] == 0][F].median()
MED_M = DF[DF["maligno"] == 1][F].median()
MEDIA, DESV = DF[F].mean(), DF[F].std(ddof=0)
R_PER = DF["mean perimeter"] / (2 * np.pi * DF["mean radius"])
R_ARE = DF["mean area"] / (np.pi * DF["mean radius"] ** 2)
LIM_PER = (R_PER.quantile(0.005) * 0.97, R_PER.quantile(0.995) * 1.03)
LIM_ARE = (R_ARE.quantile(0.005) * 0.97, R_ARE.quantile(0.995) * 1.03)


def limitar(f, valor):
    lo, hi, _ = META["rangos"][f]
    return float(min(max(valor, lo), hi))


def reiniciar():
    for f in F:
        st.session_state[f"v_{f}"] = float(MED[f])
    st.session_state["ejemplo"] = None


def cargar_ejemplo(i):
    for f in F:
        st.session_state[f"v_{f}"] = limitar(f, META["ejemplos"][i]["valores"][f])
    st.session_state["ejemplo"] = i


def ajustar_tamano():
    r = st.session_state["v_mean radius"]
    st.session_state["v_mean perimeter"] = limitar("mean perimeter", round(2 * np.pi * r * R_PER.median(), 1))
    st.session_state["v_mean area"] = limitar("mean area", round(np.pi * r * r * R_ARE.median(), 0))


if f"v_{F[0]}" not in st.session_state:
    reiniciar()


# ---------------------------------------------------------------------------
# Utilidades de cálculo
# ---------------------------------------------------------------------------
def probabilidades(filas: pd.DataFrame) -> dict:
    """Probabilidad de malignidad (0-100) de cada modelo para cada fila."""
    return {n: m.predict_proba(filas[F])[:, 1] * 100 for n, m in MODELOS.items()}


def metricas_umbral(y, p, t):
    pred = p >= t
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    esp = tn / (tn + fp) if tn + fp else np.nan
    f1 = 2 * prec * rec / (prec + rec) if (prec == prec and rec == rec and prec + rec > 0) else np.nan
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec,
            "especificidad": esp, "f1": f1, "accuracy": (tp + tn) / len(y)}


def wilson(k, n, z=1.96):
    """Intervalo de confianza del 95 % para una proporción."""
    if n == 0:
        return np.nan, np.nan
    ph = k / n
    den = 1 + z * z / n
    centro = ph + z * z / (2 * n)
    margen = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))
    return (centro - margen) / den, (centro + margen) / den


def pct(x, d=1):
    return "—" if x != x else f"{100 * x:.{d}f} %"


def estilo(fig, alto=None):
    fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=55, b=10), font=dict(size=13))
    if alto:
        fig.update_layout(height=alto)
    return fig


def fig_matriz(cm, titulo, alto=330):
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Purples", aspect="auto",
                    x=["Predicho benigno", "Predicho maligno"], y=["Real benigno", "Real maligno"], title=titulo)
    fig.update_coloraxes(showscale=False)
    fig.update_traces(textfont_size=22)
    return estilo(fig, alto)


def bandas(u):
    """Límite inferior de la zona de incertidumbre para el umbral u."""
    return max(u - 20, 0)


def clasificar(p, u):
    if p >= u:
        return "alto", "Patrón compatible con malignidad", \
            "Se recomienda confirmar con estudios clínicos y la valoración de un especialista."
    if p >= bandas(u):
        return "medio", "Zona de incertidumbre", \
            "El resultado no es concluyente: se sugiere seguimiento cercano y estudios complementarios."
    return "bajo", "Patrón compatible con benignidad", \
        "Se recomienda continuar con el seguimiento médico habitual."


def medidor(p, u):
    ini = bandas(u)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p, number={"suffix": " %", "valueformat": ".1f", "font": {"color": "#1f2937"}},
        title={"text": "Probabilidad de malignidad", "font": {"color": "#374151", "size": 18}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#1f2937", "thickness": 0.25},
               "steps": [{"range": [0, ini], "color": "#bbf7d0"}, {"range": [ini, u], "color": "#fde68a"},
                         {"range": [u, 100], "color": "#fecaca"}],
               "threshold": {"line": {"color": COLOR_MALIGNO, "width": 4}, "thickness": 0.85, "value": u}}))
    return estilo(fig, 280)


def contribuciones(modelo, fila: dict, base: float):
    """Cuánto baja la probabilidad si cada grupo de variables tomara el valor de un caso benigno típico."""
    resultado = []
    for grupo, variables in GRUPOS.items():
        alterada = dict(fila)
        for f in variables:
            alterada[f] = float(MED_B[f])
        p2 = modelo.predict_proba(pd.DataFrame([alterada], columns=F))[0, 1] * 100
        resultado.append((grupo, base - p2))
    return sorted(resultado, key=lambda t: t[1])


def percentil(f, valor):
    return 100 * float((DF[f] <= valor).mean())


def informe_html(valores, p, u, modelo, P):
    clase, titulo, texto = clasificar(p, u)
    filas = "".join(f"<tr><td>{NOMBRE[f]}</td><td style='text-align:right'>{valores[f]:.{VARIABLES[f]['decimales']}f}</td></tr>"
                    for f in F)
    votos = sum(v >= u for v in P.values())
    por_modelo = "".join(f"<tr><td>{n}</td><td style='text-align:right'>{v:.1f} %</td></tr>" for n, v in P.items())
    return f"""<!DOCTYPE html><html lang="es"><head><meta charset="utf-8"><title>Informe MamaIA</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;max-width:760px;margin:30px auto;color:#1f2937}}
h1{{color:#be185d}}table{{border-collapse:collapse;width:100%;margin:8px 0 18px}}
td,th{{border-bottom:1px solid #e5e7eb;padding:6px 8px;text-align:left}}
.caja{{border:2px solid #be185d;border-radius:12px;padding:14px 18px;background:#fdf2f8}}
.aviso{{background:#fffbeb;border-left:4px solid #f59e0b;padding:10px 14px;margin-top:20px;font-size:.9rem}}</style></head>
<body><h1>🎗️ Informe de evaluación · MamaIA</h1>
<p>Generado el {datetime.now().strftime('%d/%m/%Y %H:%M')} · Modelo: <b>{modelo}</b> · Umbral de decisión: <b>{u} %</b></p>
<div class="caja"><h2 style="margin:0">{titulo}</h2>
<p style="font-size:1.4rem;margin:6px 0"><b>{p:.1f} %</b> de probabilidad de malignidad</p>
<p style="margin:0">{texto}</p></div>
<h3>Consenso de modelos</h3><p>{votos} de {len(P)} modelos clasifican la muestra como maligna.</p>
<table><tr><th>Modelo</th><th style='text-align:right'>Probabilidad de malignidad</th></tr>{por_modelo}</table>
<h3>Medidas ingresadas</h3><table><tr><th>Variable</th><th style='text-align:right'>Valor</th></tr>{filas}</table>
<div class="aviso">⚕️ Resultado orientativo generado por un modelo entrenado con el dataset Breast Cancer Wisconsin
(569 casos). No es un diagnóstico: solo un profesional de la salud puede diagnosticar.</div></body></html>"""


# ---------------------------------------------------------------------------
# Estado actual de la muestra (se usa en varias pestañas)
# ---------------------------------------------------------------------------
ACTUAL = {f: float(st.session_state[f"v_{f}"]) for f in F}
X_ACT = pd.DataFrame([ACTUAL], columns=F)

# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎗️ MamaIA")
    st.caption("Apoyo al tamizaje de cáncer de mama")
    modelo_sel = st.selectbox(
        "Modelo de predicción", NM, index=NM.index(G), key="modelo_sel",
        format_func=lambda n: f" {n} (ganador)" if n == G else n,
        help="Todos los modelos se entrenaron con los mismos datos. El ganador se eligió por validación cruzada.")
    U = st.slider(
        "Umbral de decisión (%)", 10, 90, 50, 5, key="umbral",
        help="Probabilidad a partir de la cual una muestra se considera «maligna». "
             "Bajarlo detecta más casos malignos (mayor recall) pero produce más falsas alarmas.")
    if U != 50:
        st.caption(f"⚙️ Umbral modificado ({U} %). El estándar es 50 %. Revisa la pestaña «Umbral de decisión».")
    t_sel = M[modelo_sel]["test"]
    c1, c2 = st.columns(2)
    c1.metric("Recall", pct(t_sel["recall"], 0), help="De cada 100 muestras malignas, cuántas detecta (conjunto de prueba, umbral 50 %).")
    c2.metric("AUC", f"{t_sel['auc']:.3f}", help="Capacidad de separar benignas de malignas (1.0 = perfecta).")
    st.markdown('<div class="aviso">⚕️ <b>Herramienta educativa.</b> No diagnostica ni reemplaza la opinión '
                'de un profesional de la salud.</div>', unsafe_allow_html=True)
    st.caption(f"Modelos entrenados el {META['version']['fecha']} · scikit-learn {META['version']['sklearn']}")

P_ACT = {n: float(v[0]) for n, v in probabilidades(X_ACT).items()}
p_act = P_ACT[modelo_sel]

# ---------------------------------------------------------------------------
# Encabezado
# ---------------------------------------------------------------------------
t_gan = M[G]["test"]
st.markdown("".join([
    '<div class="hero"><div class="hero-top"><span class="logo">🎗️</span><div><h1>MamaIA</h1>',
    '<p>Apoyo al tamizaje de cáncer de mama: clasifica masas mamarias como benignas o malignas con aprendizaje automático</p>',
    '</div></div><div class="chips">',
    f'<span class="chip"><b>{META["n"]}</b> casos reales de entrenamiento y prueba</span>',
    f'<span class="chip"><b>{len(NM)}</b> modelos comparados</span>',
    f'<span class="chip">Ganador: <b>{G}</b></span>',
    f'<span class="chip">Recall en prueba: <b>{pct(t_gan["recall"], 1)}</b></span>',
    f'<span class="chip">AUC: <b>{t_gan["auc"]:.3f}</b></span>',
    '</div></div>']), unsafe_allow_html=True)

t1, t2, t3, t4, t5, t6 = st.tabs([
    "🔬 Evaluar muestra", "📂 Análisis por lote", "📊 Exploración de datos",
    "🤖 Modelos y métricas", "🎚️ Umbral de decisión", "📘 Metodología y ética"])

# ===========================================================================
# 1. EVALUAR MUESTRA
# ===========================================================================
with t1:
    st.markdown("#### Evalúa una muestra")
    st.caption("Ingresa las medidas del núcleo celular obtenidas de la biopsia por aspiración con aguja fina. "
               "El resultado se actualiza al instante. ¿Quieres verlo en acción? Carga un caso real de ejemplo.")

    colb = st.columns([1, 1, 1, 0.7])
    for i, ej in enumerate(META["ejemplos"]):
        colb[i].button(ej["titulo"], on_click=cargar_ejemplo, args=(i,), key=f"ej{i}", **ancho(st.button))
    colb[3].button("↺ Restablecer", on_click=reiniciar, key="reset", **ancho(st.button))

    ej_i = st.session_state.get("ejemplo")
    if ej_i is not None:
        ej = META["ejemplos"][ej_i]
        if all(abs(ACTUAL[f] - ej["valores"][f]) <= 10 ** -VARIABLES[f]["decimales"] for f in F):
            st.info(f"Cargaste un caso real del conjunto de prueba (el modelo no lo vio al entrenar). "
                    f"Su diagnóstico real fue: **{ej['real']}**.")

    izq, der = st.columns([1.1, 1], gap="large")

    with izq:
        for titulo, variables in GRUPOS_UI:
            with st.container(border=True):
                st.markdown(f"**{titulo}**")
                cols = st.columns(2)
                for j, f in enumerate(variables):
                    lo, hi, _ = META["rangos"][f]
                    d = VARIABLES[f]["decimales"]
                    cols[j % 2].slider(NOMBRE[f], float(lo), float(hi), key=f"v_{f}", step=10.0 ** -d,
                                       format=f"%.{d}f", help=VARIABLES[f]["ayuda"])
                if titulo.startswith("📏"):
                    r_p = ACTUAL["mean perimeter"] / (2 * np.pi * ACTUAL["mean radius"])
                    r_a = ACTUAL["mean area"] / (np.pi * ACTUAL["mean radius"] ** 2)
                    if not (LIM_PER[0] <= r_p <= LIM_PER[1] and LIM_ARE[0] <= r_a <= LIM_ARE[1]):
                        st.warning("Radio, perímetro y área están muy relacionados en un núcleo real, y estos valores "
                                   "no encajan entre sí. El resultado podría no ser confiable.")
                        st.button("Ajustar perímetro y área al radio", on_click=ajustar_tamano, key="ajustar")

    with der:
        clase, titulo_v, texto_v = clasificar(p_act, U)
        st.plotly_chart(medidor(p_act, U), key="medidor", **ancho(st.plotly_chart))
        st.markdown(f'<div class="veredicto {clase}"><h3>{titulo_v}</h3><p>{texto_v}</p></div>', unsafe_allow_html=True)
        votos = sum(v >= U for v in P_ACT.values())
        k1, k2 = st.columns(2)
        k1.metric("Consenso de modelos", f"{votos} de {len(P_ACT)}", help="Cuántos de los modelos clasifican la muestra como maligna con el umbral actual.")
        if len(P_ACT) > 0:
            dispersion = max(P_ACT.values()) - min(P_ACT.values())
            k2.metric("Acuerdo entre modelos", "Alto" if dispersion < 15 else ("Medio" if dispersion < 35 else "Bajo"),
                      help="Qué tan parecidas son las probabilidades que dan los distintos modelos.")
        if 0 < votos < len(P_ACT):
            st.caption("⚠️ Los modelos no coinciden entre sí: interpreta este resultado con más cautela.")
        st.markdown(f'<span class="pequeno">Modelo en uso: <b>{modelo_sel}</b> · Umbral: <b>{U} %</b></span>',
                    unsafe_allow_html=True)
        st.download_button("📄 Descargar informe (HTML, imprimible a PDF)", key="informe",
                           data=informe_html(ACTUAL, p_act, U, modelo_sel, P_ACT).encode("utf-8"),
                           file_name="informe_mamaia.html", mime="text/html", **ancho(st.download_button))

    st.divider()
    st.markdown("#### ¿Por qué este resultado?")
    e1, e2 = st.columns(2, gap="large")

    with e1:
        contrib = contribuciones(MODELOS[modelo_sel], ACTUAL, p_act)
        fig = go.Figure(go.Bar(
            x=[c[1] for c in contrib], y=[c[0] for c in contrib], orientation="h",
            marker_color=[COLOR_MALIGNO if c[1] > 0 else COLOR_BENIGNO for c in contrib],
            hovertemplate="%{y}: %{x:+.1f} puntos<extra></extra>"))
        fig.update_layout(title="Qué empuja el resultado hacia «maligno»", xaxis_title="Puntos porcentuales de probabilidad")
        fig.add_vline(x=0, line_color="#6b7280")
        st.plotly_chart(estilo(fig, 360), key="contrib", **ancho(st.plotly_chart))
        st.caption("Compara tu muestra contra un caso benigno típico: cuánto bajaría la probabilidad si ese grupo de "
                   "medidas fuera normal. Barras rojas = esa característica sube el riesgo; azules = lo baja. "
                   "Los grupos se afectan entre sí, así que los efectos no suman exactamente.")

    with e2:
        categorias = [NOMBRE[f] for f in F]
        radar = go.Figure()
        for nombre, valores, color, relleno in [
                ("Benigno típico", MED_B, COLOR_BENIGNO, "rgba(14,165,233,0.12)"),
                ("Maligno típico", MED_M, COLOR_MALIGNO, "rgba(190,24,93,0.12)"),
                ("Tu muestra", ACTUAL, "#f59e0b", "rgba(245,158,11,0.30)")]:
            r = [percentil(f, valores[f]) for f in F]
            radar.add_trace(go.Scatterpolar(r=r + [r[0]], theta=categorias + [categorias[0]], name=nombre,
                                            fill="toself", fillcolor=relleno,
                                            line=dict(color=color, width=3 if nombre == "Tu muestra" else 2)))
        radar.update_layout(title="Perfil de la muestra", polar=dict(radialaxis=dict(range=[0, 100], ticksuffix="")),
                            legend=dict(orientation="h", y=-0.12))
        st.plotly_chart(estilo(radar, 360), key="radar", **ancho(st.plotly_chart))
        st.caption("Cada eje muestra en qué percentil queda la medida respecto a los 569 casos reales "
                   "(0 = de las más bajas, 100 = de las más altas).")

    s1, s2 = st.columns(2, gap="large")
    with s1:
        orden = sorted(P_ACT.items(), key=lambda t: t[1])
        fig = go.Figure(go.Bar(x=[v for _, v in orden], y=[n for n, _ in orden], orientation="h",
                               marker_color=[COLOR_MALIGNO if v >= U else COLOR_BENIGNO for _, v in orden],
                               text=[f"{v:.1f} %" for _, v in orden], textposition="outside",
                               hovertemplate="%{y}: %{x:.1f} %<extra></extra>"))
        fig.add_vline(x=U, line_dash="dash", line_color="#111827", annotation_text=f"Umbral {U} %")
        fig.update_layout(title="¿Qué dice cada modelo?", xaxis=dict(range=[0, 112], title="Probabilidad de malignidad (%)"))
        st.plotly_chart(estilo(fig, 360), key="consenso", **ancho(st.plotly_chart))
    with s2:
        z_muestra = (X_ACT.iloc[0] - MEDIA) / DESV
        distancias = np.sqrt((((DF[F] - MEDIA) / DESV - z_muestra) ** 2).sum(axis=1))
        vecinos = DF.loc[distancias.nsmallest(7).index, "maligno"].tolist()
        st.markdown("##### Casos reales más parecidos")
        st.metric("De los 7 casos reales más parecidos", f"{sum(vecinos)} fueron malignos",
                  help="Se comparan las diez medidas estandarizadas contra los 569 casos del dataset.")
        st.markdown(f"<div style='font-size:2rem;letter-spacing:6px'>{''.join('🔴' if v else '🔵' for v in vecinos)}</div>",
                    unsafe_allow_html=True)
        st.caption("🔵 benigno · 🔴 maligno · ordenados del más parecido al menos parecido. "
                   "Es una segunda opinión sencilla y fácil de entender, independiente de los modelos.")
        st.markdown('<div class="aviso">⚕️ Resultado orientativo. Solo un profesional de la salud puede diagnosticar.</div>',
                    unsafe_allow_html=True)

    with st.expander("ℹ️ Cómo leer esta pantalla"):
        st.markdown("""
- **Medidor:** probabilidad de que la masa sea maligna según el modelo elegido. La línea rosa es el umbral de decisión.
- **Consenso:** si varios algoritmos distintos coinciden, el resultado es más sólido.
- **Qué empuja el resultado:** muestra qué características de la muestra elevan o reducen el riesgo.
- **Casos parecidos:** compara tu muestra con casos reales ya diagnosticados.
- **Importante:** el modelo solo ve diez medidas de una biopsia. No ve imágenes, historia clínica ni antecedentes.
""")

# ===========================================================================
# 2. ANÁLISIS POR LOTE
# ===========================================================================
with t2:
    st.markdown("#### Analiza muchas muestras a la vez")
    st.caption("Sube un archivo CSV con una fila por muestra. Reconoce los nombres de columna en español "
               "(Radio, Textura, Perímetro…) o en inglés (radius_mean, mean radius…).")

    plantilla = pd.DataFrame([{NOMBRE[f]: e["valores"][f] for f in F} for e in META["ejemplos"]])
    cb1, cb2 = st.columns([1, 2])
    cb1.download_button("⬇️ Descargar plantilla CSV", key="plantilla",
                        data=plantilla.to_csv(index=False).encode("utf-8-sig"),
                        file_name="plantilla_mamaia.csv", mime="text/csv", **ancho(st.download_button))
    usar_demo = cb2.toggle(f"Usar las {len(Y_TEST)} muestras de prueba del dataset (demostración con diagnóstico real)",
                           key="demo_lote")
    archivo = st.file_uploader("Archivo CSV", type=["csv"], key="archivo_lote")

    datos_lote, etiquetas = None, None
    if archivo is not None:
        try:
            contenido = archivo.getvalue()
            try:
                texto = contenido.decode("utf-8-sig")
            except UnicodeDecodeError:
                texto = contenido.decode("latin-1")
            crudo = pd.read_csv(io.StringIO(texto), sep=None, engine="python")
            mapa = mapear_columnas(crudo.columns)
            faltan = [NOMBRE[f] for f in F if f not in mapa.values()]
            if faltan:
                st.error(f"Faltan estas columnas en el archivo: **{', '.join(faltan)}**. "
                         "Descarga la plantilla para ver el formato esperado.")
            elif len(crudo) > 20000:
                st.error("El archivo tiene más de 20,000 filas. Divídelo en partes más pequeñas.")
            else:
                datos_lote = crudo.rename(columns=mapa)[F].apply(
                    lambda s: pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce"))
                col_real = [c for c in crudo.columns if limpiar(c) in ALIAS_ETIQUETA]
                if col_real:
                    etiquetas = etiqueta_binaria(crudo[col_real[0]])
        except Exception as error:  # noqa: BLE001
            st.error(f"No se pudo leer el archivo: {error}")
    elif usar_demo:
        datos_lote = DF.loc[META["idx_test"], F].reset_index(drop=True)
        etiquetas = pd.Series(Y_TEST)

    if datos_lote is not None:
        validas = datos_lote.notna().all(axis=1)
        if (~validas).any():
            st.warning(f"Se omitieron {int((~validas).sum())} filas con valores vacíos o no numéricos.")
        datos_ok = datos_lote[validas].reset_index(drop=True)
        etiquetas_ok = etiquetas[validas].reset_index(drop=True) if etiquetas is not None else None

        if datos_ok.empty:
            st.error("No quedaron filas válidas para analizar.")
        else:
            P_LOTE = probabilidades(datos_ok)
            p = P_LOTE[modelo_sel]
            votos_lote = sum((v >= U).astype(int) for v in P_LOTE.values())
            fuera = np.zeros(len(datos_ok), dtype=bool)
            for f in F:
                lo, hi, _ = META["rangos"][f]
                fuera |= ((datos_ok[f] < lo) | (datos_ok[f] > hi)).to_numpy()

            res = pd.DataFrame({
                "N.º": np.arange(1, len(datos_ok) + 1),
                "Probabilidad de malignidad": np.round(p, 1),
                "Clasificación": np.where(p >= U, "🔴 Maligna", np.where(p >= bandas(U), "🟡 Incierta", "🔵 Benigna")),
                "Modelos que dicen «maligna»": [f"{int(v)} de {len(P_LOTE)}" for v in votos_lote],
                "Alerta": np.where(fuera, "Fuera del rango entrenado", ""),
            })
            if etiquetas_ok is not None:
                res["Diagnóstico real"] = np.where(etiquetas_ok == 1, "Maligno", "Benigno")
            for f in F:
                res[NOMBRE[f]] = datos_ok[f].to_numpy()

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Muestras analizadas", len(res))
            m2.metric("Clasificadas malignas", int((p >= U).sum()))
            m3.metric("Clasificadas benignas", int((p < U).sum()))
            m4.metric("Probabilidad promedio", f"{p.mean():.1f} %")
            if fuera.any():
                st.warning(f"{int(fuera.sum())} filas tienen valores fuera del rango visto al entrenar; "
                           "para ellas la predicción es una extrapolación poco confiable.")

            ordenar = st.checkbox("Ordenar de mayor a menor probabilidad", value=True, key="orden_lote")
            mostrar = res.sort_values("Probabilidad de malignidad", ascending=False) if ordenar else res
            st.dataframe(mostrar, hide_index=True, key="tabla_lote", **ancho(st.dataframe),
                         column_config={"Probabilidad de malignidad": st.column_config.ProgressColumn(
                             "Probabilidad de malignidad", min_value=0, max_value=100, format="%.1f %%")})
            st.download_button("⬇️ Descargar resultados (CSV)", key="descarga_lote",
                               data=res.to_csv(index=False).encode("utf-8-sig"),
                               file_name="resultados_mamaia.csv", mime="text/csv")

            g1, g2 = st.columns(2, gap="large")
            with g1:
                fig = px.histogram(x=p, nbins=20, title="Distribución de probabilidades",
                                   labels={"x": "Probabilidad de malignidad (%)", "y": "Muestras"},
                                   color_discrete_sequence=[COLOR_MALIGNO])
                fig.add_vline(x=U, line_dash="dash", line_color="#111827", annotation_text=f"Umbral {U} %")
                fig.update_layout(yaxis_title="Muestras", bargap=0.05)
                st.plotly_chart(estilo(fig, 330), key="hist_lote", **ancho(st.plotly_chart))
            if etiquetas_ok is not None:
                with g2:
                    mm = metricas_umbral(etiquetas_ok.to_numpy(), p, U)
                    st.plotly_chart(fig_matriz([[mm["tn"], mm["fp"]], [mm["fn"], mm["tp"]]],
                                               "Resultado frente al diagnóstico real"), key="cm_lote", **ancho(st.plotly_chart))
                a1, a2, a3 = st.columns(3)
                a1.metric("Exactitud", pct(mm["accuracy"]))
                a2.metric("Recall (malignas detectadas)", pct(mm["recall"]))
                a3.metric("Precisión", pct(mm["precision"]))
                if usar_demo and archivo is None:
                    st.caption("Estas muestras no se usaron para entrenar los modelos.")
    else:
        st.info("Sube un archivo o activa la demostración para ver los resultados.")

# ===========================================================================
# 3. EXPLORACIÓN DE DATOS
# ===========================================================================
@st.cache_resource
def ajustar_pca():
    pca = PCA(n_components=2, random_state=42).fit((DF[F] - MEDIA) / DESV)
    return pca, pca.transform((DF[F] - MEDIA) / DESV)


@st.cache_data
def auc_univariable():
    return pd.Series({f: max(roc_auc_score(DF["maligno"], DF[f]), 1 - roc_auc_score(DF["maligno"], DF[f])) for f in F})


with t3:
    n_mal = int(DF["maligno"].sum())
    st.markdown("#### Conoce los datos")
    with st.expander("📋 Ficha técnica del dataset", expanded=False):
        st.markdown(f"""
| Campo | Detalle |
|---|---|
| **Nombre** | Breast Cancer Wisconsin (Diagnostic) |
| **Fuente** | UCI Machine Learning Repository · https://archive.ics.uci.edu/dataset/17 |
| **Origen** | Hospitales de la Universidad de Wisconsin (Madison, EE. UU.), muestras de aspiración con aguja fina de masas mamarias |
| **Tamaño** | {META['n']} filas × {len(F)} variables + 1 etiqueta |
| **Variables** | Diez medidas promedio de los núcleos celulares, calculadas a partir de imágenes digitalizadas de la muestra |
| **Etiqueta** | `maligno`: 1 = maligno ({n_mal}), 0 = benigno ({META['n'] - n_mal}) |
| **Tarea** | Clasificación binaria supervisada |
| **Licencia** | CC BY 4.0 |
""")
    k = st.columns(6)
    k[0].metric("Registros", META["n"])
    k[1].metric("Variables", len(F))
    k[2].metric("Benignos", f"{META['n'] - n_mal} ({100 * (META['n'] - n_mal) / META['n']:.0f} %)")
    k[3].metric("Malignos", f"{n_mal} ({100 * n_mal / META['n']:.0f} %)")
    k[4].metric("Valores nulos", int(DF.isnull().sum().sum()))
    k[5].metric("Duplicados", int(DF.duplicated().sum()))

    DFc = DF.assign(Clase=DF["maligno"].map({0: "Benigno", 1: "Maligno"}))
    colores = {"Benigno": COLOR_BENIGNO, "Maligno": COLOR_MALIGNO}

    st.markdown("##### Distribución de cada variable según la clase")
    v = st.selectbox("Variable", F, format_func=lambda f: NOMBRE[f], key="eda_var")
    h1, h2 = st.columns(2, gap="large")
    with h1:
        fig = px.histogram(DFc, x=v, color="Clase", barmode="overlay", opacity=0.7, nbins=30,
                           color_discrete_map=colores, labels={v: NOMBRE[v]}, title=f"Histograma · {NOMBRE[v]}")
        fig.add_vline(x=ACTUAL[v], line_dash="dash", line_color="#f59e0b", annotation_text="Tu muestra")
        fig.update_layout(yaxis_title="Casos")
        st.plotly_chart(estilo(fig, 340), key="eda_hist", **ancho(st.plotly_chart))
    with h2:
        fig = px.box(DFc, x="Clase", y=v, color="Clase", color_discrete_map=colores, points="outliers",
                     labels={v: NOMBRE[v]}, title=f"Diagrama de caja · {NOMBRE[v]}")
        fig.update_layout(showlegend=False)
        st.plotly_chart(estilo(fig, 340), key="eda_box", **ancho(st.plotly_chart))

    c1, c2 = st.columns(2, gap="large")
    with c1:
        corr = DF[F].corr().round(2)
        corr.index = corr.columns = [NOMBRE[f] for f in F]
        fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        title="Mapa de correlación")
        fig.update_traces(textfont_size=10)
        st.plotly_chart(estilo(fig, 520), key="eda_corr", **ancho(st.plotly_chart))
        altas = [(a, b, corr.loc[a, b]) for i, a in enumerate(corr.columns) for b in corr.columns[i + 1:]
                 if abs(corr.loc[a, b]) >= 0.9]
        if altas:
            st.caption("Pares casi redundantes (|r| ≥ 0.90): " + "; ".join(f"{a}–{b} ({r:.2f})" for a, b, r in altas)
                       + ". Por eso se agrupan las variables al explicar cada predicción.")
    with c2:
        sep = auc_univariable().sort_values()
        fig = go.Figure(go.Bar(x=sep.values, y=[NOMBRE[f] for f in sep.index], orientation="h",
                               marker_color=COLOR_MALIGNO, text=[f"{x:.2f}" for x in sep.values], textposition="outside"))
        fig.update_layout(title="¿Cuánto separa cada variable por sí sola?", xaxis=dict(range=[0.5, 1.03], title="AUC de la variable sola"))
        st.plotly_chart(estilo(fig, 330), key="eda_auc", **ancho(st.plotly_chart))
        bal = DFc["Clase"].value_counts().reset_index()
        bal.columns = ["Clase", "n"]
        fig = px.pie(bal, names="Clase", values="n", hole=0.55, color="Clase", color_discrete_map=colores,
                     title="Balance de clases")
        st.plotly_chart(estilo(fig, 280), key="eda_bal", **ancho(st.plotly_chart))
        st.caption(f"Proporción {100 * (1 - n_mal / META['n']):.0f}/{100 * n_mal / META['n']:.0f}: moderadamente desbalanceado. "
                   "No se aplicó SMOTE; se evalúa con recall y F1 en lugar de solo exactitud.")

    st.markdown("##### Relación entre dos variables y mapa de todas las muestras")
    d1, d2 = st.columns(2, gap="large")
    with d1:
        sx, sy = st.columns(2)
        vx = sx.selectbox("Eje X", F, index=F.index("mean concave points"), format_func=lambda f: NOMBRE[f], key="eda_x")
        vy = sy.selectbox("Eje Y", F, index=F.index("mean area"), format_func=lambda f: NOMBRE[f], key="eda_y")
        fig = px.scatter(DFc, x=vx, y=vy, color="Clase", color_discrete_map=colores, opacity=0.65,
                         labels={vx: NOMBRE[vx], vy: NOMBRE[vy]}, title="Dispersión")
        fig.add_trace(go.Scatter(x=[ACTUAL[vx]], y=[ACTUAL[vy]], mode="markers", name="Tu muestra",
                                 marker=dict(symbol="star", size=18, color="#facc15", line=dict(color="black", width=1.5))))
        st.plotly_chart(estilo(fig, 400), key="eda_disp", **ancho(st.plotly_chart))
    with d2:
        pca, coords = ajustar_pca()
        ev = pca.explained_variance_ratio_
        fig = px.scatter(x=coords[:, 0], y=coords[:, 1], color=DFc["Clase"], color_discrete_map=colores, opacity=0.65,
                         labels={"x": f"Componente 1 ({100 * ev[0]:.0f} %)", "y": f"Componente 2 ({100 * ev[1]:.0f} %)", "color": "Clase"},
                         title="Mapa PCA de las 10 variables en 2 dimensiones")
        punto = pca.transform((X_ACT[F] - MEDIA) / DESV)
        fig.add_trace(go.Scatter(x=[punto[0, 0]], y=[punto[0, 1]], mode="markers", name="Tu muestra",
                                 marker=dict(symbol="star", size=18, color="#facc15", line=dict(color="black", width=1.5))))
        st.plotly_chart(estilo(fig, 440), key="eda_pca", **ancho(st.plotly_chart))
        st.caption(f"Las dos componentes resumen el {100 * ev.sum():.0f} % de la información: las clases se separan bien, "
                   "con una zona de traslape donde el modelo se equivoca más.")

    st.markdown("##### Valores atípicos, estadísticas y decisiones de preprocesamiento")
    q1, q3 = DF[F].quantile(0.25), DF[F].quantile(0.75)
    iqr = q3 - q1
    atipicos = ((DF[F] < q1 - 1.5 * iqr) | (DF[F] > q3 + 1.5 * iqr)).sum()
    resumen = pd.DataFrame({
        "Variable": [NOMBRE[f] for f in F],
        "Media benignos": [DF.loc[DF.maligno == 0, f].mean() for f in F],
        "Media malignos": [DF.loc[DF.maligno == 1, f].mean() for f in F],
        "Veces mayor en malignos": [DF.loc[DF.maligno == 1, f].mean() / DF.loc[DF.maligno == 0, f].mean() for f in F],
        "Valores atípicos (IQR)": [int(atipicos[f]) for f in F],
    }).round(3)
    st.dataframe(resumen, hide_index=True, key="eda_resumen", **ancho(st.dataframe))
    st.markdown("""
| Paso | Decisión | Justificación |
|---|---|---|
| **Valores nulos** | No hay; no se imputó nada | Verificado en los 569 registros |
| **Duplicados** | No hay; no se eliminó nada | Verificado |
| **Codificación** | No necesaria | Todas las variables son numéricas; la etiqueta ya es 0/1 |
| **Valores atípicos** | Se conservaron | Son clínicamente plausibles (tumores grandes o irregulares) y justo los más informativos |
| **Escalado** | `StandardScaler` dentro de un `Pipeline` para KNN y SVM | Esos algoritmos dependen de la escala; el escalador se ajusta solo con el entrenamiento de cada pliegue (sin fuga de datos) |
| **Balanceo** | No se aplicó | Desbalance moderado; se prioriza recall y F1 |
| **Partición** | 80 % entrenamiento / 20 % prueba, estratificada, `random_state=42` | Mantiene la proporción de clases y permite reproducir los resultados |
""")
    st.dataframe(DF[F].rename(columns=NOMBRE).describe().T.round(3), key="eda_desc", **ancho(st.dataframe))

# ===========================================================================
# 4. MODELOS Y MÉTRICAS
# ===========================================================================
EXPLICACIONES = {
    "KNN": ("Mira los K casos más parecidos y decide por mayoría. No «aprende» una regla: compara.",
            r"d(x,x_i)=\sqrt{\textstyle\sum_{j}(x_j-x_{ij})^2}"),
    "SVM": ("Busca la frontera que separa las clases con el mayor margen posible; con un kernel puede ser curva.",
            r"\max_{w,b}\ \dfrac{2}{\lVert w\rVert}\quad\text{sujeto a}\quad y_i(w^\top x_i+b)\ge 1"),
    "Random Forest": ("Entrena muchos árboles con datos y variables al azar y vota. Reduce el sobreajuste de un solo árbol.",
                      r"\hat y=\operatorname{moda}\{T_1(x),\dots,T_B(x)\}"),
}

with t4:
    st.markdown("#### Comparación de algoritmos")
    cv_g = M[G]["cv"]["f1"]
    ordenados = sorted(NM, key=lambda n: -M[n]["cv"]["f1"]["media"])
    segundo = M[ordenados[1]]["cv"]["f1"]["media"] if ordenados[0] == G else M[ordenados[0]]["cv"]["f1"]["media"]
    tg = M[G]["test"]
    fn_g, tp_g = tg["cm"][1]
    lo_r, hi_r = wilson(tp_g, tp_g + fn_g)
    st.success(
        f"**Modelo seleccionado: {G}.** Tuvo el mayor F1 medio en validación cruzada repetida "
        f"({cv_g['media']:.3f} ± {cv_g['desv']:.3f}). En el conjunto de prueba alcanzó exactitud {pct(tg['accuracy'])}, "
        f"recall {pct(tg['recall'])} y AUC {tg['auc']:.3f}, con {fn_g} falsos negativos de {tp_g + fn_g} casos malignos.")
    if cv_g["media"] - segundo < cv_g["desv"]:
        st.info("**Lectura honesta:** la ventaja del ganador sobre el siguiente modelo es menor que la variabilidad de la "
                "validación cruzada, así que varios algoritmos rinden prácticamente igual. Se eligió por el criterio definido "
                "de antemano (F1 en validación cruzada), no por el resultado en prueba.")
    st.caption(f"El conjunto de prueba solo tiene {META['n_test']} muestras ({tp_g + fn_g} malignas): el recall de {pct(tg['recall'])} "
               f"tiene un intervalo de confianza del 95 % de {pct(lo_r, 0)} a {pct(hi_r, 0)}.")

    filas = []
    for n in NM:
        r = M[n]
        filas.append({
            "Modelo": ("⭐ " if n == G else "") + n,
            "Hiperparámetros óptimos": ", ".join(f"{k}={v}" for k, v in r["params"].items()),
            "F1 CV": r["cv"]["f1"]["media"], "± CV": r["cv"]["f1"]["desv"],
            "Exactitud": r["test"]["accuracy"], "Precisión": r["test"]["precision"], "Recall": r["test"]["recall"],
            "Especificidad": r["test"]["especificidad"], "F1": r["test"]["f1"], "AUC": r["test"]["auc"],
            "Falsos neg.": r["test"]["cm"][1][0], "Falsos pos.": r["test"]["cm"][0][1]})
    tabla = pd.DataFrame(filas)
    estilo_tabla = (tabla.style.highlight_max(subset=["F1 CV", "Exactitud", "Precisión", "Recall", "Especificidad", "F1", "AUC"], color="#fbcfe8")
                    .highlight_min(subset=["Falsos neg.", "Falsos pos."], color="#fbcfe8")
                    .format({c: "{:.3f}" for c in ["F1 CV", "± CV", "Exactitud", "Precisión", "Recall", "Especificidad", "F1", "AUC"]}))
    st.dataframe(estilo_tabla, hide_index=True, key="tabla_modelos", **ancho(st.dataframe))
    st.caption("Métricas de prueba con umbral 50 %. «F1 CV» es el promedio de una validación cruzada de 5 pliegues repetida 3 veces. "
               "En rosa, el mejor valor de cada columna. En salud, un falso negativo (tumor maligno no detectado) es más grave que un falso positivo.")

    m1, m2 = st.columns(2, gap="large")
    with m1:
        largo = pd.DataFrame([{"Modelo": n, "Métrica": mt, "Valor": M[n]["test"][k]}
                              for n in NM for mt, k in [("Exactitud", "accuracy"), ("Precisión", "precision"),
                                                        ("Recall", "recall"), ("F1", "f1"), ("AUC", "auc")]])
        fig = px.bar(largo, x="Modelo", y="Valor", color="Métrica", barmode="group", title="Métricas en el conjunto de prueba",
                     color_discrete_sequence=["#6d28d9", "#a78bfa", COLOR_MALIGNO, "#f472b6", "#0ea5e9"])
        fig.update_yaxes(range=[0.75, 1.0])
        fig.update_xaxes(title=None)
        fig.update_layout(legend=dict(orientation="h", y=-0.45, title=None))
        st.plotly_chart(estilo(fig, 420), key="m_metricas", **ancho(st.plotly_chart))
    with m2:
        fig = go.Figure()
        for n in NM:
            fig.add_scatter(x=M[n]["roc"]["fpr"], y=M[n]["roc"]["tpr"], name=f"{n} (AUC {M[n]['test']['auc']:.3f})",
                            line=dict(width=4 if n == G else 2))
        fig.add_scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="gray"), name="Azar")
        fig.update_layout(title="Curvas ROC", xaxis_title="Tasa de falsos positivos", yaxis_title="Tasa de verdaderos positivos",
                          legend=dict(orientation="h", y=-0.3))
        st.plotly_chart(estilo(fig, 400), key="m_roc", **ancho(st.plotly_chart))

    n1, n2 = st.columns(2, gap="large")
    with n1:
        fig = go.Figure()
        fig.add_bar(x=NM, y=[M[n]["base"]["f1"] for n in NM], name="Parámetros por defecto", marker_color="#c4b5fd")
        fig.add_bar(x=NM, y=[M[n]["test"]["f1"] for n in NM], name="Optimizado (GridSearchCV)", marker_color=COLOR_MALIGNO)
        fig.update_layout(barmode="group", title="Efecto de optimizar hiperparámetros (F1 en prueba)", legend=dict(orientation="h", y=-0.3))
        fig.update_yaxes(range=[0.75, 1.0])
        st.plotly_chart(estilo(fig, 380), key="m_opt", **ancho(st.plotly_chart))
    with n2:
        fig = go.Figure()
        fig.add_bar(x=NM, y=[M[n]["acc_train"] for n in NM], name="Entrenamiento", marker_color="#c4b5fd")
        fig.add_bar(x=NM, y=[M[n]["test"]["accuracy"] for n in NM], name="Prueba", marker_color=COLOR_MALIGNO)
        fig.update_layout(barmode="group", title="Sobreajuste: exactitud en entrenamiento vs. prueba", legend=dict(orientation="h", y=-0.3))
        fig.update_yaxes(range=[0.85, 1.0])
        st.plotly_chart(estilo(fig, 380), key="m_over", **ancho(st.plotly_chart))
    brechas = {n: M[n]["acc_train"] - M[n]["test"]["accuracy"] for n in NM}
    peor = max(brechas, key=brechas.get)
    st.caption(f"Una brecha grande entre entrenamiento y prueba indica sobreajuste. La mayor es la de {peor} "
               f"({100 * brechas[peor]:.1f} puntos); la del ganador es {"prácticamente nula" if abs(brechas[G]) < 0.005 else f"{100 * brechas[G]:.1f} puntos"}.")

    o1, o2 = st.columns(2, gap="large")
    with o1:
        fig = go.Figure(go.Scatter(
            x=[M[n]["cv"]["f1"]["media"] for n in ordenados[::-1]], y=ordenados[::-1], mode="markers",
            marker=dict(size=11, color=COLOR_MALIGNO),
            error_x=dict(type="data", array=[M[n]["cv"]["f1"]["desv"] for n in ordenados[::-1]], color="#9ca3af")))
        fig.update_layout(title="F1 en validación cruzada (media ± desviación)", xaxis_title="F1")
        st.plotly_chart(estilo(fig, 360), key="m_cv", **ancho(st.plotly_chart))
        st.caption("Si las barras se traslapan, la diferencia entre modelos no es estadísticamente clara.")
    with o2:
        sel = st.selectbox("Matriz de confusión de", NM, index=NM.index(G), key="cm_modelo")
        st.plotly_chart(fig_matriz(M[sel]["test"]["cm"], f"Matriz de confusión · {sel}", 330), key="m_cm", **ancho(st.plotly_chart))

    i1, i2 = st.columns(2, gap="large")
    with i1:
        imp = pd.DataFrame({"Variable": [NOMBRE[f] for f in F], "Importancia": [META["importancia"][f]["media"] for f in F],
                            "Desv": [META["importancia"][f]["desv"] for f in F]}).sort_values("Importancia")
        fig = go.Figure(go.Bar(x=imp["Importancia"], y=imp["Variable"], orientation="h", marker_color=COLOR_MALIGNO,
                               error_x=dict(type="data", array=imp["Desv"], color="#9ca3af")))
        fig.update_layout(title=f"¿Qué variables usa {G}? (importancia por permutación)", xaxis_title="Caída del AUC al desordenar la variable")
        st.plotly_chart(estilo(fig, 380), key="m_imp", **ancho(st.plotly_chart))
        st.caption("Las variables muy correlacionadas (radio, perímetro, área) se reparten la importancia, por lo que cada una parece menor.")
    with i2:
        c = META["curva_aprendizaje"]
        fig = go.Figure()
        fig.add_scatter(x=c["n"], y=c["train"], name="Entrenamiento", line=dict(color="#a78bfa", width=3),
                        error_y=dict(type="data", array=c["train_desv"], color="#ddd6fe"))
        fig.add_scatter(x=c["n"], y=c["val"], name="Validación", line=dict(color=COLOR_MALIGNO, width=3),
                        error_y=dict(type="data", array=c["val_desv"], color="#fbcfe8"))
        fig.update_layout(title=f"Curva de aprendizaje de {G}", xaxis_title="Muestras de entrenamiento", yaxis_title="F1",
                          legend=dict(orientation="h", y=-0.3))
        st.plotly_chart(estilo(fig, 380), key="m_curva", **ancho(st.plotly_chart))
        st.caption("Si la validación sigue subiendo al añadir datos, más casos reales mejorarían el modelo.")

    st.markdown("##### Cómo funciona cada algoritmo")
    try:
        from entrenar import espacios_de_busqueda
        mallas = {n: v[1] for n, v in espacios_de_busqueda().items()}
    except Exception:  # noqa: BLE001
        mallas = {}
    for n in NM:
        texto, formula = EXPLICACIONES[n]
        with st.expander(("⭐ " if n == G else "") + n):
            st.markdown(texto)
            st.latex(formula)
            if n in mallas:
                st.markdown("**Hiperparámetros probados:** " + "; ".join(
                    f"`{k.replace('modelo__', '')}` ∈ {{{', '.join(map(str, v))}}}" for k, v in mallas[n].items())
                    + f" → {M[n]['combinaciones']} combinaciones × 5 pliegues.")
            st.markdown(f"**Elegido:** {', '.join(f'`{k}={v}`' for k, v in M[n]['params'].items())} · "
                        f"búsqueda en {M[n]['tiempo_busqueda_s']} s.")

# ===========================================================================
# 5. UMBRAL DE DECISIÓN
# ===========================================================================
with t5:
    st.markdown("#### ¿Qué pasa si cambiamos el umbral?")
    st.caption("Un modelo entrega una probabilidad; el umbral decide a partir de cuál se considera «maligna». "
               "Cambia el umbral y el modelo en la barra lateral para ver el equilibrio entre no dejar escapar casos malignos y evitar falsas alarmas.")
    p_test = np.array(M[modelo_sel]["proba_test"]) * 100
    mu = metricas_umbral(Y_TEST, p_test, U)
    st.markdown(f"**Modelo:** {modelo_sel} · **Umbral:** {U} % · **Conjunto de prueba:** {len(Y_TEST)} muestras ({int(Y_TEST.sum())} malignas)")

    q = st.columns(5)
    q[0].metric("Recall", pct(mu["recall"]), help="Malignas detectadas / malignas reales")
    q[1].metric("Precisión", pct(mu["precision"]), help="Malignas reales / todas las que el modelo marcó como malignas")
    q[2].metric("Especificidad", pct(mu["especificidad"]), help="Benignas bien clasificadas / benignas reales")
    q[3].metric("Falsos negativos", mu["fn"], help="Tumores malignos que el modelo dejó pasar")
    q[4].metric("Falsas alarmas", mu["fp"], help="Muestras benignas marcadas como malignas")

    u1, u2 = st.columns(2, gap="large")
    with u1:
        ts = np.arange(0, 101, 1)
        curvas = [metricas_umbral(Y_TEST, p_test, t) for t in ts]
        fig = go.Figure()
        for clave, nombre, color in [("recall", "Recall", COLOR_MALIGNO), ("precision", "Precisión", "#6d28d9"),
                                     ("especificidad", "Especificidad", COLOR_BENIGNO)]:
            fig.add_scatter(x=ts, y=[c[clave] for c in curvas], name=nombre, line=dict(color=color, width=3), connectgaps=False)
        fig.add_vline(x=U, line_dash="dash", line_color="#111827", annotation_text=f"Umbral {U} %")
        fig.update_layout(title="Cómo cambian las métricas con el umbral", xaxis_title="Umbral de decisión (%)",
                          yaxis=dict(range=[0, 1.02], title=""), legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(estilo(fig, 400), key="u_curvas", **ancho(st.plotly_chart))
    with u2:
        st.plotly_chart(fig_matriz([[mu["tn"], mu["fp"]], [mu["fn"], mu["tp"]]], f"Matriz de confusión con umbral {U} %", 400),
                        key="u_cm", **ancho(st.plotly_chart))

    st.markdown(f"Con un umbral de **{U} %**, el modelo deja pasar **{mu['fn']}** tumores malignos y genera **{mu['fp']}** falsas alarmas "
                f"en estas {len(Y_TEST)} muestras.")
    tabla_u = pd.DataFrame([{"Umbral": f"{t} %", "Recall": c["recall"], "Precisión": c["precision"], "Especificidad": c["especificidad"],
                             "Falsos negativos": c["fn"], "Falsas alarmas": c["fp"]}
                            for t in (10, 20, 30, 40, 50, 60, 70, 80, 90) for c in [metricas_umbral(Y_TEST, p_test, t)]])
    st.dataframe(tabla_u.style.format({"Recall": "{:.3f}", "Precisión": "{:.3f}", "Especificidad": "{:.3f}"}, na_rep="—"),
                 hide_index=True, key="tabla_umbral", **ancho(st.dataframe))
    st.markdown('<div class="tarjeta"><b>Criterio clínico:</b> en un tamizaje conviene un umbral que priorice el recall, porque una falsa alarma '
                'se resuelve con estudios adicionales, mientras que un falso negativo retrasa el tratamiento. '
                'El costo es más estudios innecesarios. La decisión final siempre es del profesional de la salud.</div>', unsafe_allow_html=True)

# ===========================================================================
# 6. METODOLOGÍA Y ÉTICA
# ===========================================================================
with t6:
    st.markdown("#### El problema y por qué importa")
    st.markdown("""
El cáncer de mama es el cáncer más diagnosticado en el mundo: se estimaron unos 2.3 millones de casos nuevos en 2020 (Sung et al., 2021).
En México, el INEGI registró 8,451 defunciones por cáncer de mama en 2024 (cifra preliminar), con una tasa de 18.7 por cada 100 mil
mujeres de 20 años y más, y lo ubicó como la primera causa de muerte entre los tumores malignos (INEGI, 2025).
Detectar a tiempo cambia el pronóstico, y los modelos de aprendizaje automático pueden ayudar a priorizar casos y ofrecer una segunda lectura.
**MamaIA** muestra cómo se construye ese apoyo de forma transparente: clasifica una masa ya detectada como benigna o maligna a partir de
medidas de una biopsia por aspiración. **No previene el cáncer ni diagnostica.**
""")

    st.markdown("#### Flujo del proyecto")
    st.graphviz_chart("""
digraph {
  rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fillcolor="#fdf2f8", color="#be185d", fontname="Helvetica", fontsize=11];
  edge [color="#6d28d9"];
  A [label="1. Problema\\ny dataset UCI"]; B [label="2. EDA"]; C [label="3. Preprocesamiento\\nPipeline + escalado"];
  D [label="4. Partición\\n80/20 estratificada"]; E [label="5. Entrenamiento\\n3 algoritmos"]; F [label="6. GridSearchCV\\n+ validación cruzada"];
  G [label="7. Evaluación\\nen prueba"]; H [label="8. Selección\\ndel modelo"]; I [label="9. Aplicación\\nStreamlit"];
  A -> B -> C -> D -> E -> F -> G -> H -> I;
}
""")

    st.markdown("#### Ciclo de vida del proyecto")
    st.markdown(f"""
| # | Etapa | Qué se hizo en MamaIA |
|---|---|---|
| 1 | Definición del problema | Clasificación binaria: masa benigna (0) o maligna (1) |
| 2 | Recolección de datos | Breast Cancer Wisconsin (Diagnostic), UCI; archivo `{META.get('archivo_origen', 'CSV del proyecto')}`; {META['n']} registros |
| 3 | Análisis exploratorio | Histogramas, cajas, correlaciones, balance, PCA y valores atípicos (pestaña «Exploración de datos») |
| 4 | Preprocesamiento | Sin nulos ni duplicados; `StandardScaler` dentro de `Pipeline` |
| 5 | División | {META['n_train']} de entrenamiento y {META['n_test']} de prueba, estratificado, `random_state={META['version']['semilla']}` |
| 6 | Selección de modelos | {len(NM)} algoritmos: {', '.join(NM)} |
| 7 | Entrenamiento | `GridSearchCV` (5 pliegues, F1) y validación cruzada repetida 5×3 |
| 8 | Evaluación | Exactitud, precisión, recall, especificidad, F1, AUC-ROC, matriz de confusión, importancia por permutación y curva de aprendizaje |
| 9 | Despliegue | Esta aplicación en Streamlit, con predicción en tiempo real y análisis por lote |
| 10 | Monitoreo | Pendiente: en un uso real habría que vigilar el desempeño con casos nuevos |
""")

    st.markdown("#### Ética, límites y riesgos")
    st.markdown("""
- **No es un diagnóstico.** Solo un profesional de la salud, con estudios clínicos, puede diagnosticar.
- **Un solo origen de datos.** Todas las muestras vienen de un hospital de EE. UU.; el modelo podría no generalizarse a otras poblaciones, equipos o laboratorios. El dataset no incluye edad, origen ni otros datos demográficos, así que **no se puede auditar si el desempeño es equitativo entre grupos**.
- **Prevalencia distinta a la real.** En el dataset 37 % de las muestras son malignas; en un programa de tamizaje la proporción es mucho menor. Por eso la probabilidad que muestra la aplicación **no equivale al riesgo real de una persona**, y en la práctica habría más falsas alarmas de las que muestra la prueba.
- **Entrada limitada.** El modelo no ve imágenes de mamografía ni historia clínica: usa diez medidas ya calculadas de una biopsia por aspiración.
- **Muestra de prueba pequeña.** Con 114 muestras, las métricas tienen incertidumbre amplia (ver intervalo de confianza en «Modelos y métricas»).
- **Privacidad.** Los valores ingresados se procesan durante la sesión y no se guardan.
- **Riesgo de exceso de confianza.** Por eso la aplicación muestra el consenso de modelos, una zona de incertidumbre y avisos de valores fuera de rango.
""")

    st.markdown("#### Reproducibilidad")
    v = META["version"]
    st.markdown(f"""
Cualquier persona puede repetir los resultados: `pip install -r requirements.txt`, luego `python entrenar.py` y `streamlit run app.py`.
Semilla aleatoria: **{v['semilla']}** · Python {v['python']} · scikit-learn {v['sklearn']} · última ejecución: {v['fecha']}.
""")

    st.markdown("#### Referencias (APA 7)")
    st.markdown("""
Géron, A. (2022). *Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow* (3.ª ed.). O'Reilly Media.

Instituto Nacional de Estadística y Geografía. (2025). *Estadísticas a propósito del Día Internacional de la Lucha contra el Cáncer de Mama (19 de octubre)* (Comunicado de prensa núm. 138/25). https://en.www.inegi.org.mx/contenidos/saladeprensa/aproposito/2025/EAP_CancerMama_25.pdf

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.

Russell, S., & Norvig, P. (2020). *Inteligencia artificial: Un enfoque moderno* (4.ª ed.). Pearson Educación.

Street, W. N., Wolberg, W. H., & Mangasarian, O. L. (1993). Nuclear feature extraction for breast tumor diagnosis. *IS&T/SPIE International Symposium on Electronic Imaging: Science and Technology, 1905*, 861–870.

Sung, H., Ferlay, J., Siegel, R. L., Laversanne, M., Soerjomataram, I., Jemal, A., & Bray, F. (2021). Global cancer statistics 2020: GLOBOCAN estimates of incidence and mortality worldwide for 36 cancers in 185 countries. *CA: A Cancer Journal for Clinicians, 71*(3), 209–249. https://doi.org/10.3322/caac.21660

Wolberg, W. H., Street, W. N., & Mangasarian, O. L. (1995). *Breast Cancer Wisconsin (Diagnostic)* [Conjunto de datos]. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B
""")

    st.markdown("#### Declaración de uso de inteligencia artificial")
    st.markdown("""
**Uso:** se utilizaron herramientas digitales de apoyo durante el desarrollo y revisión del código.
**Verificación:** los resultados deben obtenerse ejecutando `entrenar.py`; se pueden reproducir con la semilla indicada.
""")
