
# streamlit_app.py
import math
import numpy as np
import pandas as pd
import streamlit as st

from typing import Dict, Tuple, List, Optional

# =========================
# Config padrão (editável)
# =========================
DEFAULT_CONFIG = {
    'metas': {
        'mencoes': 500,
        'engajamentos': 6000,
        'sentimento': 70.0,
        'brandfit': 7.0
    },
    'pesos_iniciais': {
        'mencoes': 0.5,
        'engajamentos': 2.0,
        'sentimento': 1.0,
        'brandfit': 0.75
    },
    'caps': {'ratio_cap': 2.0},
    'logistic_k': 4.0
}

# =========================
# Funções utilitárias
# =========================
def normalize_features(mencoes, engaj, sentimento, brandfit, CONFIG):
    metas = CONFIG['metas']; caps = CONFIG['caps']
    r_menc = min(max(0.0, mencoes / max(1.0, metas['mencoes'])), caps['ratio_cap'])
    r_eng  = min(max(0.0, engaj   / max(1.0, metas['engajamentos'])), caps['ratio_cap'])
    r_sent = min(max(0.0, sentimento / max(1e-6, metas['sentimento'])), caps['ratio_cap'])
    r_bfit = min(max(0.0, brandfit / max(1e-6, metas['brandfit'])), caps['ratio_cap'])

    menc_log = math.log1p(mencoes) / math.log1p(metas['mencoes'])
    eng_log  = math.log1p(engaj)   / math.log1p(metas['engajamentos'])

    sent_smooth = math.tanh(r_sent)
    bfit_smooth = math.tanh(r_bfit)

    return np.array([r_menc, menc_log, r_eng, eng_log, r_sent, sent_smooth, r_bfit, bfit_smooth], dtype=float)

def compute_composite(features, CONFIG):
    pesos = CONFIG['pesos_iniciais']
    r_menc, menc_log, r_eng, eng_log, r_sent, sent_smooth, r_bfit, bfit_smooth = features

    comp_menc = 0.5*r_menc + 0.5*menc_log
    comp_eng  = 0.5*r_eng  + 0.5*eng_log
    comp_sent = 0.5*r_sent + 0.5*sent_smooth
    comp_bfit = 0.5*r_bfit + 0.5*bfit_smooth

    w = np.array([pesos['mencoes'], pesos['engajamentos'], pesos['sentimento'], pesos['brandfit']], dtype=float)
    w = w / (w.sum() if w.sum() != 0 else 1.0)

    composite = w[0]*comp_menc + w[1]*comp_eng + w[2]*comp_sent + w[3]*comp_bfit
    comps = {'mencoes': comp_menc, 'engajamentos': comp_eng, 'sentimento': comp_sent, 'brandfit': comp_bfit}
    weights = {'mencoes': w[0], 'engajamentos': w[1], 'sentimento': w[2], 'brandfit': w[3]}
    return composite, comps, weights

def composite_to_score(composite, CONFIG):
    k = CONFIG['logistic_k']
    prob_like = 1.0 / (1.0 + math.exp(-k*(composite - 1.0)))
    return float(100.0 * prob_like)

def score_to_composite_target(score_target, CONFIG):
    k = CONFIG['logistic_k']
    p = max(1e-6, min(1 - 1e-6, score_target/100.0))
    return 1.0 + (1.0/k) * math.log(p/(1.0 - p))

def classify(score):
    if score <= 40:  return 'FRIA'
    if score <= 74:  return 'MORNA'
    return 'QUENTE'

def analyze_drivers(values, features, composite, comps, weights, score, CONFIG):
    metas = CONFIG['metas']
    ratios = {'mencoes': features[0], 'engajamentos': features[2], 'sentimento': features[4], 'brandfit': features[6]}
    contribs = {k: weights[k]*comps[k] for k in comps.keys()}
    contribs_sorted = sorted(contribs.items(), key=lambda x: x[1], reverse=True)

    ups   = [k for k,_ in contribs_sorted if ratios[k] >= 1.0]
    downs = [k for k,_ in contribs_sorted if ratios[k] < 1.0]

    recs = []
    if score < 75:
        composite_target = score_to_composite_target(75.0, CONFIG)
        delta_needed = max(0.0, composite_target - composite)
        gaps = [(k, (1.0 - min(1.0, ratios[k])), weights[k]) for k in ratios if ratios[k] < 1.0]
        gaps_sorted = sorted(gaps, key=lambda x: (x[1]*x[2]), reverse=True)
        for k, _, _ in gaps_sorted[:2]:
            if k in ['mencoes', 'engajamentos']:
                recs.append(f"- Aumente **{k}** até ~{int(metas[k]):,} (alvo de meta).".replace(',', '.'))
            elif k == 'sentimento':
                recs.append(f"- Eleve **sentimento** para ≥ {metas['sentimento']:.0f} via criativos de valência positiva.")
            elif k == 'brandfit':
                recs.append(f"- Suba **brandfit** para ≥ {metas['brandfit']:.1f} alinhando mensagem e territórios de marca.")
        if delta_needed > 0:
            recs.append(f"- Ganho composto necessário ~{delta_needed:.2f} para chegar a score ≈ 75.")
    else:
        recs.append('- **Manter pilares ≥ meta** e escalar formatos/canais vencedores.')
        if ratios['brandfit'] < 1.0:
            recs.append('- **Ajustar brandfit**: refine narrativa/CTAs para aderir mais ao território da marca.')

    txt = []
    if ups:   txt.append('🔼 **Puxaram o score para cima:** ' + ', '.join([u.capitalize() for u in ups]))
    if downs: txt.append('🔽 **Seguraram o score:** ' + ', '.join([d.capitalize() for d in downs]))
    if recs:
        txt.append('🛠️ **Recomendações:**'); txt.extend(recs)
    return '\n'.join(txt) if txt else 'Sem destaques relevantes; pilares próximos da meta.'

# ===== Gauge helpers (Plotly e Matplotlib) =====
def normalize(value: float, min_v: float, max_v: float) -> float:
    if max_v == min_v:
        return 0.0
    x = (value - min_v) / (max_v - min_v)
    return max(0.0, min(1.0, x))

def weighted_temperature(signals: Dict[str, float],
                         weights: Dict[str, float],
                         bounds: Optional[Dict[str, Tuple[float, float]]] = None) -> float:
    if not signals or not weights:
        return 0.0
    total, wsum = 0.0, 0.0
    for k, w in weights.items():
        if k not in signals:
            continue
        v = signals[k]
        if bounds and k in bounds:
            v = normalize(v, bounds[k][0], bounds[k][1])
        total += w * v
        wsum  += w
    if wsum == 0:
        return 0.0
    score_0_1 = total / wsum
    return float(round(score_0_1 * 100.0, 2))

# --- Plotly (retorna a Figure) ---
def render_gauge_plotly_figure(score: float, title: str='Temperatura da Pauta',
                               bands: Optional[List[Tuple[float, float, str]]] = None):
    try:
        import plotly.graph_objects as go
    except Exception as e:
        st.error('Plotly não está instalado. Rode: pip install plotly')
        return None
    if bands is None:
        bands = [(0, 40, '#D9534F'), (40, 70, '#F0AD4E'), (70, 100, '#5CB85C')]
    gauge_steps = [dict(range=[a, b], color=c) for (a, b, c) in bands]
    fig = go.Figure(go.Indicator(
        mode='gauge+number',
        value=max(0, min(100, score)),
        title={'text': title},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': '#808080'},
            'steps': gauge_steps
        }
    ))
    fig.update_layout(margin=dict(l=30, r=30, t=40, b=30), height=300, width=300)
    return fig

# --- Matplotlib (exibe com st.pyplot) ---
def render_gauge_matplotlib(score: float, title: str='Temperatura da Pauta',
                            bands: Optional[List[Tuple[float, float, str]]]=None):
    import matplotlib.pyplot as plt
    import numpy as np
    if bands is None:
        bands = [(0, 40, '#D9534F'), (40, 70, '#F0AD4E'), (70, 100, '#5CB85C')]
    score = max(0, min(100, score))
    fig = plt.figure(figsize=(5.5, 3), dpi=150)
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9], polar=True)
    ax.set_theta_zero_location('S')
    ax.set_theta_direction(-1)
    ax.set_yticklabels([]); ax.set_xticklabels([]); ax.set_ylim(0, 1)

    def deg2rad(d): return np.deg2rad(d)
    def val2deg(v): return 180 * (v / 100.0)
    for a, b, color in bands:
        theta1 = deg2rad(val2deg(a))
        theta2 = deg2rad(val2deg(b))
        ax.bar(x=(theta1 + theta2) / 2.0, height=1.0, width=(theta2 - theta1),
               bottom=0.0, linewidth=0, color=color, alpha=0.75)
    theta = deg2rad(val2deg(score))
    ax.arrow(theta, 0.0, 0.0, 0.95, width=0.02, head_width=0.06, head_length=0.08,
             length_includes_head=True, color='#808080')
    circ = plt.Circle((0.0, 0.0), 0.08, transform=ax.transData._b, color='#FFFFFF', zorder=10)
    ax.add_artist(circ)
    ax.text(0.0, -0.18, title, ha='center', va='center', fontsize=11, transform=ax.transAxes)
    ax.text(0.5, -0.05, f'{score:.0f}', ha='center', va='center', fontsize=16, transform=ax.transAxes)
    return fig

def pct_of_goal(value, goal):
    try:
        return max(0.0, min(100.0, (float(value) / float(goal)) * 100.0))
    except Exception:
        return 0.0

# =========================
# App
# =========================
st.set_page_config(page_title="Temperatura da Pauta", page_icon="🔥", layout="wide")
st.title("🔥 Temperatura da Pauta — Score e Diagnóstico")

# ---- Sidebar: Config ----
st.sidebar.header("Configurações")
with st.sidebar.expander("Metas (alvos)", expanded=True):
    meta_menc = st.number_input("Meta de Menções", min_value=0, value=int(DEFAULT_CONFIG['metas']['mencoes']), step=50)
    meta_eng  = st.number_input("Meta de Engajamentos", min_value=0, value=int(DEFAULT_CONFIG['metas']['engajamentos']), step=250)
    meta_sent = st.number_input("Meta de Sentimento", min_value=0.0, max_value=100.0, value=float(DEFAULT_CONFIG['metas']['sentimento']), step=1.0)
    meta_bfit = st.number_input("Meta de Brandfit", min_value=0.0, max_value=10.0, value=float(DEFAULT_CONFIG['metas']['brandfit']), step=0.1)

with st.sidebar.expander("Pesos (importância relativa)", expanded=True):
    peso_menc = st.number_input("Peso - Menções",  value=float(DEFAULT_CONFIG['pesos_iniciais']['mencoes']), step=0.1, format="%.2f")
    peso_eng  = st.number_input("Peso - Engajamentos", value=float(DEFAULT_CONFIG['pesos_iniciais']['engajamentos']), step=0.1, format="%.2f")
    peso_sent = st.number_input("Peso - Sentimento", value=float(DEFAULT_CONFIG['pesos_iniciais']['sentimento']), step=0.1, format="%.2f")
    peso_bfit = st.number_input("Peso - Brandfit", value=float(DEFAULT_CONFIG['pesos_iniciais']['brandfit']), step=0.1, format="%.2f")

ratio_cap = st.sidebar.slider("Cap de razão (limite superior)", 1.0, 5.0, float(DEFAULT_CONFIG['caps']['ratio_cap']), 0.1)
log_k     = st.sidebar.slider("Inclinação logística (k)", 1.0, 8.0, float(DEFAULT_CONFIG['logistic_k']), 0.5)

CONFIG = {
    'metas': {'mencoes': meta_menc, 'engajamentos': meta_eng, 'sentimento': meta_sent, 'brandfit': meta_bfit},
    'pesos_iniciais': {'mencoes': peso_menc, 'engajamentos': peso_eng, 'sentimento': peso_sent, 'brandfit': peso_bfit},
    'caps': {'ratio_cap': ratio_cap},
    'logistic_k': log_k
}

# ---- Inputs principais (substitui input())
st.subheader("Entradas da Pauta")
col_a, col_b = st.columns(2)
with col_a:
    pauta = st.text_input("Nome da pauta", value="Minha Pauta")
    menc_ig = st.number_input("Menções no Instagram", min_value=0, value=120, step=10)
    eng_ig  = st.number_input("Engajamentos no Instagram", min_value=0, value=3000, step=100)
with col_b:
    menc_tw = st.number_input("Menções no Twitter/X", min_value=0, value=80, step=10)
    eng_tw  = st.number_input("Engajamentos no Twitter/X", min_value=0, value=1800, step=100)
    brand   = st.number_input("Brandfit (0-10)", min_value=0.0, max_value=10.0, value=7.5, step=0.1)
    sent    = st.number_input("Sentimento (0-100)", min_value=0.0, max_value=100.0, value=72.0, step=1.0)

menc_total = menc_ig + menc_tw
eng_total  = eng_ig + eng_tw

# ---- Cálculos
feats = normalize_features(menc_total, eng_total, sent, brand, CONFIG)
composite, comps, weights = compute_composite(feats, CONFIG)
score = composite_to_score(composite, CONFIG)
classe = classify(score)
values = {'mencoes': menc_total, 'engajamentos': eng_total, 'sentimento': sent, 'brandfit': brand}
analise = analyze_drivers(values, feats, composite, comps, weights, score, CONFIG)

# ---- Header com KPIs
st.markdown("---")
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Score Final", f"{score:.1f}")
kpi2.metric("Classificação", classe)
kpi3.metric("Menções (IG+X)", f"{menc_total:,}".replace(",", "."))
kpi4.metric("Engajamentos (IG+X)", f"{eng_total:,}".replace(",", "."))
kpi5.metric("Sent/Brandfit", f"{sent:.0f} / {brand:.1f}")

# ---- Análise textual
st.subheader("Análise dos Drivers")
st.markdown(analise)

# ---- Gauges
st.subheader("Gauges")
try:
    import plotly
    use_plotly = True
except Exception:
    use_plotly = False

if use_plotly:
    g_main = render_gauge_plotly_figure(score, title='Temperatura da Pauta')
    cols = st.columns(3)
    with cols[0]:
        if g_main: st.plotly_chart(g_main, use_container_width=False)
    # mini-gauges de progresso vs meta
    gauge_vals = {
        'Menções': pct_of_goal(menc_total, meta_menc),
        'Engajamentos': pct_of_goal(eng_total, meta_eng),
        'Sentimento': pct_of_goal(sent, meta_sent),
        'Brandfit': pct_of_goal(brand, meta_bfit),
    }
    c1, c2, c3, c4 = st.columns(4)
    mini_figs = []
    for nome, v in gauge_vals.items():
        mini_figs.append(render_gauge_plotly_figure(v, title=nome))
    for c, fig in zip([c1, c2, c3, c4], mini_figs):
        with c:
            if fig: st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Plotly não detectado. Mostrando gauge via Matplotlib.")
    fig = render_gauge_matplotlib(score, title='Temperatura da Pauta')
    st.pyplot(fig)

# ---- Tabela de componentes e pesos normalizados
st.subheader("Componentes & Pesos")
df = pd.DataFrame({
    'Componente': ['Menções', 'Engajamentos', 'Sentimento', 'Brandfit'],
    'Valor Composto': [comps['mencoes'], comps['engajamentos'], comps['sentimento'], comps['brandfit']],
    'Peso Normalizado': [weights['mencoes'], weights['engajamentos'], weights['sentimento'], weights['brandfit']],
})
st.dataframe(df.style.format({'Valor Composto': '{:.3f}', 'Peso Normalizado': '{:.2%}'}), use_container_width=True)

# ---- Rodapé
st.markdown("---")
st.caption("Creative Data • Temperatura da Pauta — App Streamlit")
