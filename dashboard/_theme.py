"""Design system — tokens, Plotly template, custom CSS, and HTML primitives.

Goal: make the app feel like a product (think Linear / Vercel / Tecton) rather
than a dark-themed Streamlit demo. Built on a small set of primitives:

    - `kpi_card(label, value, ..., spark=...)`: premium metric card with sparkline
    - `hero_forecast(...)`:                       the big headline KPI
    - `brand_header(...)`:                        wordmark + status pill
    - `panel_open()` / `panel_close()`:           glass container wrappers
    - `section(title, caption)`:                  consistent section heads
"""

from __future__ import annotations

from typing import Iterable

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st


# ─── Design tokens ───────────────────────────────────────────────────────────
BG             = "#07090F"   # deepest bg, behind gradient
SURFACE        = "#0E1320"   # first-elevation panel
SURFACE_ELEV   = "#141A2A"   # second elevation (hovered)
BORDER         = "#1F2636"
BORDER_SUBTLE  = "rgba(255,255,255,0.06)"
BORDER_STRONG  = "rgba(255,255,255,0.12)"
TEXT           = "#F4F6FB"
TEXT_MUTED     = "#8B93A7"
TEXT_DIM       = "#5B6476"

ACCENT         = "#8B7CFF"
ACCENT_2       = "#5B8CFF"
ACCENT_SOFT    = "rgba(139, 124, 255, 0.16)"
ACCENT_GLOW    = "rgba(139, 124, 255, 0.35)"
PREDICTED      = "#22D3B3"
PREDICTED_SOFT = "rgba(34, 211, 179, 0.18)"
ACTUAL         = "#F472B6"
ACTUAL_SOFT    = "rgba(244, 114, 182, 0.18)"
HIGHLIGHT      = "#FFD166"
POSITIVE       = "#22D3B3"
NEGATIVE       = "#F472B6"

SHAPE_ACTUAL     = "x-thin"
SHAPE_PREDICTED  = "circle"
SHAPE_SELECTION  = "diamond"

SERIES_PALETTE = [ACCENT, PREDICTED, ACTUAL, HIGHLIGHT, ACCENT_2, "#C084FC"]


# ─── Plotly template ─────────────────────────────────────────────────────────
def register_plotly_template() -> None:
    tpl = go.layout.Template()
    tpl.layout = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, family="Inter, system-ui, sans-serif", size=13),
        title=dict(font=dict(size=15, color=TEXT_MUTED), x=0, xanchor="left", y=0.98),
        colorway=SERIES_PALETTE,
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.04)",
            zerolinecolor="rgba(255,255,255,0.04)",
            linecolor="rgba(255,255,255,0.08)",
            tickfont=dict(color=TEXT_MUTED, size=11),
            title=dict(font=dict(color=TEXT_MUTED, size=12)),
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.04)",
            zerolinecolor="rgba(255,255,255,0.04)",
            linecolor="rgba(255,255,255,0.08)",
            tickfont=dict(color=TEXT_MUTED, size=11),
            title=dict(font=dict(color=TEXT_MUTED, size=12)),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(255,255,255,0.08)",
            borderwidth=0,
            font=dict(color=TEXT_MUTED, size=11),
        ),
        margin=dict(l=48, r=24, t=40, b=48),
        hoverlabel=dict(
            bgcolor=SURFACE_ELEV,
            bordercolor=BORDER_STRONG,
            font=dict(color=TEXT, family="Inter, sans-serif", size=12),
        ),
    )
    pio.templates["forecasting"] = tpl
    pio.templates.default = "forecasting"


# ─── CSS ─────────────────────────────────────────────────────────────────────
_CSS = f"""
<style>
  @import url('https://rsms.me/inter/inter.css');
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

  html, body, [class*="css"], [data-testid="stAppViewContainer"] {{
    font-family: 'Inter', ui-sans-serif, system-ui, -apple-system, sans-serif !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    font-feature-settings: 'cv11','ss01','ss03';
  }}

  /* Ambient gradient mesh — fixed, sits behind content. */
  [data-testid="stAppViewContainer"]::before {{
    content: '';
    position: fixed;
    inset: 0;
    background:
      radial-gradient(900px 560px at 80% -8%,  rgba(139, 124, 255, 0.20), transparent 60%),
      radial-gradient(800px 500px at -10% 20%, rgba(34, 211, 179, 0.08), transparent 55%),
      radial-gradient(600px 400px at 110% 90%, rgba(244, 114, 182, 0.06), transparent 60%),
      linear-gradient(180deg, {BG} 0%, #04060A 100%);
    pointer-events: none;
    z-index: 0;
  }}
  /* Faint grid overlay */
  [data-testid="stAppViewContainer"]::after {{
    content: '';
    position: fixed;
    inset: 0;
    background-image:
      linear-gradient(rgba(255,255,255,0.015) 1px, transparent 1px),
      linear-gradient(90deg, rgba(255,255,255,0.015) 1px, transparent 1px);
    background-size: 48px 48px;
    mask-image: radial-gradient(ellipse at 50% 30%, #000 30%, transparent 75%);
    -webkit-mask-image: radial-gradient(ellipse at 50% 30%, #000 30%, transparent 75%);
    pointer-events: none;
    z-index: 0;
  }}
  [data-testid="stAppViewContainer"] > * {{ position: relative; z-index: 1; }}

  .block-container {{
    padding-top: 1.4rem !important;
    padding-bottom: 4rem !important;
    max-width: 1320px;
  }}

  /* Headings */
  h1 {{ font-weight: 700 !important; letter-spacing: -0.028em; font-size: 2.2rem; }}
  h2 {{ font-weight: 650 !important; letter-spacing: -0.02em; }}
  h3 {{ font-weight: 600 !important; letter-spacing: -0.012em; }}

  /* Section rule */
  hr {{ border: none; height: 1px; background:
        linear-gradient(90deg, transparent, {BORDER_SUBTLE} 30%, {BORDER_SUBTLE} 70%, transparent); }}

  /* ── BRAND HEADER ──────────────────────────────────────────────────────── */
  .brand-row {{
    display: flex; align-items: center; justify-content: space-between;
    margin: 0 0 28px 0;
  }}
  .brand {{
    display: flex; align-items: center; gap: 12px;
    font-family: 'Inter', sans-serif; font-weight: 600;
    letter-spacing: -0.02em; font-size: 0.95rem; color: {TEXT};
  }}
  .brand-mark {{
    width: 28px; height: 28px; border-radius: 8px;
    background: conic-gradient(from 220deg at 50% 50%,
                {ACCENT} 0deg, {PREDICTED} 140deg, {ACCENT_2} 280deg, {ACCENT} 360deg);
    box-shadow: 0 8px 24px -8px {ACCENT_GLOW}, inset 0 0 0 1px rgba(255,255,255,0.1);
    position: relative;
  }}
  .brand-mark::after {{
    content: ''; position: absolute; inset: 6px; border-radius: 4px;
    background: {BG}; box-shadow: inset 0 0 0 1px rgba(255,255,255,0.08);
  }}
  .brand-sub {{ color: {TEXT_MUTED}; font-weight: 400; font-size: 0.8rem; margin-left: 6px; }}

  .status-pill {{
    display: inline-flex; align-items: center; gap: 8px;
    padding: 6px 12px; border-radius: 999px;
    background: {SURFACE}; border: 1px solid {BORDER_SUBTLE};
    font-size: 0.72rem; color: {TEXT_MUTED}; font-weight: 500;
    letter-spacing: 0.02em;
  }}
  .status-dot {{
    width: 6px; height: 6px; border-radius: 999px;
    background: {POSITIVE}; box-shadow: 0 0 0 3px rgba(34,211,179,0.18);
    animation: pulse 2.4s ease-in-out infinite;
  }}
  @keyframes pulse {{
    0%, 100% {{ opacity: 1; box-shadow: 0 0 0 3px rgba(34,211,179,0.18); }}
    50%      {{ opacity: 0.6; box-shadow: 0 0 0 6px rgba(34,211,179,0.05); }}
  }}

  /* ── EYEBROW ─────────────────────────────────────────────────────────── */
  .eyebrow {{
    display: inline-flex; align-items: center; gap: 8px;
    text-transform: uppercase; letter-spacing: 0.14em;
    font-size: 0.68rem; font-weight: 500; color: {TEXT_DIM};
    margin-bottom: 10px;
  }}
  .eyebrow-dot {{
    display: inline-block; width: 5px; height: 5px; border-radius: 999px;
    background: {ACCENT}; box-shadow: 0 0 10px {ACCENT_GLOW};
  }}

  .page-title {{ margin: 2px 0 6px 0; }}
  .page-sub   {{ color: {TEXT_MUTED}; font-size: 0.98rem; margin: 0 0 22px 0; max-width: 640px; }}

  /* ── SIDEBAR ─────────────────────────────────────────────────────────── */
  [data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {SURFACE} 0%, {BG} 100%) !important;
    border-right: 1px solid {BORDER_SUBTLE};
  }}
  [data-testid="stSidebarNav"] ul {{ padding-top: 12px; }}
  [data-testid="stSidebarNav"] a {{
    border-radius: 10px; transition: all 180ms ease;
    font-weight: 500; color: {TEXT_MUTED};
  }}
  [data-testid="stSidebarNav"] a:hover {{
    background: rgba(255,255,255,0.03) !important;
    color: {TEXT} !important;
  }}
  [data-testid="stSidebarNav"] a[aria-current="page"] {{
    background: linear-gradient(135deg, {ACCENT_SOFT}, rgba(139,124,255,0.04)) !important;
    color: {TEXT} !important;
    box-shadow: inset 0 0 0 1px rgba(139,124,255,0.28), 0 0 24px -12px {ACCENT_GLOW};
  }}

  /* ── PREMIUM KPI CARDS ───────────────────────────────────────────────── */
  .kpi-grid   {{ display: grid; grid-template-columns: 1.5fr 1fr 1fr 1fr; gap: 14px; margin: 4px 0 28px 0; }}
  @media (max-width: 980px) {{ .kpi-grid {{ grid-template-columns: 1fr 1fr; }} }}

  .kpi {{
    position: relative;
    padding: 18px 20px;
    border-radius: 14px;
    background:
      linear-gradient(180deg, rgba(20,26,42,0.9), rgba(14,19,32,0.9));
    backdrop-filter: blur(20px);
    border: 1px solid {BORDER_SUBTLE};
    transition: transform 220ms cubic-bezier(.2,.7,.2,1), border-color 180ms ease, box-shadow 220ms ease;
    overflow: hidden;
    min-height: 120px;
  }}
  .kpi::before {{
    content: '';
    position: absolute; inset: 0;
    border-radius: 14px;
    padding: 1px;
    background: linear-gradient(135deg, rgba(139,124,255,0.28), transparent 45%);
    -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
            mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    -webkit-mask-composite: xor; mask-composite: exclude;
    pointer-events: none;
  }}
  .kpi:hover {{
    transform: translateY(-1px);
    border-color: rgba(139,124,255,0.22);
    box-shadow: 0 14px 38px -22px {ACCENT_GLOW};
  }}
  .kpi-label {{
    display: flex; align-items: center; gap: 8px;
    text-transform: uppercase; letter-spacing: 0.1em;
    font-size: 0.66rem; color: {TEXT_DIM}; font-weight: 500;
    margin-bottom: 10px;
  }}
  .kpi-value {{
    font-variant-numeric: tabular-nums;
    font-weight: 650;
    font-size: 1.9rem;
    color: {TEXT};
    line-height: 1;
    letter-spacing: -0.02em;
    white-space: nowrap;
  }}
  .kpi-unit {{
    font-size: 0.78rem;
    color: {TEXT_MUTED};
    font-weight: 500;
    margin-left: 6px;
    letter-spacing: 0;
    text-transform: lowercase;
  }}
  .kpi-sub {{
    margin-top: 8px; font-size: 0.78rem; color: {TEXT_MUTED};
    display: flex; align-items: center; gap: 6px;
  }}
  .kpi-delta-pos {{ color: {POSITIVE}; }}
  .kpi-delta-neg {{ color: {NEGATIVE}; }}
  .kpi-spark    {{ margin-top: 10px; opacity: 0.85; }}

  .kpi-hero {{ grid-column: 1 / 2; }}
  .kpi-hero .kpi-value {{ font-size: 3.2rem; font-weight: 700; letter-spacing: -0.035em; }}
  .kpi-hero {{ min-height: 150px; padding: 22px 24px; }}
  .kpi-hero::before {{
    background: linear-gradient(135deg, rgba(139,124,255,0.55), rgba(34,211,179,0.3) 45%, transparent 75%);
  }}

  /* Keep Streamlit-native st.metric consistent with the kpi style */
  [data-testid="stMetric"] {{
    background: linear-gradient(180deg, rgba(20,26,42,0.85), rgba(14,19,32,0.85));
    backdrop-filter: blur(18px);
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 14px;
    padding: 16px 20px;
    transition: all 180ms ease;
    position: relative; overflow: hidden;
  }}
  [data-testid="stMetric"]::before {{
    content: ''; position: absolute; inset: 0; border-radius: 14px; padding: 1px;
    background: linear-gradient(135deg, rgba(139,124,255,0.22), transparent 50%);
    -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
            mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    -webkit-mask-composite: xor; mask-composite: exclude;
    pointer-events: none;
  }}
  [data-testid="stMetric"]:hover {{ transform: translateY(-1px); }}
  [data-testid="stMetricLabel"] p {{
    color: {TEXT_DIM} !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-size: 0.66rem !important;
    font-weight: 500 !important;
  }}
  [data-testid="stMetricValue"] {{
    color: {TEXT} !important;
    font-size: 1.85rem !important;
    font-weight: 650 !important;
    letter-spacing: -0.02em !important;
    font-variant-numeric: tabular-nums;
  }}

  /* ── SECTIONS / PANELS ────────────────────────────────────────────────── */
  .section-head {{
    display: flex; align-items: baseline; justify-content: space-between;
    margin: 28px 0 10px 0;
  }}
  .section-title {{ color: {TEXT}; font-weight: 600; font-size: 1.05rem; letter-spacing: -0.012em; }}
  .section-cap   {{ color: {TEXT_MUTED}; font-size: 0.85rem; margin-top: 2px; }}

  .glass-panel {{
    background: linear-gradient(180deg, rgba(20,26,42,0.5), rgba(14,19,32,0.5));
    backdrop-filter: blur(18px);
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 16px;
    padding: 18px 20px;
  }}

  /* ── TABS — pill-ish segmented control ────────────────────────────────── */
  [data-baseweb="tab-list"] {{
    gap: 6px !important;
    padding: 4px !important;
    background: {SURFACE};
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 12px;
    width: fit-content;
  }}
  button[data-baseweb="tab"] {{
    height: 34px !important;
    padding: 0 14px !important;
    border-radius: 8px !important;
    color: {TEXT_MUTED} !important;
    font-weight: 500 !important;
    transition: all 160ms ease !important;
    background: transparent !important;
  }}
  button[data-baseweb="tab"] > div > p {{ font-weight: 500 !important; }}
  button[data-baseweb="tab"]:hover {{ color: {TEXT} !important; background: rgba(255,255,255,0.03) !important; }}
  button[data-baseweb="tab"][aria-selected="true"] {{
    background: linear-gradient(135deg, {ACCENT_SOFT}, rgba(91,140,255,0.10)) !important;
    color: {TEXT} !important;
    box-shadow: inset 0 0 0 1px rgba(139,124,255,0.28);
  }}
  button[data-baseweb="tab"][aria-selected="true"] > div > p {{ color: {TEXT} !important; font-weight: 600 !important; }}
  [data-baseweb="tab-highlight"] {{ display: none !important; }}
  [data-baseweb="tab-border"]    {{ display: none !important; }}

  /* ── INPUTS: selectbox / slider / radio ───────────────────────────────── */
  .stSelectbox label, .stSlider label, .stRadio label {{
    color: {TEXT_DIM} !important;
    font-size: 0.7rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }}

  .stSelectbox [data-baseweb="select"] > div {{
    background: {SURFACE} !important;
    border: 1px solid {BORDER_SUBTLE} !important;
    border-radius: 10px !important;
    min-height: 44px !important;
    transition: border-color 160ms ease, box-shadow 160ms ease;
  }}
  .stSelectbox [data-baseweb="select"] > div:hover {{ border-color: {BORDER_STRONG} !important; }}
  .stSelectbox [data-baseweb="select"] > div[aria-expanded="true"] {{
    border-color: {ACCENT} !important;
    box-shadow: 0 0 0 3px {ACCENT_SOFT};
  }}

  /* Slider track */
  .stSlider [data-baseweb="slider"] [role="slider"] {{
    background: {ACCENT} !important;
    border: 3px solid {SURFACE} !important;
    box-shadow: 0 0 0 1px {ACCENT}, 0 0 18px {ACCENT_GLOW} !important;
    width: 18px !important; height: 18px !important;
  }}
  .stSlider [data-baseweb="slider"] > div > div > div:nth-child(1) > div {{
    background: linear-gradient(90deg, {ACCENT}, {ACCENT_2}) !important;
    height: 5px !important;
  }}
  .stSlider [data-baseweb="slider"] > div > div > div:nth-child(2) > div {{
    background: rgba(255,255,255,0.06) !important;
    height: 5px !important;
  }}

  /* Radio */
  .stRadio [role="radiogroup"] {{ gap: 6px !important; }}
  .stRadio label[data-baseweb="radio"] {{
    background: {SURFACE};
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 10px;
    padding: 8px 12px !important;
    transition: all 160ms ease;
  }}
  .stRadio label[data-baseweb="radio"]:hover {{ border-color: {BORDER_STRONG}; }}

  /* ── DATAFRAMES ──────────────────────────────────────────────────────── */
  [data-testid="stDataFrame"] {{
    border: 1px solid {BORDER_SUBTLE} !important;
    border-radius: 12px !important;
    overflow: hidden;
    background: rgba(14,19,32,0.5);
  }}
  [data-testid="stDataFrame"] [role="columnheader"] {{
    background: rgba(255,255,255,0.02) !important;
    color: {TEXT_DIM} !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.7rem !important;
    font-weight: 500 !important;
  }}

  /* Hide Streamlit chrome we don't want */
  #MainMenu, footer, header {{ visibility: hidden; }}

  /* Plotly chart container — subtle panel */
  [data-testid="stPlotlyChart"] {{
    background: linear-gradient(180deg, rgba(20,26,42,0.35), rgba(14,19,32,0.35));
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 14px;
    padding: 8px 4px 4px 4px;
  }}
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


# ─── HTML primitives ─────────────────────────────────────────────────────────
def _spark_svg(values: Iterable[float], *, color: str = PREDICTED,
               width: int = 140, height: int = 32, fill: str | None = None) -> str:
    vals = list(values)
    if len(vals) < 2:
        return ""
    mn, mx = min(vals), max(vals)
    rng = (mx - mn) or 1.0
    n = len(vals)
    pts = [
        (i * (width - 2) / (n - 1) + 1,
         height - 2 - ((v - mn) / rng) * (height - 4))
        for i, v in enumerate(vals)
    ]
    path = f"M {pts[0][0]:.1f} {pts[0][1]:.1f} " + " ".join(
        f"L {x:.1f} {y:.1f}" for x, y in pts[1:]
    )
    fill_area = ""
    if fill:
        area = (path + f" L {pts[-1][0]:.1f} {height} L {pts[0][0]:.1f} {height} Z")
        fill_area = f'<path d="{area}" fill="url(#spark-grad)" opacity="0.55"/>'
    grad = (f'<defs><linearGradient id="spark-grad" x1="0" x2="0" y1="0" y2="1">'
            f'<stop offset="0%" stop-color="{color}" stop-opacity="0.5"/>'
            f'<stop offset="100%" stop-color="{color}" stop-opacity="0"/>'
            f'</linearGradient></defs>') if fill else ""
    last_x, last_y = pts[-1]
    return (
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'xmlns="http://www.w3.org/2000/svg">{grad}{fill_area}'
        f'<path d="{path}" fill="none" stroke="{color}" stroke-width="1.6" '
        f'stroke-linecap="round" stroke-linejoin="round"/>'
        f'<circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="2.4" fill="{color}"/>'
        f'</svg>'
    )


def kpi_card(label: str, value: str, *,
             unit: str | None = None,
             spark: Iterable[float] | None = None,
             spark_color: str = PREDICTED, fill: bool = True,
             sub: str | None = None,
             delta: float | None = None,
             delta_suffix: str = "",
             hero: bool = False) -> str:
    """Return the HTML for one KPI card. Render via st.markdown with unsafe_allow_html=True."""
    unit_html = f'<span class="kpi-unit">{unit}</span>' if unit else ""
    spark_html = ""
    if spark is not None:
        spark_html = f'<div class="kpi-spark">{_spark_svg(spark, color=spark_color, fill=fill)}</div>'
    sub_html = ""
    if sub:
        sub_html = f'<div class="kpi-sub">{sub}</div>'
    if delta is not None:
        cls = "kpi-delta-pos" if delta >= 0 else "kpi-delta-neg"
        arrow = "▲" if delta >= 0 else "▼"
        sub_html = (f'<div class="kpi-sub"><span class="{cls}">{arrow} '
                    f'{abs(delta):,.2f}{delta_suffix}</span>'
                    + (f' <span style="color:{TEXT_DIM};">vs actual</span>' if delta_suffix == "" else "")
                    + "</div>")
    hero_cls = " kpi-hero" if hero else ""
    return (
        f'<div class="kpi{hero_cls}">'
        f'  <div class="kpi-label">{label}</div>'
        f'  <div class="kpi-value">{value}{unit_html}</div>'
        f'  {sub_html}{spark_html}'
        f'</div>'
    )


def render_kpi_row(cards_html: Iterable[str]) -> None:
    st.markdown(f'<div class="kpi-grid">{"".join(cards_html)}</div>', unsafe_allow_html=True)


def hero_forecast(forecast: float, actual: float, lower: float, upper: float,
                  spark_values: Iterable[float], *, unit: str = "") -> None:
    """Top-of-page premium KPI strip: Forecast hero + Actual / P10 / P90."""
    delta = forecast - actual
    interval_width = upper - lower
    cards = [
        kpi_card("Forecast",     f"{forecast:,.1f}", unit=unit,
                 spark=spark_values, spark_color=PREDICTED, fill=True,
                 delta=delta, hero=True),
        kpi_card("Actual",       f"{actual:,.1f}", unit=unit,
                 spark=spark_values, spark_color=ACTUAL, fill=False,
                 sub="held-out ground truth"),
        kpi_card("P10 bound",    f"{lower:,.1f}", unit=unit,
                 sub=f"{interval_width:,.1f} {unit} interval width" if unit else f"{interval_width:,.1f} interval width"),
        kpi_card("P90 bound",    f"{upper:,.1f}", unit=unit,
                 sub="90th percentile"),
    ]
    render_kpi_row(cards)


def brand_header(status_text: str = "Models loaded · 22 tests passing") -> None:
    st.markdown(
        f"""
        <div class="brand-row">
          <div class="brand">
            <div class="brand-mark"></div>
            <span>Forecast Foundry</span>
            <span class="brand-sub">multi-domain demand forecasting</span>
          </div>
          <div class="status-pill">
            <span class="status-dot"></span>
            <span>{status_text}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_header(eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="eyebrow"><span class="eyebrow-dot"></span>{eyebrow}</div>
        <h1 class="page-title">{title}</h1>
        <p class="page-sub">{subtitle}</p>
        """,
        unsafe_allow_html=True,
    )


def section(title: str, caption: str | None = None) -> None:
    cap = f'<div class="section-cap">{caption}</div>' if caption else ""
    st.markdown(
        f'<div class="section-head"><div>'
        f'<div class="section-title">{title}</div>{cap}'
        f'</div></div>',
        unsafe_allow_html=True,
    )


def hero_metric(label: str, value: str, caption: str | None = None) -> None:
    """Backwards-compat alias — renders a single KPI card."""
    st.markdown(kpi_card(label, value, sub=caption), unsafe_allow_html=True)


def apply(page_title: str = "Demand Forecasting") -> None:
    register_plotly_template()
    inject_css()
