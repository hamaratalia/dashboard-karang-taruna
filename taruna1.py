import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import ttest_ind
from datetime import datetime
import io
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG HALAMAN
# ============================================================

st.set_page_config(
    page_title="Dashboard Keuangan Karang Taruna",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS — v3.0 (semua overlap fix)
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800;900&display=swap');

*, html, body, [class*="css"] { font-family: 'Poppins', sans-serif !important; }

.stApp { background: #F0FAF4; }
.block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 2rem !important;
    max-width: 1400px !important;
}

/* ======= HEADER ======= */
.main-header {
    background: linear-gradient(135deg, #0D5C2E 0%, #1A8A47 65%, #27AE60 100%);
    padding: 28px 36px;
    border-radius: 18px;
    margin-bottom: 20px;
    color: white;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(13,92,46,0.28);
}
.main-header::after {
    content: '🌾';
    position: absolute; right: 36px; top: 50%;
    transform: translateY(-50%);
    font-size: 88px; opacity: 0.10; line-height: 1;
    pointer-events: none;
}
.main-header h1 { font-size: 1.75rem; font-weight: 900; margin: 0 0 6px 0; line-height: 1.25; }
.main-header p  { font-size: 0.88rem; margin: 0; opacity: 0.82; }

/* ======= WELCOME ======= */
.welcome-box {
    background: white;
    border-radius: 20px;
    padding: 48px 36px;
    text-align: center;
    box-shadow: 0 6px 28px rgba(13,92,46,0.10);
    border: 2px dashed #A9DFBF;
    margin-bottom: 20px;
}
.welcome-icon  { font-size: 4rem; margin-bottom: 12px; display: block; }
.welcome-title { font-size: 1.45rem; font-weight: 900; color: #0D5C2E; margin-bottom: 10px; }
.welcome-desc  {
    font-size: 0.90rem; color: #4A6050; line-height: 1.75;
    max-width: 500px; margin: 0 auto 24px;
}
.welcome-steps {
    display: grid; grid-template-columns: repeat(3,1fr);
    gap: 12px; max-width: 620px; margin: 0 auto; text-align: left;
}
.step-item {
    background: #F0FAF4; border-radius: 12px;
    padding: 14px 16px; border: 1.5px solid #C8EDD4;
}
.step-num {
    background: #27AE60; color: white; border-radius: 50%;
    width: 24px; height: 24px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.76rem; font-weight: 800; margin-bottom: 7px;
}
.step-title { font-size: 0.78rem; font-weight: 800; color: #0D5C2E; margin-bottom: 3px; }
.step-desc  { font-size: 0.72rem; color: #5A7260; line-height: 1.5; }

/* ======= SECTION TITLE ======= */
.section-title {
    font-size: 0.95rem; font-weight: 800; color: #0D5C2E;
    padding: 8px 14px;
    border-left: 5px solid #27AE60;
    background: #E8F8EF;
    border-radius: 0 10px 10px 0;
    margin: 16px 0 12px 0;
    display: block;
    line-height: 1.4;
}

/* ======= KPI CARDS ======= */
.kpi-card {
    background: white;
    border-radius: 14px;
    padding: 18px 14px;
    text-align: center;
    box-shadow: 0 3px 12px rgba(0,0,0,0.07);
    border-top: 4px solid;
    height: 100%;
    box-sizing: border-box;
    overflow: hidden;
}
.kpi-icon  { font-size: 1.75rem; margin-bottom: 7px; display: block; }
.kpi-label {
    font-size: 0.65rem; font-weight: 700; color: #5A7260;
    text-transform: uppercase; letter-spacing: 0.4px; margin-bottom: 5px;
    word-break: break-word;
}
.kpi-value { font-size: 1.15rem; font-weight: 900; color: #1A2D20; line-height: 1.2; margin-bottom: 4px; }
.kpi-sub   { font-size: 0.65rem; color: #8BA88F; line-height: 1.4; }
.kpi-delta { font-size: 0.63rem; margin-top: 4px; font-weight: 700; }

/* ======= ANOMALI ======= */
.anomali-box {
    background: #FFF8E1; border: 2px solid #F39C12;
    border-radius: 12px; padding: 12px 14px; margin: 8px 0;
    display: flex; gap: 10px; align-items: flex-start;
}
.anomali-icon { font-size: 1.4rem; flex-shrink: 0; }
.anomali-title { font-size: 0.80rem; font-weight: 800; color: #7D3C00; margin-bottom: 3px; }
.anomali-text  { font-size: 0.72rem; color: #7D4E00; line-height: 1.5; }

/* ======= PREDIKSI BOX ======= */
.pred-box {
    background: linear-gradient(135deg, #0D5C2E 0%, #1A8A47 65%, #27AE60 100%);
    border-radius: 16px;
    padding: 24px 26px;
    color: white;
    box-shadow: 0 8px 26px rgba(13,92,46,0.28);
    position: relative; overflow: hidden;
    margin-bottom: 16px;
}
.pred-box::after {
    content: '🔮'; position: absolute;
    right: 24px; top: 20px;
    font-size: 56px; opacity: 0.10; line-height: 1;
    pointer-events: none;
}
.pred-box-title { font-size: 1.10rem; font-weight: 800; margin-bottom: 4px; }
.pred-box-sub   { font-size: 0.80rem; opacity: 0.80; margin-bottom: 18px; }
.pred-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 10px; }
.pred-item {
    background: rgba(255,255,255,0.14);
    border: 1.5px solid rgba(255,255,255,0.22);
    border-radius: 11px; padding: 13px 10px; text-align: center;
}
.pred-item-icon  { font-size: 1.4rem; display: block; margin-bottom: 4px; }
.pred-item-label { font-size: 0.68rem; opacity: 0.82; font-weight: 600; margin-bottom: 4px; }
.pred-item-value { font-size: 1.00rem; font-weight: 900; line-height: 1.2; }
.pred-item-delta { font-size: 0.64rem; opacity: 0.72; margin-top: 3px; }

/* ======= FORECAST CARDS (3 bulan) ======= */
.fc-card {
    background: white;
    border-radius: 14px;
    padding: 16px 14px;
    text-align: center;
    box-shadow: 0 3px 12px rgba(0,0,0,0.07);
    border-top: 4px solid #27AE60;
    flex: 1;
}
.fc-month { font-size: 0.70rem; font-weight: 700; color: #5A7260; text-transform: uppercase; margin-bottom: 8px; }
.fc-value { font-size: 1.05rem; font-weight: 900; color: #0D5C2E; margin-bottom: 3px; }
.fc-ci    { font-size: 0.65rem; color: #8BA88F; }

/* ======= HEALTH ======= */
.health-row { margin-bottom: 18px; }
.health-meta {
    display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;
}
.health-label { font-size: 0.82rem; font-weight: 700; color: #1A2D20; }
.health-score { font-size: 0.82rem; font-weight: 900; }
.health-bar-bg  { background: #D5F5E3; border-radius: 10px; height: 12px; overflow: hidden; }
.health-bar-fill { height: 100%; border-radius: 10px; transition: width 0.6s ease; }
.health-sub { font-size: 0.70rem; color: #5A7260; margin-top: 4px; }

/* ======= SCORE CIRCLE ======= */
.score-circle {
    border-radius: 16px; padding: 24px 14px; text-align: center; height: 100%;
    display: flex; flex-direction: column; justify-content: center; align-items: center;
    min-height: 180px;
}
.score-emoji { font-size: 3rem; margin-bottom: 7px; }
.score-num   { font-size: 2.6rem; font-weight: 900; line-height: 1; }
.score-label { font-size: 0.85rem; font-weight: 800; margin-top: 7px; }
.score-sub   { font-size: 0.68rem; color: #5A7260; margin-top: 7px; line-height: 1.4; text-align: center; }

/* ======= TIP CARDS ======= */
.tip-card {
    background: white; border-radius: 12px;
    padding: 14px 14px;
    box-shadow: 0 2px 9px rgba(0,0,0,0.07);
    border-left: 5px solid;
    display: flex; gap: 10px; align-items: flex-start;
    margin-bottom: 10px;
}
.tip-icon  { font-size: 1.5rem; flex-shrink: 0; padding-top: 1px; }
.tip-title { font-size: 0.80rem; font-weight: 800; color: #1A2D20; margin-bottom: 3px; }
.tip-text  { font-size: 0.72rem; color: #4A6050; line-height: 1.55; }

/* ======= A/B CARDS ======= */
.ab-card { border-radius: 12px; padding: 18px 14px; text-align: center; }
.ab-card.awal  { background: #E8F8EF; border: 2px solid #A9DFBF; }
.ab-card.akhir { background: #EBF5FB; border: 2px solid #AED6F1; }
.ab-period { font-size: 0.76rem; font-weight: 700; color: #1A2D20; margin-bottom: 7px; }
.ab-value  { font-size: 1.35rem; font-weight: 900; margin-bottom: 4px; }
.ab-card.awal  .ab-value { color: #27AE60; }
.ab-card.akhir .ab-value { color: #2980B9; }
.ab-desc { font-size: 0.68rem; color: #5A7260; }

/* ======= DIVIDER ======= */
.custom-divider {
    height: 2px; background: linear-gradient(90deg, #27AE60 0%, transparent 100%);
    border: none; border-radius: 2px; margin: 16px 0;
}

/* ======= FOOTER ======= */
.footer-box {
    background: linear-gradient(135deg, #0D5C2E, #1A8A47);
    color: white; text-align: center;
    padding: 18px; border-radius: 14px;
    margin-top: 28px; font-size: 0.78rem;
}
.footer-box span { opacity: 0.68; font-size: 0.70rem; display: block; margin-top: 4px; }

/* ======= DOWNLOAD BUTTON ======= */
.stDownloadButton button {
    background: #27AE60 !important; color: white !important;
    border-radius: 10px !important; font-weight: 700 !important;
    border: none !important; width: 100% !important;
}

/* ======= TABS ======= */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: white;
    border-radius: 12px;
    padding: 6px 8px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.07);
    margin-bottom: 16px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    padding: 8px 14px !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
    color: #4A6050 !important;
}
.stTabs [aria-selected="true"] {
    background: #27AE60 !important;
    color: white !important;
}

/* ======= SIDEBAR ======= */

/* Sembunyikan tombol collapse sidebar (penyebab teks "keyboard_double_arrow_right") */
[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"],
button[aria-label="Close sidebar"],
button[aria-label="Open sidebar"] {
    display: none !important;
    visibility: hidden !important;
}

/* Sembunyikan span ikon material (teks "double_arrow_right", "upload", dsb)
   yang muncul karena font Material Icons tidak ter-load */
[data-testid="stSidebar"] button > span:first-child:not(:only-child),
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button > span:first-child,
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button > p:first-child {
    display: none !important;
}

/* Pastikan teks "Upload" / "Browse files" di dalam tombol tetap tampil */
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button > span:last-child,
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button > p:last-child {
    display: inline !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A4D25 0%, #0F5529 40%, #145C31 100%) !important;
}
[data-testid="stSidebar"] > div {
    padding-top: 0.5rem !important;
    padding-bottom: 1.5rem !important;
}
[data-testid="stSidebar"] label {
    color: rgba(255,255,255,0.95) !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] .stMarkdown {
    color: rgba(255,255,255,0.85) !important;
}
[data-testid="stSidebar"] h3 {
    color: white !important; font-weight: 800 !important;
    font-size: 0.95rem !important; margin-bottom: 8px !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: white !important; border-radius: 8px !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] span,
[data-testid="stSidebar"] [data-baseweb="select"] div {
    color: #1A2D20 !important;
}

/* Sidebar File Uploader — sembunyikan teks "Upload" di luar kotak */
[data-testid="stSidebar"] [data-testid="stFileUploader"] > label {
    display: none !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,0.08) !important;
    border: 2px dashed rgba(255,255,255,0.35) !important;
    border-radius: 12px !important;
    padding: 18px 10px !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]:hover {
    background: rgba(255,255,255,0.13) !important;
    border-color: #A9DFBF !important;
}
/* Teks di dalam dropzone (instruksi & limit) — putih agar terbaca */
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] span,
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] p,
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] small {
    color: rgba(255,255,255,0.85) !important;
}
/* Tombol Browse di dalam dropzone — sembunyikan SEMUA teks bawaan,
   ganti dengan teks custom via ::after agar tidak ada "uploadUpload" */
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
    background: #27AE60 !important;
    color: transparent !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 0 !important;
    font-weight: 700 !important;
    padding: 6px 18px !important;
    margin-top: 6px !important;
    cursor: pointer !important;
    position: relative !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button * {
    font-size: 0 !important;
    color: transparent !important;
    display: inline !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button::after {
    content: "📂 Pilih File";
    font-size: 0.82rem !important;
    color: white !important;
    font-family: 'Poppins', sans-serif !important;
    font-weight: 700 !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] {
    background: rgba(255,255,255,0.15) !important;
    border-radius: 8px !important;
    padding: 8px 10px !important;
    border: 1px solid rgba(255,255,255,0.3) !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] span,
[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] p,
[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] small,
[data-testid="stSidebar"] [data-testid="stFileUploaderFileName"] {
    color: white !important;
    opacity: 1 !important;
}
/* Ikon file (thumbnail) */
[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] svg {
    fill: #A9DFBF !important;
    color: #A9DFBF !important;
}

/* Sidebar Info Card */
.sidebar-info {
    background: rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 14px;
    font-size: 0.78rem;
    color: rgba(255,255,255,0.9);
    line-height: 1.7;
    border: 1px solid rgba(255,255,255,0.14);
}
.sidebar-info strong { color: #A9DFBF; }
.sidebar-badge {
    display: inline-block;
    background: rgba(39,174,96,0.3);
    color: #A9DFBF;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.70rem;
    font-weight: 700;
    margin-top: 6px;
}
.sidebar-divider {
    border: none;
    border-top: 1px dashed rgba(255,255,255,0.22);
    margin: 14px 0;
}
.sidebar-logo {
    text-align: center;
    padding: 18px 0 16px;
    border-bottom: 1px dashed rgba(255,255,255,0.2);
    margin-bottom: 16px;
}
.sidebar-logo .logo-icon { font-size: 3rem; line-height: 1.1; display: block; }
.sidebar-logo .logo-title {
    font-size: 1.05rem; font-weight: 900;
    color: white; margin-top: 6px; display: block;
}
.sidebar-logo .logo-sub {
    font-size: 0.76rem; color: #A9DFBF;
    font-weight: 500; display: block; margin-top: 2px;
}
.sidebar-nav {
    background: rgba(255,255,255,0.07);
    border-radius: 10px; padding: 12px 14px;
    margin-top: 12px; font-size: 0.76rem;
    color: rgba(255,255,255,0.8); line-height: 2.0;
}
.sidebar-footer {
    margin-top: 20px;
    text-align: center;
    font-size: 0.72rem;
    color: rgba(255,255,255,0.45);
    padding-bottom: 8px;
}

/* YoY comparison box */
.yoy-box {
    background: white; border-radius: 12px; padding: 16px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.07);
    border-left: 4px solid #2980B9;
    margin-bottom: 12px;
}
.yoy-title { font-size: 0.82rem; font-weight: 800; color: #1A2D20; margin-bottom: 8px; }
.yoy-val   { font-size: 1.10rem; font-weight: 900; }
.yoy-delta { font-size: 0.70rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# KONSTANTA & HELPERS
# ============================================================

MONTH_ID = {
    1:"Januari", 2:"Februari", 3:"Maret",    4:"April",
    5:"Mei",     6:"Juni",     7:"Juli",      8:"Agustus",
    9:"September",10:"Oktober",11:"November",12:"Desember"
}
MONTH_SHORT = {
    1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr",  5:"Mei", 6:"Jun",
    7:"Jul", 8:"Agu", 9:"Sep", 10:"Okt", 11:"Nov", 12:"Des"
}
CLR_GREEN  = "#27AE60"
CLR_RED    = "#E74C3C"
CLR_BLUE   = "#2980B9"
CLR_GOLD   = "#D4A017"
CLR_DARK   = "#0D5C2E"
CLR_ORANGE = "#E67E22"

LAYOUT = dict(
    font=dict(family="Poppins, sans-serif", size=11, color="#1A2D20"),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=12, r=12, t=44, b=12),
)


def rupiah(n, singkat=False):
    """Format angka sebagai Rupiah. Nilai negatif ditampilkan dengan tanda minus."""
    try:
        n = float(n)
        if np.isnan(n): n = 0.0
    except Exception:
        n = 0.0
    n = int(n)
    negatif = n < 0
    absn = abs(n)
    if singkat:
        if absn >= 1_000_000_000: teks = f"Rp {absn/1_000_000_000:.1f} M"
        elif absn >= 1_000_000:   teks = f"Rp {absn/1_000_000:.1f} jt"
        elif absn >= 1_000:       teks = f"Rp {absn/1_000:.0f} rb"
        else:                      teks = "Rp " + f"{absn:,}".replace(",",".")
    else:
        teks = "Rp " + f"{absn:,}".replace(",",".")
    return ("-" + teks) if negatif else teks


def clr_health(v):
    return CLR_GREEN if v >= 70 else (CLR_GOLD if v >= 40 else CLR_RED)

def emoji_health(v):
    return "🟢" if v >= 70 else ("🟡" if v >= 40 else "🔴")


# ============================================================
# BERSIHKAN DATA
# ============================================================

def bersihkan_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().dropna(how='all').reset_index(drop=True)
    df.columns = df.columns.str.strip()

    col_map = {}
    for col in df.columns:
        c = col.lower().strip()
        if   any(k in c for k in ['tanggal','date','tgl']):
            col_map.setdefault('Tanggal',    col)
        elif any(k in c for k in ['pemasukan','income','masuk','debit_masuk','kredit_masuk']):
            col_map.setdefault('Pemasukan',  col)
        elif any(k in c for k in ['pengeluaran','expense','keluar','kredit','bayar','debit_keluar']):
            col_map.setdefault('Pengeluaran',col)
        elif any(k in c for k in ['saldo','balance','kas','neraca']):
            col_map.setdefault('Saldo',      col)
        elif any(k in c for k in ['keterangan','ket','description','catatan','uraian','remark']):
            col_map.setdefault('Keterangan', col)
        elif any(k in c for k in ['kategori','category','jenis','type']):
            col_map.setdefault('Kategori',   col)

    df = df.rename(columns={v: k for k, v in col_map.items()})

    missing = [c for c in ['Tanggal','Pemasukan','Pengeluaran'] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Kolom wajib tidak ditemukan: **{', '.join(missing)}**\n\n"
            f"Kolom yang tersedia: {', '.join(df.columns.tolist())}"
        )

    for col in ['Pemasukan','Pengeluaran','Saldo']:
        if col not in df.columns:
            continue
        df[col] = (
            df[col].astype(str)
            .str.replace(r'\(([0-9.,]+)\)', r'-\1', regex=True)
            .str.replace(r'[Rp\s$]|IDR', '', regex=True)
            .str.replace(r'\.(?=\d{3}(?:[.,]|$))', '', regex=True)
            .str.replace(',', '.', regex=False)
            .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df['Tanggal'] = pd.to_datetime(df['Tanggal'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Tanggal']).sort_values('Tanggal').reset_index(drop=True)

    if df.empty:
        raise ValueError("Tidak ada baris valid setelah pembersihan. Periksa format kolom Tanggal.")

    df['Tahun']       = df['Tanggal'].dt.year
    df['Bulan']       = df['Tanggal'].dt.month
    df['Hari']        = df['Tanggal'].dt.day
    df['BulanNama']   = df['Bulan'].map(MONTH_ID)
    df['BulanPendek'] = df['Bulan'].map(MONTH_SHORT)

    if 'Saldo' not in df.columns or df['Saldo'].abs().sum() == 0:
        df['Saldo'] = (df['Pemasukan'] - df['Pengeluaran']).cumsum()

    if 'Keterangan' not in df.columns:
        df['Keterangan'] = ''
    if 'Kategori' not in df.columns:
        df['Kategori'] = 'Umum'

    return df


# ============================================================
# AGREGASI BULANAN
# ============================================================

def agregasi_bulanan(df: pd.DataFrame, tahun: int) -> pd.DataFrame:
    d = df[df['Tahun'] == tahun]
    if d.empty:
        return pd.DataFrame()
    m = (
        d.groupby(['Bulan','BulanNama','BulanPendek'])
        .agg(
            Pemasukan    = ('Pemasukan',  'sum'),
            Pengeluaran  = ('Pengeluaran','sum'),
            SaldoAkhir   = ('Saldo',      'last'),
            JmlTransaksi = ('Pemasukan',  'count')
        )
        .reset_index().sort_values('Bulan')
    )
    m['Surplus']    = m['Pemasukan'] - m['Pengeluaran']
    m['RasioBeban'] = np.where(m['Pemasukan'] > 0,
                               m['Pengeluaran'] / m['Pemasukan'] * 100, 100.0)
    m['Status']     = m['Surplus'].apply(lambda x: '✅ Surplus' if x >= 0 else '⚠️ Defisit')
    return m


# ============================================================
# DETEKSI ANOMALI
# ============================================================

def deteksi_anomali(monthly: pd.DataFrame) -> list:
    anomali = []
    if len(monthly) < 3:
        return anomali
    for col, label in [('Pengeluaran','Pengeluaran'),('Pemasukan','Pemasukan')]:
        mean_v = monthly[col].mean()
        std_v  = monthly[col].std()
        if std_v < 1:
            continue
        for _, row in monthly.iterrows():
            z = (row[col] - mean_v) / std_v
            if abs(z) > 2.0:
                arah = "sangat TINGGI 📈" if z > 0 else "sangat RENDAH 📉"
                anomali.append({
                    'bulan': row['BulanNama'], 'kolom': label,
                    'nilai': row[col], 'arah': arah, 'z': z,
                })
    return anomali


# ============================================================
# PREDIKSI — MODEL HYBRID + HOLT'S DOUBLE EXPONENTIAL
# ============================================================

def _exp_smoothing(values: list, alpha: float = 0.45) -> float:
    """Single Exponential Smoothing (SES)."""
    if not values: return 0.0
    s = float(values[0])
    for v in values[1:]:
        s = alpha * float(v) + (1 - alpha) * s
    return s


def _holt_smoothing(values: list, alpha: float = 0.5, beta: float = 0.3, periods_ahead: int = 3):
    """
    [P1] Holt's Double Exponential Smoothing (level + trend).
    Lebih akurat daripada Linear Regression untuk data non-linear dengan tren.
    Returns: (forecasts list, levels list, trends list, fitted list)
    """
    n = len(values)
    if n < 2:
        return [values[-1]] * periods_ahead, list(values), [0.0] * n, list(values)

    # Inisialisasi
    L = [float(values[0])]
    T = [float(values[1]) - float(values[0])]

    fitted = [L[0] + T[0]]  # one-step ahead fitted

    for i in range(1, n):
        v = float(values[i])
        L_prev, T_prev = L[-1], T[-1]
        L_new = alpha * v + (1 - alpha) * (L_prev + T_prev)
        T_new = beta  * (L_new - L_prev) + (1 - beta) * T_prev
        L.append(L_new)
        T.append(T_new)
        if i < n - 1:
            fitted.append(L_new + T_new)

    # Forecast h steps ahead
    forecasts = []
    for h in range(1, periods_ahead + 1):
        f = L[-1] + h * T[-1]
        forecasts.append(max(0.0, f))

    return forecasts, L, T, fitted


def _mape(actual, predicted):
    """Mean Absolute Percentage Error."""
    a, p = np.array(actual, dtype=float), np.array(predicted, dtype=float)
    mask = a != 0
    if mask.sum() == 0: return 100.0
    return float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100)


def prediksi(monthly: pd.DataFrame, tahun: int) -> dict:
    """
    [P1-P6] Model Prediksi Hybrid Akurat:
    - n >= 4: Holt's Double Exponential Smoothing (utama) + LR (validasi)
    - n = 2-3: WMA + ETS hybrid
    - n = 1: rata-rata sederhana

    Output tambahan:
    - forecast_3bln: prediksi 3 bulan ke depan
    - ci_80, ci_95: confidence interval
    - mape_pem, mape_pen: MAPE accuracy
    - r2_pem, r2_pen: R² dari LR (bila n >= 4)
    """
    n        = len(monthly)
    avg_pem  = float(monthly['Pemasukan'].mean())
    avg_pen  = float(monthly['Pengeluaran'].mean())
    last_sal = float(monthly['SaldoAkhir'].iloc[-1])
    last_bln = int(monthly['Bulan'].iloc[-1])
    next_bln = (last_bln % 12) + 1
    next_thn = tahun if last_bln < 12 else tahun + 1

    pem_vals = monthly['Pemasukan'].values.tolist()
    pen_vals = monthly['Pengeluaran'].values.tolist()

    mae_pem = mae_pen = r2_pem = r2_pen = mape_pem = mape_pen = 0.0
    ci_pem_80 = ci_pen_80 = ci_pem_95 = ci_pen_95 = 0.0
    forecast_pem = forecast_pen = []
    conf = "Rendah ⚠️"
    model_name = "—"

    if n < 2:
        pred_pem, pred_pen = avg_pem, avg_pen
        forecast_pem = [avg_pem] * 3
        forecast_pen = [avg_pen] * 3
        conf = "Rendah ⚠️"
        model_name = "Rata-rata"

    elif n < 4:
        # WMA + ETS hybrid
        weights  = np.arange(1, n+1, dtype=float)
        wma_pem  = float(np.average(pem_vals, weights=weights))
        wma_pen  = float(np.average(pen_vals, weights=weights))
        ets_pem  = _exp_smoothing(pem_vals)
        ets_pen  = _exp_smoothing(pen_vals)
        pred_pem = max(0.0, 0.60 * ets_pem + 0.40 * wma_pem)
        pred_pen = max(0.0, 0.60 * ets_pen + 0.40 * wma_pen)
        forecast_pem = [pred_pem] * 3
        forecast_pen = [pred_pen] * 3
        conf = "Rendah ⚠️"
        model_name = "WMA + ETS"

    else:
        # === Holt's Double Exponential Smoothing [P1] ===
        # Optimasi alpha/beta sederhana via grid search kecil
        best_err = float('inf')
        best_ap, best_bp, best_ae, best_be = 0.5, 0.3, 0.5, 0.3
        for ap in [0.2, 0.35, 0.5, 0.65, 0.8]:
            for bp in [0.1, 0.2, 0.3, 0.4]:
                _, _, _, fitted_p = _holt_smoothing(pem_vals, ap, bp)
                if len(fitted_p) == 0: continue
                min_len = min(len(pem_vals)-1, len(fitted_p))
                err = np.mean(np.abs(np.array(pem_vals[1:min_len+1]) - np.array(fitted_p[:min_len])))
                if err < best_err:
                    best_err = err
                    best_ap, best_bp = ap, bp

        best_err = float('inf')
        for ae in [0.2, 0.35, 0.5, 0.65, 0.8]:
            for be in [0.1, 0.2, 0.3, 0.4]:
                _, _, _, fitted_e = _holt_smoothing(pen_vals, ae, be)
                if len(fitted_e) == 0: continue
                min_len = min(len(pen_vals)-1, len(fitted_e))
                err = np.mean(np.abs(np.array(pen_vals[1:min_len+1]) - np.array(fitted_e[:min_len])))
                if err < best_err:
                    best_err = err
                    best_ae, best_be = ae, be

        fc_pem, Lp, Tp, fitted_p = _holt_smoothing(pem_vals, best_ap, best_bp, 3)
        fc_pen, Le, Te, fitted_e = _holt_smoothing(pen_vals, best_ae, best_be, 3)

        pred_pem = fc_pem[0]
        pred_pen = fc_pen[0]
        forecast_pem = fc_pem
        forecast_pen = fc_pen

        # Residual errors untuk confidence interval [P2]
        res_p = np.array(pem_vals[1:]) - np.array([Lp[i] + Tp[i] for i in range(min(len(Lp), len(pem_vals))-1)])
        res_e = np.array(pen_vals[1:]) - np.array([Le[i] + Te[i] for i in range(min(len(Le), len(pen_vals))-1)])
        std_p = float(np.std(res_p)) if len(res_p) > 0 else avg_pem * 0.1
        std_e = float(np.std(res_e)) if len(res_e) > 0 else avg_pen * 0.1

        # CI 80% (z=1.28) dan 95% (z=1.96) — melebar sesuai horizon [P2]
        ci_pem_80 = 1.28 * std_p * (1 ** 0.5)
        ci_pem_95 = 1.96 * std_p * (1 ** 0.5)
        ci_pen_80 = 1.28 * std_e * (1 ** 0.5)
        ci_pen_95 = 1.96 * std_e * (1 ** 0.5)

        # Validasi dengan LR
        xs   = np.arange(1, n+1, dtype=float).reshape(-1,1)
        sp   = max(2, int(n * 0.75))
        if sp < n:
            m_p = LinearRegression().fit(xs[:sp], pem_vals[:sp])
            m_e = LinearRegression().fit(xs[:sp], pen_vals[:sp])
            mae_pem = mean_absolute_error(pem_vals[sp:], m_p.predict(xs[sp:]))
            mae_pen = mean_absolute_error(pen_vals[sp:], m_e.predict(xs[sp:]))

        m_p_full = LinearRegression().fit(xs, pem_vals)
        m_e_full = LinearRegression().fit(xs, pen_vals)
        r2_pem = max(0.0, r2_score(pem_vals, m_p_full.predict(xs)))
        r2_pen = max(0.0, r2_score(pen_vals, m_e_full.predict(xs)))

        # MAPE [P5]
        min_len_p = min(len(pem_vals)-1, len(fitted_p))
        min_len_e = min(len(pen_vals)-1, len(fitted_e))
        if min_len_p > 0:
            mape_pem = _mape(pem_vals[1:min_len_p+1], fitted_p[:min_len_p])
        if min_len_e > 0:
            mape_pen = _mape(pen_vals[1:min_len_e+1], fitted_e[:min_len_e])

        # Confidence level [P1]
        avg_r2  = (r2_pem + r2_pen) / 2
        cv_pem  = (monthly['Pemasukan'].std() / avg_pem * 100) if avg_pem > 0 else 100
        avg_mape = (mape_pem + mape_pen) / 2
        if avg_r2 >= 0.80 and cv_pem < 20 and avg_mape < 15:
            conf = "Tinggi 🎯"
        elif avg_r2 >= 0.50 or cv_pem < 40 or avg_mape < 30:
            conf = "Sedang 📊"
        else:
            conf = "Rendah ⚠️"

        model_name = f"Holt's DES (α={best_ap}, β={best_bp})"

    # Bulan-nama untuk forecast 3 bulan
    def next_month_name(base_bln, offset):
        bln = ((base_bln - 1 + offset) % 12) + 1
        thn = tahun + (last_bln - 1 + offset) // 12 - (last_bln - 1) // 12
        if last_bln == 12 and offset > 0:
            thn = tahun + (offset - 1) // 12 + 1
        return MONTH_ID[bln], bln

    fc3 = []
    for i in range(3):
        bn_nama, bn_num = next_month_name(last_bln, i + 1)
        fp = forecast_pem[i] if i < len(forecast_pem) else pred_pem
        fe = forecast_pen[i] if i < len(forecast_pen) else pred_pen
        ci_p = ci_pem_80 * ((i+1)**0.5)
        ci_e = ci_pen_80 * ((i+1)**0.5)
        fc3.append({
            'bulan': bn_nama, 'bulan_num': bn_num,
            'pemasukan': round(fp), 'pengeluaran': round(fe),
            'surplus': round(fp - fe),
            'ci_pem_80': round(ci_p), 'ci_pen_80': round(ci_e),
        })

    return {
        'pemasukan':   round(pred_pem),
        'pengeluaran': round(pred_pen),
        'surplus':     round(pred_pem - pred_pen),
        'saldo':       round(last_sal + pred_pem - pred_pen),
        'bulan_nama':  fc3[0]['bulan'] if fc3 else MONTH_ID[next_bln],
        'tahun':       next_thn,
        'confidence':  conf,
        'model':       model_name,
        'mae_pem':     mae_pem,
        'mae_pen':     mae_pen,
        'mape_pem':    mape_pem,
        'mape_pen':    mape_pen,
        'r2_pem':      r2_pem,
        'r2_pen':      r2_pen,
        'avg_pem':     avg_pem,
        'avg_pen':     avg_pen,
        'n_bulan':     n,
        'ci_pem_80':   ci_pem_80,
        'ci_pem_95':   ci_pem_95,
        'ci_pen_80':   ci_pen_80,
        'ci_pen_95':   ci_pen_95,
        'forecast_3bln': fc3,
    }


# ============================================================
# HEALTH SCORE
# ============================================================

def hitung_health(monthly: pd.DataFrame) -> dict:
    ti, to = float(monthly['Pemasukan'].sum()), float(monthly['Pengeluaran'].sum())
    n = len(monthly)

    tabungan  = max(0.0, min(100.0, (ti - to) / ti * 100)) if ti > 0 else 0.0
    stabilitas = float((monthly['Surplus'] >= 0).sum()) / n * 100 if n > 0 else 0.0

    if n >= 2:
        f = float(monthly['SaldoAkhir'].iloc[0])
        l = float(monthly['SaldoAkhir'].iloc[-1])
        if f <= 0:
            ref = max(1.0, float(monthly['Pengeluaran'].mean()))
            pertumbuhan = max(0.0, min(100.0, (l / ref) * 50))
        else:
            pct_grow = (l - f) / f * 100
            pertumbuhan = max(0.0, min(100.0, pct_grow * 2))
    else:
        pertumbuhan = 50.0

    avg_pen  = float(monthly['Pengeluaran'].mean())
    last_sal = float(monthly['SaldoAkhir'].iloc[-1]) if n > 0 else 0.0
    if avg_pen > 0 and last_sal > 0:
        bln_aman = last_sal / avg_pen
        cadangan = min(100.0, bln_aman / 3 * 100)
    elif last_sal <= 0:
        bln_aman, cadangan = 0.0, 0.0
    else:
        bln_aman, cadangan = 0.0, 50.0

    return {
        'tabungan':    round(tabungan,    1),
        'stabilitas':  round(stabilitas,  1),
        'pertumbuhan': round(pertumbuhan, 1),
        'cadangan':    round(cadangan,    1),
        'bln_aman':    round(bln_aman,    1),
    }


# ============================================================
# SIDEBAR — v3.0 Bersih
# ============================================================

@st.cache_data(show_spinner=False)
def _baca_cepat(file_bytes: bytes) -> pd.DataFrame:
    """Baca file Excel minimal untuk mengisi filter tahun & bulan di sidebar."""
    try:
        raw = pd.read_excel(io.BytesIO(file_bytes), header=0)
        if not any(k in ' '.join(raw.columns.str.lower()) for k in ['tanggal','date','tgl']):
            raw = pd.read_excel(io.BytesIO(file_bytes), header=1)
        return bersihkan_data(raw)
    except Exception:
        return pd.DataFrame()


def render_sidebar():
    with st.sidebar:
        # Logo
        st.markdown("""
        <div class="sidebar-logo">
            <span class="logo-icon">🌾</span>
            <span class="logo-title">Dashboard Keuangan</span>
            <span class="logo-sub">Karang Taruna Desa · v3.0</span>
        </div>
        """, unsafe_allow_html=True)

        # Upload
        st.markdown("### 📂 Data Keuangan")
        uploaded = st.file_uploader(
            "Upload file rekap Excel (.xlsx/.xls)",
            type=['xlsx','xls'],
            help="Kolom wajib: Tanggal, Pemasukan, Pengeluaran. Opsional: Saldo, Keterangan, Kategori"
        )
        tahun_sel = None
        bulan_sel = "Semua Bulan"

        if uploaded:
            # Baca data langsung dari file — tidak bergantung session_state
            df_preview = _baca_cepat(uploaded.getvalue())

            if not df_preview.empty:
                # Simpan ke session_state agar main() pakai tanpa baca ulang
                st.session_state['df_clean'] = df_preview

                st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
                st.markdown("### 🔍 Filter Tampilan")

                tahun_list = sorted(df_preview['Tahun'].unique(), reverse=True)
                tahun_sel  = st.selectbox(
                    "📅 Filter Tahun",
                    options=tahun_list,
                    format_func=lambda x: f"📆 {x}",
                    help="Pilih tahun data yang ingin ditampilkan"
                )

                bulan_tersedia = sorted(
                    df_preview[df_preview['Tahun'] == tahun_sel]['Bulan'].unique()
                )
                bulan_opts = ['Semua Bulan'] + [MONTH_ID[b] for b in bulan_tersedia]
                bulan_sel  = st.selectbox(
                    "🗓️ Filter Bulan",
                    options=bulan_opts,
                    help="Pilih bulan spesifik atau tampilkan semua bulan"
                )

                # Info ringkas sesuai filter aktif
                df_fil = df_preview[df_preview['Tahun'] == tahun_sel]
                if bulan_sel != 'Semua Bulan':
                    bln_num = {v: k for k, v in MONTH_ID.items()}.get(bulan_sel)
                    if bln_num:
                        df_fil = df_fil[df_fil['Bulan'] == bln_num]

                total_pem     = df_fil['Pemasukan'].sum()
                total_pen     = df_fil['Pengeluaran'].sum()
                n_trx         = len(df_fil)
                n_bln         = df_fil['Bulan'].nunique()
                surplus       = total_pem - total_pen
                label_periode = bulan_sel if bulan_sel != 'Semua Bulan' else f"Tahun {tahun_sel}"

                st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="sidebar-info">
                    <strong>📊 {label_periode}</strong><br>
                    🔢 {n_trx} transaksi · {n_bln} bulan<br>
                    📥 Masuk: <strong>{rupiah(total_pem, True)}</strong><br>
                    📤 Keluar: <strong>{rupiah(total_pen, True)}</strong><br>
                    💎 Selisih: <strong>{rupiah(surplus, True)}</strong><br>
                    <br>
                    <strong>🕐 Diperbarui</strong><br>
                    {datetime.now().strftime('%d %b %Y · %H:%M WIB')}
                    <br>
                    <span class="sidebar-badge">{'✅ Surplus' if surplus >= 0 else '⚠️ Defisit'}</span>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                <hr class="sidebar-divider">
                <div class="sidebar-nav">
                    <strong style="color:#A9DFBF;">🗺️ Navigasi Tab</strong><br>
                    📌 <strong>Ringkasan</strong> — KPI & Kesehatan<br>
                    📈 <strong>Grafik</strong> — Visualisasi data<br>
                    🔮 <strong>Prediksi</strong> — Forecast akurat<br>
                    📋 <strong>Detail</strong> — Tabel & transaksi<br>
                    💡 <strong>Tips</strong> — Saran keuangan
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("❌ Gagal membaca file. Pastikan ada kolom Tanggal, Pemasukan, Pengeluaran.")

        # Footer sidebar (tidak pakai position:absolute — [L2])
        st.markdown("""
        <div class="sidebar-footer">
            🌿 Transparansi &amp; Kemajuan Bersama
        </div>
        """, unsafe_allow_html=True)

    return uploaded, tahun_sel, bulan_sel


# ============================================================
# WELCOME SCREEN
# ============================================================

def render_welcome():
    st.markdown("""
    <div class="welcome-box">
        <span class="welcome-icon">📊</span>
        <div class="welcome-title">Selamat Datang di Dashboard Keuangan!</div>
        <div class="welcome-desc">
            Dashboard ini membantu warga desa dan pemuda Karang Taruna
            memantau keuangan organisasi secara mudah, jelas, dan transparan —
            tanpa perlu memahami ilmu data.
        </div>
        <div class="welcome-steps">
            <div class="step-item">
                <div class="step-num">1</div>
                <div class="step-title">📂 Upload Data Excel</div>
                <div class="step-desc">Klik tombol upload di sidebar kiri, lalu pilih file Excel keuangan Anda.</div>
            </div>
            <div class="step-item">
                <div class="step-num">2</div>
                <div class="step-title">🔍 Pilih Tahun / Bulan</div>
                <div class="step-desc">Pilih tahun atau bulan yang ingin Anda lihat di filter sidebar.</div>
            </div>
            <div class="step-item">
                <div class="step-num">3</div>
                <div class="step-title">📈 Lihat Hasilnya</div>
                <div class="step-desc">Grafik, prediksi 3 bulan, dan saran keuangan tampil otomatis.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">📋 Format File Excel yang Dibutuhkan</div>',
                unsafe_allow_html=True)
    contoh = pd.DataFrame({
        'Tanggal':     ['01/01/2024','07/01/2024','14/01/2024','21/01/2024'],
        'Pemasukan':   ['500.000','750.000','1.200.000','300.000'],
        'Pengeluaran': ['200.000','400.000','600.000','150.000'],
        'Saldo':       ['300.000','650.000','1.250.000','1.400.000'],
        'Keterangan':  ['Iuran Warga','Donasi','Dana Desa','Arisan'],
        'Kategori':    ['Iuran','Donasi','Dana Pemerintah','Sosial'],
    })
    st.dataframe(contoh, use_container_width=True, hide_index=True)
    st.info(
        "💡 **Tips:** Kolom **Tanggal, Pemasukan, Pengeluaran** wajib ada. "
        "Kolom **Saldo, Keterangan, Kategori** boleh ada atau tidak ada. "
        "Angka boleh pakai titik ribuan (1.000.000) atau tanpa titik (1000000)."
    )


# ============================================================
# KPI CARDS
# ============================================================

def _delta_label(monthly, col):
    if len(monthly) < 2: return ""
    last = monthly[col].iloc[-1]
    prev = monthly[col].iloc[-2]
    if prev == 0: return ""
    d = (last - prev) / prev * 100
    arrow = "▲" if d >= 0 else "▼"
    color = "#27AE60" if d >= 0 else "#E74C3C"
    return (f'<div class="kpi-delta" style="color:{color};">'
            f'{arrow} {abs(d):.1f}% vs bulan lalu</div>')


def render_kpi(monthly, pred):
    ti  = monthly['Pemasukan'].sum()
    to  = monthly['Pengeluaran'].sum()
    sur = ti - to
    sal = monthly['SaldoAkhir'].iloc[-1]
    pct = (sur / ti * 100) if ti > 0 else 0.0

    kartu = [
        {'w': CLR_GREEN, 'icon':'💰',
         'label':'Saldo Kas Sekarang', 'nilai': rupiah(sal, True),
         'sub':'Uang tunai tersedia', 'delta':''},
        {'w': CLR_BLUE,  'icon':'📥',
         'label':'Total Pemasukan',    'nilai': rupiah(ti, True),
         'sub':'Semua uang masuk', 'delta': _delta_label(monthly,'Pemasukan')},
        {'w': CLR_RED,   'icon':'📤',
         'label':'Total Pengeluaran',  'nilai': rupiah(to, True),
         'sub':'Semua uang keluar', 'delta': _delta_label(monthly,'Pengeluaran')},
        {'w': CLR_GREEN if sur >= 0 else CLR_RED,
         'icon':'💚' if sur >= 0 else '❤️',
         'label':'Surplus / Defisit',
         'nilai': rupiah(sur, True),
         'sub':'Keuangan sehat! 👍' if sur >= 0 else 'Pengeluaran > pemasukan ⚠️',
         'delta':''},
        {'w': CLR_GOLD,  'icon':'🏦',
         'label':'Persentase Ditabung', 'nilai': f"{max(0, pct):.1f}%",
         'sub': f"dari pemasukan{'  ·  >20% tercapai 👍' if pct >= 20 else '  ·  Target >20%'}",
         'delta':''},
    ]

    cols = st.columns(5)
    for col, k in zip(cols, kartu):
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="border-top-color:{k['w']};">
                <span class="kpi-icon">{k['icon']}</span>
                <div class="kpi-label">{k['label']}</div>
                <div class="kpi-value">{k['nilai']}</div>
                <div class="kpi-sub">{k['sub']}</div>
                {k['delta']}
            </div>""", unsafe_allow_html=True)


# ============================================================
# PREDIKSI BOX + FORECAST 3 BULAN [N1, P3]
# ============================================================

def render_prediksi(pred, monthly):
    sur = pred['surplus']
    t   = "+" if sur >= 0 else ""
    dp  = ((pred['pemasukan'] - pred['avg_pem']) / pred['avg_pem'] * 100) if pred['avg_pem'] > 0 else 0
    de  = ((pred['pengeluaran']- pred['avg_pen'])/ pred['avg_pen'] * 100) if pred['avg_pen'] > 0 else 0

    # Kotak prediksi bulan 1
    st.markdown(f"""
    <div class="pred-box">
        <div class="pred-box-title">🔮 Prediksi Keuangan Bulan {pred['bulan_nama']} {pred['tahun']}</div>
        <div class="pred-box-sub">
            Model: <strong>{pred['model']}</strong> &nbsp;·&nbsp;
            Data: {pred['n_bulan']} bulan &nbsp;·&nbsp;
            Kepercayaan: <strong>{pred['confidence']}</strong>
        </div>
        <div class="pred-grid">
            <div class="pred-item">
                <span class="pred-item-icon">📥</span>
                <div class="pred-item-label">Perkiraan Pemasukan</div>
                <div class="pred-item-value">{rupiah(pred['pemasukan'],True)}</div>
                <div class="pred-item-delta">{"▲" if dp>=0 else "▼"} {abs(dp):.1f}% vs rata-rata</div>
            </div>
            <div class="pred-item">
                <span class="pred-item-icon">📤</span>
                <div class="pred-item-label">Perkiraan Pengeluaran</div>
                <div class="pred-item-value">{rupiah(pred['pengeluaran'],True)}</div>
                <div class="pred-item-delta">{"▲" if de>=0 else "▼"} {abs(de):.1f}% vs rata-rata</div>
            </div>
            <div class="pred-item">
                <span class="pred-item-icon">{"✅" if sur>=0 else "⚠️"}</span>
                <div class="pred-item-label">Perkiraan {"Surplus" if sur>=0 else "Defisit"}</div>
                <div class="pred-item-value">{t}{rupiah(sur,True)}</div>
                <div class="pred-item-delta">{"Keuangan aman 👍" if sur>=0 else "Siapkan cadangan!"}</div>
            </div>
            <div class="pred-item">
                <span class="pred-item-icon">💰</span>
                <div class="pred-item-label">Prediksi Saldo Akhir</div>
                <div class="pred-item-value">{rupiah(pred['saldo'],True)}</div>
                <div class="pred-item-delta">Setelah bulan {pred['bulan_nama']}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if pred['n_bulan'] < 3:
        st.warning(
            f"⚠️ **Data hanya {pred['n_bulan']} bulan** — prediksi masih kasar. "
            "Semakin banyak data, prediksi semakin akurat."
        )

    # Forecast 3 Bulan [P3]
    st.markdown('<div class="section-title">📅 Forecast 3 Bulan ke Depan</div>',
                unsafe_allow_html=True)
    fc3 = pred.get('forecast_3bln', [])
    if fc3:
        cols = st.columns(3)
        for i, fc in enumerate(fc3):
            with cols[i]:
                sur_fc = fc['surplus']
                clr = CLR_GREEN if sur_fc >= 0 else CLR_RED
                ci_p = fc.get('ci_pem_80', 0)
                ci_e = fc.get('ci_pen_80', 0)
                st.markdown(f"""
                <div class="kpi-card" style="border-top-color:{clr};">
                    <span class="kpi-icon">{'📥' if i==0 else ('📆' if i==1 else '🗓️')}</span>
                    <div class="kpi-label">Bulan {i+1} — {fc['bulan']}</div>
                    <div class="kpi-value" style="color:{clr};">
                        {'+'if sur_fc>=0 else ''}{rupiah(sur_fc, True)}
                    </div>
                    <div class="kpi-sub">
                        📥 {rupiah(fc['pemasukan'],True)}<br>
                        📤 {rupiah(fc['pengeluaran'],True)}
                    </div>
                    <div class="kpi-delta" style="color:#8BA88F; font-size:0.60rem; margin-top:5px;">
                        CI±{rupiah(ci_p, True)} (80%)
                    </div>
                </div>""", unsafe_allow_html=True)

    # Grafik forecast dengan confidence band [P4]
    if len(monthly) >= 3:
        st.plotly_chart(chart_forecast(monthly, pred), use_container_width=True)

    # Penjelasan prediksi — pakai HTML box (bukan expander agar tidak ada artefak teks)
    ikon_sur  = "✅" if sur >= 0 else "⚠️"
    pesan_sur = (
        "Lebih banyak uang masuk dari keluar — keuangan aman!"
        if sur >= 0 else
        "Pengeluaran lebih besar dari pemasukan — siapkan tabungan cadangan!"
    )
    st.markdown(f"""
    <div style="
        background:#E8F8EF; border-radius:14px; padding:18px 20px;
        border-left:5px solid #27AE60; margin-top:4px;
        font-size:0.85rem; color:#1A2D20; line-height:1.75;
    ">
        <div style="font-weight:800; font-size:0.92rem; color:#0D5C2E; margin-bottom:10px;">
            💬 Ringkasan Prediksi — {pred['bulan_nama']} {pred['tahun']}
        </div>
        💵 Perkiraan <strong>Pemasukan</strong>: <strong>{rupiah(pred['pemasukan'])}</strong><br>
        💸 Perkiraan <strong>Pengeluaran</strong>: <strong>{rupiah(pred['pengeluaran'])}</strong><br>
        {ikon_sur} {pesan_sur}<br>
        💰 Prediksi <strong>Saldo Akhir</strong>: <strong>{rupiah(pred['saldo'])}</strong><br>
        📊 Kepercayaan Model: <strong>{pred['confidence']}</strong>
        <div style="
            margin-top:12px; padding-top:10px;
            border-top:1px dashed #A9DFBF;
            font-size:0.76rem; color:#5A7260;
        ">
            ⚠️ Ini perkiraan berbasis tren historis. Tetap catat semua transaksi
            agar prediksi bulan berikutnya semakin akurat!
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Anomali [A]
    anomali = deteksi_anomali(monthly)
    if anomali:
        st.markdown('<div class="section-title">⚠️ Anomali Keuangan Terdeteksi</div>',
                    unsafe_allow_html=True)
        for a in anomali:
            icon = "📈" if "TINGGI" in a['arah'] else "📉"
            st.markdown(f"""
            <div class="anomali-box">
                <div class="anomali-icon">{icon}</div>
                <div>
                    <div class="anomali-title">Bulan {a['bulan']} — {a['kolom']} {a['arah']}</div>
                    <div class="anomali-text">
                        Nilai tercatat: <strong>{rupiah(a['nilai'], True)}</strong> —
                        sangat jauh dari rata-rata (z-score = {a['z']:.1f}).
                        Periksa kembali transaksi bulan ini.
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)


# ============================================================
# GRAFIK
# ============================================================

def chart_saldo(monthly):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly['BulanNama'], y=monthly['SaldoAkhir'],
        mode='lines+markers', name='Saldo Aktual',
        line=dict(color=CLR_GREEN, width=3),
        fill='tozeroy', fillcolor='rgba(39,174,96,0.12)',
        marker=dict(size=9, color=CLR_GREEN),
        hovertemplate='<b>%{x}</b><br>Saldo: Rp %{y:,.0f}<extra></extra>'
    ))
    if len(monthly) >= 2:
        xs    = np.arange(len(monthly), dtype=float).reshape(-1,1)
        trend = LinearRegression().fit(xs, monthly['SaldoAkhir'].values).predict(xs)
        fig.add_trace(go.Scatter(
            x=monthly['BulanNama'], y=trend,
            mode='lines', name='Tren',
            line=dict(color=CLR_GOLD, width=2, dash='dot'),
            hovertemplate='<b>%{x}</b><br>Tren: Rp %{y:,.0f}<extra></extra>'
        ))
    fig.update_layout(
        title=dict(text="📈 Pergerakan Saldo Kas + Garis Tren",
                   font=dict(size=13, color=CLR_DARK)),
        yaxis=dict(tickformat=",", tickprefix="Rp ", gridcolor="#E8F8EF"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        legend=dict(orientation="h", y=1.08, x=0),
        **LAYOUT
    )
    return fig


def chart_bulanan(monthly):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=monthly['BulanPendek'], y=monthly['Pemasukan'],
        name='📥 Pemasukan', marker_color=CLR_GREEN, marker_line_width=0,
        hovertemplate='%{x}<br>Pemasukan: Rp %{y:,.0f}<extra></extra>'
    ))
    fig.add_trace(go.Bar(
        x=monthly['BulanPendek'], y=monthly['Pengeluaran'],
        name='📤 Pengeluaran', marker_color=CLR_RED, marker_line_width=0,
        hovertemplate='%{x}<br>Pengeluaran: Rp %{y:,.0f}<extra></extra>'
    ))
    fig.update_layout(
        title=dict(text="📊 Pemasukan vs Pengeluaran",
                   font=dict(size=13, color=CLR_DARK)),
        barmode='group',
        yaxis=dict(tickformat=",", gridcolor="#E8F8EF"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        legend=dict(orientation="h", y=1.08, x=0),
        **LAYOUT
    )
    return fig


def chart_surplus(monthly):
    colors = [CLR_GREEN if s >= 0 else CLR_RED for s in monthly['Surplus']]
    fig = go.Figure(go.Bar(
        x=monthly['BulanPendek'], y=monthly['Surplus'],
        marker_color=colors, marker_line_width=0,
        hovertemplate='%{x}<br>Rp %{y:,.0f}<extra></extra>'
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#1A2D20", line_width=1.5,
                  annotation_text="Titik Impas",
                  annotation_position="top right",
                  annotation_font_size=10)
    fig.update_layout(
        title=dict(text="💎 Surplus / Defisit Tiap Bulan",
                   font=dict(size=13, color=CLR_DARK)),
        yaxis=dict(tickformat=",", gridcolor="#E8F8EF"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        showlegend=False, **LAYOUT
    )
    return fig


def chart_donut(ti, to):
    if ti <= 0 and to <= 0:
        return go.Figure()
    if ti >= to:
        sur    = ti - to
        labels = ['Pengeluaran', 'Surplus / Ditabung']
        values = [to, sur]
        colors = [CLR_RED, CLR_GREEN]
        pct    = sur / ti * 100 if ti > 0 else 0
        center = f"<b>{pct:.1f}%</b><br><span style='font-size:10px'>ditabung</span>"
    else:
        kelebihan = to - ti
        labels = ['Ditanggung Pemasukan', 'Kelebihan Pengeluaran ⚠️']
        values = [ti, kelebihan]
        colors = [CLR_BLUE, CLR_RED]
        pct    = kelebihan / to * 100 if to > 0 else 0
        center = f"<b style='color:#E74C3C;'>⚠️</b><br><span style='font-size:10px'>defisit {pct:.1f}%</span>"

    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.62,
        marker=dict(colors=colors, line=dict(color='white', width=3)),
        hovertemplate='<b>%{label}</b><br>Rp %{value:,.0f} (%{percent})<extra></extra>'
    ))
    fig.add_annotation(
        text=center, x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color=CLR_DARK)
    )
    fig.update_layout(
        title=dict(text="🥧 Komposisi Penggunaan Uang",
                   font=dict(size=13, color=CLR_DARK)),
        legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center"),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Poppins, sans-serif", size=11),
        margin=dict(l=12, r=12, t=44, b=12),
    )
    return fig


def chart_forecast(monthly, pred):
    """[P4] Grafik forecast pemasukan + CI band."""
    n = len(monthly)
    fc3 = pred.get('forecast_3bln', [])
    if not fc3:
        return go.Figure()

    hist_x = monthly['BulanNama'].tolist()
    hist_p = monthly['Pemasukan'].tolist()
    hist_e = monthly['Pengeluaran'].tolist()

    fc_x   = [f['bulan'] for f in fc3]
    fc_p   = [f['pemasukan'] for f in fc3]
    fc_e   = [f['pengeluaran'] for f in fc3]

    ci_p_hi = [f['pemasukan'] + f.get('ci_pem_80',0) * ((i+1)**0.5) for i, f in enumerate(fc3)]
    ci_p_lo = [max(0, f['pemasukan'] - f.get('ci_pem_80',0) * ((i+1)**0.5)) for i, f in enumerate(fc3)]
    ci_e_hi = [f['pengeluaran'] + f.get('ci_pen_80',0) * ((i+1)**0.5) for i, f in enumerate(fc3)]
    ci_e_lo = [max(0, f['pengeluaran'] - f.get('ci_pen_80',0) * ((i+1)**0.5)) for i, f in enumerate(fc3)]

    fig = make_subplots(rows=1, cols=2,
        subplot_titles=("Forecast Pemasukan", "Forecast Pengeluaran"),
        horizontal_spacing=0.12)

    # ── Panel 1: Pemasukan ──
    fig.add_trace(go.Scatter(
        x=hist_x, y=hist_p, name='Aktual Pemasukan',
        line=dict(color=CLR_GREEN, width=2.5),
        marker=dict(size=7), mode='lines+markers',
        hovertemplate='%{x}: Rp %{y:,.0f}<extra></extra>'
    ), row=1, col=1)
    # CI upper
    fig.add_trace(go.Scatter(
        x=fc_x, y=ci_p_hi, name='CI Upper', mode='lines',
        line=dict(width=0), showlegend=False,
        hoverinfo='skip'
    ), row=1, col=1)
    # CI lower + fill
    fig.add_trace(go.Scatter(
        x=fc_x, y=ci_p_lo, name='CI 80% Pemasukan', mode='lines',
        line=dict(width=0), fill='tonexty',
        fillcolor='rgba(39,174,96,0.18)', showlegend=True,
        hoverinfo='skip'
    ), row=1, col=1)
    # Forecast line
    fig.add_trace(go.Scatter(
        x=fc_x, y=fc_p, name='Forecast Pemasukan', mode='lines+markers',
        line=dict(color=CLR_GREEN, width=2, dash='dot'),
        marker=dict(size=9, symbol='diamond', color=CLR_GREEN),
        hovertemplate='%{x}: Rp %{y:,.0f}<extra></extra>'
    ), row=1, col=1)

    # ── Panel 2: Pengeluaran ──
    fig.add_trace(go.Scatter(
        x=hist_x, y=hist_e, name='Aktual Pengeluaran',
        line=dict(color=CLR_RED, width=2.5),
        marker=dict(size=7), mode='lines+markers',
        hovertemplate='%{x}: Rp %{y:,.0f}<extra></extra>'
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=fc_x, y=ci_e_hi, mode='lines',
        line=dict(width=0), showlegend=False, hoverinfo='skip'
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=fc_x, y=ci_e_lo, name='CI 80% Pengeluaran', mode='lines',
        line=dict(width=0), fill='tonexty',
        fillcolor='rgba(231,76,60,0.18)', showlegend=True,
        hoverinfo='skip'
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=fc_x, y=fc_e, name='Forecast Pengeluaran', mode='lines+markers',
        line=dict(color=CLR_RED, width=2, dash='dot'),
        marker=dict(size=9, symbol='diamond', color=CLR_RED),
        hovertemplate='%{x}: Rp %{y:,.0f}<extra></extra>'
    ), row=1, col=2)

    fig.update_layout(
        title=dict(text=f"🔮 Visualisasi Forecast 3 Bulan ({pred['model']})",
                   font=dict(size=13, color=CLR_DARK)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Poppins, sans-serif", size=11, color="#1A2D20"),
        margin=dict(l=12, r=12, t=80, b=80),
        legend=dict(
            orientation="h",
            y=-0.22, x=0.5, xanchor="center",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#D5F5E3", borderwidth=1,
            font=dict(size=10)
        )
    )
    # Perbaiki posisi subtitle agar tidak bertabrakan dengan judul
    for ann in fig.layout.annotations:
        ann.update(y=ann.y - 0.04, font=dict(size=11, color="#1A2D20"))
    fig.update_yaxes(tickformat=",", tickprefix="Rp ", gridcolor="#E8F8EF")
    fig.update_xaxes(gridcolor="rgba(0,0,0,0)")
    return fig


def chart_akurasi(monthly):
    n = len(monthly)
    if n < 4: return None
    xs = np.arange(1, n+1, dtype=float).reshape(-1,1)
    sp = max(2, int(n * 0.75))
    if sp >= n: return None

    mp = LinearRegression().fit(xs[:sp], monthly['Pemasukan'].values[:sp])
    me = LinearRegression().fit(xs[:sp], monthly['Pengeluaran'].values[:sp])
    pp = mp.predict(xs[sp:])
    pe = me.predict(xs[sp:])
    xl = monthly['BulanNama'].values[sp:]

    fig = make_subplots(rows=1, cols=2,
        subplot_titles=("Akurasi Prediksi Pemasukan","Akurasi Prediksi Pengeluaran"))
    for col_idx, (actuals, preds, clr, name) in enumerate([
        (monthly['Pemasukan'].values[sp:],  pp, CLR_GREEN, 'Pemasukan'),
        (monthly['Pengeluaran'].values[sp:], pe, CLR_RED,   'Pengeluaran'),
    ], 1):
        fig.add_trace(go.Scatter(
            x=xl, y=actuals, name=f'Aktual {name}',
            line=dict(color=clr, width=2.5),
            marker=dict(size=8), mode='lines+markers',
            showlegend=(col_idx==1)
        ), row=1, col=col_idx)
        fig.add_trace(go.Scatter(
            x=xl, y=preds, name=f'Prediksi {name}',
            line=dict(color=clr, width=2.5, dash='dot'),
            marker=dict(size=8, symbol='diamond'),
            showlegend=(col_idx==1)
        ), row=1, col=col_idx)

    fig.update_layout(
        title=dict(text="🤖 Akurasi Model: Data Aktual vs Prediksi",
                   font=dict(size=13, color=CLR_DARK)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Poppins, sans-serif", size=11, color="#1A2D20"),
        margin=dict(l=12, r=12, t=80, b=80),
        legend=dict(
            orientation="h",
            y=-0.22, x=0.5, xanchor="center",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#D5F5E3", borderwidth=1,
            font=dict(size=10)
        )
    )
    for ann in fig.layout.annotations:
        ann.update(y=ann.y - 0.04, font=dict(size=11, color="#1A2D20"))
    fig.update_yaxes(tickformat=",", tickprefix="Rp ", gridcolor="#E8F8EF")
    fig.update_xaxes(gridcolor="rgba(0,0,0,0)")
    return fig


def chart_kategori(df, tahun):
    d = df[df['Tahun'] == tahun]
    if 'Kategori' not in d.columns:
        return None
    unique_kat = d['Kategori'].dropna().unique()
    if len(unique_kat) <= 1 and (len(unique_kat) == 0 or unique_kat[0] == 'Umum'):
        return None
    kat = (
        d.groupby('Kategori')
        .agg(Pengeluaran=('Pengeluaran','sum'), Pemasukan=('Pemasukan','sum'))
        .reset_index().sort_values('Pengeluaran', ascending=True)
    )
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=kat['Kategori'], x=kat['Pengeluaran'],
        orientation='h', name='📤 Pengeluaran',
        marker_color=CLR_RED, marker_line_width=0,
        hovertemplate='%{y}<br>Pengeluaran: Rp %{x:,.0f}<extra></extra>'
    ))
    fig.add_trace(go.Bar(
        y=kat['Kategori'], x=kat['Pemasukan'],
        orientation='h', name='📥 Pemasukan',
        marker_color=CLR_GREEN, marker_line_width=0,
        hovertemplate='%{y}<br>Pemasukan: Rp %{x:,.0f}<extra></extra>'
    ))
    fig.update_layout(
        title=dict(text="📂 Pemasukan & Pengeluaran per Kategori",
                   font=dict(size=13, color=CLR_DARK)),
        barmode='group',
        xaxis=dict(tickformat=",", gridcolor="#E8F8EF"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        legend=dict(orientation="h", y=1.08, x=0),
        **LAYOUT
    )
    return fig


def chart_yoy(df, tahun_list):
    """[N2] Year-over-Year comparison chart."""
    records = []
    for th in tahun_list:
        m = agregasi_bulanan(df, th)
        if m.empty: continue
        for _, row in m.iterrows():
            records.append({
                'Tahun': str(th), 'Bulan': row['Bulan'],
                'BulanPendek': row['BulanPendek'],
                'Pemasukan': row['Pemasukan'], 'Pengeluaran': row['Pengeluaran'],
            })
    if not records:
        return None
    df_yoy = pd.DataFrame(records)
    colors_list = [CLR_GREEN, CLR_BLUE, CLR_GOLD, CLR_ORANGE]
    fig = go.Figure()
    for i, th in enumerate(tahun_list):
        sub = df_yoy[df_yoy['Tahun']==str(th)].sort_values('Bulan')
        clr = colors_list[i % len(colors_list)]
        fig.add_trace(go.Scatter(
            x=sub['BulanPendek'], y=sub['Pemasukan'],
            mode='lines+markers', name=f'Pemasukan {th}',
            line=dict(color=clr, width=2),
            marker=dict(size=7),
            hovertemplate=f'%{{x}} {th}: Rp %{{y:,.0f}}<extra></extra>'
        ))
    fig.update_layout(
        title=dict(text="📊 Perbandingan Pemasukan Tahun ke Tahun (YoY)",
                   font=dict(size=13, color=CLR_DARK)),
        yaxis=dict(tickformat=",", tickprefix="Rp ", gridcolor="#E8F8EF"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        legend=dict(orientation="h", y=1.08, x=0),
        **LAYOUT
    )
    return fig


# ============================================================
# HEALTH BARS
# ============================================================

def render_health(monthly):
    h = hitung_health(monthly)
    bars = [
        {'label':'💰 Tingkat Tabungan', 'val': h['tabungan'],
         'sub': f"{h['tabungan']:.1f}% dari pemasukan berhasil ditabung  ·  Target sehat: di atas 20%"},
        {'label':'📅 Bulan Keuangan Positif', 'val': h['stabilitas'],
         'sub': f"{h['stabilitas']:.0f}% bulan berjalan surplus  ·  Semakin tinggi semakin stabil"},
        {'label':'📈 Pertumbuhan Saldo', 'val': h['pertumbuhan'],
         'sub': f"Skor pertumbuhan saldo: {h['pertumbuhan']:.1f}%  ·  Tandanya kas makin bertambah"},
        {'label':'🛡️ Cadangan Kas', 'val': h['cadangan'],
         'sub': f"Cukup untuk ≈ {h['bln_aman']} bulan ke depan  ·  Idealnya ≥ 2–3 bulan"},
    ]
    col_bars, col_score = st.columns([3, 1])
    with col_bars:
        for b in bars:
            c = clr_health(b['val'])
            st.markdown(f"""
            <div class="health-row">
                <div class="health-meta">
                    <span class="health-label">{b['label']}</span>
                    <span class="health-score" style="color:{c};">{emoji_health(b['val'])} {b['val']:.0f}%</span>
                </div>
                <div class="health-bar-bg">
                    <div class="health-bar-fill" style="width:{b['val']}%; background:{c};"></div>
                </div>
                <div class="health-sub">{b['sub']}</div>
            </div>""", unsafe_allow_html=True)

    with col_score:
        avg = float(np.mean([h['tabungan'],h['stabilitas'],h['pertumbuhan'],h['cadangan']]))
        c   = clr_health(avg)
        emj = "😄" if avg >= 70 else ("😐" if avg >= 40 else "😟")
        sts = "Sangat Sehat" if avg >= 70 else ("Perlu Perhatian" if avg >= 40 else "Kritis")
        st.markdown(f"""
        <div class="score-circle" style="background:{c}18; border:3px solid {c};">
            <div class="score-emoji">{emj}</div>
            <div class="score-num" style="color:{c};">{avg:.0f}%</div>
            <div class="score-label" style="color:{c};">{sts}</div>
            <div class="score-sub">Skor Kesehatan<br>Keuangan Keseluruhan</div>
        </div>""", unsafe_allow_html=True)


# ============================================================
# A/B TEST
# ============================================================

def render_ab(df, tahun):
    dy = df[df['Tahun'] == tahun]
    a  = dy[dy['Hari'] <= 15]['Pemasukan'].dropna()
    k  = dy[dy['Hari'] >  15]['Pemasukan'].dropna()

    aa  = float(a.mean()) if len(a) > 0 else 0.0
    ak  = float(k.mean()) if len(k) > 0 else 0.0
    aa  = 0.0 if np.isnan(aa) else aa
    ak  = 0.0 if np.isnan(ak) else ak
    denom = max(1.0, (aa + ak) / 2)
    pct   = abs(aa - ak) / denom * 100

    sig, pv = False, 1.0
    if len(a) >= 2 and len(k) >= 2:
        _, pv = ttest_ind(a, k, equal_var=False)
        sig   = bool(pv < 0.05)

    c1, c2, c3 = st.columns([2, 2, 3])
    with c1:
        st.markdown(f"""
        <div class="ab-card awal">
            <div class="ab-period">📅 Awal Bulan (Tgl 1 – 15)</div>
            <div class="ab-value">{rupiah(aa, True)}</div>
            <div class="ab-desc">Rata-rata pemasukan per transaksi</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="ab-card akhir">
            <div class="ab-period">📅 Akhir Bulan (Tgl 16 – 31)</div>
            <div class="ab-value">{rupiah(ak, True)}</div>
            <div class="ab-desc">Rata-rata pemasukan per transaksi</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        if len(a) < 2 or len(k) < 2:
            st.info(
                f"ℹ️ **Data belum cukup** untuk uji statistik "
                f"(butuh ≥ 2 transaksi per periode). Selisih saat ini: **{pct:.1f}%**."
            )
        elif sig:
            lb = "awal bulan" if aa > ak else "akhir bulan"
            st.warning(
                f"⚠️ **Perbedaan signifikan ({pct:.1f}%)**\n\n"
                f"Pemasukan lebih banyak di **{lb}**. "
                "Rencanakan pengeluaran besar saat kas sedang penuh."
            )
        else:
            st.success(
                f"✅ **Pemasukan merata sepanjang bulan** (selisih hanya {pct:.1f}%)\n\n"
                "Bagus! Arus kas Anda stabil dan mudah dikelola."
            )


# ============================================================
# TIPS
# ============================================================

def render_tips(monthly, pred):
    ti  = monthly['Pemasukan'].sum()
    to  = monthly['Pengeluaran'].sum()
    r   = to / ti if ti > 0 else 1.0
    sur = pred['surplus']
    tips = []

    if r > 0.85:
        tips.append({'icon':'⚠️','w':'#E74C3C','judul':'Pengeluaran Terlalu Besar!',
            'isi':f'Lebih dari 85% pemasukan habis untuk pengeluaran ({r*100:.0f}%). '
                  'Coba kurangi belanja yang tidak mendesak, atau cari sumber pemasukan tambahan.'})
    elif r > 0.70:
        tips.append({'icon':'💡','w':CLR_GOLD,'judul':'Pengeluaran Perlu Diperhatikan',
            'isi':f'Pengeluaran sudah {r*100:.0f}% dari pemasukan. Masih aman, '
                  'tapi mulailah menabung lebih banyak agar ada cadangan dana mendadak.'})
    else:
        tips.append({'icon':'✅','w':'#27AE60','judul':'Keuangan Sangat Sehat!',
            'isi':f'Pengeluaran hanya {r*100:.0f}% dari pemasukan. Luar biasa! '
                  'Pertahankan pola ini dan pertimbangkan menyisihkan sebagian untuk kegiatan produktif desa.'})

    if sur < 0:
        tips.append({'icon':'🔴','w':'#E74C3C','judul':f'Waspadai Bulan {pred["bulan_nama"]}!',
            'isi':f'Prediksi bulan depan: pengeluaran lebih besar dari pemasukan '
                  f'(defisit ≈ {rupiah(abs(sur),True)}). Siapkan dana cadangan dari sekarang!'})
    else:
        tips.append({'icon':'🎉','w':'#27AE60','judul':f'Bulan {pred["bulan_nama"]} Terlihat Bagus!',
            'isi':f'Perkiraan surplus ≈ {rupiah(sur,True)} bulan depan. '
                  'Manfaatkan untuk menambah tabungan atau kegiatan sosial bagi warga desa.'})

    best = monthly.loc[monthly['Pemasukan'].idxmax()]
    tips.append({'icon':'🏆','w':'#2980B9','judul':f'Bulan Terkuat: {best["BulanNama"]}',
        'isi':f'Pemasukan tertinggi terjadi di bulan {best["BulanNama"]} ({rupiah(best["Pemasukan"],True)}). '
              'Jadwalkan kegiatan atau pembelian penting saat kas sedang penuh.'})

    tips.append({'icon':'📒','w':'#27AE60','judul':'Selalu Catat Setiap Transaksi',
        'isi':'Catat semua pemasukan dan pengeluaran tepat waktu dengan tanggal yang benar. '
              'Rekap mingguan sangat membantu agar tidak ada yang terlewat dan laporan selalu akurat.'})

    cols = st.columns(2)
    for i, t in enumerate(tips):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="tip-card" style="border-color:{t['w']};">
                <div class="tip-icon">{t['icon']}</div>
                <div>
                    <div class="tip-title">{t['judul']}</div>
                    <div class="tip-text">{t['isi']}</div>
                </div>
            </div>""", unsafe_allow_html=True)


# ============================================================
# TABEL REKAP
# ============================================================

def render_tabel(monthly, tahun):
    tampil = monthly[[
        'BulanNama','Pemasukan','Pengeluaran','Surplus',
        'RasioBeban','SaldoAkhir','JmlTransaksi','Status'
    ]].copy()
    tampil.columns = [
        '📅 Bulan','📥 Pemasukan','📤 Pengeluaran',
        '💎 Surplus/Defisit','📊 Rasio Beban','💰 Saldo Akhir',
        '🔢 Transaksi','📊 Status'
    ]
    for col in ['📥 Pemasukan','📤 Pengeluaran','💎 Surplus/Defisit','💰 Saldo Akhir']:
        tampil[col] = tampil[col].apply(rupiah)
    tampil['📊 Rasio Beban'] = tampil['📊 Rasio Beban'].apply(lambda x: f"{x:.1f}%")

    st.dataframe(tampil, use_container_width=True, hide_index=True)

    buf = io.BytesIO()
    monthly.to_excel(buf, index=False, engine='openpyxl')
    buf.seek(0)
    st.download_button(
        label     = "⬇️  Download Rekap Excel",
        data      = buf.getvalue(),
        file_name = f"Rekap_Keuangan_Karang_Taruna_{tahun}.xlsx",
        mime      = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


# ============================================================
# AKURASI MODEL — expander terpisah di tab Prediksi [L3]
# ============================================================

def render_akurasi(monthly, pred):
    if len(monthly) < 4:
        st.info("ℹ️ Butuh minimal 4 bulan data untuk menampilkan grafik validasi prediksi.")
        return

    fig_ak = chart_akurasi(monthly)
    if fig_ak:
        st.plotly_chart(fig_ak, use_container_width=True)

    # ── Penjelasan sederhana (tanpa istilah teknis) ──────────
    conf     = pred.get('confidence', 'Rendah ⚠️')
    model    = pred.get('model', '—')
    n_bulan  = pred.get('n_bulan', 0)

    # Tentukan warna & pesan sesuai tingkat kepercayaan
    if 'Tinggi' in conf:
        emoji_conf = "🟢"
        pesan_conf = "Prediksi sangat bisa diandalkan."
    elif 'Sedang' in conf:
        emoji_conf = "🟡"
        pesan_conf = "Prediksi cukup bisa diandalkan, namun bisa sedikit meleset."
    else:
        emoji_conf = "🔴"
        pesan_conf = "Prediksi masih kasar karena data belum cukup banyak."

    st.markdown(f"""
    <div style="
        background: white;
        border-radius: 14px;
        padding: 20px 22px;
        box-shadow: 0 3px 12px rgba(0,0,0,0.07);
        border-left: 5px solid #27AE60;
        margin-top: 8px;
    ">
        <div style="font-size:0.92rem; font-weight:800; color:#0D5C2E; margin-bottom:14px;">
            💬 Seberapa Akurat Prediksi Ini?
        </div>
        <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:14px;">
            <div style="background:#F0FAF4; border-radius:10px; padding:14px; text-align:center;">
                <div style="font-size:1.6rem;">{emoji_conf}</div>
                <div style="font-size:0.72rem; font-weight:700; color:#5A7260; margin:6px 0 3px;">Tingkat Kepercayaan</div>
                <div style="font-size:0.90rem; font-weight:900; color:#0D5C2E;">{conf}</div>
            </div>
            <div style="background:#F0FAF4; border-radius:10px; padding:14px; text-align:center;">
                <div style="font-size:1.6rem;">📅</div>
                <div style="font-size:0.72rem; font-weight:700; color:#5A7260; margin:6px 0 3px;">Dari Data</div>
                <div style="font-size:0.90rem; font-weight:900; color:#0D5C2E;">{n_bulan} bulan</div>
            </div>
            <div style="background:#F0FAF4; border-radius:10px; padding:14px; text-align:center;">
                <div style="font-size:1.6rem;">🤖</div>
                <div style="font-size:0.72rem; font-weight:700; color:#5A7260; margin:6px 0 3px;">Metode</div>
                <div style="font-size:0.90rem; font-weight:900; color:#0D5C2E;">{model}</div>
            </div>
        </div>
        <div style="
            margin-top:14px;
            background:#E8F8EF;
            border-radius:10px;
            padding:12px 16px;
            font-size:0.82rem;
            color:#1A2D20;
            line-height:1.7;
        ">
            📌 <strong>Cara membaca grafik di atas:</strong> Garis <em>penuh</em> adalah data nyata,
            garis <em>putus-putus</em> adalah prediksi. Semakin berdekatan kedua garis, semakin akurat modelnya.<br><br>
            {emoji_conf} {pesan_conf}
            Semakin banyak data yang dicatat setiap bulan, prediksi akan semakin mendekati kenyataan.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# MAIN — Menggunakan st.tabs() [L1]
# ============================================================

def main():
    uploaded, tahun_sel, bulan_sel = render_sidebar()

    tgl = datetime.now().strftime('%d %B %Y')
    st.markdown(f"""
    <div class="main-header">
        <h1>💰 Dashboard Keuangan Karang Taruna</h1>
        <p>Pantau keuangan desa dengan mudah, jelas, dan transparan 🤝 &nbsp;·&nbsp; {tgl}</p>
    </div>
    """, unsafe_allow_html=True)

    if not uploaded:
        render_welcome()
        st.markdown("""
        <div class="footer-box">
            🌾 Dashboard Keuangan Karang Taruna &nbsp;·&nbsp; Transparansi &amp; Kemajuan Bersama
            <span>Upload file Excel di sidebar untuk mulai memantau keuangan Anda</span>
        </div>""", unsafe_allow_html=True)
        return

    # ── Ambil df dari session_state (sudah diisi oleh sidebar via _baca_cepat) ──
    if 'df_clean' in st.session_state:
        df = st.session_state['df_clean']
    else:
        try:
            raw = pd.read_excel(uploaded, header=0)
            if not any(k in ' '.join(raw.columns.str.lower()) for k in ['tanggal','date','tgl']):
                raw = pd.read_excel(uploaded, header=1)
            df = bersihkan_data(raw)
            st.session_state['df_clean'] = df
        except ValueError as ve:
            st.error(f"❌ {ve}")
            return
        except Exception as e:
            st.error(
                f"❌ Gagal membaca file: {e}\n\n"
                "Pastikan ada kolom: **Tanggal | Pemasukan | Pengeluaran**"
            )
            return

    if df.empty:
        st.warning("⚠️ Data kosong atau tidak bisa dibaca. Periksa kembali format file Excel Anda.")
        return

    tahun_list  = sorted(df['Tahun'].unique(), reverse=True)
    tahun_aktif = tahun_sel if (tahun_sel and tahun_sel in tahun_list) else tahun_list[0]
    monthly     = agregasi_bulanan(df, tahun_aktif)

    if monthly.empty:
        st.warning(f"⚠️ Tidak ada data untuk tahun {tahun_aktif}.")
        return

    # Filter bulan (jika dipilih)
    if bulan_sel != 'Semua Bulan':
        bln_num = {v: k for k, v in MONTH_ID.items()}.get(bulan_sel)
        if bln_num:
            monthly_filtered = monthly[monthly['Bulan'] == bln_num]
            if monthly_filtered.empty:
                st.warning(f"⚠️ Tidak ada data untuk bulan {bulan_sel}.")
                monthly_filtered = monthly
        else:
            monthly_filtered = monthly
    else:
        monthly_filtered = monthly

    pred = prediksi(monthly, tahun_aktif)

    # ── TABS NAVIGASI [L1] ────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📌 Ringkasan",
        "📈 Grafik",
        "🔮 Prediksi",
        "📋 Detail",
        "💡 Tips & Analisis"
    ])

    # ══════════════════════════════════════════════════════════
    # TAB 1: RINGKASAN
    # ══════════════════════════════════════════════════════════
    with tab1:
        st.markdown(
            f'<div class="section-title">📌 Ringkasan Keuangan Tahun {tahun_aktif}'
            + (f' — {bulan_sel}' if bulan_sel != 'Semua Bulan' else '') + '</div>',
            unsafe_allow_html=True
        )
        render_kpi(monthly_filtered, pred)

        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">🏥 Kondisi Kesehatan Keuangan</div>',
            unsafe_allow_html=True
        )
        st.caption(
            "Skor 0–100%  ·  🟢 Sangat Sehat (70–100%)  ·  "
            "🟡 Perlu Perhatian (40–70%)  ·  🔴 Kritis (0–40%)"
        )
        render_health(monthly_filtered)

    # ══════════════════════════════════════════════════════════
    # TAB 2: GRAFIK
    # ══════════════════════════════════════════════════════════
    with tab2:
        st.markdown(
            f'<div class="section-title">📈 Grafik Keuangan — Tahun {tahun_aktif}</div>',
            unsafe_allow_html=True
        )

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(chart_saldo(monthly_filtered), use_container_width=True)
        with c2:
            st.plotly_chart(
                chart_donut(monthly_filtered['Pemasukan'].sum(),
                            monthly_filtered['Pengeluaran'].sum()),
                use_container_width=True
            )

        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(chart_bulanan(monthly_filtered), use_container_width=True)
        with c4:
            st.plotly_chart(chart_surplus(monthly_filtered), use_container_width=True)

        # Grafik Kategori
        fig_kat = chart_kategori(df, tahun_aktif)
        if fig_kat:
            st.plotly_chart(fig_kat, use_container_width=True)

        # YoY jika multi-tahun [N2]
        if len(tahun_list) > 1:
            st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-title">📊 Perbandingan Tahun ke Tahun (YoY)</div>',
                unsafe_allow_html=True
            )
            fig_yoy = chart_yoy(df, tahun_list[:4])  # maks 4 tahun
            if fig_yoy:
                st.plotly_chart(fig_yoy, use_container_width=True)

    # ══════════════════════════════════════════════════════════
    # TAB 3: PREDIKSI
    # ══════════════════════════════════════════════════════════
    with tab3:
        st.markdown(
            f'<div class="section-title">🔮 Prediksi Keuangan — {pred["bulan_nama"]} {pred["tahun"]} dan Seterusnya</div>',
            unsafe_allow_html=True
        )
        render_prediksi(pred, monthly)

        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">🤖 Akurasi & Validasi Model</div>',
            unsafe_allow_html=True
        )
        st.caption("Seberapa dekat prediksi model dengan data nyata masa lalu.")
        render_akurasi(monthly, pred)

    # ══════════════════════════════════════════════════════════
    # TAB 4: DETAIL
    # ══════════════════════════════════════════════════════════
    with tab4:
        st.markdown(
            f'<div class="section-title">📋 Rekap Keuangan Bulanan — Tahun {tahun_aktif}</div>',
            unsafe_allow_html=True
        )
        st.caption("Rincian lengkap pemasukan, pengeluaran, rasio beban, dan saldo setiap bulan")
        render_tabel(monthly, tahun_aktif)

        # Detail transaksi bulan terpilih
        if bulan_sel != 'Semua Bulan':
            bln_num = {v: k for k, v in MONTH_ID.items()}.get(bulan_sel)
            if bln_num:
                dd = df[(df['Tahun'] == tahun_aktif) & (df['Bulan'] == bln_num)]
                if not dd.empty:
                    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="section-title">🔎 Detail Transaksi: {bulan_sel} {tahun_aktif}</div>',
                        unsafe_allow_html=True
                    )
                    dc = ['Tanggal','Pemasukan','Pengeluaran','Saldo']
                    if 'Keterangan' in dd.columns: dc.append('Keterangan')
                    if 'Kategori'   in dd.columns: dc.append('Kategori')
                    st.dataframe(dd[dc], use_container_width=True, hide_index=True)

        # A/B Test pola awal vs akhir bulan
        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">🔬 Pola Pemasukan: Awal vs Akhir Bulan</div>',
            unsafe_allow_html=True
        )
        st.caption("Apakah pemasukan lebih banyak di awal bulan (tgl 1–15) atau akhir bulan (tgl 16–31)?")
        render_ab(df, tahun_aktif)

    # ══════════════════════════════════════════════════════════
    # TAB 5: TIPS & ANALISIS
    # ══════════════════════════════════════════════════════════
    with tab5:
        st.markdown(
            '<div class="section-title">💡 Saran &amp; Tips Keuangan</div>',
            unsafe_allow_html=True
        )
        render_tips(monthly, pred)

    # ── FOOTER ───────────────────────────────────────────────
    st.markdown(f"""
    <div class="footer-box">
        🌾 Dashboard Keuangan Karang Taruna &nbsp;·&nbsp;
        Transparansi &amp; Kemajuan Bersama &nbsp;·&nbsp;
        Dibuat dengan ❤️ untuk Desa
        <span>Data Tahun {tahun_aktif} &nbsp;·&nbsp;
        Diperbarui {datetime.now().strftime('%d %B %Y pukul %H:%M WIB')} &nbsp;·&nbsp; v3.0</span>
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()