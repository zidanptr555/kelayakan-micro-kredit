"""
app.py — Aplikasi Web Klasifikasi Kelayakan Kredit Mikro
Jalankan: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle, json, os, io, base64
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# shap di-import hanya saat dibutuhkan (lazy import)
# untuk menghindari crash saat module belum siap

# ── Konfigurasi halaman ────────────────────────────────────
st.set_page_config(
    page_title="KreditCheck — Klasifikasi Kelayakan Kredit Mikro",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS kustom ─────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: #0f1117;
    border-right: 1px solid #1e2130;
  }
  section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
  section[data-testid="stSidebar"] .stRadio label { color: #94a3b8 !important; }
  section[data-testid="stSidebar"] hr { border-color: #1e2130 !important; }

  /* Main background */
  .main .block-container { padding-top: 2rem; max-width: 1100px; }

  /* Hero */
  .hero-wrap {
    background: linear-gradient(135deg, #0f1117 0%, #1a1f2e 60%, #0f1117 100%);
    border-radius: 16px;
    padding: 3rem 3.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
  }
  .hero-wrap::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(99,130,244,0.15) 0%, transparent 70%);
    border-radius: 50%;
  }
  .hero-eyebrow {
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.12em;
    color: #6382f4; text-transform: uppercase; margin-bottom: 0.75rem;
  }
  .hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem; line-height: 1.15;
    color: #f1f5f9; margin: 0 0 1rem;
  }
  .hero-title span { color: #6382f4; }
  .hero-sub {
    font-size: 1rem; color: #94a3b8;
    line-height: 1.7; max-width: 520px; margin-bottom: 2rem;
  }
  .hero-pills { display: flex; gap: 10px; flex-wrap: wrap; }
  .hero-pill {
    background: rgba(99,130,244,0.12);
    border: 1px solid rgba(99,130,244,0.25);
    color: #a5b4fc; font-size: 0.78rem; font-weight: 500;
    padding: 5px 14px; border-radius: 999px;
  }

  /* Feature cards */
  .feat-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 16px; margin-bottom: 2.5rem; }
  .feat-card {
    background: #fff; border: 1px solid #e8edf5;
    border-radius: 14px; padding: 1.5rem;
    transition: box-shadow 0.2s, transform 0.2s;
  }
  .feat-card:hover { box-shadow: 0 8px 24px rgba(0,0,0,0.08); transform: translateY(-2px); }
  .feat-icon {
    width: 40px; height: 40px; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem; margin-bottom: 1rem;
  }
  .feat-icon.blue   { background: #eff2ff; }
  .feat-icon.green  { background: #f0fdf4; }
  .feat-icon.amber  { background: #fffbeb; }
  .feat-name  { font-size: 0.95rem; font-weight: 600; color: #1e293b; margin-bottom: 4px; }
  .feat-desc  { font-size: 0.83rem; color: #64748b; line-height: 1.55; }

  /* Steps */
  .steps-wrap { margin-bottom: 2rem; }
  .step-item  { display: flex; gap: 16px; align-items: flex-start; margin-bottom: 20px; }
  .step-dot   {
    width: 32px; height: 32px; border-radius: 50%; flex-shrink: 0;
    background: #0f1117; color: #6382f4;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.8rem; font-weight: 700; border: 1.5px solid #6382f4;
  }
  .step-title { font-size: 0.95rem; font-weight: 600; color: #1e293b; }
  .step-desc  { font-size: 0.83rem; color: #64748b; margin-top: 2px; }

  /* Result boxes */
  .result-box  { padding: 1.75rem; border-radius: 14px; text-align: center; margin: 1.25rem 0; }
  .result-layak { background: #f0fdf4; border: 1.5px solid #22c55e; }
  .result-tolak { background: #fff1f2; border: 1.5px solid #f43f5e; }
  .result-perlu { background: #fffbeb; border: 1.5px solid #f59e0b; }
  .result-title    { font-size: 1.5rem; font-weight: 700; margin: 0; letter-spacing: -0.02em; }
  .result-subtitle { font-size: 0.9rem; color: #64748b; margin-top: 6px; }

  /* Chat */
  .chat-user {
    background: #eff2ff; border-radius: 14px 14px 4px 14px;
    padding: 11px 16px; margin: 8px 0; font-size: 0.88rem;
    color: #1e293b; text-align: right; margin-left: 20%;
  }
  .chat-bot {
    background: #f8fafc; border: 1px solid #e8edf5;
    border-radius: 14px 14px 14px 4px;
    padding: 11px 16px; margin: 8px 0; font-size: 0.88rem;
    color: #1e293b; margin-right: 20%;
  }

  /* Section headings */
  .section-head {
    font-family: 'DM Serif Display', serif;
    font-size: 1.4rem; color: #1e293b; margin-bottom: 0.25rem;
  }
  .section-sub { font-size: 0.85rem; color: #94a3b8; margin-bottom: 1.5rem; }

  /* Streamlit overrides */
  .stButton > button {
    background: #0f1117 !important; color: #fff !important;
    border: none !important; border-radius: 10px !important;
    font-weight: 500 !important; padding: 0.6rem 1.5rem !important;
    transition: opacity 0.2s !important;
  }
  .stButton > button:hover { opacity: 0.85 !important; }

  /* Nav buttons di sidebar */
  section[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    color: #64748b !important;
    border: none !important;
    border-left: 2px solid transparent !important;
    border-radius: 0 8px 8px 0 !important;
    font-weight: 400 !important;
    font-size: 0.88rem !important;
    padding: 0.5rem 1rem !important;
    text-align: left !important;
    width: 100% !important;
    margin-bottom: 2px !important;
    transition: all 0.15s !important;
  }
  section[data-testid="stSidebar"] .stButton > button:hover {
    background: #1e2130 !important;
    color: #e2e8f0 !important;
    opacity: 1 !important;
    border-left: 2px solid #6382f4 !important;
  }
  div[data-testid="stForm"] { border: none !important; padding: 0 !important; }
  @media (max-width: 700px) {
    .feat-grid { grid-template-columns: 1fr; }
    .hero-title { font-size: 1.8rem; }
    .hero-wrap  { padding: 2rem 1.5rem; }
  }
</style>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────
# Load model & artefak
# ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    base = "model_artifacts"

    # Folder belum ada — training belum dijalankan
    if not os.path.exists(base):
        return None

    required_files = [
        "best_model.pkl", "imputer.pkl", "shap_explainer.pkl",
        "features.pkl", "top_features.pkl", "metadata.json"
    ]
    missing = [f for f in required_files if not os.path.exists(f"{base}/{f}")]
    if missing:
        return {"error": f"File berikut tidak ditemukan di model_artifacts/: {', '.join(missing)}"}

    try:
        arts = {}
        arts["model"]    = pickle.load(open(f"{base}/best_model.pkl",     "rb"))
        arts["imputer"]  = pickle.load(open(f"{base}/imputer.pkl",        "rb"))
        arts["explainer"]= pickle.load(open(f"{base}/shap_explainer.pkl", "rb"))
        arts["features"] = pickle.load(open(f"{base}/features.pkl",       "rb"))
        arts["top_feats"]= pickle.load(open(f"{base}/top_features.pkl",   "rb"))
        arts["meta"]     = json.load(open(f"{base}/metadata.json"))
        return arts
    except Exception as e:
        return {"error": str(e)}

arts = load_artifacts()

# Tampilkan panduan setup jika model belum ada
if arts is None:
    st.error("📂 Folder `model_artifacts/` tidak ditemukan.")
    st.info("""
**Langkah yang perlu dilakukan:**
1. Buka `01_training_colab.ipynb` di Google Colab
2. Jalankan semua cell dari atas ke bawah
3. Download `model_artifacts.zip` yang dihasilkan di akhir notebook
4. Ekstrak zip tersebut ke folder yang sama dengan `app.py` ini:
```
D:\\apk kelayakan micro kredit\\
├── app.py
├── requirements.txt
└── model_artifacts\\          ← ekstrak zip di sini
    ├── best_model.pkl
    ├── imputer.pkl
    ├── shap_explainer.pkl
    ├── features.pkl
    ├── top_features.pkl
    └── metadata.json
```
5. Jalankan ulang: `streamlit run app.py`
    """)
    st.stop()

if isinstance(arts, dict) and "error" in arts:
    st.error(f"❌ Gagal memuat model: {arts['error']}")
    st.warning("Pastikan semua file di folder `model_artifacts/` lengkap dan tidak rusak.")
    st.stop()


# ────────────────────────────────────────────────────────────
# Fungsi prediksi
# ────────────────────────────────────────────────────────────
def predict(input_dict: dict, arts: dict) -> dict:
    """Prediksi menggunakan fitur Credit Risk Dataset (laotse)."""
    features = arts["features"]
    df_in    = pd.DataFrame([input_dict])
    eps      = 1e-6

    # Feature engineering — sesuai Credit Risk Dataset
    df_in["loan_percent_income"]    = df_in["loan_amnt"] / (df_in["person_income"] + eps)
    df_in["income_per_emp_year"]    = df_in["person_income"] / (df_in["person_emp_length"] + 1)
    df_in["loan_to_emp_ratio"]      = df_in["loan_amnt"] / (df_in["person_emp_length"] + 1)
    df_in["age_emp_ratio"]          = df_in["person_emp_length"] / (df_in["person_age"] + eps)
    df_in["monthly_payment"]        = (df_in["loan_amnt"] * df_in["loan_int_rate"] / 100) / 12
    df_in["payment_income_ratio"]   = df_in["monthly_payment"] / (df_in["person_income"] / 12 + eps)
    df_in["credit_hist_age_ratio"]  = df_in["cb_person_cred_hist_length"] / (df_in["person_age"] + eps)

    # Align ke semua fitur training
    for col in features:
        if col not in df_in.columns:
            df_in[col] = 0
    df_aligned = df_in[features]

    # Impute — pakai SimpleImputer baru agar kompatibel semua versi Python
    from sklearn.impute import SimpleImputer
    imputer_fresh = SimpleImputer(strategy="median")
    X_imp = imputer_fresh.fit_transform(df_aligned)
    X_df  = pd.DataFrame(X_imp, columns=features)

    prob      = arts["model"].predict_proba(X_df)[0]
    threshold = arts["meta"].get("threshold", 0.5)
    label     = int(prob[1] >= threshold)

    # SHAP
    sv       = arts["explainer"].shap_values(X_df)
    shap_ser = pd.Series(sv[0], index=features).abs().sort_values(ascending=False)

    return {
        "label"       : label,
        "prob_default": float(prob[1]),
        "prob_layak"  : float(prob[0]),
        "shap_values" : sv[0],
        "shap_series" : shap_ser,
        "features_df" : X_df,
    }


# ────────────────────────────────────────────────────────────
# Chatbot — Gemini API
# ────────────────────────────────────────────────────────────
def build_prompt(prediction: dict, input_data: dict) -> str:
    label    = "TIDAK LAYAK" if prediction["label"] == 1 else "LAYAK"
    prob_pct = prediction["prob_default"] * 100
    top3     = prediction["shap_series"].head(3).index.tolist()

    return f"""Kamu adalah asisten keuangan bernama KreditBot untuk aplikasi kelayakan kredit mikro di Indonesia.
Bantu pengguna memahami hasil analisis kredit mereka dan berikan saran konkret.
Gunakan Bahasa Indonesia yang ramah, jelas, dan tidak menggurui.

[KONTEKS ANALISIS]
Hasil prediksi          : {label}
Probabilitas gagal bayar: {prob_pct:.1f}%
Faktor utama penentu    : {', '.join(top3)}

[DATA PEMOHON]
Pendapatan tahunan : Rp {input_data.get('person_income', 0):,.0f}
Jumlah pinjaman    : Rp {input_data.get('loan_amnt', 0):,.0f}
Suku bunga         : {input_data.get('loan_int_rate', 0):.1f}%
Usia               : {input_data.get('person_age', 0):.0f} tahun
Lama bekerja       : {input_data.get('person_emp_length', 0):.0f} tahun
Rasio pinjaman/pendapatan: {input_data.get('loan_percent_income', 0):.2%}

Jika TIDAK LAYAK: berikan 3 saran konkret untuk meningkatkan kelayakan kredit.
Jika LAYAK: berikan tips mengelola kredit dengan baik.
Tetap supportif dan fokus pada solusi."""


def chat_with_gemini(messages: list, system_prompt: str) -> str:
    """Fungsi chat menggunakan Groq API (gratis, tanpa billing)."""
    try:
        from groq import Groq
        api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
        if not api_key:
            return "⚠️ API key Groq belum dikonfigurasi. Tambahkan GROQ_API_KEY di file .streamlit/secrets.toml"

        client = Groq(api_key=api_key)

        # Susun pesan dengan system prompt
        groq_messages = [{"role": "system", "content": system_prompt}]
        for m in messages:
            role = "user" if m["role"] == "user" else "assistant"
            groq_messages.append({"role": role, "content": m["content"]})

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=groq_messages,
            temperature=0.7,
            max_tokens=600,
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"Terjadi kesalahan: {e}"


# ────────────────────────────────────────────────────────────
# Sidebar — navigasi
# ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:0.5rem 0 1.75rem;">
      <div style="display:flex;align-items:center;gap:12px;">
        <!-- Logo SVG -->
        <svg width="40" height="40" viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
          <rect width="40" height="40" rx="10" fill="#1a1a1a"/>
          <rect x="1" y="1" width="38" height="38" rx="9" stroke="url(#silver)" stroke-width="1"/>
          <rect x="8" y="14" width="24" height="15" rx="2.5" stroke="#C0C0C0" stroke-width="1.5"/>
          <rect x="8" y="17" width="24" height="4" fill="#C0C0C0" opacity="0.25"/>
          <circle cx="13" cy="25" r="2" fill="#C0C0C0" opacity="0.6"/>
          <circle cx="19" cy="25" r="2" fill="#C0C0C0" opacity="0.4"/>
          <rect x="14" y="11" width="5" height="4" rx="1" stroke="#C0C0C0" stroke-width="1.2"/>
          <rect x="21" y="11" width="5" height="4" rx="1" stroke="#C0C0C0" stroke-width="1.2"/>
          <defs>
            <linearGradient id="silver" x1="0" y1="0" x2="40" y2="40" gradientUnits="userSpaceOnUse">
              <stop offset="0%" stop-color="#888"/>
              <stop offset="50%" stop-color="#e8e8e8"/>
              <stop offset="100%" stop-color="#888"/>
            </linearGradient>
          </defs>
        </svg>
        <div>
          <div style="font-size:1.1rem;font-weight:700;color:#f1f5f9;
                      letter-spacing:-0.02em;line-height:1.2;">KreditCheck</div>
          <div style="font-size:0.68rem;color:#64748b;letter-spacing:0.05em;
                      text-transform:uppercase;margin-top:2px;">Micro Credit AI</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Custom navigasi HTML
    if "page" not in st.session_state:
        st.session_state["page"] = "Beranda"

    st.markdown('<div style="font-size:0.7rem;font-weight:600;letter-spacing:0.1em;color:#475569;text-transform:uppercase;margin-bottom:10px;">Navigasi</div>', unsafe_allow_html=True)

    nav_items = [
        ("Beranda", """<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>"""),
        ("Prediksi Kredit", """<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="5" width="20" height="14" rx="2"/><line x1="2" y1="10" x2="22" y2="10"/></svg>"""),
        ("Simulasi What-If", """<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>"""),
        ("Upload CSV", """<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>"""),
        ("Riwayat", """<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="12 8 12 12 14 14"/><path d="M3.05 11a9 9 0 1 1 .5 4m-.5 5v-5h5"/></svg>"""),
        ("Konsultasi AI", """<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>"""),
        ("Info Model", """<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>"""),
    ]

    for nav_name, nav_icon in nav_items:
        is_active = st.session_state["page"] == nav_name
        active_style = "background:#1e2130;color:#f1f5f9;border-left:2px solid #6382f4;" if is_active else "background:transparent;color:#64748b;border-left:2px solid transparent;"
        if st.sidebar.button(
            nav_name,
            key=f"nav_{nav_name}",
            use_container_width=True,
        ):
            st.session_state["page"] = nav_name
            st.rerun()

        st.markdown(f"""
        <style>
        div[data-testid="stSidebarContent"] div[data-testid="stButton"] button[kind="secondary"] {{
            text-align: left !important;
            justify-content: flex-start !important;
        }}
        </style>
        """, unsafe_allow_html=True)

    page_map = {
        "Beranda"         : "🏠 Beranda",
        "Prediksi Kredit" : "📋 Prediksi Kredit",
        "Simulasi What-If": "📈 Simulasi What-If",
        "Upload CSV"      : "📂 Upload CSV",
        "Riwayat"         : "🕓 Riwayat",
        "Konsultasi AI"   : "💬 Konsultasi AI",
        "Info Model"      : "📊 Info Model",
    }
    page = page_map[st.session_state["page"]]

    st.markdown('<hr style="border-color:#1e2130;margin:1.5rem 0;">', unsafe_allow_html=True)

    if arts:
        meta = arts["meta"]
        st.markdown(f"""
        <div style="font-size:0.7rem;font-weight:600;letter-spacing:0.1em;color:#475569;
                    text-transform:uppercase;margin-bottom:12px;">Performa Model</div>
        <div style="display:flex;flex-direction:column;gap:10px;">
          <div style="background:#1e2130;border-radius:8px;padding:10px 12px;">
            <div style="font-size:0.7rem;color:#64748b;">Model</div>
            <div style="font-size:0.9rem;font-weight:600;color:#e2e8f0;margin-top:2px;">{meta['model_name']}</div>
          </div>
          <div style="background:#1e2130;border-radius:8px;padding:10px 12px;">
            <div style="font-size:0.7rem;color:#64748b;">AUC-ROC</div>
            <div style="font-size:1.1rem;font-weight:700;color:#6382f4;margin-top:2px;">{meta['auc_roc']:.4f}</div>
          </div>
          <div style="background:#1e2130;border-radius:8px;padding:10px 12px;">
            <div style="font-size:0.7rem;color:#64748b;">F1-Score</div>
            <div style="font-size:1.1rem;font-weight:700;color:#6382f4;margin-top:2px;">{meta['f1_score']:.4f}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Model belum dimuat.")


# ────────────────────────────────────────────────────────────
# Halaman: Beranda
# ────────────────────────────────────────────────────────────
if page == "🏠 Beranda":
    # Hero
    st.markdown("""
    <div class="hero-wrap">
      <div class="hero-eyebrow">Sistem Analisis Kredit Mikro</div>
      <h1 class="hero-title">KreditCheck —<br><span>Prediksi Kelayakan</span><br>Berbasis AI</h1>
      <p class="hero-sub">
        Analisis kelayakan kredit secara cepat dan transparan menggunakan
        XGBoost dan SHAP Explainability. Setiap keputusan disertai penjelasan
        faktor-faktor yang mempengaruhinya.
      </p>
      <div class="hero-pills">
        <span class="hero-pill">XGBoost Classifier</span>
        <span class="hero-pill">SHAP Explainability</span>
        <span class="hero-pill">Konsultasi AI</span>
        <span class="hero-pill">Analisis Fairness</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature cards
    st.markdown("""
    <div class="feat-grid">
      <div class="feat-card">
        <div class="feat-icon blue">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#6382f4" stroke-width="2">
            <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
          </svg>
        </div>
        <div class="feat-name">Prediksi Akurat</div>
        <div class="feat-desc">Model XGBoost terlatih pada data kredit nyata dengan AUC-ROC tinggi</div>
      </div>
      <div class="feat-card">
        <div class="feat-icon green">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#16a34a" stroke-width="2">
            <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
          </svg>
        </div>
        <div class="feat-name">Transparan & Explainable</div>
        <div class="feat-desc">SHAP menjelaskan secara visual faktor apa yang menentukan hasil keputusan</div>
      </div>
      <div class="feat-card">
        <div class="feat-icon amber">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#d97706" stroke-width="2">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
          </svg>
        </div>
        <div class="feat-name">Konsultasi AI</div>
        <div class="feat-desc">Chatbot AI memberikan saran personal berdasarkan hasil analisis kredit Anda</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Steps
    st.markdown('<div class="section-head">Cara Penggunaan</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Empat langkah sederhana untuk mendapatkan analisis kelayakan kredit</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="steps-wrap">
      <div class="step-item">
        <div class="step-dot">1</div>
        <div>
          <div class="step-title">Buka halaman Prediksi Kredit</div>
          <div class="step-desc">Isi data pemohon pada form yang tersedia — data pribadi dan detail pinjaman</div>
        </div>
      </div>
      <div class="step-item">
        <div class="step-dot">2</div>
        <div>
          <div class="step-title">Klik tombol Analisis Kelayakan</div>
          <div class="step-desc">Model memproses data dan menampilkan hasil prediksi beserta probabilitasnya</div>
        </div>
      </div>
      <div class="step-item">
        <div class="step-dot">3</div>
        <div>
          <div class="step-title">Baca penjelasan SHAP</div>
          <div class="step-desc">Pahami faktor mana yang paling berpengaruh terhadap keputusan model</div>
        </div>
      </div>
      <div class="step-item">
        <div class="step-dot">4</div>
        <div>
          <div class="step-title">Konsultasi di halaman Konsultasi AI</div>
          <div class="step-desc">Tanya chatbot untuk mendapat saran konkret meningkatkan kelayakan kredit</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────
# Halaman: Prediksi Kredit
# ────────────────────────────────────────────────────────────
elif page == "📋 Prediksi Kredit":
    st.markdown("## 📋 Prediksi Kelayakan Kredit")
    st.markdown("Isi data pemohon di bawah ini untuk mendapatkan analisis kelayakan.")

    if not arts:
        st.error("Model belum tersedia. Jalankan notebook training dan taruh folder model_artifacts/ di direktori yang sama.")
        st.stop()

    with st.form("credit_form"):
        st.markdown("### Data Pribadi")
        c1, c2, c3 = st.columns(3)
        with c1:
            age      = st.number_input("Usia (tahun)", 18, 80, 30)
            emp_len  = st.number_input("Lama Bekerja (tahun)", 0, 50, 5)
        with c2:
            income   = st.number_input("Pendapatan Tahunan (Rp)", 1_000_000, 5_000_000_000,
                                        60_000_000, step=1_000_000)
            home     = st.selectbox("Status Tempat Tinggal",
                                     ["rent", "own", "mortgage", "other"])
        with c3:
            cred_hist = st.number_input("Panjang Riwayat Kredit (tahun)", 0, 30, 3)
            has_default = st.selectbox("Pernah Gagal Bayar Sebelumnya?", ["Tidak (N)", "Ya (Y)"])

        st.markdown("### Data Pinjaman")
        c4, c5, c6 = st.columns(3)
        with c4:
            loan_amnt   = st.number_input("Jumlah Pinjaman (Rp)", 500_000, 500_000_000,
                                           10_000_000, step=500_000)
            loan_intent = st.selectbox("Tujuan Pinjaman",
                                        ["PERSONAL", "EDUCATION", "MEDICAL",
                                         "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
        with c5:
            int_rate    = st.slider("Suku Bunga (%/tahun)", 1.0, 30.0, 11.0, 0.1)
            loan_grade  = st.selectbox("Grade Kredit", ["A","B","C","D","E","F","G"])
        with c6:
            st.markdown("**Estimasi otomatis:**")
            loan_pct    = round(loan_amnt / (income + 1), 4)
            monthly_pay = round((loan_amnt * int_rate / 100) / 12, 0)
            st.metric("Rasio Pinjaman/Pendapatan", f"{loan_pct:.2%}")
            st.metric("Estimasi Cicilan/Bulan", f"Rp {monthly_pay:,.0f}")

        submitted = st.form_submit_button("🔍 Analisis Kelayakan", use_container_width=True)

    if submitted:
        grade_map   = {"A":0,"B":1,"C":2,"D":3,"E":4,"F":5,"G":6}
        home_map    = {"rent":0,"own":1,"mortgage":2,"other":3}
        intent_map  = {"PERSONAL":0,"EDUCATION":1,"MEDICAL":2,
                       "VENTURE":3,"HOMEIMPROVEMENT":4,"DEBTCONSOLIDATION":5}
        input_data = {
            "person_age"                 : age,
            "person_income"              : income,
            "person_emp_length"          : emp_len,
            "loan_amnt"                  : loan_amnt,
            "loan_int_rate"              : int_rate,
            "loan_percent_income"        : loan_pct,
            "cb_person_cred_hist_length" : cred_hist,
            "cb_person_default_on_file"  : 1 if "Y" in has_default else 0,
            "loan_grade"                 : grade_map[loan_grade],
            "loan_intent"                : intent_map[loan_intent],
            "person_home_ownership"      : home_map[home],
        }
        st.session_state["last_prediction"] = None
        st.session_state["last_input"]      = input_data

        with st.spinner("Menganalisis data..."):
            result = predict(input_data, arts)

        st.session_state["last_prediction"] = result

        # ── Tampilkan hasil ──────────────────────────────────
        prob_pct = result["prob_default"] * 100

        if result["label"] == 0:
            css_class, icon, verdict = "result-layak", "✅", "LAYAK"
            color_hex = "#4caf50"
        elif prob_pct < 70:
            css_class, icon, verdict = "result-perlu", "⚠️", "PERLU TINJAUAN"
            color_hex = "#ff9800"
        else:
            css_class, icon, verdict = "result-tolak", "❌", "TIDAK LAYAK"
            color_hex = "#f44336"

        st.markdown(f"""
        <div class="result-box {css_class}">
          <p class="result-title" style="color:{color_hex};">{icon} {verdict}</p>
          <p class="result-subtitle">Probabilitas gagal bayar: <strong>{prob_pct:.1f}%</strong></p>
        </div>""", unsafe_allow_html=True)

        # Metrik ringkas
        loan_amnt_val = input_data.get("loan_amnt", 0)
        income_val    = input_data.get("person_income", 1)
        int_rate_val  = input_data.get("loan_int_rate", 0)
        monthly_pay   = (loan_amnt_val * int_rate_val / 100) / 12

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Probabilitas Gagal Bayar",    f"{prob_pct:.1f}%")
        m2.metric("Rasio Pinjaman/Pendapatan",   f"{loan_amnt_val/income_val:.2%}")
        m3.metric("Estimasi Cicilan/Bulan",      f"Rp {monthly_pay:,.0f}")
        m4.metric("Suku Bunga",                  f"{int_rate_val:.1f}%")

        # ── SHAP chart ───────────────────────────────────────
        st.markdown("### 🔍 Faktor Penentu Keputusan (SHAP)")
        st.caption("Batang merah = meningkatkan risiko | Batang hijau = menurunkan risiko")

        top_feats = result["shap_series"].head(12)
        sv_raw    = pd.Series(result["shap_values"], index=arts["features"])
        sv_top    = sv_raw.loc[top_feats.index]

        fig, ax = plt.subplots(figsize=(8, 5))
        colors  = ["#E24B4A" if v > 0 else "#3B9B5C" for v in sv_top.values]
        bars    = ax.barh(range(len(sv_top)), sv_top.values, color=colors, edgecolor="white")
        ax.set_yticks(range(len(sv_top)))
        ax.set_yticklabels(sv_top.index.tolist(), fontsize=9)
        ax.axvline(0, color="gray", lw=0.8, linestyle="--")
        ax.set_xlabel("SHAP Value (pengaruh terhadap prediksi)")
        ax.set_title("Kontribusi Fitur terhadap Prediksi", fontweight="bold")
        ax.invert_yaxis()

        red_patch   = mpatches.Patch(color="#E24B4A", label="Meningkatkan risiko")
        green_patch = mpatches.Patch(color="#3B9B5C", label="Menurunkan risiko")
        ax.legend(handles=[red_patch, green_patch], fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # ── Skor Kredit 0–100 ────────────────────────────────
        st.markdown("### Skor Kredit")
        credit_score = int((1 - result["prob_default"]) * 100)
        if credit_score >= 75:
            score_color, score_label = "#22c55e", "Sangat Baik"
        elif credit_score >= 60:
            score_color, score_label = "#84cc16", "Baik"
        elif credit_score >= 45:
            score_color, score_label = "#f59e0b", "Cukup"
        elif credit_score >= 30:
            score_color, score_label = "#f97316", "Kurang"
        else:
            score_color, score_label = "#f43f5e", "Buruk"

        fig_score, ax_score = plt.subplots(figsize=(6, 2))
        ax_score.barh([""], [credit_score], color=score_color, height=0.5)
        ax_score.barh([""], [100 - credit_score], color="#f1f5f9", height=0.5, left=credit_score)
        ax_score.set_xlim(0, 100)
        for thresh, col in [(30,"#f43f5e"),(45,"#f97316"),(60,"#f59e0b"),(75,"#84cc16")]:
            ax_score.axvline(thresh, color=col, lw=1.2, linestyle="--", alpha=0.6)
        ax_score.text(credit_score/2, 0, f"{credit_score}", va="center", ha="center",
                      fontsize=14, fontweight="bold",
                      color="white" if credit_score > 15 else score_color)
        ax_score.set_xlabel("Skor (0 = risiko tinggi, 100 = risiko rendah)")
        ax_score.set_title(f"Skor Kredit: {credit_score}/100 — {score_label}", fontweight="bold")
        ax_score.spines[["top","right","left"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_score)
        plt.close()

        # ── Simpan ke riwayat ─────────────────────────────────
        if "prediction_history" not in st.session_state:
            st.session_state["prediction_history"] = []
        st.session_state["prediction_history"].append({
            "No"                    : len(st.session_state["prediction_history"]) + 1,
            "Usia"                  : input_data.get("person_age", "-"),
            "Pendapatan (Rp)"       : f"{input_data.get('person_income',0):,.0f}",
            "Pinjaman (Rp)"         : f"{input_data.get('loan_amnt',0):,.0f}",
            "Suku Bunga (%)"        : input_data.get("loan_int_rate", "-"),
            "Skor Kredit"           : credit_score,
            "Status"                : "LAYAK" if result["label"] == 0 else "TIDAK LAYAK",
            "Prob. Gagal Bayar (%)" : f"{result['prob_default']*100:.1f}",
        })

        # ── Download PDF ──────────────────────────────────────
        st.markdown("### Download Laporan")
        try:
            from fpdf import FPDF
            pdf = FPDF()
            pdf.add_page()
            pdf.set_margins(20, 20, 20)

            # Header
            pdf.set_fill_color(15, 17, 23)
            pdf.rect(0, 0, 210, 30, "F")
            pdf.set_font("Helvetica", "B", 16)
            pdf.set_text_color(255, 255, 255)
            pdf.set_xy(20, 10)
            pdf.cell(0, 10, "KreditCheck — Laporan Analisis Kredit", ln=True)

            pdf.set_text_color(0, 0, 0)
            pdf.set_xy(20, 38)

            # Data pemohon
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "Data Pemohon", ln=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.set_x(20)
            rows = [
                ("Usia", f"{input_data.get('person_age','-')} tahun"),
                ("Pendapatan Tahunan", f"Rp {input_data.get('person_income',0):,.0f}"),
                ("Jumlah Pinjaman", f"Rp {input_data.get('loan_amnt',0):,.0f}"),
                ("Suku Bunga", f"{input_data.get('loan_int_rate',0):.1f}%"),
                ("Lama Bekerja", f"{input_data.get('person_emp_length',0):.0f} tahun"),
            ]
            for label, val in rows:
                pdf.set_x(20)
                pdf.cell(70, 7, label, border="B")
                pdf.cell(0, 7, val, border="B", ln=True)

            pdf.ln(8)
            # Hasil prediksi
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_x(20)
            pdf.cell(0, 8, "Hasil Analisis", ln=True)
            pdf.set_font("Helvetica", "B", 14)
            pdf.set_x(20)
            status_text = "LAYAK" if result["label"] == 0 else "TIDAK LAYAK"
            if result["label"] == 0:
                pdf.set_text_color(34, 197, 94)
            else:
                pdf.set_text_color(244, 63, 94)
            pdf.cell(0, 10, f"Status: {status_text}", ln=True)
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Helvetica", "", 10)
            pdf.set_x(20)
            pdf.cell(0, 7, f"Probabilitas Gagal Bayar : {result['prob_default']*100:.1f}%", ln=True)
            pdf.set_x(20)
            pdf.cell(0, 7, f"Skor Kredit              : {credit_score}/100 ({score_label})", ln=True)

            pdf.ln(6)
            # Top 5 faktor
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_x(20)
            pdf.cell(0, 8, "Faktor Utama Penentu Keputusan (SHAP)", ln=True)
            pdf.set_font("Helvetica", "", 10)
            for i, (feat, val) in enumerate(result["shap_series"].head(5).items(), 1):
                direction = "Meningkatkan risiko" if result["shap_values"][list(result["shap_series"].index).index(feat)] > 0 else "Menurunkan risiko"
                pdf.set_x(20)
                pdf.cell(0, 6, f"{i}. {feat} — {direction}", ln=True)

            pdf.ln(6)
            pdf.set_font("Helvetica", "I", 8)
            pdf.set_text_color(150, 150, 150)
            pdf.set_x(20)
            pdf.cell(0, 5, "Laporan ini dihasilkan oleh KreditCheck — Sistem Klasifikasi Kelayakan Kredit Mikro berbasis ML", ln=True)

            pdf_bytes = pdf.output()
            st.download_button(
                "Download Laporan PDF",
                data=bytes(pdf_bytes),
                file_name=f"laporan_kredit_skor{credit_score}.pdf",
                mime="application/pdf",
            )
        except ImportError:
            st.info("Install fpdf2 untuk fitur download PDF: `pip install fpdf2`")
        except Exception as e:
            st.warning(f"PDF tidak bisa dibuat: {e}")

        st.info("💬 Buka halaman **Konsultasi AI** untuk mendapatkan saran personal berdasarkan hasil ini.")


# ────────────────────────────────────────────────────────────
# Halaman: Konsultasi AI
# ────────────────────────────────────────────────────────────
elif page == "💬 Konsultasi AI":
    st.markdown("## 💬 Konsultasi AI — KreditBot")

    prediction = st.session_state.get("last_prediction")
    input_data = st.session_state.get("last_input", {})

    if prediction is None:
        st.warning("Lakukan prediksi terlebih dahulu di halaman **Prediksi Kredit**.")
        st.stop()

    # Tampilkan ringkasan prediksi
    prob_pct = prediction["prob_default"] * 100
    verdict  = "LAYAK ✅" if prediction["label"] == 0 else "TIDAK LAYAK ❌"
    st.markdown(f"""
    <div style="background:#f8f9fa;border-radius:10px;padding:12px 16px;
                border-left:4px solid #534AB7;margin-bottom:1rem;">
      <strong>Hasil prediksi:</strong> {verdict} &nbsp;|&nbsp;
      Probabilitas gagal bayar: <strong>{prob_pct:.1f}%</strong>
    </div>""", unsafe_allow_html=True)

    # Inisialisasi chat history
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
        system_prompt = build_prompt(prediction, input_data)
        # Pesan pembuka otomatis
        opening = chat_with_gemini(
            [{"role": "user", "content": "Halo, bisa jelaskan hasil analisis kredit saya?"}],
            system_prompt
        )
        st.session_state["chat_history"] = [
            {"role": "user",      "content": "Halo, bisa jelaskan hasil analisis kredit saya?"},
            {"role": "assistant", "content": opening},
        ]

    # Tampilkan riwayat chat
    for msg in st.session_state["chat_history"]:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">👤 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bot">🤖 {msg["content"]}</div>', unsafe_allow_html=True)

    # Input chat
    st.markdown("---")
    user_input = st.chat_input("Tanya sesuatu tentang hasil analisis atau cara meningkatkan kredit...")
    if user_input:
        st.session_state["chat_history"].append({"role": "user", "content": user_input})
        system_prompt = build_prompt(prediction, input_data)
        gemini_history = [
            {"role": "user" if m["role"]=="user" else "model", "content": m["content"]}
            for m in st.session_state["chat_history"]
        ]
        with st.spinner("KreditBot sedang menjawab..."):
            response = chat_with_gemini(gemini_history, system_prompt)
        st.session_state["chat_history"].append({"role": "assistant", "content": response})
        st.rerun()

    # Tombol pertanyaan cepat
    st.markdown("**Pertanyaan cepat:**")
    q_cols = st.columns(3)
    quick_qs = [
        "Apa yang perlu saya perbaiki?",
        "Berapa lama waktu perbaikan kredit?",
        "Apakah penghasilan saya cukup?",
    ]
    for i, (col, q) in enumerate(zip(q_cols, quick_qs)):
        if col.button(q, key=f"qq_{i}"):
            st.session_state["chat_history"].append({"role": "user", "content": q})
            system_prompt = build_prompt(prediction, input_data)
            gemini_history = [
                {"role": "user" if m["role"]=="user" else "model", "content": m["content"]}
                for m in st.session_state["chat_history"]
            ]
            resp = chat_with_gemini(gemini_history, system_prompt)
            st.session_state["chat_history"].append({"role": "assistant", "content": resp})
            st.rerun()

    if st.button("🗑️ Reset Percakapan"):
        st.session_state.pop("chat_history", None)
        st.rerun()



# ────────────────────────────────────────────────────────────
# Halaman: Simulasi What-If
# ────────────────────────────────────────────────────────────
elif page == "📈 Simulasi What-If":
    st.markdown("## Simulasi What-If — Analisis Sensitivitas")
    st.markdown("Geser slider untuk melihat bagaimana perubahan variabel mempengaruhi hasil prediksi secara real-time.")

    if not arts:
        st.error("Model belum tersedia.")
        st.stop()

    # Ambil nilai terakhir dari prediksi sebelumnya sebagai baseline
    last_input = st.session_state.get("last_input", {})

    st.markdown("### Parameter Simulasi")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Data Pribadi**")
        sim_age      = st.slider("Usia (tahun)", 18, 70,
                                  int(last_input.get("person_age", 30)))
        sim_income   = st.slider("Pendapatan Tahunan (juta Rp)", 12, 500,
                                  int(last_input.get("person_income", 60_000_000) / 1_000_000)) * 1_000_000
        sim_emp      = st.slider("Lama Bekerja (tahun)", 0, 40,
                                  int(last_input.get("person_emp_length", 5)))
        sim_cred_hist= st.slider("Riwayat Kredit (tahun)", 0, 30,
                                  int(last_input.get("cb_person_cred_hist_length", 3)))
    with c2:
        st.markdown("**Data Pinjaman**")
        sim_loan     = st.slider("Jumlah Pinjaman (juta Rp)", 1, 100,
                                  int(last_input.get("loan_amnt", 10_000_000) / 1_000_000)) * 1_000_000
        sim_rate     = st.slider("Suku Bunga (%)", 1.0, 30.0,
                                  float(last_input.get("loan_int_rate", 11.0)), 0.5)
        sim_default  = st.selectbox("Pernah Gagal Bayar?", ["Tidak (N)", "Ya (Y)"],
                                     index=0 if last_input.get("cb_person_default_on_file", 0) == 0 else 1)
        sim_grade    = st.selectbox("Grade Kredit", ["A","B","C","D","E","F","G"],
                                     index=last_input.get("loan_grade", 0))

    # Hitung prediksi real-time
    grade_map  = {"A":0,"B":1,"C":2,"D":3,"E":4,"F":5,"G":6}
    sim_input  = {
        "person_age"                 : sim_age,
        "person_income"              : sim_income,
        "person_emp_length"          : sim_emp,
        "loan_amnt"                  : sim_loan,
        "loan_int_rate"              : sim_rate,
        "loan_percent_income"        : sim_loan / (sim_income + 1),
        "cb_person_cred_hist_length" : sim_cred_hist,
        "cb_person_default_on_file"  : 1 if "Y" in sim_default else 0,
        "loan_grade"                 : grade_map[sim_grade],
        "loan_intent"                : last_input.get("loan_intent", 0),
        "person_home_ownership"      : last_input.get("person_home_ownership", 0),
    }

    try:
        sim_result = predict(sim_input, arts)
        prob_pct   = sim_result["prob_default"] * 100

        # Hasil real-time
        st.markdown("---")
        st.markdown("### Hasil Simulasi Real-time")

        col_res1, col_res2, col_res3, col_res4 = st.columns(4)
        verdict_color = "#22c55e" if sim_result["label"] == 0 else "#f43f5e"
        verdict_text  = "LAYAK" if sim_result["label"] == 0 else "TIDAK LAYAK"

        col_res1.markdown(f"""
        <div style="background:#f8fafc;border-radius:10px;padding:14px;text-align:center;
                    border:1.5px solid {verdict_color};">
          <div style="font-size:0.75rem;color:#64748b;">Status</div>
          <div style="font-size:1.2rem;font-weight:700;color:{verdict_color};margin-top:4px;">{verdict_text}</div>
        </div>""", unsafe_allow_html=True)
        col_res2.metric("Prob. Gagal Bayar", f"{prob_pct:.1f}%")
        col_res3.metric("Rasio Pinjaman/Pendapatan", f"{sim_loan/sim_income:.2%}")
        col_res4.metric("Cicilan Est./Bulan", f"Rp {(sim_loan * sim_rate/100)/12:,.0f}")

        # Gauge chart probabilitas
        fig, ax = plt.subplots(figsize=(8, 2.5))
        bar_color = "#22c55e" if prob_pct < 50 else "#f59e0b" if prob_pct < 70 else "#f43f5e"
        ax.barh(["Probabilitas Gagal Bayar"], [prob_pct], color=bar_color,
                height=0.4, left=0)
        ax.barh(["Probabilitas Gagal Bayar"], [100 - prob_pct], color="#f1f5f9",
                height=0.4, left=prob_pct)
        ax.set_xlim(0, 100)
        ax.axvline(50, color="#94a3b8", lw=1, linestyle="--", alpha=0.5)
        ax.text(prob_pct - 2 if prob_pct > 10 else prob_pct + 2,
                0, f"{prob_pct:.1f}%", va="center",
                ha="right" if prob_pct > 10 else "left",
                fontsize=12, fontweight="bold", color="white" if prob_pct > 15 else bar_color)
        ax.set_xlabel("Probabilitas (%)")
        ax.set_title("Tingkat Risiko Kredit", fontweight="bold", fontsize=11)
        ax.spines[["top","right","left"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.info("Geser slider di atas untuk melihat perubahan prediksi secara real-time — tidak perlu klik tombol apapun.")

    except Exception as e:
        st.error(f"Error simulasi: {e}")


# ────────────────────────────────────────────────────────────
# Halaman: Upload CSV
# ────────────────────────────────────────────────────────────
elif page == "📂 Upload CSV":
    st.markdown("## Prediksi Massal — Upload CSV")
    st.markdown("Upload file CSV berisi data beberapa pemohon sekaligus untuk diprediksi secara batch.")

    if not arts:
        st.error("Model belum tersedia.")
        st.stop()

    # Template download
    st.markdown("### Download Template CSV")
    template_df = pd.DataFrame([{
        "person_age": 30, "person_income": 60000000,
        "person_emp_length": 5, "loan_amnt": 10000000,
        "loan_int_rate": 11.0, "cb_person_cred_hist_length": 3,
        "cb_person_default_on_file": 0, "loan_grade": 1,
        "loan_intent": 0, "person_home_ownership": 0,
    }] * 3)
    csv_template = template_df.to_csv(index=False)
    st.download_button(
        "Download Template CSV",
        csv_template,
        "template_pemohon.csv",
        "text/csv",
    )
    st.caption("Isi file template dengan data pemohon, lalu upload di bawah.")

    st.markdown("### Upload File CSV")
    uploaded = st.file_uploader("Pilih file CSV", type=["csv"])

    if uploaded:
        try:
            df_upload = pd.read_csv(uploaded)
            st.markdown(f"**{len(df_upload)} pemohon terdeteksi**")
            st.dataframe(df_upload.head(5), use_container_width=True)

            if st.button("Proses Prediksi Semua", use_container_width=True):
                results_list = []
                progress = st.progress(0)
                status   = st.empty()

                for i, row in df_upload.iterrows():
                    status.text(f"Memproses pemohon {i+1} dari {len(df_upload)}...")
                    try:
                        row_dict = row.to_dict()
                        row_dict["loan_percent_income"] = row_dict.get("loan_amnt", 0) / (row_dict.get("person_income", 1) + 1)
                        res = predict(row_dict, arts)
                        results_list.append({
                            "No"                   : i + 1,
                            "Usia"                 : int(row_dict.get("person_age", 0)),
                            "Pendapatan (Rp)"      : f"{row_dict.get('person_income',0):,.0f}",
                            "Pinjaman (Rp)"        : f"{row_dict.get('loan_amnt',0):,.0f}",
                            "Suku Bunga (%)"       : row_dict.get("loan_int_rate", 0),
                            "Status"               : "LAYAK" if res["label"] == 0 else "TIDAK LAYAK",
                            "Prob. Gagal Bayar (%)" : f"{res['prob_default']*100:.1f}",
                        })
                    except Exception:
                        results_list.append({
                            "No": i+1, "Status": "ERROR",
                            "Prob. Gagal Bayar (%)": "-"
                        })
                    progress.progress((i + 1) / len(df_upload))

                status.empty()
                progress.empty()
                df_results = pd.DataFrame(results_list)

                # Ringkasan
                st.markdown("### Hasil Prediksi")
                n_layak   = (df_results["Status"] == "LAYAK").sum()
                n_tolak   = (df_results["Status"] == "TIDAK LAYAK").sum()
                r1, r2, r3 = st.columns(3)
                r1.metric("Total Pemohon", len(df_results))
                r2.metric("Layak", n_layak, delta=f"{n_layak/len(df_results)*100:.0f}%")
                r3.metric("Tidak Layak", n_tolak, delta=f"-{n_tolak/len(df_results)*100:.0f}%")

                # Tabel hasil dengan warna
                st.dataframe(
                    df_results.style.apply(
                        lambda x: ["background-color:#f0fdf4;color:#166534" if v == "LAYAK"
                                   else "background-color:#fff1f2;color:#9f1239" if v == "TIDAK LAYAK"
                                   else "" for v in x],
                        subset=["Status"]
                    ),
                    use_container_width=True
                )

                # Download hasil
                csv_out = df_results.to_csv(index=False)
                st.download_button(
                    "Download Hasil Prediksi (CSV)",
                    csv_out,
                    "hasil_prediksi_kredit.csv",
                    "text/csv",
                )

        except Exception as e:
            st.error(f"Error membaca file: {e}")
            st.info("Pastikan format CSV sesuai template yang disediakan.")

# ────────────────────────────────────────────────────────────
# Halaman: Riwayat Prediksi
# ────────────────────────────────────────────────────────────
elif page == "🕓 Riwayat":
    st.markdown("## Riwayat Prediksi Sesi Ini")

    history = st.session_state.get("prediction_history", [])

    if not history:
        st.info("Belum ada prediksi dalam sesi ini. Lakukan prediksi di halaman Prediksi Kredit terlebih dahulu.")
    else:
        # Ringkasan
        n_total = len(history)
        n_layak = sum(1 for h in history if h["Status"] == "LAYAK")
        n_tolak = n_total - n_layak
        avg_score = sum(h["Skor Kredit"] for h in history) / n_total

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Total Prediksi", n_total)
        r2.metric("Layak", n_layak)
        r3.metric("Tidak Layak", n_tolak)
        r4.metric("Rata-rata Skor", f"{avg_score:.0f}/100")

        # Pie chart
        if n_total > 1:
            fig_pie, ax_pie = plt.subplots(figsize=(4, 3))
            ax_pie.pie([n_layak, n_tolak],
                       labels=["Layak", "Tidak Layak"],
                       colors=["#22c55e", "#f43f5e"],
                       autopct="%1.0f%%", startangle=90,
                       textprops={"fontsize": 10})
            ax_pie.set_title("Distribusi Hasil", fontweight="bold")
            col_pie, col_empty = st.columns([1, 2])
            with col_pie:
                st.pyplot(fig_pie)
            plt.close()

        # Tabel riwayat
        st.markdown("### Detail Riwayat")
        df_hist = pd.DataFrame(history)
        st.dataframe(
            df_hist.style.apply(
                lambda x: ["background-color:#f0fdf4;color:#166534" if v == "LAYAK"
                           else "background-color:#fff1f2;color:#9f1239" if v == "TIDAK LAYAK"
                           else "" for v in x],
                subset=["Status"]
            ),
            use_container_width=True
        )

        # Download riwayat
        csv_hist = pd.DataFrame(history).to_csv(index=False)
        col_dl1, col_dl2 = st.columns([1, 3])
        with col_dl1:
            st.download_button(
                "Download Riwayat CSV",
                csv_hist,
                "riwayat_prediksi.csv",
                "text/csv",
            )
        with col_dl2:
            if st.button("Hapus Riwayat"):
                st.session_state["prediction_history"] = []
                st.rerun()


# ────────────────────────────────────────────────────────────
# Halaman: Info Model
# ────────────────────────────────────────────────────────────
elif page == "📊 Info Model":
    st.markdown("## 📊 Informasi Model")

    if arts:
        meta = arts["meta"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Model",     meta["model_name"])
        c2.metric("AUC-ROC",   f"{meta['auc_roc']:.4f}")
        c3.metric("F1-Score",  f"{meta['f1_score']:.4f}")

        st.markdown("### Fitur Paling Berpengaruh")
        st.markdown("Berdasarkan analisis SHAP pada data uji:")
        for i, feat in enumerate(meta["top_features"][:10], 1):
            st.markdown(f"**{i}.** `{feat}`")
    else:
        st.info("Jalankan notebook training terlebih dahulu untuk melihat info model.")

    st.markdown("---")
    st.markdown("### Dataset yang Digunakan")
    st.markdown("""
    | Dataset | Sumber | Ukuran | Keterangan |
    |---|---|---|---|
    | Home Credit Default Risk | Kaggle | 307K baris | Dataset utama training |
    | German Credit (UCI) | Kaggle / UCI | 1.000 baris | Validasi eksternal |
    """)

    st.markdown("### Cara Deploy ke Internet (Streamlit Cloud)")
    st.code("""
# 1. Push kode ke GitHub:
#    - app.py
#    - requirements.txt
#    - model_artifacts/ (folder hasil training)

# 2. Buka https://share.streamlit.io
# 3. Connect repo GitHub
# 4. Tambahkan GEMINI_API_KEY di Settings > Secrets:
#    GEMINI_API_KEY = "isi_api_key_kamu"
# 5. Deploy — aplikasi langsung online dan bisa dibuka dari HP!
    """, language="bash")
