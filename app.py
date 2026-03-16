"""
app.py — Aplikasi Web Klasifikasi Kelayakan Kredit Mikro
Jalankan: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle, json, os
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

    st.markdown('<div style="font-size:0.7rem;font-weight:600;letter-spacing:0.1em;color:#475569;text-transform:uppercase;margin-bottom:8px;">Navigasi</div>', unsafe_allow_html=True)
    page = st.radio("", ["🏠 Beranda", "📋 Prediksi Kredit", "💬 Konsultasi AI", "📊 Info Model"],
                    label_visibility="collapsed")

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
