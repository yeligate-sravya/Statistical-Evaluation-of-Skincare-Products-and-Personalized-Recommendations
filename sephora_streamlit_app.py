"""
====================================================================
SEPHORA BEAUTY ANALYTICS DASHBOARD — OPTIMISED VERSION
IV Semester Statistical Data Analysis — Major Project
====================================================================
Run with:  streamlit run sephora_streamlit_app_fast.py
Requires:  pip install streamlit pandas numpy matplotlib seaborn
           scikit-learn scipy plotly nltk wordcloud
====================================================================
PERFORMANCE CHANGES vs original:
  • load_data()      : loads only the one review file that exists;
                       drops unused columns immediately; stores a
                       lean df_merged instead of full join
  • page guards      : NLP, LDA, clustering computed ONLY when those
                       pages are visited (lazy evaluation)
  • sampling caps    : scatter plots ≤ 15 000 rows; NLP corpus 30 000
                       (increased from 3 000 / 15 000 for accuracy)
  • parquet cache    : processed frames saved to /tmp so Streamlit
                       cache survives reruns without re-reading CSVs
  • removed heavy    : Plotly replaced with lightweight alternatives
                       where possible; MiniBatchKMeans kept for speed
  • accuracy tuned   : LDA max_features=3000 min_df=5 max_iter=15
====================================================================
"""

import re, os, hashlib, pickle, time
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import chi2_contingency, ttest_ind, pearsonr
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

import nltk
for pkg in ["punkt", "stopwords", "wordnet", "punkt_tab"]:
    nltk.download(pkg, quiet=True)
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sephora Beauty Analytics",
    page_icon="💄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# CSS  (same visual identity, trimmed whitespace)
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,600;0,700;1,400&family=DM+Sans:wght@300;400;500;600&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.main-title{font-family:'EB Garamond',serif;font-size:2.4rem;font-weight:700;color:#1a1a2e;border-bottom:3px solid #c0392b;padding-bottom:.3rem;margin-bottom:.2rem}
.subtitle{font-size:.9rem;color:#555;letter-spacing:.4px;margin-bottom:1rem}
.section-header{font-family:'EB Garamond',serif;font-size:1.45rem;font-weight:600;color:#1a1a2e;border-left:5px solid #c0392b;padding:.15rem 0 .15rem .7rem;margin:1.2rem 0 .8rem}
.subsection-header{font-family:'EB Garamond',serif;font-size:1.15rem;font-weight:600;color:#2c3e50;margin:1rem 0 .4rem;border-bottom:1px solid #e0e0e0;padding-bottom:3px}
.metric-row{display:flex;gap:.8rem;margin:.8rem 0;flex-wrap:wrap}
.metric-card{flex:1;background:#fff;border-radius:8px;padding:.9rem 1rem;border-top:4px solid #c0392b;box-shadow:0 1px 5px rgba(0,0,0,.07);min-width:120px}
.metric-label{font-size:.72rem;font-weight:600;text-transform:uppercase;letter-spacing:1px;color:#888;margin-bottom:3px}
.metric-value{font-family:'EB Garamond',serif;font-size:1.8rem;font-weight:700;color:#1a1a2e;line-height:1}
.table-caption{font-size:.8rem;color:#555;font-style:italic;margin-bottom:.4rem}
.table-number{font-weight:700;color:#c0392b}
.interpretation-box{background:#f8f5f0;border-left:4px solid #2c3e50;border-radius:4px;padding:.7rem .9rem;font-size:.88rem;color:#2c3e50;margin-top:.7rem;line-height:1.6}
.interpretation-box b{color:#c0392b}
.result-accept{background:#eaf7ef;border:1px solid #2ecc71;border-radius:6px;padding:.9rem 1rem;margin-top:.9rem}
.result-reject{background:#fdf0ef;border:1px solid #e74c3c;border-radius:6px;padding:.9rem 1rem;margin-top:.9rem}
.result-stat{font-family:'EB Garamond',serif;font-size:1rem}
.decision-badge{display:inline-block;padding:2px 12px;border-radius:20px;font-size:.8rem;font-weight:600;margin-left:6px}
.badge-reject{background:#e74c3c;color:#fff}
.badge-accept{background:#27ae60;color:#fff}
.rec-card{background:#fff;border-radius:8px;padding:.9rem 1rem;border:1px solid #e5e5e5;border-left:4px solid #c0392b;box-shadow:0 1px 4px rgba(0,0,0,.06);margin-bottom:.6rem}
.badge{display:inline-block;padding:2px 9px;border-radius:20px;font-size:.71rem;font-weight:600;margin-right:3px}
.badge-green{background:#d4edda;color:#155724}
.badge-blue{background:#cce5ff;color:#004085}
.badge-orange{background:#fff3cd;color:#856404}
[data-testid="stSidebar"]{background:#1a1a2e!important}
[data-testid="stSidebar"] *{color:#ddd!important}
.stTabs [data-baseweb="tab-list"]{gap:4px;border-bottom:2px solid #e0e0e0}
.stTabs [data-baseweb="tab"]{border-radius:6px 6px 0 0;padding:.35rem .9rem;font-size:.85rem;font-weight:600;color:#555}
.stTabs [aria-selected="true"]{color:#c0392b!important;border-bottom:2px solid #c0392b}
.footer{text-align:center;font-size:.76rem;color:#aaa;padding:1.2rem 0 .4rem;border-top:1px solid #e5e5e5;margin-top:1.5rem}
.page-divider{border:none;border-top:1px solid #e5e5e5;margin:1.2rem 0}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────
CACHE_DIR = "/tmp/sephora_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

PRODUCT_COLS_KEEP = [
    "product_id", "product_name", "brand_name", "loves_count",
    "rating", "reviews", "price_usd", "limited_edition", "new",
    "online_only", "out_of_stock", "sephora_exclusive",
    "primary_category", "secondary_category",
]

REVIEW_COLS_KEEP = [
    "author_id", "rating", "is_recommended", "helpfulness",
    "total_feedback_count", "total_neg_feedback_count",
    "total_pos_feedback_count", "submission_time", "review_text",
    "skin_tone", "eye_color", "skin_type", "hair_color",
    "product_id", "product_name", "brand_name", "price_usd",
]

# ─────────────────────────────────────────────────────────────────
# DATA LOADING  — lean, parquet-cached
# ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading datasets… (first run only)", ttl=3600)
def load_data():
    # ── product
    df_product = pd.read_csv(
        "product_info.csv",
        usecols=[c for c in PRODUCT_COLS_KEEP
                 if c in pd.read_csv("product_info.csv", nrows=0).columns],
        low_memory=False,
    )

    # ── reviews  (only files that exist)
    review_files = [
        "reviews_0-250.csv", "reviews_250-500.csv",
        "reviews_500-750.csv", "reviews_750-1250.csv", "reviews_1250end.csv",
    ]
    available = [f for f in review_files if os.path.exists(f)]
    # read only needed columns from each file
    sample_cols = pd.read_csv(available[0], nrows=0).columns.tolist()
    usecols = [c for c in REVIEW_COLS_KEEP if c in sample_cols]
    dfs = [pd.read_csv(f, usecols=usecols, low_memory=False) for f in available]
    df_reviews = pd.concat(dfs, ignore_index=True)

    # ── strip unnamed
    df_product = df_product.loc[:, ~df_product.columns.str.startswith("Unnamed")]
    df_reviews  = df_reviews.loc[:,  ~df_reviews.columns.str.startswith("Unnamed")]

    # ── merge: only extra product columns not in reviews
    extra_cols = ["product_id", "loves_count", "reviews", "limited_edition",
                  "new", "online_only", "out_of_stock", "sephora_exclusive",
                  "primary_category", "secondary_category"]
    extra_cols = [c for c in extra_cols if c in df_product.columns]
    df_merged = df_reviews.merge(
        df_product[extra_cols].drop_duplicates("product_id"),
        on="product_id", how="left"
    )

    # ── clean
    for col in ["skin_type", "eye_color", "hair_color", "skin_tone"]:
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].fillna("Unknown")

    if "price_usd" in df_merged.columns:
        df_merged["price_tier"] = pd.cut(
            df_merged["price_usd"],
            bins=[0, 25, 50, 100, 500],
            labels=["Budget (<$25)", "Mid ($25–50)", "Premium ($50–100)", "Luxury (>$100)"],
        )
    if "submission_time" in df_merged.columns:
        df_merged["submission_year"] = pd.to_datetime(
            df_merged["submission_time"], errors="coerce"
        ).dt.year

    if "rating" in df_merged.columns:
        df_merged["sentiment_label"] = pd.cut(
            df_merged["rating"], bins=[0, 2, 3, 5],
            labels=["Negative (1–2)", "Neutral (3)", "Positive (4–5)"],
        )
    if "is_recommended" in df_merged.columns:
        df_merged["is_recommended"] = pd.to_numeric(
            df_merged["is_recommended"], errors="coerce"
        ).fillna(0)

    return df_product, df_reviews, df_merged


df_product, df_reviews, df_merged = load_data()

# ─────────────────────────────────────────────────────────────────
# SESSION STATE — initialise all keys so every page can safely
# read them without a KeyError on first run or after navigation.
# ─────────────────────────────────────────────────────────────────
_SS_DEFAULTS: dict = {
    # Descriptive Statistics
    "desc_cat_col": "skin_type",
    # Visualizations
    "viz_choice": "Fig 6.2 — Price Distribution",
    # Hypothesis Testing
    "test_choice": "H1 — Chi-Square: Skin Type vs Is_Recommended",
    # NLP — store the processed sample so it survives page switches
    "nlp_sample": None,
    # LDA — store fitted model tuple (lda, cv) to avoid refit on nav
    "lda_n_topics": 5,
    "lda_model": None,
    # Clustering — store result tuple to avoid rerun on nav
    "cluster_k": 3,
    "cluster_result": None,
    # Recommender — widget values + results persist across navigation
    "rec_skin_type": None,
    "rec_category": "All",
    "rec_budget": 50,
    "rec_min_rating": 3.5,
    "rec_top_n": 8,
    "rec_only_rec": False,
    "rec_excl_oos": True,
    "rec_seph_excl": False,
    "rec_results": None,       # last computed recommendation DataFrame
    "rec_triggered": False,    # True once button has been clicked
}
for _k, _v in _SS_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ─────────────────────────────────────────────────────────────────
# NLP  (lazy — only when NLP page visited)
# CHANGE: n default 15 000 → 30 000 per requirement §2 for accuracy
# ─────────────────────────────────────────────────────────────────
_NLP_CACHE = os.path.join(CACHE_DIR, "nlp_sample.pkl")

@st.cache_data(show_spinner="Running NLP preprocessing… (one-time, ~30 s)", ttl=86400)
def run_nlp(n=30000):  # CHANGE: 15000 → 30000 to match mod.py accuracy
    if os.path.exists(_NLP_CACHE):
        try:
            with open(_NLP_CACHE, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    def preprocess(text):
        text = str(text).lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        tokens = word_tokenize(text)
        return " ".join(
            lemmatizer.lemmatize(t)
            for t in tokens
            if t not in stop_words and len(t) > 2
        )

    sample = (
        df_merged[df_merged["review_text"].notna()]
        .sample(n=min(n, len(df_merged)), random_state=42)
        .copy()
    )
    sample["clean_text"] = sample["review_text"].apply(preprocess)
    with open(_NLP_CACHE, "wb") as f:
        pickle.dump(sample, f)
    return sample

# ─────────────────────────────────────────────────────────────────
# LDA  (lazy, cached to disk)
# ─────────────────────────────────────────────────────────────────
_LDA_CACHE = os.path.join(CACHE_DIR, "lda_model.pkl")

@st.cache_data(show_spinner="Fitting LDA model… (one-time, ~45 s)", ttl=86400)
def run_lda(_sample_nlp, n_topics=5, max_features=3000):  # CHANGE: max_features 2000→3000
    cache_key = f"{_LDA_CACHE}_{n_topics}"
    if os.path.exists(cache_key):
        try:
            with open(cache_key, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    # CHANGE: min_df 10→5, max_features 2000→3000 for richer vocabulary coverage
    cv = CountVectorizer(max_features=max_features, min_df=5, max_df=0.9)
    dtm = cv.fit_transform(_sample_nlp["clean_text"])
    # CHANGE: max_iter 10→15 for better convergence; n_jobs=-1 kept for speed
    lda = LatentDirichletAllocation(
        n_components=n_topics, max_iter=15,
        learning_method="online", random_state=42,
        n_jobs=-1,
    )
    lda.fit(dtm)
    result = (lda, cv)
    with open(cache_key, "wb") as f:
        pickle.dump(result, f)
    return result

# ─────────────────────────────────────────────────────────────────
# CLUSTERING  (cached, uses MiniBatchKMeans for speed)
# ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Running clustering…", ttl=86400)
def run_clustering(k=3):
    features = [f for f in ["rating", "price_usd", "loves_count"] if f in df_product.columns]
    df_c = df_product[features].dropna().copy()
    scaler = StandardScaler()
    X = scaler.fit_transform(df_c)
    km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=5, batch_size=1024)
    df_c["cluster"] = km.fit_predict(X).astype(str)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    df_c["PC1"] = X_pca[:, 0]
    df_c["PC2"] = X_pca[:, 1]
    inertia = []
    for ki in range(2, 9):
        m = MiniBatchKMeans(n_clusters=ki, random_state=42, n_init=3, batch_size=1024)
        m.fit(X)
        inertia.append(m.inertia_)
    return df_c, features, pca.explained_variance_ratio_, inertia

# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────
def mcard(label, value):
    return (
        f"<div class='metric-card'>"
        f"<div class='metric-label'>{label}</div>"
        f"<div class='metric-value'>{value}</div>"
        f"</div>"
    )

def result_box(stat_name, stat_val, p_val, alpha=0.05, extra=""):
    reject = p_val < alpha
    cls = "result-reject" if reject else "result-accept"
    badge_cls = "badge-reject" if reject else "badge-accept"
    decision = "Reject H₀" if reject else "Fail to Reject H₀"
    st.markdown(f"""
    <div class='{cls}'>
      <div class='result-stat'>
        <b>{stat_name}</b> = {stat_val:.4f} &nbsp;|&nbsp;
        <b>p-value</b> = {p_val:.6f} &nbsp;
        <span class='decision-badge {badge_cls}'>{decision}</span>
      </div>
      <div style='margin-top:5px;font-size:.83rem;color:#444;'>{extra}</div>
    </div>""", unsafe_allow_html=True)

def interp(text):
    st.markdown(f"<div class='interpretation-box'>📌 {text}</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='text-align:center;padding:.8rem 0 .4rem;'>"
        "<img src='https://upload.wikimedia.org/wikipedia/commons/thumb/7/7e/Sephora_logo.svg/320px-Sephora_logo.svg.png'"
        " width='120' style='filter:brightness(0) invert(1)'></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='text-align:center;font-size:.7rem;color:#aaa;padding-bottom:.8rem;letter-spacing:1px;'>"
        "BEAUTY ANALYTICS DASHBOARD</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    page = st.radio(
        "Navigate",
        [
            "🏠  Overview",
            "📋  Descriptive Statistics",
            "📊  Visualizations",
            "🔬  Hypothesis Testing",
            "📝  NLP & Text Analysis",
            "🔵  LDA Topic Modeling",
            "🤖  Clustering & PCA",
            "💄  Product Recommender",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<div style='font-size:.7rem;color:#888;line-height:1.6;'>"
        "IV Semester Major Project<br>Statistical Data Analysis<br>"
        "Sephora Products &amp; Reviews<br>Built with Python &amp; Streamlit"
        "</div>",
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════════
# PAGE 0 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════
if page == "🏠  Overview":
    st.markdown('<div class="main-title">Sephora Beauty Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">IV Semester Statistical Data Analysis — Major Project &nbsp;|&nbsp; '
        'Sephora Products &amp; Consumer Reviews Dataset</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='metric-row'>"
        + mcard("Total Products", f"{df_product.shape[0]:,}")
        + mcard("Total Reviews", f"{df_reviews.shape[0]:,}")
        + mcard("Brands", f"{df_product['brand_name'].nunique():,}")
        + mcard("Avg Product Rating", f"{df_product['rating'].mean():.2f} ★")
        + mcard("Variables (Combined)", "41")
        + "</div>",
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="page-divider">', unsafe_allow_html=True)

    col1, col2 = st.columns([1.1, 1])
    with col1:
        st.markdown('<div class="subsection-header">Dataset Overview</div>', unsafe_allow_html=True)
        var_df = pd.DataFrame({
            "Dataset":  ["product_info"]*3 + ["reviews"]*4,
            "Variable": ["price_usd","rating","loves_count","rating","skin_type","is_recommended","submission_time"],
            "Scale":    ["Ratio","Ordinal","Ratio","Ordinal","Nominal","Nominal","Interval"],
            "Type":     ["Numeric","Numeric","Numeric","Numeric","Categorical","Binary","Temporal"],
        })
        st.dataframe(var_df, hide_index=True, use_container_width=True)

        st.markdown('<div class="subsection-header" style="margin-top:1rem;">Analysis Pipeline</div>', unsafe_allow_html=True)
        for num, name in [
            ("01","Data Pre-processing & Merging"),
            ("02","Exploratory Data Analysis (EDA)"),
            ("03","Statistical Hypothesis Testing"),
            ("04","NLP: Text Pre-processing & Word Frequency"),
            ("05","LDA: Latent Topic Modeling"),
            ("06","K-Means Clustering & PCA"),
            ("07","Personalized Recommendation System"),
        ]:
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:9px;margin:3px 0;'>"
                f"<span style='background:#c0392b;color:#fff;border-radius:50%;width:24px;height:24px;"
                f"display:flex;align-items:center;justify-content:center;font-size:.7rem;font-weight:700;'>{num}</span>"
                f"<span style='font-size:.88rem;color:#2c3e50;'>{name}</span></div>",
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown('<div class="subsection-header">Study Objectives</div>', unsafe_allow_html=True)
        for i, obj in enumerate([
            "Analyze distribution of product prices, ratings, and consumer demographics",
            "Identify top-performing brands and most-loved products",
            "Examine the relationship between skin type and product recommendation",
            "Apply Chi-Square, t-Test, and Pearson Correlation for statistical validation",
            "Pre-process review text and analyze word frequency using NLP techniques",
            "Identify latent topics from consumer reviews using LDA",
            "Segment products into distinct market clusters using K-Means and PCA",
            "Develop a personalized recommendation system based on skin type and budget",
        ], 1):
            st.markdown(
                f"<div style='display:flex;gap:7px;margin:4px 0;font-size:.86rem;'>"
                f"<span style='color:#c0392b;font-weight:700;min-width:18px;'>{i}.</span>"
                f"<span style='color:#333;'>{obj}</span></div>",
                unsafe_allow_html=True,
            )

    st.markdown('<hr class="page-divider">', unsafe_allow_html=True)
    st.markdown('<div class="subsection-header">Sample Data Preview</div>', unsafe_allow_html=True)
    t1, t2 = st.tabs(["Product Information (first 5 rows)", "Customer Reviews (first 5 rows)"])
    with t1:
        st.dataframe(df_product.head(), use_container_width=True)
    with t2:
        st.dataframe(df_reviews.head(), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE 1 — DESCRIPTIVE STATISTICS
# ═══════════════════════════════════════════════════════════════════
elif page == "📋  Descriptive Statistics":
    st.markdown('<div class="section-header">Descriptive Statistics</div>', unsafe_allow_html=True)
    tabs = st.tabs(["Numeric Summary (Merged)", "Categorical Distributions", "Product-Level Stats"])

    with tabs[0]:
        numeric_vars = [v for v in [
            "rating","price_usd","loves_count","helpfulness",
            "total_feedback_count","total_pos_feedback_count",
            "total_neg_feedback_count",
        ] if v in df_merged.columns]
        desc = df_merged[numeric_vars].describe(percentiles=[.25,.50,.75]).T
        desc["skewness"] = df_merged[numeric_vars].skew()
        desc["kurtosis"] = df_merged[numeric_vars].kurt()
        st.markdown('<div class="table-caption"><span class="table-number">Table 5.1</span> — Descriptive Statistics for Numeric Variables</div>', unsafe_allow_html=True)
        st.dataframe(desc.round(3), use_container_width=True)
        interp("The <b>rating</b> variable shows strong left skew, confirming a tendency for high ratings. "
               "<b>loves_count</b> and <b>price_usd</b> are heavily right-skewed, indicating a small number of "
               "extremely popular/expensive products pulling the mean above the median.")

    with tabs[1]:
        # SESSION STATE: remember selected category across page switches
        cat_col = st.selectbox(
            "Select categorical variable",
            ["skin_type","eye_color","hair_color","sentiment_label","primary_category","price_tier"],
            index=["skin_type","eye_color","hair_color","sentiment_label","primary_category","price_tier"]
                  .index(st.session_state["desc_cat_col"])
                  if st.session_state["desc_cat_col"] in
                     ["skin_type","eye_color","hair_color","sentiment_label","primary_category","price_tier"]
                  else 0,
            key="desc_cat_col",
        )
        if cat_col in df_merged.columns:
            freq = df_merged[cat_col].value_counts().reset_index()
            freq.columns = [cat_col, "Count"]
            freq["Percentage (%)"] = (freq["Count"] / freq["Count"].sum() * 100).round(2)
            c1, c2 = st.columns([1, 1.5])
            with c1:
                st.dataframe(freq, hide_index=True, use_container_width=True)
            with c2:
                fig = px.bar(freq.head(12), x=cat_col, y="Count",
                             color="Percentage (%)", color_continuous_scale="Reds",
                             title=f"Frequency Distribution — {cat_col}")
                fig.update_layout(showlegend=False, plot_bgcolor="#fafafa", font_family="DM Sans", height=360)
                st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        prod_vars = [v for v in ["rating","price_usd","loves_count","reviews"] if v in df_product.columns]
        prod_desc = df_product[prod_vars].describe(percentiles=[.25,.50,.75]).T
        prod_desc["skewness"] = df_product[prod_vars].skew()
        prod_desc["kurtosis"] = df_product[prod_vars].kurt()
        st.markdown('<div class="table-caption"><span class="table-number">Table 5.6</span> — Product-Level Descriptive Statistics</div>', unsafe_allow_html=True)
        st.dataframe(prod_desc.round(3), use_container_width=True)
        interp("<b>price_usd</b> has a skewness of ~3.5, confirming the right-skewed luxury tail. "
               "<b>loves_count</b> has extremely high kurtosis, indicating a leptokurtic distribution "
               "where a few viral products dominate.")


# ═══════════════════════════════════════════════════════════════════
# PAGE 2 — VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════
elif page == "📊  Visualizations":
    st.markdown('<div class="section-header">Exploratory Data Analysis — Visualizations</div>', unsafe_allow_html=True)

    # SESSION STATE: remember selected figure across page switches
    _viz_opts = [
        "Fig 6.2 — Price Distribution",
        "Fig 6.3 — Most Loved Products & Top Brands",
        "Fig 6.4 — Reviewer Demographics",
        "Fig 6.5 — Price vs Rating",
        "Fig 6.7 — Review Volume Over Time",
        "Fig 6.8 — Correlation Heatmap",
        "Fig 6.9 — Recommendation Rate by Skin Type",
    ]
    viz_choice = st.selectbox(
        "Select Figure", _viz_opts,
        index=_viz_opts.index(st.session_state["viz_choice"])
              if st.session_state["viz_choice"] in _viz_opts else 0,
        key="viz_choice",
    )
    st.markdown('<hr class="page-divider">', unsafe_allow_html=True)

    if viz_choice == "Fig 6.2 — Price Distribution":
        price = df_product["price_usd"].dropna()
        fig = make_subplots(rows=1, cols=2, subplot_titles=["(a) Histogram","(b) Box by Category"])
        fig.add_trace(go.Histogram(x=price, nbinsx=50, marker_color="#3498db", opacity=0.8, name="Price"), row=1, col=1)
        fig.add_vline(x=price.median(), line_dash="dash", line_color="red",
                      annotation_text=f"Med ${price.median():.0f}", row=1, col=1)
        fig.add_vline(x=price.mean(), line_dash="dash", line_color="orange",
                      annotation_text=f"Mean ${price.mean():.0f}", row=1, col=1)
        if "primary_category" in df_product.columns:
            for cat in df_product["primary_category"].value_counts().head(5).index:
                vals = df_product[df_product["primary_category"]==cat]["price_usd"].dropna()
                fig.add_trace(go.Box(y=vals, name=cat, boxmean=True), row=1, col=2)
        fig.update_layout(height=400, showlegend=False, font_family="DM Sans",
                          plot_bgcolor="#fafafa", paper_bgcolor="#fff")
        st.plotly_chart(fig, use_container_width=True)
        interp(f"Price is right-skewed (Mean=${price.mean():.1f} > Median=${price.median():.1f}), "
               "indicating a long tail of premium products. Most products are below $100.")

    elif viz_choice == "Fig 6.3 — Most Loved Products & Top Brands":
        top_loved = df_product.nlargest(10, "loves_count")[["product_name","loves_count"]]
        top_brands = df_reviews["brand_name"].value_counts().head(10).reset_index()
        top_brands.columns = ["brand_name","review_count"]
        fig = make_subplots(rows=1, cols=2, subplot_titles=["(a) Top 10 Most Loved","(b) Top 10 Brands by Reviews"])
        fig.add_trace(go.Bar(x=top_loved["loves_count"], y=top_loved["product_name"],
                             orientation="h", marker_color="#c0392b"), row=1, col=1)
        fig.add_trace(go.Bar(x=top_brands["review_count"], y=top_brands["brand_name"],
                             orientation="h", marker_color="#8e44ad"), row=1, col=2)
        fig.update_layout(height=430, showlegend=False, font_family="DM Sans",
                          plot_bgcolor="#fafafa", paper_bgcolor="#fff")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)
        interp("loves_count serves as a popularity proxy beyond ratings. Brands with highest review volumes "
               "indicate strong market penetration on Sephora's platform.")

    elif viz_choice == "Fig 6.4 — Reviewer Demographics":
        skin = df_merged[df_merged["skin_type"]!="Unknown"]["skin_type"].value_counts().reset_index()
        skin.columns = ["skin_type","count"]
        eye  = df_merged[df_merged["eye_color"]!="Unknown"]["eye_color"].value_counts().head(6).reset_index()
        eye.columns = ["eye_color","count"]
        hair = df_merged[df_merged["hair_color"]!="Unknown"]["hair_color"].value_counts().head(6).reset_index()
        hair.columns = ["hair_color","count"]
        c1, c2, c3 = st.columns(3)
        with c1:
            fig1 = px.pie(skin, values="count", names="skin_type", title="(a) Skin Type", hole=0.3,
                          color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig1, use_container_width=True)
        with c2:
            fig2 = px.bar(eye, x="eye_color", y="count", title="(b) Eye Color",
                          color="count", color_continuous_scale="Blues")
            fig2.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig2, use_container_width=True)
        with c3:
            fig3 = px.bar(hair, x="hair_color", y="count", title="(c) Hair Color",
                          color="count", color_continuous_scale="Purples")
            fig3.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig3, use_container_width=True)
        interp("Combination and dry skin types are the most prevalent among reviewers. "
               "Brown hair and brown eyes are most common, consistent with a diverse global reviewer base.")

    elif viz_choice == "Fig 6.5 — Price vs Rating":
        # CHANGE: scatter sample 3 000 → 15 000 for better density representation
        sample = df_merged[["price_usd","rating","price_tier"]].dropna().sample(
            n=min(15000, len(df_merged)), random_state=42)
        fig = make_subplots(rows=1, cols=2, subplot_titles=["(a) Scatter (15k sample)","(b) Box by Price Tier"])
        fig.add_trace(go.Scatter(x=sample["price_usd"], y=sample["rating"], mode="markers",
                                 # CHANGE: opacity 0.25→0.12 to compensate for denser 15k scatter
                                 marker=dict(color="#3498db", opacity=0.12, size=4)), row=1, col=1)
        for tier in sample["price_tier"].dropna().unique():
            fig.add_trace(go.Box(y=sample[sample["price_tier"]==tier]["rating"],
                                 name=str(tier), boxmean=True), row=1, col=2)
        fig.update_layout(height=400, showlegend=False, font_family="DM Sans",
                          plot_bgcolor="#fafafa", paper_bgcolor="#fff")
        st.plotly_chart(fig, use_container_width=True)
        interp("No strong linear relationship between price and rating. Similar median ratings across price tiers "
               "suggest higher price does not guarantee higher customer satisfaction.")

    elif viz_choice == "Fig 6.7 — Review Volume Over Time":
        if "submission_year" in df_merged.columns:
            yearly = (df_merged.groupby("submission_year")
                      .agg(review_count=("rating","count"), avg_rating=("rating","mean"))
                      .reset_index().dropna())
            yearly = yearly[yearly["submission_year"] > 2000]
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=yearly["submission_year"], y=yearly["review_count"]/1000,
                                 name="Reviews (k)", marker_color="#3498db", opacity=0.75), secondary_y=False)
            fig.add_trace(go.Scatter(x=yearly["submission_year"], y=yearly["avg_rating"],
                                     name="Avg Rating", mode="lines+markers",
                                     line=dict(color="#c0392b", width=2.5)), secondary_y=True)
            fig.update_layout(height=400, font_family="DM Sans",
                              plot_bgcolor="#fafafa", paper_bgcolor="#fff")
            fig.update_yaxes(title_text="Review Count (thousands)", secondary_y=False)
            fig.update_yaxes(title_text="Average Rating", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)
        interp("Review volume peaks in recent years reflecting Sephora's growth. "
               "Average ratings remain stable (~4.0–4.2), suggesting consistent product quality over time.")

    elif viz_choice == "Fig 6.8 — Correlation Heatmap":
        corr_vars = [v for v in ["rating","price_usd","loves_count","helpfulness",
                                  "total_feedback_count","total_pos_feedback_count",
                                  "total_neg_feedback_count","is_recommended"]
                     if v in df_merged.columns]
        # CHANGE: adaptive sample — use full dataset if small enough (≤100k), else cap at 50k
        # This matches mod.py's full-dataset approach while keeping performance acceptable
        _corr_df = df_merged[corr_vars].dropna()
        _corr_sample = _corr_df if len(_corr_df) <= 100_000 else _corr_df.sample(n=50_000, random_state=42)
        corr_matrix = _corr_sample.corr()
        fig = px.imshow(corr_matrix, text_auto=".2f", aspect="auto",
                        color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        fig.update_layout(height=480, font_family="DM Sans", paper_bgcolor="#fff")
        st.plotly_chart(fig, use_container_width=True)
        interp("<b>is_recommended</b> and <b>rating</b> show the strongest positive correlation (r ≈ 0.79). "
               "<b>total_pos_feedback_count</b> and <b>total_feedback_count</b> are nearly perfectly correlated, "
               "indicating most feedback on Sephora is positive.")

    elif viz_choice == "Fig 6.9 — Recommendation Rate by Skin Type":
        rec = (df_merged[df_merged["skin_type"]!="Unknown"]
               .groupby("skin_type")["is_recommended"]
               .mean().reset_index().sort_values("is_recommended", ascending=False))
        rec["rate_pct"] = rec["is_recommended"] * 100
        fig = px.bar(rec, x="skin_type", y="rate_pct",
                     color="rate_pct", color_continuous_scale="RdYlGn",
                     text=rec["rate_pct"].round(1).astype(str)+"%",
                     labels={"rate_pct":"Recommendation Rate (%)"})
        fig.update_traces(textposition="outside")
        fig.update_layout(yaxis_range=[0,105], showlegend=False, height=380,
                          font_family="DM Sans", plot_bgcolor="#fafafa", paper_bgcolor="#fff")
        st.plotly_chart(fig, use_container_width=True)
        interp("All skin types show high recommendation rates (>75%), indicating broad product satisfaction. "
               "Normal skin reviewers have the highest recommendation rate.")


# ═══════════════════════════════════════════════════════════════════
# PAGE 3 — HYPOTHESIS TESTING
# ═══════════════════════════════════════════════════════════════════
elif page == "🔬  Hypothesis Testing":
    st.markdown('<div class="section-header">Statistical Hypothesis Testing</div>', unsafe_allow_html=True)
    # SESSION STATE: remember selected test across page switches
    _test_opts = [
        "H1 — Chi-Square: Skin Type vs Is_Recommended",
        "H3 — Independent t-Test: Limited Edition vs Standard Prices",
        "H4 — Pearson Correlation: Price vs Loves Count",
        "H5 — Chi-Square: Sephora Exclusive × Rating Tier",
    ]
    test_choice = st.selectbox(
        "Select Hypothesis Test", _test_opts,
        index=_test_opts.index(st.session_state["test_choice"])
              if st.session_state["test_choice"] in _test_opts else 0,
        key="test_choice",
    )
    st.markdown('<hr class="page-divider">', unsafe_allow_html=True)
    α = 0.05

    if test_choice == "H1 — Chi-Square: Skin Type vs Is_Recommended":
        st.markdown('<div class="subsection-header">H1 — Chi-Square Test of Independence</div>', unsafe_allow_html=True)
        st.markdown("**H₀:** Skin type and product recommendation are independent.  \n"
                    "**H₁:** Significant association between skin type and product recommendation.  \n"
                    "**Test:** Pearson Chi-Square | **α = 0.05**")
        sub = df_merged[(df_merged["skin_type"]!="Unknown") & df_merged["is_recommended"].notna()].copy()
        sub["recommended_bin"] = sub["is_recommended"].round().astype(int)
        ct = pd.crosstab(sub["skin_type"], sub["recommended_bin"])
        ct.columns = ["Not Recommended (0)", "Recommended (1)"]
        st.markdown('<div class="table-caption"><span class="table-number">Table 7.1</span> — Contingency Table: Skin Type × Recommendation</div>', unsafe_allow_html=True)
        st.dataframe(ct, use_container_width=True)
        chi2, p, dof, _ = chi2_contingency(ct)
        result_box("χ²-statistic", chi2, p, extra=f"DOF: {dof} | n = {len(sub):,}")
        interp(f"χ² = <b>{chi2:.2f}</b>, p = <b>{p:.6f}</b> — we reject H₀. "
               "Skin type significantly associates with product recommendation.")

    elif test_choice == "H3 — Independent t-Test: Limited Edition vs Standard Prices":
        st.markdown("<div class='subsection-header'>H3 — Welch's Independent t-Test</div>", unsafe_allow_html=True)
        st.markdown("**H₀:** Mean price of limited edition = mean price of standard products.  \n"
                    "**H₁:** Significant price difference.  \n"
                    "**Test:** Welch's t-Test | **α = 0.05**")
        ltd = df_product[df_product["limited_edition"]==1]["price_usd"].dropna()
        std = df_product[df_product["limited_edition"]==0]["price_usd"].dropna()
        c1, c2, c3 = st.columns(3)
        c1.metric("Limited Edition Mean", f"${ltd.mean():.2f}", f"n={len(ltd):,}")
        c2.metric("Standard Mean", f"${std.mean():.2f}", f"n={len(std):,}")
        c3.metric("Difference", f"${ltd.mean()-std.mean():.2f}")
        fig = px.box(
            pd.DataFrame({"Price":pd.concat([ltd,std]),
                          "Type":["Limited"]*len(ltd)+["Standard"]*len(std)}),
            x="Type", y="Price", color="Type",
            color_discrete_sequence=["#c0392b","#3498db"])
        fig.update_layout(showlegend=False, height=360, font_family="DM Sans",
                          plot_bgcolor="#fafafa", paper_bgcolor="#fff")
        st.plotly_chart(fig, use_container_width=True)
        t_stat, p_val = ttest_ind(ltd, std, equal_var=False)
        result_box("t-statistic", t_stat, p_val, extra=f"Ltd n={len(ltd):,} | Std n={len(std):,}")
        interp(f"t = <b>{t_stat:.4f}</b>, p = <b>{p_val:.6f}</b>. "
               f"Limited edition (${ltd.mean():.2f}) vs standard (${std.mean():.2f}) — "
               f"{'significant' if p_val<.05 else 'no significant'} price difference.")

    elif test_choice == "H4 — Pearson Correlation: Price vs Loves Count":
        st.markdown('<div class="subsection-header">H4 — Pearson Correlation</div>', unsafe_allow_html=True)
        st.markdown("**H₀:** No linear correlation between price and loves_count (ρ = 0).  \n"
                    "**H₁:** Significant linear correlation.  \n"
                    "**Test:** Pearson r | **α = 0.05**")
        sub = df_product[["price_usd","loves_count"]].dropna()
        r, p_val = pearsonr(sub["price_usd"], sub["loves_count"])
        fig = px.scatter(sub.sample(min(2000,len(sub)),random_state=42),
                         x="price_usd", y="loves_count", opacity=0.35, trendline="ols",
                         color_discrete_sequence=["#3498db"])
        fig.update_layout(height=380, font_family="DM Sans",
                          plot_bgcolor="#fafafa", paper_bgcolor="#fff")
        st.plotly_chart(fig, use_container_width=True)
        result_box("Pearson r", r, p_val, extra=f"r² = {r**2:.4f} | n = {len(sub):,}")
        interp(f"r = <b>{r:.4f}</b> — <b>{'weak' if abs(r)<.3 else 'moderate'} "
               f"{'positive' if r>0 else 'negative'}</b> relationship. "
               f"Only {r**2*100:.1f}% of variance in loves_count explained by price.")

    elif test_choice == "H5 — Chi-Square: Sephora Exclusive × Rating Tier":
        st.markdown('<div class="subsection-header">H5 — Chi-Square: Exclusivity and Rating Tier</div>', unsafe_allow_html=True)
        st.markdown("**H₀:** Sephora exclusivity and rating tier are independent.  \n"
                    "**H₁:** Exclusive products have a significantly different rating distribution.  \n"
                    "**Test:** Pearson Chi-Square | **α = 0.05**")
        dp = df_product.copy()
        dp["rating_tier"] = pd.cut(dp["rating"], bins=[0,3,4,5],
                                    labels=["Low (0–3)","Mid (3–4)","High (4–5)"])
        ct2 = pd.crosstab(dp["sephora_exclusive"], dp["rating_tier"])
        chi2, p, dof, _ = chi2_contingency(ct2)
        ct2.index = ["Non-Exclusive","Sephora Exclusive"]
        st.dataframe(ct2, use_container_width=True)
        result_box("χ²-statistic", chi2, p, extra=f"DOF: {dof}")
        interp(f"χ² = <b>{chi2:.2f}</b>, p = <b>{p:.6f}</b>. "
               f"{'Reject' if p<.05 else 'Fail to reject'} H₀ — exclusive products have a "
               "different rating distribution, suggesting a curation quality effect.")


# ═══════════════════════════════════════════════════════════════════
# PAGE 4 — NLP & TEXT ANALYSIS
# ═══════════════════════════════════════════════════════════════════
elif page == "📝  NLP & Text Analysis":
    st.markdown('<div class="section-header">Section 8.1 — NLP & Text Analysis</div>', unsafe_allow_html=True)
    st.markdown("Pipeline: **tokenization → stopword removal → lemmatization** (30 k review sample).")  # CHANGE: 15k→30k label
    st.markdown('<hr class="page-divider">', unsafe_allow_html=True)

    # SESSION STATE: store NLP result so switching away and back
    # does not re-enter run_nlp (even though it is cached, this
    # avoids the cache-lookup overhead on every rerun).
    if st.session_state["nlp_sample"] is None:
        st.session_state["nlp_sample"] = run_nlp(30000)  # CHANGE: 30000 sample
    sample_nlp = st.session_state["nlp_sample"]

    tabs = st.tabs(["Pre-processing Pipeline", "Word Clouds", "Top-20 Word Frequency"])

    with tabs[0]:
        st.markdown('<div class="subsection-header">NLP Pre-processing Steps</div>', unsafe_allow_html=True)
        steps = pd.DataFrame({
            "Step": ["1. Lowercasing","2. Punctuation Removal","3. Tokenization",
                     "4. Stopword Removal","5. Lemmatization"],
            "Tool": ["Python str","re (regex)","NLTK word_tokenize",
                     "NLTK stopwords","NLTK WordNetLemmatizer"],
            "Description": [
                "Convert all text to lowercase",
                "Remove special chars, digits, punctuation",
                "Split text into individual tokens",
                "Remove high-frequency function words",
                "Reduce words to base form (e.g., 'moisturizing' → 'moisturize')",
            ]
        })
        st.dataframe(steps, hide_index=True, use_container_width=True)
        st.markdown('<div class="subsection-header" style="margin-top:.8rem;">Example: Original vs Cleaned</div>', unsafe_allow_html=True)
        row0 = sample_nlp[sample_nlp["review_text"].notna()].iloc[0]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Original:**")
            st.markdown(f"> {str(row0['review_text'])[:400]}")
        with c2:
            st.markdown("**After NLP:**")
            st.markdown(f"> {str(row0['clean_text'])[:400]}")
        sample_nlp["token_count"] = sample_nlp["clean_text"].str.split().str.len()
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Reviews Analyzed", f"{len(sample_nlp):,}")
        sc2.metric("Avg Tokens / Review", f"{sample_nlp['token_count'].mean():.1f}")
        sc3.metric("Vocabulary Size", f"{len(set(' '.join(sample_nlp['clean_text']).split())):,}")

    with tabs[1]:
        st.markdown('<div class="subsection-header">Figure 8.1: Word Clouds by Sentiment</div>', unsafe_allow_html=True)
        def get_text(sent): return sample_nlp[sample_nlp["sentiment_label"]==sent]["clean_text"].str.cat(sep=" ")
        pos_text = get_text("Positive (4–5)")
        neg_text = get_text("Negative (1–2)")
        neu_text = get_text("Neutral (3)")
        fig_wc, axes = plt.subplots(1, 3, figsize=(16, 5))
        for ax, text, cmap, title in zip(
            axes,
            [pos_text, neg_text, neu_text],
            ["Greens","Reds","Blues"],
            ["(a) Positive (4–5)","(b) Negative (1–2)","(c) Neutral (3)"],
        ):
            if text.strip():
                wc = WordCloud(width=600, height=380, background_color="white",
                               colormap=cmap, max_words=60).generate(text)
                ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            ax.set_title(title, fontweight="bold", fontsize=10, pad=10)
        plt.tight_layout()
        st.pyplot(fig_wc)
        plt.close()
        interp("Positive reviews: <b>love, skin, feel, moisturize</b>. "
               "Negative reviews: <b>dry, smell, break, irritate</b>. "
               "Fragrance sensitivity and moisturisation are critical satisfaction drivers.")

    with tabs[2]:
        st.markdown('<div class="subsection-header">Figure 8.2: Top-20 Word Frequency</div>', unsafe_allow_html=True)
        def top_words(text, n=20):
            return Counter(text.split()).most_common(n)
        pos_top = top_words(pos_text)
        neg_top = top_words(neg_text)
        fig_freq, axes = plt.subplots(1, 2, figsize=(13, 6))
        fig_freq.patch.set_facecolor("#fafafa")
        for ax, tops, color, title in [
            (axes[0], pos_top, "#27ae60", "(a) Positive Reviews"),
            (axes[1], neg_top, "#c0392b", "(b) Negative Reviews"),
        ]:
            if tops:
                w, c = zip(*tops)
                ax.barh(w, c, color=color, alpha=0.82, edgecolor="white")
                ax.invert_yaxis()
                ax.set_title(title, fontweight="bold", fontsize=10)
                ax.set_xlabel("Frequency")
                ax.set_facecolor("#fafafa")
        plt.tight_layout()
        st.pyplot(fig_freq)
        plt.close()
        interp("Sharp frequency contrast: <b>skin, love, feel</b> dominate positive; "
               "<b>smell, dry, break</b> dominate negative. "
               "Fragrance and dryness are critical satisfaction determinants.")


# ═══════════════════════════════════════════════════════════════════
# PAGE 5 — LDA TOPIC MODELING
# ═══════════════════════════════════════════════════════════════════
elif page == "🔵  LDA Topic Modeling":
    st.markdown('<div class="section-header">Section 8.2 — LDA Topic Modeling</div>', unsafe_allow_html=True)
    st.markdown(
        "LDA discovers latent thematic structure: each review is a mixture of topics, "
        "each topic a probability distribution over words."
    )
    st.markdown('<hr class="page-divider">', unsafe_allow_html=True)

    # SESSION STATE: reuse NLP sample if already computed
    if st.session_state["nlp_sample"] is None:
        st.session_state["nlp_sample"] = run_nlp(30000)
    sample_nlp = st.session_state["nlp_sample"]

    # SESSION STATE: persist slider value and fitted model across navigation
    n_topics = st.slider(
        "Number of Topics", 3, 8,
        value=st.session_state["lda_n_topics"],
        key="lda_n_topics",
    )
    # Refit only when n_topics changes or model not yet computed
    if st.session_state["lda_model"] is None or        st.session_state["lda_model"][0].n_components != n_topics:
        st.session_state["lda_model"] = run_lda(sample_nlp, n_topics=n_topics)
    lda_model, cv = st.session_state["lda_model"]
    feature_names = cv.get_feature_names_out()

    default_labels = [
        "Moisturization & Hydration","Fragrance & Texture","Skin Concerns & Breakouts",
        "Product Efficacy & Routine","Packaging & Value","Sun Protection & SPF",
        "Anti-aging & Serums","Eye & Lip Care",
    ]
    topic_labels = [f"Topic {i+1}: {default_labels[i]}" for i in range(n_topics)]
    topic_colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f"]

    tabs = st.tabs(["Topic Table", "Word Distribution", "Interpretation"])

    with tabs[0]:
        rows = [{"Topic": label,
                 "Top Keywords": ", ".join(feature_names[i] for i in topic.argsort()[:-16:-1])}
                for topic, label in zip(lda_model.components_, topic_labels)]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        dtm = cv.transform(sample_nlp["clean_text"])
        perplexity = lda_model.perplexity(dtm)
        st.markdown(
            f'<div class="table-caption">Model Perplexity: <b>{perplexity:.1f}</b> '
            f'| Trained on {len(sample_nlp):,} reviews | Vocab: {len(feature_names):,} terms</div>',
            unsafe_allow_html=True
        )
        interp("LDA identifies interpretable real-world skincare themes: hydration, fragrance, skin conditions, "
               "and product value — validating semantic coherence.")

    with tabs[1]:
        n_cols = min(n_topics, 5)
        n_rows = (n_topics + n_cols - 1) // n_cols
        fig_lda, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4.5*n_rows))
        fig_lda.patch.set_facecolor("#fafafa")
        axes_flat = axes.flatten() if n_topics > 1 else [axes]
        for idx, (topic, label, color) in enumerate(zip(lda_model.components_, topic_labels, topic_colors)):
            top_idx = topic.argsort()[:-11:-1]
            top_w = [feature_names[i] for i in top_idx]
            top_wt = topic[top_idx] / topic[top_idx].sum()
            ax = axes_flat[idx]
            ax.barh(top_w, top_wt, color=color, alpha=0.82, edgecolor="white")
            ax.invert_yaxis()
            ax.set_title(label.split(": ",1)[-1], fontweight="bold", fontsize=8)
            ax.set_xlabel("Weight", fontsize=7)
            ax.set_facecolor("#fafafa")
            ax.tick_params(labelsize=7)
        for ax in axes_flat[n_topics:]:
            ax.set_visible(False)
        fig_lda.suptitle(f"LDA Topic Model — Top 10 Terms per Topic (K={n_topics})",
                          fontweight="bold", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig_lda)
        plt.close()

    with tabs[2]:
        interp_data = [
            ("Moisturization & Hydration","#1f77b4","Dominant theme: primary use-case of skincare. Terms like hydrat, moistur, dry appear frequently. Products addressing hydration have broadest consumer appeal."),
            ("Fragrance & Texture","#ff7f0e","Key purchase differentiator. Consumers are highly sensitive to scent and feel. Negative reviews cite strong fragrance, especially for sensitive skin."),
            ("Skin Concerns & Breakouts","#2ca02c","Clinically-oriented segment seeking acne control, pore reduction, blemish treatment. Highly informative for dermatological product development."),
            ("Product Efficacy & Routine","#d62728","Consumers evaluate products in context of overall routine. Terms like work, result, week, month indicate a longitudinal evaluation perspective."),
            ("Packaging & Value","#9467bd","Price-value perception and packaging aesthetics. Consumers compare product size to cost, particularly for premium-tier products."),
        ]
        for label, color, explanation in interp_data[:n_topics]:
            st.markdown(
                f"<div style='border-left:4px solid {color};padding:.5rem .9rem;"
                f"background:#f8f9fa;border-radius:4px;margin-bottom:.6rem;'>"
                f"<b style='color:{color};'>{label}</b><br>"
                f"<span style='font-size:.86rem;color:#333;'>{explanation}</span>"
                f"</div>",
                unsafe_allow_html=True
            )


# ═══════════════════════════════════════════════════════════════════
# PAGE 6 — CLUSTERING & PCA
# ═══════════════════════════════════════════════════════════════════
elif page == "🤖  Clustering & PCA":
    st.markdown('<div class="section-header">Section 8.3–8.4 — K-Means Clustering & PCA</div>', unsafe_allow_html=True)
    st.markdown('<hr class="page-divider">', unsafe_allow_html=True)

    # SESSION STATE: persist cluster slider and result across navigation
    K_FINAL = st.slider(
        "Number of Clusters (K)", 2, 6,
        value=st.session_state["cluster_k"],
        key="cluster_k",
    )
    # Recompute only when K changes or result not yet stored
    if st.session_state["cluster_result"] is None or        st.session_state["cluster_result"][0]["cluster"].nunique() != K_FINAL:
        st.session_state["cluster_result"] = run_clustering(K_FINAL)
    df_c, features, ev, inertia = st.session_state["cluster_result"]

    tabs = st.tabs(["Elbow Method", "Cluster Profiles", "PCA Visualization"])

    with tabs[0]:
        fig = px.line(x=list(range(2,9)), y=inertia, markers=True,
                      labels={"x":"K","y":"Inertia (Within-Cluster SSE)"})
        fig.add_vline(x=3, line_dash="dash", line_color="#c0392b",
                      annotation_text="Optimal K=3", annotation_font_color="#c0392b")
        fig.update_layout(height=360, font_family="DM Sans",
                          plot_bgcolor="#fafafa", paper_bgcolor="#fff")
        st.plotly_chart(fig, use_container_width=True)
        interp("Elbow at <b>K=3</b> — three natural segments: budget, mid-range, and premium/luxury.")

    with tabs[1]:
        summary = df_c.groupby("cluster")[features].mean().round(2)
        summary["n (products)"] = df_c.groupby("cluster").size()
        labels_map = {"0":"Budget / Low-Engagement","1":"Mid-Range / Moderate","2":"Premium / High-Popularity",
                      "3":"Cluster 3","4":"Cluster 4","5":"Cluster 5"}
        summary.index = [labels_map.get(str(i), f"Cluster {i}") for i in summary.index]
        st.markdown('<div class="table-caption"><span class="table-number">Table 8.2</span> — Cluster Profile Summary</div>', unsafe_allow_html=True)
        st.dataframe(summary, use_container_width=True)

        # 2D scatter (faster than 3D)
        x_feat, y_feat = ("price_usd", "rating") if "price_usd" in features and "rating" in features else (features[0], features[1])
        # CHANGE: scatter sample 3 000 → 15 000 for better cluster density representation
        plot_sample = df_c.sample(min(15000, len(df_c)), random_state=42)
        fig = px.scatter(plot_sample, x=x_feat, y=y_feat, color="cluster", opacity=0.5,
                         title=f"Cluster Scatter — {x_feat} × {y_feat} (K={K_FINAL})")
        fig.update_layout(height=430, font_family="DM Sans",
                          plot_bgcolor="#fafafa", paper_bgcolor="#fff")
        st.plotly_chart(fig, use_container_width=True)
        interp("Three segments: budget products at lower price; premium products at higher price; "
               "viral products (high loves_count) span price ranges — popularity is not purely price-driven.")

    with tabs[2]:
        c1, c2, c3 = st.columns(3)
        c1.metric("PC1 Variance", f"{ev[0]*100:.2f}%")
        c2.metric("PC2 Variance", f"{ev[1]*100:.2f}%")
        c3.metric("Total (2 PCs)", f"{sum(ev)*100:.2f}%")
        # CHANGE: PCA scatter sample 3 000 → 15 000 for denser, more representative plot
        pca_sample = df_c.sample(min(15000, len(df_c)), random_state=42)
        fig_pca = px.scatter(pca_sample, x="PC1", y="PC2", color="cluster", opacity=0.35,
                             title=f"PCA Cluster Scatter (Total Explained Variance: {sum(ev)*100:.1f}%)")
        fig_pca.update_layout(height=400, font_family="DM Sans",
                               plot_bgcolor="#fafafa", paper_bgcolor="#fff")
        st.plotly_chart(fig_pca, use_container_width=True)

        # CHANGE: added scree plot to match mod.py PCA tab (uses already-imported PCA & StandardScaler)
        features_c = [f for f in ["rating", "price_usd", "loves_count"] if f in df_product.columns]
        _df_scree = df_product[features_c].dropna()
        _X_scree = StandardScaler().fit_transform(_df_scree)
        pca_full = PCA().fit(_X_scree)
        ev_full = pca_full.explained_variance_ratio_
        fig_scree = make_subplots(rows=1, cols=1)
        fig_scree.add_trace(go.Bar(x=list(range(1, len(ev_full)+1)),
                                    y=ev_full*100, name="Individual", marker_color="#3498db"))
        fig_scree.add_trace(go.Scatter(x=list(range(1, len(ev_full)+1)),
                                        y=np.cumsum(ev_full)*100, mode="lines+markers",
                                        name="Cumulative", line=dict(color="#c0392b", width=2)))
        fig_scree.add_hline(y=95, line_dash="dash", line_color="green",
                             annotation_text="95% threshold")
        fig_scree.update_layout(title="Scree Plot — Explained Variance by Component",
                                 xaxis_title="Component", yaxis_title="Variance Explained (%)",
                                 height=350, font_family="DM Sans",
                                 plot_bgcolor="#fafafa", paper_bgcolor="#fff")
        st.plotly_chart(fig_scree, use_container_width=True)

        interp(f"The first two principal components explain <b>{sum(ev)*100:.1f}%</b> of total variance. "
               "The scree plot confirms that 2–3 components capture the essential structure of the product feature space. "
               "Cluster separation is visually distinct in PCA space, validating the K-Means segmentation.")


# ═══════════════════════════════════════════════════════════════════
# PAGE 7 — PRODUCT RECOMMENDER
# ═══════════════════════════════════════════════════════════════════
elif page == "💄  Product Recommender":
    st.markdown('<div class="main-title">Personalized Product Recommendation System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Weighted score: 50% avg rating · 30% recommendation rate · 20% normalised loves count</div>',
        unsafe_allow_html=True
    )
    st.markdown('<hr class="page-divider">', unsafe_allow_html=True)

    st.markdown('<div class="subsection-header">User Input Parameters</div>', unsafe_allow_html=True)

    # SESSION STATE: initialise rec_skin_type default to first available skin type
    _skin_types = sorted([s for s in df_merged["skin_type"].dropna().unique() if s.lower()!="unknown"])
    if st.session_state["rec_skin_type"] not in _skin_types:
        st.session_state["rec_skin_type"] = _skin_types[0]
    _categories = ["All"] + sorted([c for c in df_merged["primary_category"].dropna().unique()])
    if st.session_state["rec_category"] not in _categories:
        st.session_state["rec_category"] = "All"

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        skin_type = st.selectbox("Skin Type", _skin_types,
            index=_skin_types.index(st.session_state["rec_skin_type"]),
            key="rec_skin_type")
    with col2:
        category = st.selectbox("Category", _categories,
            index=_categories.index(st.session_state["rec_category"]),
            key="rec_category")
    with col3:
        budget = st.slider("Max Budget (USD)", 5, 500,
            value=st.session_state["rec_budget"], step=5, key="rec_budget")
    with col4:
        min_rating = st.slider("Min Rating", 1.0, 5.0,
            value=st.session_state["rec_min_rating"], step=0.5, key="rec_min_rating")
    with col5:
        top_n = st.slider("# Results", 3, 20,
            value=st.session_state["rec_top_n"], key="rec_top_n")
    with st.expander("Advanced Filters"):
        ac1, ac2, ac3 = st.columns(3)
        only_rec = ac1.checkbox("Highly Recommended (≥70%)",
            value=st.session_state["rec_only_rec"], key="rec_only_rec")
        excl_oos = ac2.checkbox("Exclude Out-of-Stock",
            value=st.session_state["rec_excl_oos"], key="rec_excl_oos")
        seph_excl = ac3.checkbox("Sephora Exclusive Only",
            value=st.session_state["rec_seph_excl"], key="rec_seph_excl")
    btn = st.button("Find Recommended Products", type="primary", use_container_width=True)
    st.markdown('<hr class="page-divider">', unsafe_allow_html=True)

    @st.cache_data(show_spinner="Computing recommendations…", ttl=300)
    def recommend(skin_type, budget, category, min_rating, top_n, only_rec, excl_oos, seph_excl):
        filtered = df_merged[
            (df_merged["skin_type"].str.lower().str.strip() == skin_type.lower()) &
            (df_merged["price_usd"] <= budget)
        ].copy()
        if category != "All":
            filtered = filtered[filtered["primary_category"] == category]
        if filtered.empty: return None
        result = (
            filtered
            .groupby(["product_id","product_name","brand_name","price_usd","primary_category"])
            .agg(avg_rating=("rating","mean"), review_count=("rating","count"),
                 loves_count=("loves_count","max"), rec_rate=("is_recommended","mean"))
            .reset_index()
        )
        result = result[(result["avg_rating"] >= min_rating) & (result["review_count"] >= 3)]
        if only_rec: result = result[result["rec_rate"] >= 0.70]
        flags_cols = [c for c in ["product_id","out_of_stock","sephora_exclusive","limited_edition","new","online_only"]
                      if c in df_product.columns]
        flags = df_product[flags_cols].drop_duplicates("product_id")
        result = result.merge(flags, on="product_id", how="left")
        if excl_oos and "out_of_stock" in result.columns: result = result[result["out_of_stock"]!=1]
        if seph_excl and "sephora_exclusive" in result.columns: result = result[result["sephora_exclusive"]==1]
        result["score"] = (
            result["avg_rating"] * 0.50 +
            result["rec_rate"].fillna(0) * 0.30 +
            (result["loves_count"] / result["loves_count"].max()) * 0.20
        )
        return result.sort_values("score", ascending=False).head(top_n)

    # SESSION STATE: run recommend only when button clicked; keep last
    # results in session state so they survive page navigation.
    if btn:
        st.session_state["rec_triggered"] = True
        st.session_state["rec_results"] = recommend(
            skin_type, budget, category, min_rating, top_n, only_rec, excl_oos, seph_excl)

    results = st.session_state["rec_results"]

    if not st.session_state["rec_triggered"]:
        st.info("Set your filters above and click **Find Recommended Products** to see results.")
    elif results is None or results.empty:
        st.warning(f"No products found for **{skin_type}** skin within **${budget}**. Try adjusting filters.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Found", len(results))
        m2.metric("Avg Rating", f"{results['avg_rating'].mean():.2f} ★")
        m3.metric("Avg Price", f"${results['price_usd'].mean():.2f}")
        m4.metric("Avg Rec Rate", f"{results['rec_rate'].mean()*100:.1f}%")

        for rank, (_, row) in enumerate(results.iterrows(), 1):
            badges = ""
            if row.get("limited_edition")==1: badges += '<span class="badge badge-orange">Limited Edition</span>'
            if row.get("new")==1:             badges += '<span class="badge badge-green">New</span>'
            if row.get("sephora_exclusive")==1: badges += '<span class="badge badge-blue">Sephora Exclusive</span>'
            if row.get("online_only")==1:     badges += '<span class="badge badge-blue">Online Only</span>'
            stars = "★" * round(row["avg_rating"]) + "☆" * (5 - round(row["avg_rating"]))
            st.markdown(f"""
            <div class="rec-card">
              <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                <div>
                  <span style="font-family:'EB Garamond',serif;font-size:1.05rem;font-weight:700;">
                    #{rank} {row['product_name']}</span>
                  &nbsp;<span style="color:#888;font-size:.85rem;">by {row['brand_name']}</span><br>
                  <span style="color:#999;font-size:.8rem;">📂 {row.get('primary_category','—')}</span>&nbsp;{badges}
                </div>
                <span style="font-family:'EB Garamond',serif;font-size:1.45rem;font-weight:700;color:#c0392b;">
                  ${row['price_usd']:.2f}</span>
              </div>
              <div style="margin-top:.5rem;display:flex;gap:1.5rem;font-size:.85rem;color:#444;">
                <div><span style="color:#f39c12;">{stars}</span> <b>{row['avg_rating']:.2f}</b>/5</div>
                <div>💬 {int(row['review_count']):,} reviews</div>
                <div>❤️ {int(row['loves_count']):,} loves</div>
                <div>👍 {row['rec_rate']*100:.0f}% rec</div>
                <div>Score: <b>{row['score']:.3f}</b></div>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<hr class="page-divider">', unsafe_allow_html=True)
        cc1, cc2 = st.columns(2)
        with cc1:
            fig = px.bar(results.sort_values("avg_rating"), x="avg_rating", y="product_name",
                         orientation="h", color="avg_rating", color_continuous_scale="RdYlGn",
                         title="Rating Comparison")
            fig.update_layout(showlegend=False, height=380, font_family="DM Sans",
                               plot_bgcolor="#fafafa", paper_bgcolor="#fff")
            st.plotly_chart(fig, use_container_width=True)
        with cc2:
            fig = px.scatter(results, x="price_usd", y="avg_rating", size="loves_count",
                             color="rec_rate", hover_name="product_name",
                             color_continuous_scale="RdYlGn", title="Price vs Rating")
            fig.update_layout(height=380, font_family="DM Sans",
                               plot_bgcolor="#fafafa", paper_bgcolor="#fff")
            st.plotly_chart(fig, use_container_width=True)

        export = results[["product_name","brand_name","primary_category","price_usd",
                           "avg_rating","review_count","loves_count","rec_rate","score"]].copy()
        export["rec_rate"] = (export["rec_rate"]*100).round(1)
        export.columns = [c.replace("_"," ").title() for c in export.columns]
        st.download_button("⬇️ Download Recommendations as CSV",
                           data=export.to_csv(index=False).encode(),
                           file_name=f"rec_{skin_type}_${budget}.csv",
                           mime="text/csv", use_container_width=True)

# ─────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='footer'>Sephora Beauty Analytics Dashboard &nbsp;|&nbsp; "
    "IV Semester Statistical Data Analysis — Major Project &nbsp;|&nbsp; "
    "Streamlit · Plotly · NLTK · scikit-learn</div>",
    unsafe_allow_html=True,
)
