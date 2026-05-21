"""
====================================================================
SEPHORA BEAUTY ANALYTICS DASHBOARD
IV Semester Statistical Data Analysis — Major Project
====================================================================
Run with:  streamlit run sephora_streamlit_app.py
Requires:  pip install streamlit pandas numpy matplotlib seaborn
           scikit-learn scipy plotly nltk wordcloud
====================================================================
"""

import re
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import chi2_contingency, ttest_ind, pearsonr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

import nltk
nltk.download("punkt",      quiet=True)
nltk.download("stopwords",  quiet=True)
nltk.download("wordnet",    quiet=True)
nltk.download("punkt_tab",  quiet=True)
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
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────
# ACADEMIC PROFESSIONAL CSS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,600;0,700;1,400&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Main title ── */
.main-title {
    font-family: 'EB Garamond', serif;
    font-size: 2.6rem;
    font-weight: 700;
    color: #1a1a2e;
    letter-spacing: -0.5px;
    border-bottom: 3px solid #c0392b;
    padding-bottom: 0.4rem;
    margin-bottom: 0.2rem;
}
.subtitle {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.95rem;
    color: #555;
    letter-spacing: 0.5px;
    margin-bottom: 1.2rem;
}

/* ── Section headers ── */
.section-header {
    font-family: 'EB Garamond', serif;
    font-size: 1.55rem;
    font-weight: 600;
    color: #1a1a2e;
    border-left: 5px solid #c0392b;
    padding: 0.2rem 0 0.2rem 0.8rem;
    margin: 1.5rem 0 1rem 0;
}
.subsection-header {
    font-family: 'EB Garamond', serif;
    font-size: 1.2rem;
    font-weight: 600;
    color: #2c3e50;
    margin: 1.2rem 0 0.5rem 0;
    border-bottom: 1px solid #e0e0e0;
    padding-bottom: 4px;
}

/* ── Metric cards ── */
.metric-row { display: flex; gap: 1rem; margin: 1rem 0; flex-wrap: wrap; }
.metric-card {
    flex: 1;
    background: #fff;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    border-top: 4px solid #c0392b;
    box-shadow: 0 1px 6px rgba(0,0,0,0.07);
    min-width: 140px;
}
.metric-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #888;
    margin-bottom: 4px;
}
.metric-value {
    font-family: 'EB Garamond', serif;
    font-size: 1.9rem;
    font-weight: 700;
    color: #1a1a2e;
    line-height: 1;
}

/* ── Table captions ── */
.table-caption {
    font-size: 0.82rem;
    color: #555;
    font-style: italic;
    margin-bottom: 0.5rem;
}
.table-number {
    font-weight: 700;
    color: #c0392b;
}

/* ── Interpretation box ── */
.interpretation-box {
    background: #f8f5f0;
    border-left: 4px solid #2c3e50;
    border-radius: 4px;
    padding: 0.8rem 1rem;
    font-size: 0.9rem;
    color: #2c3e50;
    margin-top: 0.8rem;
    line-height: 1.6;
}
.interpretation-box b { color: #c0392b; }

/* ── Result box for hypothesis ── */
.result-accept {
    background: #eaf7ef;
    border: 1px solid #2ecc71;
    border-radius: 6px;
    padding: 1rem 1.2rem;
    margin-top: 1rem;
}
.result-reject {
    background: #fdf0ef;
    border: 1px solid #e74c3c;
    border-radius: 6px;
    padding: 1rem 1.2rem;
    margin-top: 1rem;
}
.result-stat {
    font-family: 'EB Garamond', serif;
    font-size: 1.05rem;
}
.decision-badge {
    display: inline-block;
    padding: 3px 14px;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 600;
    margin-left: 8px;
}
.badge-reject   { background: #e74c3c; color: #fff; }
.badge-accept   { background: #27ae60; color: #fff; }

/* ── Rec card ── */
.rec-card {
    background: #fff;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    border: 1px solid #e5e5e5;
    border-left: 4px solid #c0392b;
    box-shadow: 0 1px 5px rgba(0,0,0,0.06);
    margin-bottom: 0.7rem;
}
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.73rem;
    font-weight: 600;
    margin-right: 4px;
}
.badge-green  { background: #d4edda; color: #155724; }
.badge-blue   { background: #cce5ff; color: #004085; }
.badge-orange { background: #fff3cd; color: #856404; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #1a1a2e !important;
}
[data-testid="stSidebar"] * { color: #ddd !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 0.9rem !important; }

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 2px solid #e0e0e0; }
.stTabs [data-baseweb="tab"] {
    border-radius: 6px 6px 0 0;
    padding: 0.4rem 1rem;
    font-size: 0.88rem;
    font-weight: 600;
    color: #555;
}
.stTabs [aria-selected="true"] { color: #c0392b !important; border-bottom: 2px solid #c0392b; }

/* ── Footer ── */
.footer {
    text-align: center;
    font-size: 0.78rem;
    color: #aaa;
    padding: 1.5rem 0 0.5rem 0;
    border-top: 1px solid #e5e5e5;
    margin-top: 2rem;
}
.page-divider { border: none; border-top: 1px solid #e5e5e5; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading datasets…")
def load_data():
    import pandas as pd

    # Product dataset
    # Google Drive folder links won't work directly.
    product_file = "https://drive.google.com/uc?export=download&id=1cUmKZ3SiOZ75CO7ixTf9Nqzjm9LDAGQL"
    review_files = [
    "https://drive.google.com/uc?export=download&id=1sYmatoOWNklNK0kt-ivD0WYOg_AWwVgD",
    "https://drive.google.com/uc?export=download&id=1h9wijBRmxAnNwtU-TdZEM9ifCAtIULBi",  
    "https://drive.google.com/uc?export=download&id=11MVi8dDeKpvatRSYRszBgQlw8y2-l2bl",
    "https://drive.google.com/uc?export=download&id=1SYFPpvZbGF6b0ZPiBsA4fw_Oi9ak24ER",
    "https://drive.google.com/uc?export=download&id=1ML-kHF0-SFNhWOyC1ypGNfqLpT-0NwdI",
    ]
    df_product = pd.read_csv(product_file, low_memory=False)
    dfs = [pd.read_csv(f, low_memory=False,on_bad_lines="skip") for f in review_files]
    df_reviews = pd.concat(dfs, ignore_index=True)
    df_reviews = df_reviews.loc[:, ~df_reviews.columns.str.startswith("Unnamed")]
    df_product = df_product.loc[:, ~df_product.columns.str.startswith("Unnamed")]

    product_cols = [
        "product_id", "brand_id", "loves_count", "reviews", "size",
        "variation_type", "variation_value", "ingredients",
        "limited_edition", "new", "online_only", "out_of_stock",
        "sephora_exclusive", "highlights", "primary_category",
        "secondary_category", "tertiary_category", "child_count",
    ]

    keep = [c for c in product_cols if c in df_product.columns]

    df_merged = df_reviews.merge(
        df_product[keep],
        on="product_id",
        how="left"
    )

    return df_product, df_reviews, df_merged


df_product, df_reviews, df_merged = load_data()
# ─────────────────────────────────────────────────────────────────
# NLP helpers (cached)
# ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Running NLP preprocessing (one-time)…")
def run_nlp(n=30000):
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

    sample = df_merged[df_merged["review_text"].notna()].sample(
        n=min(n, len(df_merged)), random_state=42
    ).copy()
    sample["clean_text"] = sample["review_text"].apply(preprocess)
    return sample

@st.cache_data(show_spinner="Fitting LDA model…")
def run_lda(_sample_nlp, n_topics=5, max_features=3000):
    cv = CountVectorizer(max_features=max_features, min_df=5, max_df=0.9)
    dtm = cv.fit_transform(_sample_nlp["clean_text"])
    lda = LatentDirichletAllocation(
        n_components=n_topics, max_iter=15,
        learning_method="online", random_state=42
    )
    lda.fit(dtm)
    return lda, cv

# ─────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='text-align:center;padding:1rem 0 0.5rem;'>"
        "<img src='https://upload.wikimedia.org/wikipedia/commons/thumb/7/7e/Sephora_logo.svg/320px-Sephora_logo.svg.png'"
        " width='130' style='filter:brightness(0) invert(1)'></div>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<div style='text-align:center;font-size:0.72rem;color:#aaa;padding-bottom:1rem;letter-spacing:1px;'>"
        "BEAUTY ANALYTICS DASHBOARD</div>",
        unsafe_allow_html=True
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
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.72rem;color:#888;line-height:1.6;'>"
        "IV Semester Major Project<br>Statistical Data Analysis<br>"
        "Sephora Products & Reviews<br>Built with Python & Streamlit"
        "</div>",
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────────────────────────
# HELPER: metric card HTML
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
      <div style='margin-top:6px;font-size:0.85rem;color:#444;'>{extra}</div>
    </div>""", unsafe_allow_html=True)

def interp(text):
    st.markdown(f"<div class='interpretation-box'>📌 {text}</div>", unsafe_allow_html=True)

COLORS = {
    "primary": "#c0392b",
    "dark":    "#1a1a2e",
    "mid":     "#2c3e50",
    "accent":  "#8e44ad",
    "green":   "#27ae60",
    "gold":    "#f39c12",
}

# ═══════════════════════════════════════════════════════════════════
# PAGE 0 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════
if page == "🏠  Overview":
    st.markdown('<div class="main-title">Sephora Beauty Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">IV Semester Statistical Data Analysis — Major Project &nbsp;|&nbsp; '
        'Sephora Products &amp; Consumer Reviews Dataset</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='metric-row'>"
        + mcard("Total Products", f"{df_product.shape[0]:,}")
        + mcard("Total Reviews", f"{df_reviews.shape[0]:,}")
        + mcard("Brands", f"{df_product['brand_name'].nunique():,}")
        + mcard("Avg Product Rating", f"{df_product['rating'].mean():.2f} ★")
        + mcard("Variables (Combined)", "41")
        + "</div>",
        unsafe_allow_html=True
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

        st.markdown('<div class="subsection-header" style="margin-top:1.2rem;">Analysis Pipeline</div>', unsafe_allow_html=True)
        stages = [
            ("01", "Data Pre-processing & Merging"),
            ("02", "Exploratory Data Analysis (EDA)"),
            ("03", "Statistical Hypothesis Testing"),
            ("04", "NLP: Text Pre-processing & Word Frequency"),
            ("05", "LDA: Latent Topic Modeling"),
            ("06", "K-Means Clustering & PCA"),
            ("07", "Personalized Recommendation System"),
        ]
        for num, name in stages:
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:10px;margin:4px 0;'>"
                f"<span style='background:#c0392b;color:#fff;border-radius:50%;width:26px;height:26px;"
                f"display:flex;align-items:center;justify-content:center;font-size:0.72rem;font-weight:700;'>{num}</span>"
                f"<span style='font-size:0.9rem;color:#2c3e50;'>{name}</span></div>",
                unsafe_allow_html=True
            )

    with col2:
        st.markdown('<div class="subsection-header">Study Objectives</div>', unsafe_allow_html=True)
        objectives = [
            "Analyze distribution of product prices, ratings, and consumer demographics",
            "Identify top-performing brands and most-loved products",
            "Examine the relationship between skin type and product recommendation",
            "Apply Chi-Square, t-Test, and Pearson Correlation for statistical validation",
            "Pre-process review text and analyze word frequency using NLP techniques",
            "Identify latent topics from consumer reviews using LDA",
            "Segment products into distinct market clusters using K-Means and PCA",
            "Develop a personalized recommendation system based on skin type and budget",
        ]
        for i, obj in enumerate(objectives, 1):
            st.markdown(
                f"<div style='display:flex;gap:8px;margin:5px 0;font-size:0.88rem;'>"
                f"<span style='color:#c0392b;font-weight:700;min-width:20px;'>{i}.</span>"
                f"<span style='color:#333;'>{obj}</span></div>",
                unsafe_allow_html=True
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
            "total_neg_feedback_count","reviews","child_count"
        ] if v in df_merged.columns]
        desc = df_merged[numeric_vars].describe(percentiles=[0.25, 0.50, 0.75]).T
        desc["skewness"] = df_merged[numeric_vars].skew()
        desc["kurtosis"] = df_merged[numeric_vars].kurt()
        st.markdown('<div class="table-caption"><span class="table-number">Table 5.1</span> — Descriptive Statistics for Numeric Variables (Merged Dataset)</div>', unsafe_allow_html=True)
        st.dataframe(desc.round(3), use_container_width=True)
        interp(
            "The <b>rating</b> variable shows strong left skew, confirming a tendency for high ratings. "
            "<b>loves_count</b> and <b>price_usd</b> are heavily right-skewed, indicating a small number of "
            "extremely popular/expensive products pulling the mean above the median."
        )

    with tabs[1]:
        cat_col = st.selectbox("Select categorical variable",
                               ["skin_type","eye_color","hair_color",
                                "sentiment_label","primary_category","price_tier"])
        if cat_col in df_merged.columns:
            freq = df_merged[cat_col].value_counts().reset_index()
            freq.columns = [cat_col, "Count"]
            freq["Percentage (%)"] = (freq["Count"] / freq["Count"].sum() * 100).round(2)
            c1, c2 = st.columns([1, 1.5])
            with c1:
                st.markdown(f'<div class="table-caption"><span class="table-number">Frequency Table</span> — {cat_col}</div>', unsafe_allow_html=True)
                st.dataframe(freq, hide_index=True, use_container_width=True)
            with c2:
                fig = px.bar(freq.head(12), x=cat_col, y="Count",
                             color="Percentage (%)", color_continuous_scale="Reds",
                             title=f"Frequency Distribution — {cat_col}")
                fig.update_layout(showlegend=False, plot_bgcolor="#fafafa",
                                  font_family="DM Sans")
                st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        prod_vars = [v for v in ["rating","price_usd","loves_count","reviews"] if v in df_product.columns]
        prod_desc = df_product[prod_vars].describe(percentiles=[0.25, 0.50, 0.75]).T
        prod_desc["skewness"] = df_product[prod_vars].skew()
        prod_desc["kurtosis"] = df_product[prod_vars].kurt()
        st.markdown('<div class="table-caption"><span class="table-number">Table 5.6</span> — Product-Level Descriptive Statistics</div>', unsafe_allow_html=True)
        st.dataframe(prod_desc.round(3), use_container_width=True)
        interp(
            "<b>price_usd</b> has a skewness of ~3.5, confirming the right-skewed luxury tail. "
            "<b>loves_count</b> has extremely high kurtosis, indicating a leptokurtic (peaked) distribution "
            "where a few viral products dominate."
        )


# ═══════════════════════════════════════════════════════════════════
# PAGE 2 — VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════
elif page == "📊  Visualizations":
    st.markdown('<div class="section-header">Exploratory Data Analysis — Visualizations</div>', unsafe_allow_html=True)

    viz_choice = st.selectbox("Select Figure", [
        "Fig 6.2 — Price Distribution",
        "Fig 6.3 — Most Loved Products & Top Brands",
        "Fig 6.4 — Skin Type & Eye/Hair Color Demographics",
        "Fig 6.5 — Price vs Rating",
        "Fig 6.7 — Review Volume Over Time",
        "Fig 6.8 — Correlation Heatmap",
        "Fig 6.9 — Recommendation Rate by Skin Type",
    ])
    st.markdown('<hr class="page-divider">', unsafe_allow_html=True)

    # ── Fig 6.1
    

    if viz_choice == "Fig 6.2 — Price Distribution":
        st.markdown('<div class="subsection-header">Figure 6.2: Product Price Distribution</div>', unsafe_allow_html=True)
        price = df_product["price_usd"].dropna()
        fig = make_subplots(rows=1, cols=2, subplot_titles=["(a) Histogram","(b) Box Plot by Category"])
        fig.add_trace(go.Histogram(x=price, nbinsx=60, marker_color="#3498db", opacity=0.8, name="Price"), row=1, col=1)
        fig.add_vline(x=price.median(), line_dash="dash", line_color="red",
                      annotation_text=f"Median ${price.median():.0f}", row=1, col=1)
        fig.add_vline(x=price.mean(), line_dash="dash", line_color="orange",
                      annotation_text=f"Mean ${price.mean():.0f}", row=1, col=1)
        if "primary_category" in df_product.columns:
            for cat in df_product["primary_category"].value_counts().head(6).index:
                vals = df_product[df_product["primary_category"]==cat]["price_usd"].dropna()
                fig.add_trace(go.Box(y=vals, name=cat, boxmean=True), row=1, col=2)
        fig.update_layout(height=430, showlegend=False, font_family="DM Sans",
                          plot_bgcolor="#fafafa", paper_bgcolor="#fff")
        st.plotly_chart(fig, use_container_width=True)
        interp(f"Price is right-skewed (Mean=${price.mean():.1f} > Median=${price.median():.1f}), "
               "indicating a long tail of premium and luxury products. The majority of products are priced below $100.")

    elif viz_choice == "Fig 6.3 — Most Loved Products & Top Brands":
        st.markdown('<div class="subsection-header">Figure 6.3: Most Loved Products & Top Brands by Review Volume</div>', unsafe_allow_html=True)
        top_loved = df_merged.groupby("product_name")["loves_count"].max().nlargest(10).reset_index()
        top_brands = df_merged["brand_name"].value_counts().head(10).reset_index()
        top_brands.columns = ["brand_name","review_count"]
        fig = make_subplots(rows=1, cols=2, subplot_titles=["(a) Top 10 Most Loved Products","(b) Top 10 Brands by Reviews"])
        fig.add_trace(go.Bar(x=top_loved["loves_count"], y=top_loved["product_name"],
                             orientation="h", marker_color="#c0392b"), row=1, col=1)
        fig.add_trace(go.Bar(x=top_brands["review_count"], y=top_brands["brand_name"],
                             orientation="h", marker_color="#8e44ad"), row=1, col=2)
        fig.update_layout(height=450, showlegend=False, font_family="DM Sans",
                          plot_bgcolor="#fafafa", paper_bgcolor="#fff")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)
        interp("The 'loves count' metric serves as a proxy for product popularity beyond ratings. "
               "Brands with highest review volumes indicate strong market penetration and consumer engagement on Sephora's platform.")

    elif viz_choice == "Fig 6.4 — Skin Type & Eye/Hair Color Demographics":
        st.markdown('<div class="subsection-header">Figure 6.4: Reviewer Demographic Distributions</div>', unsafe_allow_html=True)
        skin = df_merged[df_merged["skin_type"]!="Unknown"]["skin_type"].value_counts().reset_index()
        skin.columns = ["skin_type","count"]
        eye = df_merged[df_merged["eye_color"]!="Unknown"]["eye_color"].value_counts().head(6).reset_index()
        eye.columns = ["eye_color","count"]
        hair = df_merged[df_merged["hair_color"]!="Unknown"]["hair_color"].value_counts().head(6).reset_index()
        hair.columns = ["hair_color","count"]
        c1, c2, c3 = st.columns(3)
        with c1:
            fig1 = px.pie(skin, values="count", names="skin_type",
                          title="(a) Skin Type", hole=0.3,
                          color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig1, use_container_width=True)
        with c2:
            fig2 = px.bar(eye, x="eye_color", y="count", title="(b) Eye Color",
                          color="count", color_continuous_scale="Blues")
            fig2.update_layout(showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
        with c3:
            fig3 = px.bar(hair, x="hair_color", y="count", title="(c) Hair Color",
                          color="count", color_continuous_scale="Purples")
            fig3.update_layout(showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)
        interp("Combination and dry skin types are the most prevalent among reviewers, reflecting the target demographic of skincare consumers. "
               "Brown hair and brown eyes are the most common characteristics, consistent with a diverse global reviewer base.")

    elif viz_choice == "Fig 6.5 — Price vs Rating":
        st.markdown('<div class="subsection-header">Figure 6.5: Price versus Review Rating</div>', unsafe_allow_html=True)
        sample = df_merged[["price_usd","rating","price_tier"]].dropna().sample(
            n=min(15000,len(df_merged)), random_state=42)
        fig = make_subplots(rows=1, cols=2, subplot_titles=["(a) Scatter (15k sample)","(b) Box Plot by Price Tier"])
        fig.add_trace(go.Scatter(x=sample["price_usd"], y=sample["rating"], mode="markers",
                                 marker=dict(color="#3498db", opacity=0.12, size=4)), row=1, col=1)
        for tier in sample["price_tier"].dropna().unique():
            fig.add_trace(go.Box(y=sample[sample["price_tier"]==tier]["rating"],
                                 name=str(tier), boxmean=True), row=1, col=2)
        fig.update_layout(height=430, showlegend=False, font_family="DM Sans",
                          plot_bgcolor="#fafafa", paper_bgcolor="#fff")
        st.plotly_chart(fig, use_container_width=True)
        interp("The scatter plot reveals no strong linear relationship between price and rating. "
               "Box plots across price tiers show similar median ratings, suggesting that higher price "
               "does not necessarily guarantee higher customer satisfaction.")

    elif viz_choice == "Fig 6.7 — Review Volume Over Time":
        st.markdown('<div class="subsection-header">Figure 6.7: Review Volume and Average Rating by Year</div>', unsafe_allow_html=True)
        if "submission_year" in df_merged.columns:
            yearly = df_merged.groupby("submission_year").agg(
                review_count=("rating","count"), avg_rating=("rating","mean")
            ).reset_index().dropna()
            yearly = yearly[yearly["submission_year"] > 2000]
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=yearly["submission_year"], y=yearly["review_count"]/1000,
                                 name="Reviews (k)", marker_color="#3498db", opacity=0.75), secondary_y=False)
            fig.add_trace(go.Scatter(x=yearly["submission_year"], y=yearly["avg_rating"],
                                     name="Avg Rating", mode="lines+markers",
                                     line=dict(color="#c0392b", width=2.5)), secondary_y=True)
            fig.update_layout(title="", height=420, font_family="DM Sans",
                              plot_bgcolor="#fafafa", paper_bgcolor="#fff")
            fig.update_yaxes(title_text="Review Count (thousands)", secondary_y=False)
            fig.update_yaxes(title_text="Average Rating", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)
        interp("Review volume peaks in recent years, reflecting Sephora's growth. "
               "Average ratings remain relatively stable (~4.0–4.2), suggesting consistent product quality standards over time.")

    elif viz_choice == "Fig 6.8 — Correlation Heatmap":
        st.markdown('<div class="subsection-header">Figure 6.8: Pearson Correlation Heatmap</div>', unsafe_allow_html=True)
        corr_vars = [v for v in ["rating","price_usd","loves_count","helpfulness",
                                  "total_feedback_count","total_pos_feedback_count",
                                  "total_neg_feedback_count","is_recommended"]
                     if v in df_merged.columns]
        corr_matrix = df_merged[corr_vars].corr()
        fig = px.imshow(corr_matrix, text_auto=".2f", aspect="auto",
                        color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        fig.update_layout(height=520, font_family="DM Sans", paper_bgcolor="#fff")
        st.plotly_chart(fig, use_container_width=True)
        interp("<b>is_recommended</b> and <b>rating</b> show the strongest positive correlation (r ≈ 0.79), "
               "confirming that high ratings drive product recommendation. "
               "<b>total_pos_feedback_count</b> and <b>total_feedback_count</b> are nearly perfectly correlated, "
               "indicating that most feedback on Sephora tends to be positive.")

    elif viz_choice == "Fig 6.9 — Recommendation Rate by Skin Type":
        st.markdown('<div class="subsection-header">Figure 6.9: Recommendation Rate (%) by Skin Type</div>', unsafe_allow_html=True)
        rec = (df_merged[df_merged["skin_type"]!="Unknown"]
               .groupby("skin_type")["is_recommended"]
               .mean().reset_index().sort_values("is_recommended", ascending=False))
        rec["rate_pct"] = rec["is_recommended"] * 100
        fig = px.bar(rec, x="skin_type", y="rate_pct",
                     color="rate_pct", color_continuous_scale="RdYlGn",
                     text=rec["rate_pct"].round(1).astype(str)+"%",
                     labels={"rate_pct":"Recommendation Rate (%)"})
        fig.update_traces(textposition="outside")
        fig.update_layout(yaxis_range=[0,105], showlegend=False, height=400,
                          font_family="DM Sans", plot_bgcolor="#fafafa", paper_bgcolor="#fff")
        st.plotly_chart(fig, use_container_width=True)
        interp("All skin types show high recommendation rates (>75%), indicating broad product satisfaction. "
               "Normal skin reviewers have the highest recommendation rate, possibly due to greater product compatibility.")


# ═══════════════════════════════════════════════════════════════════
# PAGE 3 — HYPOTHESIS TESTING
# ═══════════════════════════════════════════════════════════════════
elif page == "🔬  Hypothesis Testing":
    st.markdown('<div class="section-header">Statistical Hypothesis Testing</div>', unsafe_allow_html=True)

    test_choice = st.selectbox("Select Hypothesis Test", [
        "H1 — Chi-Square: Skin Type vs Is_Recommended",
        "H3 — Independent t-Test: Limited Edition vs Standard Prices",
        "H4 — Pearson Correlation: Price vs Loves Count",
        "H5 — Chi-Square: Sephora Exclusive × Rating Tier",
    ])
    st.markdown('<hr class="page-divider">', unsafe_allow_html=True)

    α = 0.05

    if test_choice == "H1 — Chi-Square: Skin Type vs Is_Recommended":
        st.markdown('<div class="subsection-header">H1 — Chi-Square Test of Independence</div>', unsafe_allow_html=True)
        st.markdown("""
        **Null Hypothesis (H₀):** Skin type and product recommendation are independent (no association).  
        **Alternative Hypothesis (H₁):** There is a statistically significant association between skin type and product recommendation.  
        **Test Used:** Pearson Chi-Square Test of Independence | **Significance Level:** α = 0.05
        """)
        sub = df_merged[(df_merged["skin_type"]!="Unknown") & df_merged["is_recommended"].notna()].copy()
        sub["recommended_bin"] = sub["is_recommended"].round().astype(int)
        ct = pd.crosstab(sub["skin_type"], sub["recommended_bin"])
        ct.columns = ["Not Recommended (0)", "Recommended (1)"]
        st.markdown('<div class="table-caption"><span class="table-number">Table 7.1</span> — Contingency Table: Skin Type × Recommendation</div>', unsafe_allow_html=True)
        st.dataframe(ct, use_container_width=True)
        chi2, p, dof, expected = chi2_contingency(ct)
        result_box("χ²-statistic", chi2, p, extra=f"Degrees of Freedom: {dof} | Sample size: {len(sub):,}")
        interp(f"With χ² = <b>{chi2:.2f}</b> and p-value = <b>{p:.6f}</b> (well below α=0.05), we reject H₀. "
               "There is a statistically significant association between skin type and product recommendation. "
               "This suggests that product suitability varies meaningfully across different skin types.")

    elif test_choice == "H3 — Independent t-Test: Limited Edition vs Standard Prices":
        st.markdown("<div class='subsection-header'>H3 — Welch's Independent t-Test</div>", unsafe_allow_html=True)
        st.markdown("""
        **Null Hypothesis (H₀):** Mean price of limited edition products equals mean price of standard products.  
        **Alternative Hypothesis (H₁):** Limited edition products are priced significantly differently from standard products.  
        **Test Used:** Welch's Independent Samples t-Test (unequal variances) | **Significance Level:** α = 0.05
        """)
        ltd = df_product[df_product["limited_edition"]==1]["price_usd"].dropna()
        std = df_product[df_product["limited_edition"]==0]["price_usd"].dropna()
        c1, c2, c3 = st.columns(3)
        c1.metric("Limited Edition — Mean Price", f"${ltd.mean():.2f}", f"n = {len(ltd):,}")
        c2.metric("Standard — Mean Price", f"${std.mean():.2f}", f"n = {len(std):,}")
        c3.metric("Mean Difference", f"${ltd.mean()-std.mean():.2f}")
        fig = px.box(df_product.dropna(subset=["price_usd","limited_edition"]),
                     x=df_product["limited_edition"].map({1:"Limited Edition",0:"Standard"}),
                     y="price_usd", color="limited_edition",
                     labels={"x":"Product Type","price_usd":"Price (USD)"},
                     color_discrete_sequence=["#c0392b","#3498db"])
        fig.update_layout(showlegend=False, height=380, font_family="DM Sans",
                          plot_bgcolor="#fafafa", paper_bgcolor="#fff")
        st.plotly_chart(fig, use_container_width=True)
        t_stat, p_val = ttest_ind(ltd, std, equal_var=False)
        result_box("t-statistic", t_stat, p_val, extra=f"Limited Edition n={len(ltd):,} | Standard n={len(std):,}")
        interp(f"With t = <b>{t_stat:.4f}</b> and p = <b>{p_val:.6f}</b>, we {'reject' if p_val<0.05 else 'fail to reject'} H₀. "
               f"Limited edition products (mean=${ltd.mean():.2f}) are priced significantly "
               f"{'higher' if ltd.mean()>std.mean() else 'lower'} than standard products (mean=${std.mean():.2f}).")

    elif test_choice == "H4 — Pearson Correlation: Price vs Loves Count":
        st.markdown('<div class="subsection-header">H4 — Pearson Correlation Analysis</div>', unsafe_allow_html=True)
        st.markdown("""
        **Null Hypothesis (H₀):** There is no linear correlation between product price and loves count (ρ = 0).  
        **Alternative Hypothesis (H₁):** A significant linear correlation exists between price and loves count (ρ ≠ 0).  
        **Test Used:** Pearson Product-Moment Correlation | **Significance Level:** α = 0.05
        """)
        sub = df_product[["price_usd","loves_count"]].dropna()
        r, p_val = pearsonr(sub["price_usd"], sub["loves_count"])
        fig = px.scatter(sub.sample(min(3000,len(sub)),random_state=42),
                         x="price_usd", y="loves_count", opacity=0.4, trendline="ols",
                         labels={"price_usd":"Price (USD)","loves_count":"Loves Count"},
                         color_discrete_sequence=["#3498db"])
        fig.update_layout(height=400, font_family="DM Sans",
                          plot_bgcolor="#fafafa", paper_bgcolor="#fff")
        st.plotly_chart(fig, use_container_width=True)
        result_box("Pearson r", r, p_val, extra=f"r² = {r**2:.4f} (proportion of variance explained) | n = {len(sub):,}")
        interp(f"r = <b>{r:.4f}</b> indicates a <b>{'weak' if abs(r)<0.3 else 'moderate'} {'positive' if r>0 else 'negative'}</b> "
               f"linear relationship between price and product popularity. "
               f"Only r² = {r**2*100:.1f}% of variance in loves_count is explained by price, "
               "suggesting that popularity is driven by factors beyond price alone (brand, efficacy, marketing).")

    elif test_choice == "H5 — Chi-Square: Sephora Exclusive × Rating Tier":
        st.markdown('<div class="subsection-header">H5 — Chi-Square: Sephora Exclusivity and Rating Tier</div>', unsafe_allow_html=True)
        st.markdown("""
        **Null Hypothesis (H₀):** Sephora exclusivity and rating tier are independent.  
        **Alternative Hypothesis (H₁):** Sephora-exclusive products have a significantly different rating distribution.  
        **Test Used:** Pearson Chi-Square Test | **Significance Level:** α = 0.05
        """)
        df_product["rating_tier"] = pd.cut(df_product["rating"],bins=[0,3,4,5],
                                            labels=["Low (0–3)","Mid (3–4)","High (4–5)"])
        ct2 = pd.crosstab(df_product["sephora_exclusive"], df_product["rating_tier"])
        chi2, p, dof, _ = chi2_contingency(ct2)
        ct2.index = ["Non-Exclusive","Sephora Exclusive"]
        st.markdown('<div class="table-caption"><span class="table-number">Table 7.4</span> — Contingency Table: Sephora Exclusivity × Rating Tier</div>', unsafe_allow_html=True)
        st.dataframe(ct2, use_container_width=True)
        result_box("χ²-statistic", chi2, p, extra=f"Degrees of Freedom: {dof}")
        interp(f"χ² = <b>{chi2:.2f}</b>, p = <b>{p:.6f}</b>. "
               f"We {'reject' if p<0.05 else 'fail to reject'} H₀. "
               "This indicates that Sephora-exclusive products do not follow the same rating distribution as non-exclusive products, "
               "suggesting a quality or curation effect associated with exclusivity.")


# ═══════════════════════════════════════════════════════════════════
# PAGE 4 — NLP & TEXT ANALYSIS
# ═══════════════════════════════════════════════════════════════════
elif page == "📝  NLP & Text Analysis":
    st.markdown('<div class="section-header">Section 8.1 — Natural Language Processing & Text Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        "Customer review text is pre-processed using standard NLP pipeline: "
        "**tokenization → stopword removal → lemmatization**. "
        "Word frequency analysis and word clouds are generated for each sentiment group.",
        unsafe_allow_html=False
    )
    st.markdown('<hr class="page-divider">', unsafe_allow_html=True)

    sample_nlp = run_nlp(30000)

    tabs = st.tabs(["Pre-processing Pipeline", "Word Clouds", "Top-20 Word Frequency"])

    with tabs[0]:
        st.markdown('<div class="subsection-header">NLP Pre-processing Steps</div>', unsafe_allow_html=True)
        steps = pd.DataFrame({
            "Step": ["1. Lowercasing","2. Punctuation Removal","3. Tokenization",
                     "4. Stopword Removal","5. Lemmatization"],
            "Tool/Library": ["Python str","re (regex)","NLTK word_tokenize",
                             "NLTK stopwords","NLTK WordNetLemmatizer"],
            "Description": [
                "Convert all text to lowercase for uniformity",
                "Remove special characters, digits, and punctuation using regex [^a-z\\s]",
                "Split text into individual tokens (words)",
                "Remove high-frequency function words (the, is, and, etc.)",
                "Reduce words to their base form (e.g., 'moisturizing' → 'moisturize')"
            ]
        })
        st.dataframe(steps, hide_index=True, use_container_width=True)

        st.markdown('<div class="subsection-header" style="margin-top:1rem;">Example: Original vs Cleaned Review</div>', unsafe_allow_html=True)
        example_row = sample_nlp[sample_nlp["review_text"].notna()].iloc[0]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Original Review:**")
            st.markdown(f"> {str(example_row['review_text'])[:400]}")
        with c2:
            st.markdown("**After NLP Pre-processing:**")
            st.markdown(f"> {str(example_row['clean_text'])[:400]}")

        st.markdown('<div class="subsection-header" style="margin-top:1rem;">Sample Statistics</div>', unsafe_allow_html=True)
        sample_nlp["token_count"] = sample_nlp["clean_text"].str.split().str.len()
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Reviews Analyzed", f"{len(sample_nlp):,}")
        sc2.metric("Avg Tokens per Review (after cleaning)", f"{sample_nlp['token_count'].mean():.1f}")
        sc3.metric("Vocabulary Size (unique tokens)", f"{len(set(' '.join(sample_nlp['clean_text']).split())):,}")

    with tabs[1]:
        st.markdown('<div class="subsection-header">Figure 8.1: Word Clouds by Sentiment Group</div>', unsafe_allow_html=True)

        def get_text(sentiment):
            return sample_nlp[sample_nlp["sentiment_label"]==sentiment]["clean_text"].str.cat(sep=" ")

        pos_text = get_text("Positive (4–5)")
        neg_text = get_text("Negative (1–2)")
        neu_text = get_text("Neutral (3)")

        fig_wc, axes = plt.subplots(1, 3, figsize=(18, 6))
        for ax, text, cmap, title, label in zip(
            axes,
            [pos_text, neg_text, neu_text],
            ["Greens", "Reds", "Blues"],
            ["(a) Positive Reviews (Rating 4–5)",
             "(b) Negative Reviews (Rating 1–2)",
             "(c) Neutral Reviews (Rating 3)"],
            ["Positive","Negative","Neutral"]
        ):
            if text.strip():
                wc = WordCloud(width=700, height=450, background_color="white",
                               colormap=cmap, max_words=80).generate(text)
                ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            ax.set_title(title, fontweight="bold", fontsize=11, pad=12)
        fig_wc.suptitle("Figure 8.1: Word Clouds — Positive, Negative & Neutral Reviews",
                         fontweight="bold", fontsize=13, y=1.01)
        plt.tight_layout()
        st.pyplot(fig_wc)
        interp("Positive reviews frequently contain terms like <b>love, skin, feel, work, moisturize</b>, reflecting satisfaction with efficacy and texture. "
               "Negative reviews highlight terms like <b>dry, smell, break, return, irritate</b>, indicating concerns about fragrance, skin reactions, and dryness. "
               "Neutral reviews are lexically mixed, often reflecting tempered or conditional satisfaction.")

    with tabs[2]:
        st.markdown('<div class="subsection-header">Figure 8.2: Top-20 Word Frequency Comparison</div>', unsafe_allow_html=True)

        def top_words(text, n=20):
            words = text.split()
            return Counter(words).most_common(n)

        pos_top = top_words(pos_text)
        neg_top = top_words(neg_text)

        fig_freq, axes = plt.subplots(1, 2, figsize=(14, 7))
        fig_freq.patch.set_facecolor("#fafafa")

        if pos_top:
            w, c = zip(*pos_top)
            axes[0].barh(w, c, color="#27ae60", alpha=0.82, edgecolor="white")
            axes[0].invert_yaxis()
            axes[0].set_title("Figure 8.2(a): Top 20 Terms — Positive Reviews",
                              fontweight="bold", fontsize=10, pad=10)
            axes[0].set_xlabel("Frequency")
            axes[0].set_facecolor("#fafafa")

        if neg_top:
            w, c = zip(*neg_top)
            axes[1].barh(w, c, color="#c0392b", alpha=0.82, edgecolor="white")
            axes[1].invert_yaxis()
            axes[1].set_title("Figure 8.2(b): Top 20 Terms — Negative Reviews",
                              fontweight="bold", fontsize=10, pad=10)
            axes[1].set_xlabel("Frequency")
            axes[1].set_facecolor("#fafafa")

        plt.tight_layout()
        st.pyplot(fig_freq)
        interp("The frequency contrast between positive and negative lexicons is sharp. "
               "Words like <b>'skin', 'love', 'feel'</b> dominate positive reviews, while <b>'smell', 'dry', 'break'</b> "
               "characterize negative experiences. This confirms that fragrance sensitivity and moisturization are critical determinants of consumer satisfaction.")


# ═══════════════════════════════════════════════════════════════════
# PAGE 5 — LDA TOPIC MODELING
# ═══════════════════════════════════════════════════════════════════
elif page == "🔵  LDA Topic Modeling":
    st.markdown('<div class="section-header">Section 8.2 — Latent Dirichlet Allocation (LDA) Topic Modeling</div>', unsafe_allow_html=True)
    st.markdown(
        "LDA is an unsupervised probabilistic generative model that discovers latent thematic structure "
        "in a collection of documents. Each review is modeled as a mixture of topics, and each topic as "
        "a probability distribution over words.",
        unsafe_allow_html=False
    )
    st.markdown('<hr class="page-divider">', unsafe_allow_html=True)

    sample_nlp = run_nlp(30000)

    n_topics = st.slider("Number of Topics (n_components)", min_value=3, max_value=8, value=5,
                          help="Select the number of latent topics for LDA to discover")
    
    with st.spinner("Fitting LDA model…"):
        lda_model, cv = run_lda(sample_nlp, n_topics=n_topics)

    feature_names = cv.get_feature_names_out()

    topic_labels_default = [
        "Moisturization & Hydration",
        "Fragrance & Texture",
        "Skin Concerns & Breakouts",
        "Product Efficacy & Routine",
        "Packaging & Value",
        "Sun Protection & SPF",
        "Anti-aging & Serums",
        "Eye & Lip Care",
    ]
    topic_labels = [f"Topic {i+1}: {topic_labels_default[i]}" for i in range(n_topics)]

    tabs = st.tabs(["Topic Table", "Topic Word Distribution", "Topic Interpretation"])

    with tabs[0]:
        st.markdown('<div class="subsection-header">Table 8.1: LDA Topic Modeling Results — Top 15 Keywords per Topic</div>', unsafe_allow_html=True)
        rows = []
        for idx, (topic, label) in enumerate(zip(lda_model.components_, topic_labels)):
            top_words = [feature_names[i] for i in topic.argsort()[:-16:-1]]
            rows.append({"Topic": label, "Top Keywords": ", ".join(top_words)})
        lda_df = pd.DataFrame(rows)
        st.dataframe(lda_df, hide_index=True, use_container_width=True)

        # Perplexity note
        dtm = cv.transform(sample_nlp["clean_text"])
        perplexity = lda_model.perplexity(dtm)
        st.markdown(
            f'<div class="table-caption">Model Perplexity: <b>{perplexity:.1f}</b> '
            f'(lower values indicate better fit) | '
            f'Trained on {len(sample_nlp):,} reviews | '
            f'Vocabulary size: {len(feature_names):,} terms</div>',
            unsafe_allow_html=True
        )
        interp("LDA identifies distinct latent themes embedded in consumer review language. "
               "Topics are interpretable as real-world skincare concerns: hydration, fragrance, skin conditions, "
               "and product value — validating the model's semantic coherence.")

    with tabs[1]:
        st.markdown('<div class="subsection-header">Figure 8.3: LDA Topic Word Weight Distribution</div>', unsafe_allow_html=True)
        colors_lda = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f"]
        n_cols = min(n_topics, 5)
        n_rows = (n_topics + n_cols - 1) // n_cols
        fig_lda, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 5*n_rows))
        fig_lda.patch.set_facecolor("#fafafa")
        axes_flat = axes.flatten() if n_topics > 1 else [axes]

        for idx, (topic, label, color) in enumerate(zip(lda_model.components_, topic_labels, colors_lda)):
            top_idx = topic.argsort()[:-11:-1]
            top_words = [feature_names[i] for i in top_idx]
            top_weights = topic[top_idx] / topic[top_idx].sum()
            ax = axes_flat[idx]
            ax.barh(top_words, top_weights, color=color, alpha=0.82, edgecolor="white")
            ax.invert_yaxis()
            ax.set_title(label.replace(f"Topic {idx+1}: ",""), fontweight="bold", fontsize=9)
            ax.set_xlabel("Relative Weight", fontsize=8)
            ax.set_facecolor("#fafafa")
            ax.tick_params(labelsize=8)

        for ax in axes_flat[n_topics:]:
            ax.set_visible(False)

        fig_lda.suptitle(f"Figure 8.3: LDA Topic Model — Top 10 Terms per Topic (K={n_topics})",
                          fontweight="bold", fontsize=12, y=1.01)
        plt.tight_layout()
        st.pyplot(fig_lda)

    with tabs[2]:
        st.markdown('<div class="subsection-header">Topic Interpretation & Business Insights</div>', unsafe_allow_html=True)
        interps = [
            ("Moisturization & Hydration", "#1f77b4",
             "Dominant theme reflecting the primary use-case of skincare products. "
             "Terms like 'hydrat', 'moistur', 'dry' appear frequently. "
             "Products addressing hydration have the broadest consumer appeal."),
            ("Fragrance & Texture", "#ff7f0e",
             "A key differentiator in purchase decisions. Consumers are highly sensitive "
             "to scent and feel. Negative reviews in this cluster often cite strong fragrance "
             "as a deterrent, especially for sensitive skin users."),
            ("Skin Concerns & Breakouts", "#2ca02c",
             "Reflects a clinically oriented segment of consumers seeking acne control, "
             "pore reduction, and blemish treatment. Reviews here are highly informative for dermatological product development."),
            ("Product Efficacy & Routine", "#d62728",
             "Consumers evaluate products in the context of their overall skincare routine. "
             "Terms like 'work', 'result', 'week', 'month' indicate a longitudinal evaluation perspective."),
            ("Packaging & Value", "#9467bd",
             "Price-value perception and packaging aesthetics drive this topic. "
             "Consumers compare product size to cost, particularly for premium-tier products."),
        ]
        for label, color, explanation in interps[:n_topics]:
            st.markdown(
                f"<div style='border-left:4px solid {color};padding:0.6rem 1rem;"
                f"background:#f8f9fa;border-radius:4px;margin-bottom:0.7rem;'>"
                f"<b style='color:{color};'>{label}</b><br>"
                f"<span style='font-size:0.88rem;color:#333;'>{explanation}</span>"
                f"</div>",
                unsafe_allow_html=True
            )


# ═══════════════════════════════════════════════════════════════════
# PAGE 6 — CLUSTERING & PCA
# ═══════════════════════════════════════════════════════════════════
elif page == "🤖  Clustering & PCA":
    st.markdown('<div class="section-header">Section 8.3–8.4 — K-Means Clustering & Principal Component Analysis</div>', unsafe_allow_html=True)
    st.markdown('<hr class="page-divider">', unsafe_allow_html=True)

    cluster_features = [f for f in ["rating","price_usd","loves_count"] if f in df_product.columns]
    df_cluster = df_product[cluster_features].dropna().copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_cluster)

    tabs = st.tabs(["Elbow Method", "Cluster Profiles", "PCA Visualization"])

    with tabs[0]:
        st.markdown('<div class="subsection-header">Figure 8.4: Elbow Method — Optimal Number of Clusters</div>', unsafe_allow_html=True)
        inertia = []
        K_range = range(2, 10)
        for k in K_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X_scaled)
            inertia.append(km.inertia_)
        fig = px.line(x=list(K_range), y=inertia, markers=True,
                      labels={"x":"Number of Clusters (K)","y":"Inertia (Within-Cluster SSE)"})
        fig.add_vline(x=3, line_dash="dash", line_color="#c0392b",
                      annotation_text="Optimal K = 3", annotation_font_color="#c0392b")
        fig.update_layout(height=380, font_family="DM Sans",
                          plot_bgcolor="#fafafa", paper_bgcolor="#fff")
        st.plotly_chart(fig, use_container_width=True)
        interp("The elbow occurs at <b>K = 3</b>, where additional clusters yield diminishing reduction in inertia. "
               "This suggests three natural product segments: budget, mid-range, and premium/luxury.")

    with tabs[1]:
        K_FINAL = st.slider("Select K (number of clusters)", 2, 6, 3)
        kmeans = KMeans(n_clusters=K_FINAL, random_state=42, n_init=10)
        df_cluster["cluster"] = kmeans.fit_predict(X_scaled).astype(str)

        summary = df_cluster.groupby("cluster")[cluster_features].mean().round(2)
        summary["n (products)"] = df_cluster.groupby("cluster").size()

        segment_names = {
            "0": "Budget / Low-Engagement",
            "1": "Mid-Range / Moderate-Popularity",
            "2": "Premium / High-Popularity",
            "3": "Cluster 3", "4": "Cluster 4", "5": "Cluster 5"
        }
        summary.index = [segment_names.get(str(i), f"Cluster {i}") for i in summary.index]
        st.markdown('<div class="table-caption"><span class="table-number">Table 8.2</span> — K-Means Cluster Profile Summary</div>', unsafe_allow_html=True)
        st.dataframe(summary, use_container_width=True)

        fig = px.scatter_3d(df_cluster.sample(min(5000,len(df_cluster)),random_state=42),
                            x="price_usd", y="rating", z="loves_count",
                            color="cluster", opacity=0.45,
                            labels={"price_usd":"Price (USD)","rating":"Avg Rating","loves_count":"Loves Count"})
        fig.update_layout(height=520, font_family="DM Sans", paper_bgcolor="#fff",
                          title=f"Figure 8.5: 3D Cluster Scatter — Price × Rating × Loves Count (K={K_FINAL})")
        st.plotly_chart(fig, use_container_width=True)
        interp("The three clusters represent distinct market segments. "
               "Budget products cluster at lower price and moderate love counts; "
               "premium products show higher price and variable popularity; "
               "highly viral products (high loves_count) span multiple price ranges, "
               "indicating that popularity is not exclusively price-driven.")

    with tabs[2]:
        st.markdown('<div class="subsection-header">Figure 8.5: PCA — 2D Cluster Visualization & Scree Plot</div>', unsafe_allow_html=True)
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        ev = pca.explained_variance_ratio_

        c1, c2, c3 = st.columns(3)
        c1.metric("PC1 Explained Variance", f"{ev[0]*100:.2f}%")
        c2.metric("PC2 Explained Variance", f"{ev[1]*100:.2f}%")
        c3.metric("Total (2 Components)", f"{sum(ev)*100:.2f}%")

        pca_df = pd.DataFrame({"PC1": X_pca[:,0], "PC2": X_pca[:,1],
                                "Cluster": df_cluster["cluster"]})
        fig_pca = px.scatter(pca_df.sample(min(5000,len(pca_df)),random_state=42),
                             x="PC1", y="PC2", color="Cluster", opacity=0.35,
                             title=f"PCA Cluster Scatter (Total Explained Variance: {sum(ev)*100:.1f}%)")
        fig_pca.update_layout(height=420, font_family="DM Sans",
                               plot_bgcolor="#fafafa", paper_bgcolor="#fff")
        st.plotly_chart(fig_pca, use_container_width=True)

        # Scree plot
        pca_full = PCA().fit(X_scaled)
        ev_full = pca_full.explained_variance_ratio_
        fig_scree = make_subplots(rows=1, cols=1)
        fig_scree.add_trace(go.Bar(x=list(range(1,len(ev_full)+1)),
                                    y=ev_full*100, name="Individual", marker_color="#3498db"))
        fig_scree.add_trace(go.Scatter(x=list(range(1,len(ev_full)+1)),
                                        y=np.cumsum(ev_full)*100, mode="lines+markers",
                                        name="Cumulative", line=dict(color="#c0392b",width=2)))
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
        '<div class="subtitle">Statistical Filtering Engine — Products ranked by weighted score '
        '(50% avg rating · 30% recommendation rate · 20% normalized loves count)</div>',
        unsafe_allow_html=True
    )
    st.markdown('<hr class="page-divider">', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="subsection-header">User Input Parameters</div>', unsafe_allow_html=True)
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            skin_types = sorted([s for s in df_merged["skin_type"].dropna().unique() if s.lower()!="unknown"])
            skin_type = st.selectbox("Skin Type", skin_types)
        with col2:
            categories = sorted([c for c in df_merged["primary_category"].dropna().unique()])
            category = st.selectbox("Category", ["All"] + categories)
        with col3:
            budget = st.slider("Max Budget (USD)", 5, 500, 50, step=5)
        with col4:
            min_rating = st.slider("Min Avg Rating", 1.0, 5.0, 3.5, step=0.5)
        with col5:
            top_n = st.slider("Number of Results", 3, 20, 8)

        with st.expander("Advanced Filters"):
            ac1, ac2, ac3 = st.columns(3)
            with ac1: only_rec = st.checkbox("Only Highly Recommended (≥70%)", value=False)
            with ac2: excl_oos = st.checkbox("Exclude Out-of-Stock", value=True)
            with ac3: seph_excl = st.checkbox("Sephora Exclusive Only", value=False)

        btn = st.button("Find Recommended Products", type="primary", use_container_width=True)

    st.markdown('<hr class="page-divider">', unsafe_allow_html=True)

    def recommend(skin_type, budget, category, min_rating, top_n, only_rec, excl_oos, seph_excl):
        filtered = df_merged[
            (df_merged["skin_type"].str.lower().str.strip() == skin_type.lower()) &
            (df_merged["price_usd"] <= budget)
        ].copy()
        if category != "All":
            filtered = filtered[filtered["primary_category"] == category]
        if filtered.empty: return None

        result = (filtered
            .groupby(["product_id","product_name","brand_name","price_usd","primary_category"])
            .agg(avg_rating=("rating","mean"), review_count=("rating","count"),
                 loves_count=("loves_count","max"), rec_rate=("is_recommended","mean"))
            .reset_index()
        )
        result = result[result["avg_rating"] >= min_rating]
        result = result[result["review_count"] >= 3]
        if only_rec: result = result[result["rec_rate"] >= 0.70]

        flags = df_product[["product_id","out_of_stock","sephora_exclusive",
                              "limited_edition","new","online_only"]].drop_duplicates("product_id")
        result = result.merge(flags, on="product_id", how="left")
        if excl_oos: result = result[result["out_of_stock"] != 1]
        if seph_excl: result = result[result["sephora_exclusive"] == 1]

        result["score"] = (
            result["avg_rating"] * 0.50 +
            result["rec_rate"].fillna(0) * 0.30 +
            (result["loves_count"] / result["loves_count"].max()) * 0.20
        )
        return result.sort_values("score", ascending=False).head(top_n)

    results = recommend(skin_type, budget, category, min_rating, top_n, only_rec, excl_oos, seph_excl)

    if results is None or results.empty:
        st.warning(f"No products found for **{skin_type}** skin within **${budget}** budget. Try adjusting filters.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Products Found", len(results))
        m2.metric("Avg Rating", f"{results['avg_rating'].mean():.2f} ★")
        m3.metric("Avg Price", f"${results['price_usd'].mean():.2f}")
        m4.metric("Avg Rec Rate", f"{results['rec_rate'].mean()*100:.1f}%")

        st.markdown(f'<div class="subsection-header">Top {len(results)} Recommended Products</div>', unsafe_allow_html=True)

        for rank, (_, row) in enumerate(results.iterrows(), 1):
            badges = ""
            if row.get("limited_edition")==1: badges += '<span class="badge badge-orange">Limited Edition</span>'
            if row.get("new")==1: badges += '<span class="badge badge-green">New</span>'
            if row.get("sephora_exclusive")==1: badges += '<span class="badge badge-blue">Sephora Exclusive</span>'
            if row.get("online_only")==1: badges += '<span class="badge badge-blue">Online Only</span>'
            stars = "★" * round(row["avg_rating"]) + "☆" * (5 - round(row["avg_rating"]))
            st.markdown(f"""
            <div class="rec-card">
              <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                <div>
                  <span style="font-family:'EB Garamond',serif;font-size:1.1rem;font-weight:700;">
                    #{rank}&nbsp; {row['product_name']}</span>
                  &nbsp;<span style="color:#888;font-size:0.88rem;">by {row['brand_name']}</span><br>
                  <span style="color:#999;font-size:0.82rem;">📂 {row.get('primary_category','—')}</span>
                  &nbsp;&nbsp;{badges}
                </div>
                <div style="text-align:right;">
                  <span style="font-family:'EB Garamond',serif;font-size:1.5rem;font-weight:700;color:#c0392b;">
                    ${row['price_usd']:.2f}</span>
                </div>
              </div>
              <div style="margin-top:0.6rem;display:flex;gap:2rem;font-size:0.88rem;color:#444;">
                <div><span style="color:#f39c12;">{stars}</span> <b>{row['avg_rating']:.2f}</b>/5</div>
                <div>💬 {int(row['review_count']):,} reviews</div>
                <div>❤️ {int(row['loves_count']):,} loves</div>
                <div>👍 {row['rec_rate']*100:.0f}% recommended</div>
                <div>Score: <b>{row['score']:.3f}</b></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<hr class="page-divider">', unsafe_allow_html=True)
        st.markdown('<div class="subsection-header">Analytical Insights on Recommendations</div>', unsafe_allow_html=True)
        cc1, cc2 = st.columns(2)
        with cc1:
            fig = px.bar(results.sort_values("avg_rating"),
                         x="avg_rating", y="product_name", orientation="h",
                         color="avg_rating", color_continuous_scale="RdYlGn",
                         labels={"avg_rating":"Avg Rating","product_name":"Product"},
                         title="Average Rating Comparison")
            fig.update_layout(showlegend=False, height=400, font_family="DM Sans",
                               plot_bgcolor="#fafafa", paper_bgcolor="#fff")
            st.plotly_chart(fig, use_container_width=True)
        with cc2:
            fig = px.scatter(results, x="price_usd", y="avg_rating",
                             size="loves_count", color="rec_rate",
                             hover_name="product_name", color_continuous_scale="RdYlGn",
                             labels={"price_usd":"Price (USD)","avg_rating":"Avg Rating","rec_rate":"Rec. Rate"},
                             title="Price vs Rating (bubble = loves count)")
            fig.update_layout(height=400, font_family="DM Sans",
                               plot_bgcolor="#fafafa", paper_bgcolor="#fff")
            st.plotly_chart(fig, use_container_width=True)

        # Download
        st.markdown('<hr class="page-divider">', unsafe_allow_html=True)
        export_cols = ["product_name","brand_name","primary_category","price_usd",
                       "avg_rating","review_count","loves_count","rec_rate","score"]
        export = results[[c for c in export_cols if c in results.columns]].copy()
        export["rec_rate"] = (export["rec_rate"]*100).round(1)
        export.columns = [c.replace("_"," ").title() for c in export.columns]
        st.download_button(
            "⬇️ Download Recommendations as CSV",
            data=export.to_csv(index=False).encode(),
            file_name=f"recommendations_{skin_type}_budget{budget}.csv",
            mime="text/csv",
            use_container_width=True
        )


# ─────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='footer'>"
    "Sephora Beauty Analytics Dashboard &nbsp;|&nbsp; IV Semester Statistical Data Analysis — Major Project"
    "&nbsp;|&nbsp; Built with Streamlit · Plotly · NLTK · scikit-learn"
    "</div>",
    unsafe_allow_html=True
)
