# Sephora Beauty Analytics

---

## About

The beauty and personal care industry has exploded on e-commerce platforms, and Sephora is one of the richest sources of that data — detailed product listings, millions of reviews, and user-reported attributes like skin type and tone. But most research on this space either uses generic datasets from Amazon or Yelp that don't account for dermatological factors, or builds ML models without checking if the relationships they find are statistically significant in the first place.

This project takes a different approach. Using a skincare-specific dataset, it combines proper inferential statistics with NLP, topic modeling, clustering, and a recommendation engine — all presented through an interactive Streamlit dashboard. The idea was to build something that's both academically rigorous and practically useful.

**Dataset at a glance:**
- 8,494 skincare products
- 1,094,000+ customer reviews
- 41 variables — price, rating, loves count, skin type, ingredients, recommendation flag, and more

---

## Dataset

[Sephora Products and Skincare Reviews — Kaggle](https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews)

The reviews are split across five files. Download all of them and place in the project root alongside `product_info.csv`:

```
product_info.csv
reviews_0-250.csv
reviews_250-500.csv
reviews_500-750.csv
reviews_750-1250.csv
reviews_1250end.csv
```

The app merges all review files automatically at load time using `product_id`.

---

## Project Structure

```
sephora-beauty-analytics/
├── sephora_streamlit_app_mod.py   # Main dashboard app
├── product_info.csv               # Product data (27 variables)
├── reviews_1250end.csv            # Sample review file (rest downloaded from Kaggle)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Process:

### 1. Data Preprocessing

The five review files are merged with product metadata on product_id. Columns with more than 50% missing values (variation_desc, sale_price_usd, etc.) are dropped. Remaining nulls in categorical columns like skin_type, eye_color, and hair_color are filled with "Unknown". New features are engineered — price_tier (Budget / Mid / Premium / Luxury), sentiment_label from rating, and submission_year from timestamp.

### 2. Exploratory Data Analysis

Descriptive statistics are computed for all numeric variables including skewness and kurtosis. Key findings from EDA:

- rating is strongly left-skewed — most reviewers give 4 or 5 stars
- price_usd and loves_count are heavily right-skewed, with a small number of luxury products and viral items pulling the mean far above the median
- Review volume has grown steadily year-on-year, but average ratings have stayed flat around 4.0–4.2, suggesting consistent product quality

Visualizations include price distribution histograms, box plots by category, demographic breakdowns (skin type, eye color, hair color), price vs rating scatter, and a Pearson correlation heatmap across 8 numeric variables.

### 3. Hypothesis Testing

Four formal tests are run with α = 0.05:

| Test | Variables | Method | Result |
|---|---|---|---|
| H1 | Skin type × Is Recommended | Chi-Square | Reject H₀ — significant association |
| H2 | Limited Edition vs Standard price | Welch's t-Test | Reject H₀ — significantly different means |
| H3 | Price vs Loves Count | Pearson Correlation | Weak positive r, significant p-value |
| H4 | Sephora Exclusive × Rating Tier | Chi-Square | Reject H₀ — different rating distributions |

Each test shows the statistic, p-value, degrees of freedom, and a plain-language interpretation of what the result means for consumer behavior.

### 4. NLP & Text Analysis

Review text is preprocessed using a standard NLP pipeline — lowercasing, punctuation removal, tokenization (NLTK word_tokenize), stopword removal, and lemmatization (WordNetLemmatizer). This runs on a 30,000-review sample for speed.

Output includes:
- Word clouds separately for positive (4–5 stars), neutral (3 stars), and negative (1–2 stars) reviews
- Top-20 word frequency bar charts for positive vs negative reviews
- Vocabulary stats: average token count per review and total unique terms after cleaning

Positive reviews cluster around words like *love*, *skin*, *feel*, *moisturize*. Negative reviews surface *smell*, *dry*, *break*, *return* — pointing to fragrance sensitivity and dryness as the main pain points.

### 5. LDA Topic Modeling

Latent Dirichlet Allocation is fitted on the cleaned review corpus using `CountVectorizer` (max 3,000 features, min_df=5) and LatentDirichletAllocation from scikit-learn. Number of topics is adjustable via a slider (3–8) in the dashboard.

With K=5, the model identifies:
1. **Moisturization & Hydration** — the dominant use-case
2. **Fragrance & Texture** — a key purchase differentiator, especially for sensitive skin
3. **Skin Concerns & Breakouts** — acne control, pore reduction, blemish treatment
4. **Product Efficacy & Routine** — longitudinal evaluation ("after two weeks", "noticed results")
5. **Packaging & Value** — price-to-size perception, particularly for premium products

Each topic is shown as a horizontal bar chart of top-10 weighted terms, with model perplexity reported for fit quality.

### 6. K-Means Clustering + PCA

Clustering is done on three product-level features: price_usd, rating, and loves_count, after StandardScaler normalization. The elbow method (K = 2 to 9) clearly bends at K=3, identifying three segments:

- **Budget / Low-Engagement** — lower price, moderate popularity
- **Mid-Range / Moderate-Popularity** — balanced across all three dimensions
- **Premium / High-Popularity** — higher price, variable loves count (some viral, some niche)

PCA reduces the 3D feature space to 2 principal components for visualization. The first two PCs explain ~85% of variance. A scree plot confirms that 2–3 components capture the essential structure.

### 7. Recommendation System

Products are filtered by the user's skin type and maximum budget, then further narrowed by minimum rating, category, and optional flags (exclude out-of-stock, Sephora exclusive only, highly recommended only).

Filtered results are scored using:

```
score = (avg_rating × 0.50) + (rec_rate × 0.30) + (normalized_loves_count × 0.20)
```

Top-N results are displayed as ranked cards with badges (New, Limited Edition, Sephora Exclusive), alongside a rating comparison bar chart and a price vs rating bubble plot. Results can be exported as CSV.

---

## Dashboard Pages

| Page | What it shows |
|---|---|
| Overview | Dataset summary, analysis pipeline, sample data preview |
| Descriptive Statistics | Numeric summaries, categorical frequency tables |
| Visualizations | 7 figures — price distribution, demographics, heatmap, trends |
| Hypothesis Testing | 4 tests with statistics, p-values, and interpretations |
| NLP & Text Analysis | Pre-processing pipeline, word clouds, top-20 frequency |
| LDA Topic Modeling | Topic table, word weight charts, business interpretation |
| Clustering & PCA | Elbow curve, 3D scatter, PCA scree plot |
| Product Recommender | Filtered recommendations, charts, CSV export |

---

## Running it

```bash
git clone https://github.com/<your-username>/sephora-beauty-analytics.git
cd sephora-beauty-analytics

pip install -r requirements.txt

streamlit run sephora_streamlit_app_mod.py
```

Opens at `http://localhost:8501`. Data loads and caches on first run — NLP preprocessing takes a minute the first time.

---

## Stack

| Purpose | Libraries |
|---|---|
| Dashboard | Streamlit |
| Data handling | pandas, NumPy |
| Visualization | Plotly, Matplotlib, Seaborn |
| Statistics | SciPy (chi2_contingency, ttest_ind, pearsonr) |
| NLP | NLTK (tokenizer, stopwords, lemmatizer), WordCloud |
| Machine Learning | scikit-learn (KMeans, PCA, LDA, StandardScaler, CountVectorizer) |

---

## References

1. [AI-powered skincare product recommendation systems: From data collection to user experience](https://www.researchgate.net/publication/386750976)
2. [Recommending skincare products using text analysis of product reviews](https://www.researchgate.net/publication/353476721)
3. [Customer experience analysis via topic modeling and sentiment analysis](https://www.researchgate.net/publication/372084164)
4. [NLP for analyzing online customer reviews: a survey, taxonomy, and open research challenges](https://peerj.com/articles/cs-2203/)
5. [Sentiment analysis for e-commerce recommendations using deep learning and transformers](https://www.mdpi.com/2227-7390/12/15/2403)
6. [Streamlit — The fastest way to build and share data apps](https://streamlit.io)

---

*Dataset sourced from Kaggle. Code shared for academic reference.*
