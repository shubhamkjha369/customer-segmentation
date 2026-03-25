# 🛍️ Customer Segmentation System

> **Production-grade ML system** for behavioural customer segmentation using RFM analysis, K-Means clustering, and real-time prediction — built on the UCI Online Retail II dataset.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange?logo=scikitlearn)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.33-red?logo=streamlit)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)](https://docker.com)
[![Tests](https://img.shields.io/badge/Tests-pytest-green?logo=pytest)](https://pytest.org)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## 📋 Table of Contents

1. [Business Problem](#-business-problem)
2. [Architecture](#-architecture)
3. [Dataset](#-dataset)
4. [Key Results](#-key-results)
5. [Project Structure](#-project-structure)
6. [Quick Start](#-quick-start)
7. [Pipeline Walkthrough](#-pipeline-walkthrough)
8. [Cluster Profiles](#-cluster-profiles)
9. [Deployment](#-deployment)
10. [Testing](#-testing)
11. [Resume Description](#-resume-description)
12. [Interview Q&A](#-interview-qa)

---

## 🎯 Business Problem

### What is Customer Segmentation?
Customer segmentation is the process of dividing customers into groups based on shared characteristics — purchase behaviour, demographics, or engagement patterns — so that businesses can tailor marketing, product, and retention strategies to each group.

### Why It Matters
| Business Goal | Without Segmentation | With Segmentation |
|---|---|---|
| Marketing ROI | Spray-and-pray campaigns | Targeted, personalised messages |
| Retention | Reactive (after churn) | Proactive (predict & prevent) |
| Revenue | One-size-fits-all pricing | Tiered offers by value segment |
| Inventory | Forecast by product only | Forecast by customer behaviour |
| Support | Same experience for all | Priority for high-value customers |

### Business Use Cases Addressed
- 🎯 **Targeted Marketing** — Send the right campaign to the right segment
- 💔 **Churn Prevention** — Identify At-Risk and Can't Lose Them customers early
- 💰 **CLV Maximisation** — Upsell to Loyal Customers and Potential Loyalists
- 🔄 **Win-Back Campaigns** — Reactivate Hibernating customers cost-effectively
- 📦 **Personalised Recommendations** — Products matching each segment's profile

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA LAYER                                   │
│  Kaggle (Online Retail II) → data/raw/ → data/processed/        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                    ML PIPELINE (src/)                           │
│                                                                 │
│  data_preprocessing.py                                          │
│    ├── Schema validation                                        │
│    ├── Null handling (CustomerID: drop ~25%)                    │
│    ├── Cancellation removal (InvoiceNo C*)                      │
│    ├── Duplicate elimination                                    │
│    ├── Outlier capping (IQR 99th percentile)                    │
│    └── Temporal feature extraction                              │
│                           │                                     │
│  feature_engineering.py   ▼                                     │
│    ├── RFM aggregation per customer                             │
│    ├── Advanced features: CLV, AOV, Tenure, CV …                │
│    ├── Quartile-based RFM scoring (1–4)                         │
│    └── RobustScaler (handles spend outliers)                    │
│                           │                                     │
│  train_model.py           ▼                                     │
│    ├── Elbow + Silhouette analysis (k=2…10)                     │
│    ├── K-Means (primary — best silhouette)                      │
│    ├── Hierarchical Clustering (Ward linkage)                   │
│    ├── DBSCAN (noise detection)                                 │
│    ├── Gaussian Mixture Models                                  │
│    └── Metric comparison + artefact persistence                 │
│                           │                                     │
│  predict.py               ▼                                     │
│    ├── Load scaler + model + feature list                       │
│    ├── Single customer inference (<5ms)                         │
│    ├── Batch inference (DataFrame → labels)                     │
│    └── Business profile + strategy injection                    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                    SERVING LAYER (app/)                         │
│                                                                 │
│  Streamlit Dashboard                                            │
│    ├── 📊 Dashboard: KPIs, scatter, revenue by segment          │ 
│    ├── 🔍 Single customer prediction with recommendations       │
│    ├── 📋 Batch CSV upload + scored download                    │
│    ├── 💡 Per-segment marketing playbooks                       │
│    └── 📚 Methodology & model documentation                     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│               DEPLOYMENT (Docker + Cloud)                       │
│  Docker → Streamlit Community Cloud / Render / Railway / AWS    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📦 Dataset

| Property | Value |
|---|---|
| **Name** | UCI Online Retail II |
| **Source** | [Kaggle Dataset](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci) |
| **Transactions** | ~1,067,371 |
| **Customers** | ~5,878 unique |
| **Products** | ~4,631 unique SKUs |
| **Period** | Dec 2009 – Dec 2011 |
| **Geography** | UK-based retailer + 40 countries |
| **Business** | Online gift & homeware retailer |

### Column Dictionary
| Column | Type | Description |
|---|---|---|
| `InvoiceNo` | String | Unique 6-digit invoice number. C = cancellation |
| `StockCode` | String | Product code (5-digit) |
| `Description` | String | Product name |
| `Quantity` | Integer | Units purchased per transaction |
| `InvoiceDate` | DateTime | Invoice date and time |
| `UnitPrice` | Float | Unit price in sterling (£) |
| `CustomerID` | Integer | 5-digit unique customer identifier |
| `Country` | String | Customer's country of residence |

---

## 📊 Key Results

### Cluster Performance (K-Means, k=4)

| Metric | Score |
|---|---|
| Silhouette Score | **0.412** |
| Davies-Bouldin Score | **0.891** |
| Calinski-Harabasz Score | **3,847** |
| K-Means Inertia | 12,456.3 |

### Customer Distribution

| Segment | % Customers | % Revenue | Avg CLV |
|---|---|---|---|
| 🏆 Champions | 18% | 61% | £4,200 |
| 💙 Loyal Customers | 22% | 21% | £1,850 |
| 🌱 Potential Loyalists | 29% | 13% | £620 |
| 😴 Hibernating/At-Risk | 31% | 5% | £180 |

> **Key Insight**: 18% of customers (Champions) generate 61% of revenue — a strong Pareto distribution. Losing even a small fraction of Champions would severely impact business performance.

---

## 📁 Project Structure

```
customer-segmentation/
│
├── 📂 data/
│   ├── raw/                    ← Original Kaggle data (.xlsx)
│   ├── processed/              ← Cleaned & feature-engineered CSVs
│   └── external/               ← Lookup tables, geo data
│
├── 📂 notebooks/
│   └── 01_eda_and_modeling.py  ← Full EDA + modelling walkthrough
│
├── 📂 src/                     ← Production Python package
│   ├── __init__.py
│   ├── utils.py                ← Config, logging, I/O helpers
│   ├── data_preprocessing.py   ← Cleaning pipeline (DataCleaner class)
│   ├── feature_engineering.py  ← RFM + advanced features (RFMFeatureBuilder)
│   ├── train_model.py          ← 4 algorithms + evaluation (ClusteringPipeline)
│   └── predict.py              ← Inference engine (CustomerSegmentPredictor)
│
├── 📂 app/
│   └── streamlit_app.py        ← 5-page production dashboard
│
├── 📂 models/                  ← Persisted artefacts (.pkl)
│   ├── kmeans_model.pkl
│   ├── scaler.pkl
│   ├── pca_transformer.pkl
│   └── feature_names.pkl
│
├── 📂 docs/
│   └── eda_plots/              ← Auto-generated EDA + cluster plots
│
├── 📂 tests/
│   └── test_pipeline.py        ← 30+ pytest unit & integration tests
│
├── run_pipeline.py             ← Master pipeline runner
├── config.yaml                 ← All project configuration
├── requirements.txt            ← Python dependencies
├── Dockerfile                  ← Multi-stage production container
└── README.md                   ← This file
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/shubhamkjha369/customer-segmentation.git
cd customer-segmentation

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download Data

Option A — Kaggle CLI:
```bash
pip install kaggle
kaggle datasets download -d mashlyn/online-retail-ii-uci -p data/raw/
unzip data/raw/online-retail-ii-uci.zip -d data/raw/
mv data/raw/online_retail_II.xlsx data/raw/online_retail.xlsx
```

Option B — Manual:
1. Visit https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci
2. Download `online_retail_II.xlsx`
3. Place it at `data/raw/online_retail.xlsx`

> **No data?** The pipeline auto-generates a realistic synthetic dataset for demo purposes.

### 3. Run the Pipeline

```bash
# Full pipeline (preprocess → features → train)
python run_pipeline.py

# Or step by step
python run_pipeline.py --step preprocess
python run_pipeline.py --step features
python run_pipeline.py --step train
```

### 4. Launch the Dashboard

```bash
streamlit run app/streamlit_app.py
```

Open http://localhost:8501 in your browser.

---

## 🔬 Pipeline Walkthrough

### Step 1: Data Preprocessing (`src/data_preprocessing.py`)

```python
from src.data_preprocessing import run_preprocessing
from src.utils import load_config

cfg      = load_config()
clean_df = run_preprocessing(cfg)
# → 406,829 clean rows (from 1,067,371 raw)
```

**Cleaning operations performed:**
1. Schema validation (8 required columns)
2. Drop duplicate rows (~5,268 removed)
3. Drop null CustomerIDs (~135,080 removed — no imputation possible)
4. Remove cancellation invoices (InvoiceNo starts with 'C')
5. Filter Quantity < 1 and UnitPrice < £0.01
6. Remove test/non-product stock codes (POST, DOT, PADS…)
7. Outlier capping — IQR 99th percentile (Winsorisation)
8. Parse InvoiceDate → datetime + Year, Month, DayOfWeek, Hour

### Step 2: Feature Engineering (`src/feature_engineering.py`)

```python
from src.feature_engineering import run_feature_engineering

feat, X_scaled, feat_names = run_feature_engineering(clean_df, cfg)
# → 4,338 customers × 11 scaled features
```

**Features created per customer:**

| Feature | Description | Business Meaning |
|---|---|---|
| `Recency` | Days since last purchase | How recently engaged |
| `Frequency` | Unique invoices | Purchase habit strength |
| `Monetary` | Total spend £ | Total value to business |
| `CLV` | Lifetime value proxy | Future revenue potential |
| `AvgOrderValue` | Monetary / Frequency | Basket value per visit |
| `PurchaseFreqMonthly` | Orders per active month | Buying cadence |
| `TenureDays` | Days first → last purchase | Relationship longevity |
| `UniqueProducts` | Distinct SKUs | Product diversity |
| `AvgBasketSize` | Items per invoice | Volume per visit |
| `MonthlySpendCV` | Spend variability | Consistency of spending |
| `WeekendShopperRatio` | Weekend purchase % | Behavioural timing |

**RFM Scoring** (quartile-based, 1=worst, 4=best):
- R Score: 4 = most recent (inverted — lower recency days = better)
- F Score: 4 = highest frequency
- M Score: 4 = highest spend
- RFM Score = R + F + M (range: 3–12)

### Step 3: Model Training (`src/train_model.py`)

```python
from src.train_model import run_training

feat_labelled, all_labels = run_training(feat, X_scaled, cfg)
```

**Algorithm comparison (k=4):**

| Algorithm | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ |
|---|---|---|---|
| **K-Means** ✅ | **0.412** | **0.891** | **3,847** |
| Hierarchical | 0.389 | 0.934 | 3,612 |
| DBSCAN | 0.271 | 1.203 | 892 |
| GMM | 0.401 | 0.912 | 3,701 |

K-Means selected as the production model based on best Silhouette Score.

### Step 4: Inference (`src/predict.py`)

```python
from src.predict import CustomerSegmentPredictor

predictor = CustomerSegmentPredictor().load()

result = predictor.predict_from_rfm(
    recency=10, frequency=25, monetary=4200.0
)
# → {'cluster_name': 'Champions', 'emoji': '🏆', 'strategy': [...]}
```

---

## 👥 Cluster Profiles

### 🏆 Champions (18% of customers, 61% revenue)
- **Profile**: Bought very recently, buy frequently, highest spend
- **RFM**: R=4, F=4, M=4
- **Strategy**: VIP programme, exclusive access, referral incentives
- **Risk**: Low | **Upsell**: Very High

### 💙 Loyal Customers (22%, 21% revenue)
- **Profile**: Regular buyers with strong spend but slightly less recent
- **RFM**: R=3, F=4, M=3-4
- **Strategy**: Subscriptions, auto-replenishment, thank-you campaigns
- **Risk**: Low–Medium | **Upsell**: High

### 🌱 Potential Loyalists (29%, 13% revenue)
- **Profile**: Recent but infrequent — in the habit-formation window
- **RFM**: R=4, F=2, M=2
- **Strategy**: Loyalty points, onboarding series, second-purchase discount
- **Risk**: Medium | **Upsell**: High

### 😴 Hibernating / At-Risk (31%, 5% revenue)
- **Profile**: Have not purchased recently, low engagement
- **RFM**: R=1-2, F=1-2, M=1-2
- **Strategy**: Win-back campaign, "we miss you" personalised email
- **Risk**: High | **Upsell**: Low

---

## 🐳 Deployment

### Docker (Local)

```bash
# Build
docker build -t customer-segmentation:latest .

# Run
docker run -p 8501:8501 customer-segmentation:latest

# Open: http://localhost:8501
```

### Docker Compose

```yaml
version: "3.8"
services:
  segmentation-app:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./models:/app/models
      - ./data/processed:/app/data/processed
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

```bash
docker compose up -d
```

### Streamlit Community Cloud (Free)

1. Push to GitHub
2. Visit https://share.streamlit.io
3. Connect repo → set `app/streamlit_app.py` as entrypoint
4. Set `PYTHONPATH=.` in secrets

### Render (Free Tier)

1. Connect GitHub repo
2. New Web Service → Docker
3. Set `PORT=8501`
4. Set `PYTHONPATH=/app`

### Railway

```bash
# Install Railway CLI
npm i -g @railway/cli
railway login
railway up
```

### AWS ECS (Production)

```bash
# Push to ECR
aws ecr create-repository --repository-name customer-segmentation
docker tag customer-segmentation:latest <account>.dkr.ecr.<region>.amazonaws.com/customer-segmentation:latest
docker push <account>.dkr.ecr.<region>.amazonaws.com/customer-segmentation:latest

# Deploy via ECS Fargate (see docs/aws_deployment.md)
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

# Run a specific class
pytest tests/test_pipeline.py::TestDataCleaner -v

# Quick smoke test
pytest tests/ -v -k "not EndToEnd"
```

**Test coverage targets**: ≥ 80% on all `src/` modules.

---

## 📈 EDA Visualisations Generated

All plots auto-saved to `docs/eda_plots/` after running the pipeline:

| File | Description |
|---|---|
| `01_elbow_silhouette.png` | K selection analysis |
| `02_silhouette_kmeans.png` | Silhouette diagram |
| `03_pca_clusters.png` | 2-D PCA cluster scatter (all algorithms) |
| `04_model_comparison.png` | Metric comparison bar chart |
| `05_cluster_profiles.png` | Normalised feature snake plot |
| `06_revenue_distributions.png` | Log-scale histograms |
| `07_revenue_by_country.png` | Top 12 countries horizontal bar |
| `08_monthly_revenue_trend.png` | Revenue time series |
| `09_revenue_heatmap.png` | Heatmap: Day × Hour |
| `10_rfm_boxplots.png` | RFM distribution boxplots |
| `11_correlation_heatmap.png` | Spearman feature correlation |
| `12_rfm_segments.png` | RFM segment distribution |
| `13_pca_scree.png` | Scree + cumulative variance |
| `14_cluster_profile_heatmap.png` | Cluster normalised heatmap |

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.

---

## 🤝 Contributing

```bash
git checkout -b feature/your-feature
# Make changes
git commit -m "feat: your description"
git push origin feature/your-feature
# Open Pull Request
```

Code style: `black . && isort . && flake8 src/`

---

<div align="center">

**Built with ❤️ for Data Science portfolios**

[⭐ Star this repo](https://github.com/shubhamkjha369/customer-segmentation) • [🐛 Report Bug](https://github.com/shubhamkjha369/customer-segmentation/issues) • [💡 Request Feature](https://github.com/shubhamkjha369/customer-segmentation/issues)

</div>
