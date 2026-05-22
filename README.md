# 📈 SentiFi — AI Market Sentiment Analyzer

> Real-time stock sentiment analysis using **FinBERT** + Reddit + Twitter, with a live dashboard and REST API.

Built with Python, FastAPI, Pandas, NumPy, Plotly Dash, and deployed on Render.

---

## 🔥 Features

- **FinBERT sentiment** — finance-specific BERT model (far better than generic models for stocks)
- **Multi-source scraping** — Reddit (wallstreetbets, stocks, investing) + Twitter/X
- **NumPy signal engine** — upvote-weighted scoring, normalization, thresholding → BUY / SELL / HOLD
- **Pandas data pipeline** — cleaning, deduplication, rolling averages, correlation analysis
- **Candlestick price chart** — 30-day OHLCV data via yFinance
- **Live Dash dashboard** — all in one dark-themed UI
- **REST API** — every feature accessible via clean endpoints
- **PostgreSQL on Render** — persistent storage of all sentiment + signal history

---

## 🗂 Project Structure

```
sentifi/
├── main.py                    # FastAPI entry point
├── requirements.txt
├── render.yaml                # Render deployment config
├── .env.example
├── app/
│   ├── db.py                  # SQLAlchemy models
│   ├── scraper/
│   │   ├── reddit.py          # PRAW scraper → Pandas DataFrame
│   │   └── twitter.py         # Tweepy v2 scraper → Pandas DataFrame
│   ├── sentiment/
│   │   └── analyzer.py        # FinBERT inference (batched)
│   ├── signals/
│   │   └── engine.py          # NumPy signal computation
│   ├── stock/
│   │   └── fetcher.py         # yFinance price data
│   ├── api/
│   │   └── routes.py          # FastAPI routes
│   └── dashboard/
│       └── app.py             # Plotly Dash UI
└── tests/
    └── test_signals.py        # Pytest unit tests
```

---

## ⚙️ Local Setup

### 1. Clone & install
```bash
git clone https://github.com/yourusername/sentifi.git
cd sentifi
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set up environment variables
```bash
cp .env.example .env
# Fill in your API keys in .env
```

**Getting API keys:**
- **Reddit**: Go to https://www.reddit.com/prefs/apps → Create app → Script type
- **Twitter/X**: Go to https://developer.twitter.com → Create project → Get Bearer Token (free tier works)

### 3. Run locally
```bash
python main.py
```

- API docs: http://localhost:8000/docs
- Dashboard: http://localhost:8000/dashboard

### 4. Run tests
```bash
pytest tests/ -v
```

---

## 🚀 Deploy to Render (Free)

### Option A — Auto deploy with render.yaml (recommended)
1. Push code to GitHub
2. Go to https://render.com → New → Blueprint
3. Connect your repo → Render reads `render.yaml` automatically
4. Add your secret env vars in the Render dashboard:
   - `REDDIT_CLIENT_ID`
   - `REDDIT_CLIENT_SECRET`
   - `TWITTER_BEARER_TOKEN`
5. Deploy → your app is live!

### Option B — Manual setup
1. New Web Service → connect GitHub repo
2. Build command: `pip install -r requirements.txt`
3. Start command: `gunicorn main:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --timeout 120`
4. Add a free PostgreSQL database → copy connection string to `DATABASE_URL` env var
5. Add remaining env vars → Deploy

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/analyze/{ticker}` | Scrape + analyze + return BUY/SELL/HOLD signal |
| GET | `/stock/{ticker}` | Historical price data + stats |
| GET | `/history/{ticker}` | Past sentiment records from DB |
| GET | `/signals/{ticker}` | Past signals from DB |
| GET | `/health` | Health check |
| GET | `/docs` | Interactive Swagger UI |
| GET | `/dashboard` | Live Dash dashboard |

### Example requests
```bash
# Analyze AAPL from both sources
curl http://localhost:8000/analyze/AAPL?sources=reddit,twitter&limit=50

# Get 30-day price data
curl http://localhost:8000/stock/TSLA?days=30

# Get past signals
curl http://localhost:8000/signals/NVDA
```

---

## 🧠 How the Signal Engine Works

```
Scraped posts
     ↓
Pandas cleaning (dedup, URL removal, length filter)
     ↓
FinBERT inference → score (-1 to +1), label, confidence
     ↓
NumPy weighting:
  - weights = upvotes / upvotes.sum()       (popularity weighting)
  - weighted_score = dot(weights, scores)   (NumPy dot product)
  - final = clip(weighted_score, -1, 1)     (NumPy clip)
     ↓
Threshold:
  final > 0.15  → BUY
  final < -0.15 → SELL
  else          → HOLD
```

---

## 📊 Tech Stack

| Layer | Tech |
|-------|------|
| Backend API | FastAPI + Uvicorn |
| Sentiment Model | FinBERT (ProsusAI/finbert) via HuggingFace |
| Data Processing | Pandas + NumPy |
| Reddit Scraping | PRAW |
| Twitter Scraping | Tweepy v2 |
| Stock Data | yFinance |
| Dashboard | Plotly Dash + Dash Bootstrap |
| Database | SQLite (dev) / PostgreSQL (prod) |
| ORM | SQLAlchemy |
| Deployment | Render + Gunicorn |

---

## ⚠️ Disclaimer

SentiFi is for **educational purposes only**. It is not financial advice. Never make investment decisions based solely on sentiment signals.

---

## 👤 Author

**Ashmit Vijay Singh** — [LinkedIn](#) | [GitHub](#)
