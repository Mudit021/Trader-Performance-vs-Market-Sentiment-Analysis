# Trader Performance vs Market Sentiment Analysis

This repository analyzes trader performance relative to Fear/Greed market sentiment. It includes data preparation, daily metrics, sentiment-driven comparison, trader segmentation, and a lightweight Streamlit dashboard for interactive exploration.

## Project structure

- `notebook.ipynb` — analysis notebook with data cleaning, metric generation, sentiment comparison, and strategy output.
- `app.py` — Streamlit dashboard for visualizing sentiment performance, trader clusters, and predictive model results.
- `historical_data.csv` — raw trade dataset containing account-level trades and PnL details.
- `fear_greed_index.csv` — daily Fear/Greed sentiment index.
- `requirements.txt` — Python dependencies required by the dashboard and notebook.

## Prerequisites

- Python 3.9 or later
- `pip` package manager
- Recommended: a Python virtual environment

## Setup

1. Open a terminal in the project root:

```bash
cd "m:\Trader Performance vs Market Sentiment Analysis"
```

2. Create and activate a virtual environment.

Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS / Linux:
```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Confirm the dataset files are present in the project root:

- `historical_data.csv`
- `fear_greed_index.csv`

## How to run

### Run the Streamlit dashboard

Launch the interactive dashboard:

```bash
streamlit run app.py
```

Then open the URL shown in your terminal (typically `http://localhost:8501`).

### Open the notebook

Open `notebook.ipynb` in Jupyter Notebook, JupyterLab, or VS Code and run the cells sequentially. The notebook contains:

- data loading and timestamp normalization
- daily PnL and trade metrics
- sentiment-based performance comparison
- trader behavior segments
- Part C actionable strategy output and a simple predictive model

## Dashboard pages

The Streamlit app includes:

- **Overview** — key metrics, sentiment performance, and daily PnL distribution
- **Sentiment analysis** — Fear vs Greed behavior comparison and summary tables
- **Trader segments** — KMeans behavioral clusters of traders and cluster exploration
- **Model & predictions** — next-day profit prediction results and strategy rules of thumb

## Summary

### Methodology
- Load trade history and Fear/Greed index data.
- Convert timestamps to normalized daily dates and merge sentiment labels with trades.
- Calculate daily metrics: PnL, win rate, trade count, average trade size, long/short ratio.
- Segment traders by frequency, performance, and cluster behavior using KMeans.
- Train a lightweight logistic regression model to predict next-day profitable days from sentiment and behavior features.

### Insights
- **Fear days** showed higher average daily PnL with greater trade volume, but also more downside tail risk.
- **Greed days** had lower average PnL, smaller average trade size, and a stronger short bias.
- High-frequency traders often have mixed outcomes; a small subset of traders delivered consistently high win rates.
- Trader clusters reveal behavioral archetypes with different trade frequency, win rates, and position sizing patterns.

### Strategy recommendations
- During Fear days, favor traders with stronger win rates and balanced long/short exposure to capture upside while limiting risk.
- During Greed days, reduce exposure to high-frequency and inconsistent traders, since this sentiment is associated with more downside tail risk.
- Use trader clusters and account segments to tailor allocation rules rather than applying a single strategy to all traders.

## Data assumptions

- `fear_greed_index.csv` must contain a UNIX-style `timestamp` column and daily sentiment labels.
- `historical_data.csv` must contain a `Timestamp IST` column in `DD-MM-YYYY HH:MM` format and a numeric `Closed PnL` column.
- The current dataset does not include an explicit `Leverage` column. If leverage is required, add it to `historical_data.csv` and update the dashboard/notebook logic accordingly.

## Troubleshooting

- If `streamlit run app.py` fails due to missing packages, reinstall dependencies:

```bash
pip install -r requirements.txt
```

- If the notebook fails at timestamp parsing, verify `historical_data.csv` has `Timestamp IST` formatted as `02-12-2024 22:50`.

- If the dashboard shows empty or inaccurate charts, confirm that both CSV files are in the same folder as `app.py`.


## Run commands summary

```bash
cd "m:\Trader Performance vs Market Sentiment Analysis"
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows PowerShell
pip install -r requirements.txt
streamlit run app.py
```
