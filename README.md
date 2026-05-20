# Trader Performance vs. Market Sentiment Analysis (Hyperliquid x Fear & Greed Index)

This project explores the relationship between derivatives traders' performance/behavior on the Hyperliquid decentralized exchange and Bitcoin market sentiment (as measured by the Bitcoin Fear & Greed Index).

## 🚀 Setup & Installation

### Prerequisites
Make sure you have Python 3.10+ installed.

### Install Dependencies
Run the following command to install the required libraries (including visualization, machine learning, and data serializing libraries):
```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit plotly pyarrow
```

---

## 🏃 How to Run

### 1. Execute the Data Analysis Pipeline
To clean the data, align the datasets, perform metrics engineering, run the K-Means clustering model, train the Random Forest predictive model, and output the processed datasets:
```bash
python analyze_data.py
```
*Outputs are saved under `processed_data/`.*

### 2. Launch the Streamlit Interactive Dashboard
To start the interactive dark-mode dashboard (featuring PnL analysis, sentiment comparisons, behavioral shift charts, clustering visualizations, and a live ML playground):
```bash
streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your web browser.

### 3. Generate Static PNG Charts
To generate the paper-ready high-resolution charts as static image files:
```bash
python generate_charts.py
```
*Outputs are saved under `output_charts/`.*

### 4. Interactive Jupyter Notebook
To run the analysis interactively cell-by-cell and inspect the code, open:
```bash
Trader_Performance_Sentiment_Analysis.ipynb
```

---

## 📊 Summary of Findings

### 1. Methodology
- **Alignment**: Integrated trade-level Hyperliquid records with daily Fear & Greed Index entries on overlapping dates (`2023-05-01` to `2025-05-01`, 479 trading days).
- **Leverage Proxy**: Modeled equity chronologically starting with a $100,000 baseline per account:
  $$\text{Equity}_t = \text{Initial Equity} + \sum \text{PnL} - \sum \text{Fees}$$
  $$\text{Leverage} = \frac{\text{Size USD}}{\text{Equity}}$$
- **Segmentation**: Clustered accounts into 3 archetypes using K-Means clustering.
- **Predictive Model**: Fit a Random Forest Classifier to forecast next-day profitability using lagged behavior/sentiment features (Accuracy: **60.4%** | ROC AUC: **61.7%**).

### 2. Key Insights
- **Sentiment-Driven Leverage Shifts**: Average equity leverage escalates by **50x** during Greed days (2.75x) compared to Fear days (0.05x).
- **Fear-Regime Contrarian Buying**: During Fear, traders scale down leverage almost to zero but maintain a high long bias (Long/Short ratio of **1.97**), allowing them to accumulation wicks without liquidation risk.
- **Trader Archetypes**: Top-performing traders are high-frequency, low-leverage scalpers ("Consistent Profit-Scalpers") who prioritize win rate (~77%) and size scaling over leverage.

### 3. Strategy Recommendations
- **Gated Leverage Limit**: Cap leverage at 5x during Greed regimes (Index ≥ 60) to protect capital from sudden sentiment-reversing liquidations.
- **Contrarian Spot-Only Buying**: Adopt a buying bias during Fear regimes (Index ≤ 40) but keep leverage at 1x/spot-only to survive high volatility wicks.
