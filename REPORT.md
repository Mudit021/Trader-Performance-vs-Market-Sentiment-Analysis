# Quantitative Research Report: Trader Performance vs. Market Sentiment

**Author:** Antigravity (Advanced Agentic AI Researcher)  
**Date:** May 20, 2026  
**Dataset Overlap:** May 1, 2023 – May 1, 2025 (479 unique trading days)

---

## 1. Methodology
To evaluate the interaction between market sentiment and derivatives trader behavior on the Hyperliquid decentralized exchange, we designed and implemented a multi-stage quantitative pipeline:

1. **Data Alignment**: We parsed transaction timestamps from `historical_data.csv` (211,224 rows) to a daily level and joined it with the daily Bitcoin Fear & Greed Index (`fear_greed_index.csv`) using an inner join, yielding **211,218 aligned trades**.
2. **Equity-Adjusted Leverage Metric**: Since raw trading records lack a leverage column, we modeled a chronological capital account for each trader starting with a baseline equity ($E_0 = \$100,000$). Running equity $E_t$ is updated recursively:
   $$E_t = E_{t-1} + \text{Closed PnL}_t - \text{Fee}_t$$
   Where $E_t$ was clipped at a lower bound of \$100 to prevent division by zero. Trade leverage is defined as:
   $$\text{Leverage}_t = \frac{\text{Trade Size (USD)}_t}{E_t}$$
3. **Sentiment Classification**: Days were binned into three sentiment regimes based on index scores: **Fear** (Score $\le 40$), **Neutral** ($41 \le \text{Score} \le 59$), and **Greed** (Score $\ge 60$).
4. **Behavioral Segmentation**: We grouped the 32 unique trading accounts into behavioral archetypes using **K-Means clustering** across five standardized variables: win rate, daily trade frequency, average trade size, average equity leverage, and average long/short ratio.
5. **Next-Day Predictability**: A **Random Forest Classifier** was trained to predict next-day profitability (binary target $PnL_{t+1} > 0$) using sentiment features (current F&G value, 3-day momentum) and lagging trader behavior. To prevent temporal data leakage, an 80/20 chronological split was utilized.

---

## 2. Key Empirical Insights

### A. Leverage & Risk Escalation During Greed
Empirical analysis indicates that market sentiment is a massive driver of leverage expansion. Average equity leverage escalates by **50x** during Greed days (**2.75x**) compared to Fear days (**0.05x**). This behavioral shift shows significant trader overconfidence during uptrends, creating systemic vulnerability to sudden price corrections.

### B. Contrarian Buying & Volatility Insulation During Fear
During Fear days, despite the market being in a downturn, traders maintain a strong **long bias (Long/Short ratio of 1.97)**. However, they manage risk by reducing leverage to nearly zero (**0.05x**, equivalent to spot trading). This contrarian "buying the dip" behavior allows top traders to accumulate wicks without incurring liquidation risk.

### C. Trader Archetypes and Win Rate Dominance
K-Means clustering classified the 32 accounts into three groups:
- **Consistent Profit-Scalpers (19 Accounts)**: Maintain high win rates (**77.0%**) and high trade frequencies (**156.7 trades/day**) while using very low leverage (**0.05x**). This group represents the most consistent and sustainable earners.
- **High-Leverage Speedrunners (1 Account)**: Utilizes extreme leverage (avg **124.8x**) and cross-margin configurations, realizing **$1.60M** in PnL but exposing the account to extreme drawdown.
- **Low-Activity Moderate Traders (12 Accounts)**: Trade less frequently (**38.7 trades/day**) with lower win rates (**66.0%**), suffering slow capital decay.

### D. Next-Day Profitability Predictability
The Random Forest model achieved an **Accuracy of 60.4%** and **ROC AUC of 61.7%** on the out-of-sample test set. Feature importances indicate that a trader's current-day PnL, win rate, and directional bias are the strongest predictors of tomorrow's profitability, suggesting high behavioral inertia.

---

## 3. Strategy & Risk Recommendations

Based on these findings, we propose two sentiment-gated rules of thumb:

1. **Sentiment-Gated Leverage Cap (Greed Regime)**:
   - *Rationale*: During Greed days, although total realized PnL is high ($4.73M), average daily PnL drops to its lowest ($4,177) and leverage increases by 50x. Extreme leverage on Greed days impairs risk-adjusted returns due to sudden market wicks.
   - *Rule*: When the Fear & Greed Index is $\ge 60$, enforce a **maximum leverage cap of 5x** across all accounts.
2. **Volatility Insulated Dip-Buying (Fear Regime)**:
   - *Rationale*: Top traders profitably buy market panic by maintaining a high long bias (1.97 L/S ratio) but scaling down leverage (0.05x).
   - *Rule*: When the Fear & Greed Index is $\le 40$, traders should increase their long exposure (target L/S ratio $\ge 2.0$) but restrict trade execution to **spot-only or 1x leverage** to survive volatility shakeouts.
