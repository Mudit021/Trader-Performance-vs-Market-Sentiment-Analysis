import pandas as pd
import numpy as np
import os
import json
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set paths
DATA_DIR = "m:\\Trader Performance vs Market Sentiment Analysis"
OUTPUT_DIR = os.path.join(DATA_DIR, "processed_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Starting Trader Performance & Sentiment Analysis Pipeline...")

# ----------------------------------------------------
# PART A: Data Preparation & Alignment
# ----------------------------------------------------
print("\n--- Part A: Data Loading and Preparation ---")
df_fg = pd.read_csv(os.path.join(DATA_DIR, "fear_greed_index.csv"))
df_hd = pd.read_csv(os.path.join(DATA_DIR, "historical_data.csv"))

# Document row/column counts and missing values / duplicates
fg_rows, fg_cols = df_fg.shape
hd_rows, hd_cols = df_hd.shape

print(f"Fear & Greed Index Shape: {fg_rows} rows, {fg_cols} columns")
print(f"Historical Data Shape: {hd_rows} rows, {hd_cols} columns")

fg_nulls = df_fg.isnull().sum().to_dict()
hd_nulls = df_hd.isnull().sum().to_dict()
fg_dups = df_fg.duplicated().sum()
hd_dups = df_hd.duplicated().sum()

print(f"Fear & Greed missing values: {fg_nulls}")
print(f"Fear & Greed duplicates: {fg_dups}")
print(f"Historical Data duplicates: {hd_dups}")

# Log dataset metadata
metadata = {
    "fear_greed": {
        "rows": fg_rows,
        "cols": fg_cols,
        "missing": fg_nulls,
        "duplicates": int(fg_dups)
    },
    "historical_data": {
        "rows": hd_rows,
        "cols": hd_cols,
        "missing": {k: int(v) for k, v in hd_nulls.items()},
        "duplicates": int(hd_dups)
    }
}
with open(os.path.join(OUTPUT_DIR, "dataset_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=4)

# Timestamps conversion
print("Converting timestamps...")
df_fg['parsed_date'] = pd.to_datetime(df_fg['date']).dt.date
df_hd['ParsedTime'] = pd.to_datetime(df_hd['Timestamp IST'], format='%d-%m-%Y %H:%M')
df_hd['parsed_date'] = df_hd['ParsedTime'].dt.date

# Sort historical data chronologically to track cumulative statistics
df_hd = df_hd.sort_values(by=['Account', 'ParsedTime', 'Trade ID']).reset_index(drop=True)

# ----------------------------------------------------
# Key Metrics Engineering
# ----------------------------------------------------
print("Engineering key metrics...")

# Calculate Running Equity & Equity Leverage per account
# Assume starting equity of $100,000 for each account
INITIAL_EQUITY = 100000.0
df_hd['Running_Equity'] = INITIAL_EQUITY
df_hd['Equity_Leverage'] = 0.0

accounts_grouped = df_hd.groupby('Account')
updated_dfs = []

for account, group in accounts_grouped:
    # Sort group to be absolutely sure of chronology
    group = group.sort_values('ParsedTime').copy()
    
    # Calculate running cumulative Closed PnL and Fees
    cum_pnl = group['Closed PnL'].cumsum()
    cum_fees = group['Fee'].cumsum()
    
    # Running Equity = Initial + realized PnL - Fees
    group['Running_Equity'] = INITIAL_EQUITY + cum_pnl - cum_fees
    
    # Clip running equity to a minimum of $100 to avoid division by zero or extreme leverage spikes
    group['Running_Equity'] = group['Running_Equity'].clip(lower=100.0)
    
    # Equity Leverage = Size USD / Running Equity
    group['Equity_Leverage'] = group['Size USD'] / group['Running_Equity']
    
    # Relative Leverage = Size USD / Mean(Size USD) for this account
    mean_size = group['Size USD'].mean()
    if mean_size == 0:
        mean_size = 1.0
    group['Relative_Leverage'] = group['Size USD'] / mean_size
    
    updated_dfs.append(group)

df_hd = pd.concat(updated_dfs).sort_values(by=['Account', 'ParsedTime']).reset_index(drop=True)

# Align datasets by date
print("Aligning datasets on date...")
# Merge F&G index into historical data
df_merged = pd.merge(df_hd, df_fg[['parsed_date', 'value', 'classification']], on='parsed_date', how='inner')
print(f"Aligned dataset shape: {df_merged.shape}")

# Save merged dataset (compressed) for analysis
df_merged.to_parquet(os.path.join(OUTPUT_DIR, "aligned_trades.parquet"), index=False)

# Create daily aggregate metrics per account
print("Aggregating metrics to daily level...")
df_merged['is_win'] = (df_merged['Closed PnL'] > 0).astype(int)
df_merged['is_loss'] = (df_merged['Closed PnL'] < 0).astype(int)
df_merged['is_close'] = (df_merged['Closed PnL'] != 0).astype(int)
df_merged['is_buy'] = (df_merged['Side'] == 'BUY').astype(int)

daily_agg = df_merged.groupby(['Account', 'parsed_date']).agg(
    daily_pnl=('Closed PnL', 'sum'),
    total_fee=('Fee', 'sum'),
    trade_count=('Trade ID', 'count'),
    win_trades=('is_win', 'sum'),
    loss_trades=('is_loss', 'sum'),
    close_trades=('is_close', 'sum'),
    buy_trades=('is_buy', 'sum'),
    total_volume_usd=('Size USD', 'sum'),
    avg_trade_size_usd=('Size USD', 'mean'),
    avg_equity_leverage=('Equity_Leverage', 'mean'),
    max_equity_leverage=('Equity_Leverage', 'max'),
    avg_relative_leverage=('Relative_Leverage', 'mean'),
    avg_running_equity=('Running_Equity', 'mean'),
    fg_value=('value', 'first'),
    fg_class=('classification', 'first')
).reset_index()

# Calculate daily win rate & long/short ratio
daily_agg['win_rate'] = daily_agg['win_trades'] / daily_agg['close_trades'].replace(0, np.nan)
# Fill NaN win rate with 0.5 (or check if no close trades)
daily_agg['win_rate'] = daily_agg['win_rate'].fillna(0.5)

# Long/Short ratio based on buy vs sell trades
daily_agg['long_short_ratio'] = daily_agg['buy_trades'] / (daily_agg['trade_count'] - daily_agg['buy_trades']).replace(0, np.nan)
daily_agg['long_short_ratio'] = daily_agg['long_short_ratio'].fillna(1.0) # default to balanced if divide by zero

daily_agg.to_csv(os.path.join(OUTPUT_DIR, "daily_account_metrics.csv"), index=False)

# ----------------------------------------------------
# PART B: Sentiment Regime Analysis
# ----------------------------------------------------
print("\n--- Part B: Analysis & Modeling ---")
print("Analyzing performance under Fear vs Greed days...")

# Define regimes
# Fear: value <= 40
# Neutral: 40 < value < 60
# Greed: value >= 60
daily_agg['regime'] = 'Neutral'
daily_agg.loc[daily_agg['fg_value'] <= 40, 'regime'] = 'Fear'
daily_agg.loc[daily_agg['fg_value'] >= 60, 'regime'] = 'Greed'

# Calculate Max Drawdown per account per regime
def get_drawdown_metrics(df_sub):
    drawdowns = []
    for acct, group in df_sub.groupby('Account'):
        # Sort chronologically
        group = group.sort_values('parsed_date')
        cum_pnl = group['daily_pnl'].cumsum()
        running_max = cum_pnl.cummax()
        dd = running_max - cum_pnl
        max_dd = dd.max()
        drawdowns.append({'Account': acct, 'max_drawdown': max_dd})
    return pd.DataFrame(drawdowns)

# Performance table per regime
regime_perf = daily_agg.groupby('regime').agg(
    total_pnl=('daily_pnl', 'sum'),
    avg_daily_pnl=('daily_pnl', 'mean'),
    median_daily_pnl=('daily_pnl', 'median'),
    avg_win_rate=('win_rate', 'mean'),
    avg_trade_count=('trade_count', 'mean'),
    avg_daily_volume=('total_volume_usd', 'mean'),
    avg_leverage=('avg_equity_leverage', 'mean'),
    avg_long_short_ratio=('long_short_ratio', 'mean')
).reset_index()

# Drawdowns by regime
dd_fear = get_drawdown_metrics(daily_agg[daily_agg['regime'] == 'Fear']).rename(columns={'max_drawdown': 'max_dd_fear'})
dd_greed = get_drawdown_metrics(daily_agg[daily_agg['regime'] == 'Greed']).rename(columns={'max_drawdown': 'max_dd_greed'})
dd_neutral = get_drawdown_metrics(daily_agg[daily_agg['regime'] == 'Neutral']).rename(columns={'max_drawdown': 'max_dd_neutral'})

dd_merged = pd.merge(dd_fear, dd_greed, on='Account', how='outer')
dd_merged = pd.merge(dd_merged, dd_neutral, on='Account', how='outer')

regime_perf.to_csv(os.path.join(OUTPUT_DIR, "regime_performance.csv"), index=False)
dd_merged.to_csv(os.path.join(OUTPUT_DIR, "account_drawdowns_by_regime.csv"), index=False)

print("\nRegime Performance Summary:")
print(regime_perf)

# ----------------------------------------------------
# Trader Clustering (Archetypes)
# ----------------------------------------------------
print("\nClustering traders into behavioral archetypes...")

# Aggregate account metrics
trader_features = daily_agg.groupby('Account').agg(
    total_pnl=('daily_pnl', 'sum'),
    avg_daily_pnl=('daily_pnl', 'mean'),
    std_daily_pnl=('daily_pnl', 'std'),
    avg_win_rate=('win_rate', 'mean'),
    avg_trade_count=('trade_count', 'mean'),
    avg_trade_size=('avg_trade_size_usd', 'mean'),
    avg_equity_leverage=('avg_equity_leverage', 'mean'),
    max_equity_leverage=('max_equity_leverage', 'max'),
    avg_long_short_ratio=('long_short_ratio', 'mean'),
    margin_crossed_ratio=('Account', lambda x: df_merged[df_merged['Account'] == x.iloc[0]]['Crossed'].mean())
).reset_index()

trader_features['std_daily_pnl'] = trader_features['std_daily_pnl'].fillna(0.0)

# Select features for K-Means clustering
features_for_clustering = [
    'avg_win_rate', 'avg_trade_count', 'avg_trade_size', 
    'avg_equity_leverage', 'avg_long_short_ratio', 'margin_crossed_ratio'
]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(trader_features[features_for_clustering])

# Set K = 3
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
trader_features['cluster'] = kmeans.fit_predict(X_scaled)

# Name clusters based on profiles
cluster_profiles = {}
for cluster_id in range(3):
    cluster_sub = trader_features[trader_features['cluster'] == cluster_id]
    mean_leverage = cluster_sub['avg_equity_leverage'].mean()
    mean_frequency = cluster_sub['avg_trade_count'].mean()
    mean_winrate = cluster_sub['avg_win_rate'].mean()
    mean_pnl = cluster_sub['total_pnl'].mean()
    
    print(f"\nCluster {cluster_id} Characteristics:")
    print(f"  Count: {len(cluster_sub)}")
    print(f"  Avg Equity Leverage: {mean_leverage:.4f}")
    print(f"  Avg Daily Trades: {mean_frequency:.2f}")
    print(f"  Avg Win Rate: {mean_winrate:.2f}")
    print(f"  Avg Total PnL: {mean_pnl:.2f}")
    
    # Assign intuitive names based on features
    if mean_leverage > 2.0 or mean_frequency > 100:
        cluster_profiles[cluster_id] = "High-Leverage Speedrunners"
    elif mean_winrate > 0.80 and mean_pnl > 100000:
        cluster_profiles[cluster_id] = "Consistent Profit-Scalpers"
    else:
        cluster_profiles[cluster_id] = "Low-Activity Moderate Traders"

trader_features['archetype'] = trader_features['cluster'].map(cluster_profiles)
# In case of overlaps, let's manually verify and make names distinct
unique_names = list(set(cluster_profiles.values()))
if len(unique_names) < 3:
    # Ensure distinct naming if K-Means boundaries are close
    sorted_clusters = sorted(cluster_profiles.keys(), key=lambda k: trader_features[trader_features['cluster'] == k]['avg_equity_leverage'].mean())
    trader_features.loc[trader_features['cluster'] == sorted_clusters[0], 'archetype'] = "Low-Activity Moderate Traders"
    trader_features.loc[trader_features['cluster'] == sorted_clusters[1], 'archetype'] = "Consistent Profit-Scalpers"
    trader_features.loc[trader_features['cluster'] == sorted_clusters[2], 'archetype'] = "High-Leverage Speedrunners"

trader_features.to_csv(os.path.join(OUTPUT_DIR, "trader_clusters.csv"), index=False)

# ----------------------------------------------------
# Predictive Modeling (Next-Day Profitability)
# ----------------------------------------------------
print("\nBuilding predictive model for next-day profitability...")

# Prepare daily predictive dataset
# Features:
# - Today's F&G Value
# - Today's F&G classification
# - Trader behavior features of today:
#   - PnL
#   - Trade Count
#   - Avg Trade Size
#   - Win Rate
#   - Avg Leverage
#   - Long/Short Ratio
# Target:
# - Tomorrow's daily PnL > 0 (Binary: 1/0)

predictive_data = []

for acct, group in daily_agg.groupby('Account'):
    group = group.sort_values('parsed_date').copy()
    
    # Target: Next-day PnL positive
    group['target_next_day_profitable'] = (group['daily_pnl'].shift(-1) > 0).astype(int)
    
    # F&G Index value change over 3 days
    group['fg_value_change_3d'] = group['fg_value'] - group['fg_value'].shift(3)
    
    # Drop the last row since tomorrow doesn't exist
    group = group.dropna(subset=['target_next_day_profitable'])
    predictive_data.append(group)

df_pred = pd.concat(predictive_data).reset_index(drop=True)

# Select feature columns
feature_cols = [
    'fg_value', 'fg_value_change_3d', 'daily_pnl', 'total_fee', 
    'trade_count', 'win_rate', 'total_volume_usd', 'avg_trade_size_usd', 
    'avg_equity_leverage', 'long_short_ratio'
]

# Handle any missing values in features
df_pred = df_pred.dropna(subset=feature_cols + ['target_next_day_profitable'])

# Chronological split to prevent time leakage (80% train, 20% test)
df_pred = df_pred.sort_values('parsed_date')
split_idx = int(len(df_pred) * 0.8)

train_data = df_pred.iloc[:split_idx]
test_data = df_pred.iloc[split_idx:]

X_train = train_data[feature_cols]
y_train = train_data['target_next_day_profitable']
X_test = test_data[feature_cols]
y_test = test_data['target_next_day_profitable']

print(f"Training data size: {len(X_train)} rows")
print(f"Test data size: {len(X_test)} rows")

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print(f"Model Accuracy on Test Set: {accuracy:.4f}")
print(f"Model ROC AUC on Test Set: {roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Get feature importances
importances = model.feature_importances_
feature_imp_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': importances
}).sort_values('Importance', ascending=False)

feature_imp_df.to_csv(os.path.join(OUTPUT_DIR, "predictive_feature_importances.csv"), index=False)

# Save test metrics
test_metrics = {
    "accuracy": float(accuracy),
    "roc_auc": float(roc_auc),
    "feature_importance": feature_imp_df.to_dict(orient="records")
}
with open(os.path.join(OUTPUT_DIR, "model_evaluation_metrics.json"), "w") as f:
    json.dump(test_metrics, f, indent=4)

# ----------------------------------------------------
# Additional Insights & Tables
# ----------------------------------------------------
print("\nComputing additional insight tables...")

# Sentiment vs win rates
sentiment_wins = daily_agg.groupby('fg_class').agg(
    total_trades=('trade_count', 'sum'),
    win_rate=('win_trades', lambda x: x.sum() / daily_agg.loc[x.index, 'close_trades'].sum()),
    avg_leverage=('avg_equity_leverage', 'mean'),
    avg_trade_size=('avg_trade_size_usd', 'mean')
).reset_index()
sentiment_wins.to_csv(os.path.join(OUTPUT_DIR, "sentiment_regime_stats.csv"), index=False)

# High leverage vs Low leverage traders daily behavior during Fear index
# Split traders by their average equity leverage (median threshold of the 32 traders)
median_leverage = trader_features['avg_equity_leverage'].median()
high_lev_traders = trader_features[trader_features['avg_equity_leverage'] >= median_leverage]['Account'].tolist()

daily_agg['leverage_group'] = 'Low Leverage'
daily_agg.loc[daily_agg['Account'].isin(high_lev_traders), 'leverage_group'] = 'High Leverage'

leverage_sentiment_interaction = daily_agg.groupby(['leverage_group', 'regime']).agg(
    avg_daily_pnl=('daily_pnl', 'mean'),
    win_rate=('win_trades', lambda x: x.sum() / daily_agg.loc[x.index, 'close_trades'].sum()),
    avg_daily_trades=('trade_count', 'mean'),
    avg_long_short_ratio=('long_short_ratio', 'mean')
).reset_index()

leverage_sentiment_interaction.to_csv(os.path.join(OUTPUT_DIR, "leverage_sentiment_interaction.csv"), index=False)

print("\nLeverage & Sentiment Interaction Summary:")
print(leverage_sentiment_interaction)

# Save trader_features directly merged with clusters
trader_features_merged = pd.merge(trader_features, dd_merged, on='Account', how='left')
trader_features_merged.to_csv(os.path.join(OUTPUT_DIR, "trader_features_with_clusters.csv"), index=False)

# Save daily aggregation with leverage group for plotting
daily_agg.to_csv(os.path.join(OUTPUT_DIR, "daily_account_metrics_with_groups.csv"), index=False)

print("\nPipeline execution complete! All processed files saved in:", OUTPUT_DIR)
