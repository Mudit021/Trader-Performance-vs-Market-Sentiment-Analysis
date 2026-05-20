import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

# Configure directories
DATA_DIR = "m:\\Trader Performance vs Market Sentiment Analysis"
CHARTS_DIR = os.path.join(DATA_DIR, "output_charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

print("Starting PNG Charts Generation Script...")

# Load datasets
df_fg = pd.read_csv(os.path.join(DATA_DIR, "fear_greed_index.csv"))
df_hd = pd.read_csv(os.path.join(DATA_DIR, "historical_data.csv"))

# Parse timestamps
df_fg['parsed_date'] = pd.to_datetime(df_fg['date']).dt.date
df_hd['ParsedTime'] = pd.to_datetime(df_hd['Timestamp IST'], format='%d-%m-%Y %H:%M')
df_hd['parsed_date'] = df_hd['ParsedTime'].dt.date

df_hd = df_hd.sort_values(by=['Account', 'ParsedTime', 'Trade ID']).reset_index(drop=True)
df_merged = pd.merge(df_hd, df_fg[['parsed_date', 'value', 'classification']], on='parsed_date', how='inner')

# Calculate running metrics
INITIAL_EQUITY = 100000.0
df_merged['Running_Equity'] = INITIAL_EQUITY
df_merged['Equity_Leverage'] = 0.0
df_merged['Relative_Leverage'] = 0.0

updated_dfs = []
for account, group in df_merged.groupby('Account'):
    group = group.sort_values('ParsedTime').copy()
    cum_pnl = group['Closed PnL'].cumsum()
    cum_fees = group['Fee'].cumsum()
    group['Running_Equity'] = INITIAL_EQUITY + cum_pnl - cum_fees
    group['Running_Equity'] = group['Running_Equity'].clip(lower=100.0)
    group['Equity_Leverage'] = group['Size USD'] / group['Running_Equity']
    
    mean_size = group['Size USD'].mean()
    group['Relative_Leverage'] = group['Size USD'] / (mean_size if mean_size > 0 else 1.0)
    updated_dfs.append(group)

df_merged = pd.concat(updated_dfs).sort_values(by=['Account', 'ParsedTime']).reset_index(drop=True)

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
    avg_running_equity=('Running_Equity', 'mean'),
    fg_value=('value', 'first'),
    fg_class=('classification', 'first')
).reset_index()

daily_agg['win_rate'] = daily_agg['win_trades'] / daily_agg['close_trades'].replace(0, np.nan)
daily_agg['win_rate'] = daily_agg['win_rate'].fillna(0.5)
daily_agg['long_short_ratio'] = daily_agg['buy_trades'] / (daily_agg['trade_count'] - daily_agg['buy_trades']).replace(0, np.nan)
daily_agg['long_short_ratio'] = daily_agg['long_short_ratio'].fillna(1.0)

# Set regimes
daily_agg['regime'] = 'Neutral'
daily_agg.loc[daily_agg['fg_value'] <= 40, 'regime'] = 'Fear'
daily_agg.loc[daily_agg['fg_value'] >= 60, 'regime'] = 'Greed'

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

# Set styles for matplotlib/seaborn
sns.set_theme(style="darkgrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["figure.dpi"] = 100

# 1. PnL and Leverage bar plots
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sns.barplot(data=regime_perf, x='regime', y='total_pnl', palette='coolwarm', ax=axes[0], order=['Fear', 'Neutral', 'Greed'])
axes[0].set_title("Total Realized PnL by Sentiment Regime")
axes[0].set_ylabel("Total PnL ($)")

sns.barplot(data=regime_perf, x='regime', y='avg_leverage', palette='coolwarm', ax=axes[1], order=['Fear', 'Neutral', 'Greed'])
axes[1].set_title("Average Equity Leverage by Sentiment Regime")
axes[1].set_ylabel("Leverage (x)")
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "pnl_leverage_by_regime.png"))
plt.close()
print("Saved: pnl_leverage_by_regime.png")

# 2. Behavioral Shifts Scatter
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.scatterplot(data=daily_agg, x='fg_value', y='long_short_ratio', hue='regime', palette='coolwarm', alpha=0.6, ax=axes[0])
axes[0].set_title("Long/Short Ratio vs. Fear & Greed Index")
axes[0].set_xlabel("Fear & Greed Index")
axes[0].set_ylabel("Long/Short Ratio (BUY / SELL)")

sns.scatterplot(data=daily_agg, x='fg_value', y='trade_count', hue='regime', palette='coolwarm', alpha=0.6, ax=axes[1])
axes[1].set_title("Daily Trade Frequency vs. Fear & Greed Index")
axes[1].set_xlabel("Fear & Greed Index")
axes[1].set_ylabel("Daily Trade Count")
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "behavioral_shifts_scatter.png"))
plt.close()
print("Saved: behavioral_shifts_scatter.png")

# 3. Trader Archetypes
trader_features = daily_agg.groupby('Account').agg(
    total_pnl=('daily_pnl', 'sum'),
    avg_daily_pnl=('daily_pnl', 'mean'),
    avg_win_rate=('win_rate', 'mean'),
    avg_trade_count=('trade_count', 'mean'),
    avg_trade_size=('avg_trade_size_usd', 'mean'),
    avg_equity_leverage=('avg_equity_leverage', 'mean'),
    avg_long_short_ratio=('long_short_ratio', 'mean')
).reset_index()

clustering_cols = ['avg_win_rate', 'avg_trade_count', 'avg_trade_size', 'avg_equity_leverage', 'avg_long_short_ratio']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(trader_features[clustering_cols])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
trader_features['cluster'] = kmeans.fit_predict(X_scaled)

sorted_clusters = sorted(range(3), key=lambda k: trader_features[trader_features['cluster'] == k]['avg_equity_leverage'].mean())
archetypes = {
    sorted_clusters[0]: "Low-Activity Moderate Traders",
    sorted_clusters[1]: "Consistent Profit-Scalpers",
    sorted_clusters[2]: "High-Leverage Speedrunners"
}
trader_features['archetype'] = trader_features['cluster'].map(archetypes)

plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=trader_features, 
    x='avg_win_rate', 
    y='avg_equity_leverage', 
    hue='archetype', 
    palette='Set2', 
    s=150, 
    alpha=0.9
)
plt.title("Trader Archetypes (Win Rate vs. Leverage)")
plt.xlabel("Average Win Rate")
plt.ylabel("Average Equity Leverage (x)")
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "trader_archetypes.png"))
plt.close()
print("Saved: trader_archetypes.png")

# 4. Feature Importances
predictive_data = []
for acct, group in daily_agg.groupby('Account'):
    group = group.sort_values('parsed_date').copy()
    group['target_next_day_profitable'] = (group['daily_pnl'].shift(-1) > 0).astype(int)
    group['fg_value_change_3d'] = group['fg_value'] - group['fg_value'].shift(3)
    group = group.dropna(subset=['target_next_day_profitable'])
    predictive_data.append(group)

df_pred = pd.concat(predictive_data).reset_index(drop=True)
feature_cols = [
    'fg_value', 'fg_value_change_3d', 'daily_pnl', 'total_fee', 
    'trade_count', 'win_rate', 'total_volume_usd', 'avg_trade_size_usd', 
    'avg_equity_leverage', 'long_short_ratio'
]
df_pred = df_pred.dropna(subset=feature_cols + ['target_next_day_profitable']).sort_values('parsed_date')
split_idx = int(len(df_pred) * 0.8)

X_train, y_train = df_pred.iloc[:split_idx][feature_cols], df_pred.iloc[:split_idx]['target_next_day_profitable']

model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
model.fit(X_train, y_train)

imp_df = pd.DataFrame({'Feature': feature_cols, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(data=imp_df, x='Importance', y='Feature', palette='viridis')
plt.title("Random Forest Feature Importances")
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "model_feature_importances.png"))
plt.close()
print("Saved: model_feature_importances.png")

print("All charts successfully generated in:", CHARTS_DIR)
