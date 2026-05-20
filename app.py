import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title='Trader Performance vs Sentiment Dashboard',
    layout='wide',
    page_icon='📈'
)

@st.cache_data
def load_data():
    df1 = pd.read_csv('fear_greed_index.csv')
    df1['datetime'] = pd.to_datetime(df1['timestamp'], unit='s', errors='coerce')
    df1['date'] = df1['datetime'].dt.normalize()
    df1['sentiment_group'] = df1['classification'].astype(str).apply(
        lambda x: 'Fear' if 'Fear' in x else ('Greed' if 'Greed' in x else 'Unknown')
    )

    df2 = pd.read_csv('historical_data.csv')
    df2['Timestamp IST'] = pd.to_datetime(df2['Timestamp IST'], format='%d-%m-%Y %H:%M', errors='coerce')
    df2['date'] = df2['Timestamp IST'].dt.normalize()
    df2['Side'] = df2['Side'].astype(str).str.upper().str.strip()
    df2['Direction'] = df2['Direction'].astype(str).str.upper().str.strip()
    df2['is_long'] = df2['Side'].isin(['BUY', 'LONG']) | df2['Direction'].isin(['BUY', 'LONG'])
    df2['is_short'] = df2['Side'].isin(['SELL', 'SHORT']) | df2['Direction'].isin(['SELL', 'SHORT'])
    df2['is_win'] = df2['Closed PnL'] > 0
    df2 = df2.merge(df1[['date', 'sentiment_group', 'classification']], on='date', how='left')
    return df1, df2

@st.cache_data
def prepare_metrics(df1, df2):
    size_metric = 'Size USD' if 'Size USD' in df2.columns else ('Size Tokens' if 'Size Tokens' in df2.columns else None)
    daily_metrics = df2.groupby('date').agg(
        num_trades=('date', 'size'),
        daily_pnl=('Closed PnL', 'sum'),
        win_rate=('is_win', 'mean'),
        avg_trade_size=(size_metric, 'mean') if size_metric else ('Closed PnL', 'size'),
        long_trades=('is_long', 'sum'),
        short_trades=('is_short', 'sum')
    ).reset_index()
    daily_metrics['long_short_ratio'] = daily_metrics['long_trades'] / daily_metrics['short_trades'].replace(0, np.nan)
    daily_metrics = daily_metrics.merge(df1[['date', 'sentiment_group', 'classification']], on='date', how='left')

    account_stats = df2.groupby('Account').agg(
        trades=('date', 'size'),
        total_pnl=('Closed PnL', 'sum'),
        win_rate=('is_win', 'mean'),
        avg_trade_size=(size_metric, 'mean') if size_metric else ('Closed PnL', 'mean'),
        long_share=('is_long', 'mean'),
        short_share=('is_short', 'mean'),
        pnl_vol=('Closed PnL', 'std')
    ).reset_index()
    account_stats['freq_segment'] = pd.qcut(account_stats['trades'].rank(method='first'), q=3, labels=['Low frequency', 'Medium frequency', 'High frequency'])
    account_stats['performance_segment'] = pd.cut(account_stats['win_rate'], bins=[-0.01, 0.4, 0.6, 1.0], labels=['Inconsistent/loser', 'Mixed', 'Consistent winner'])

    return daily_metrics, account_stats

@st.cache_data
def fit_cluster_labels(account_stats, n_clusters=3):
    features = ['trades', 'win_rate', 'avg_trade_size', 'long_share']
    scaler = StandardScaler()
    X = scaler.fit_transform(account_stats[features].fillna(0))
    labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(X)
    account_stats = account_stats.copy()
    account_stats['behavior_cluster'] = labels
    cluster_summary = account_stats.groupby('behavior_cluster').agg(
        accounts=('Account', 'count'),
        avg_trades=('trades', 'mean'),
        avg_win_rate=('win_rate', 'mean'),
        avg_trade_size=('avg_trade_size', 'mean'),
        avg_long_share=('long_share', 'mean')
    ).reset_index()
    return account_stats, cluster_summary

@st.cache_data
def train_predictive_model(daily_metrics):
    data = daily_metrics.sort_values('date').copy()
    data['next_day_pnl'] = data['daily_pnl'].shift(-1)
    data = data.dropna(subset=['next_day_pnl']).copy()
    data['next_day_profit'] = (data['next_day_pnl'] > 0).astype(int)

    features = ['num_trades', 'win_rate', 'long_short_ratio', 'avg_trade_size']
    X = data[features].fillna(0)
    y = data['next_day_profit']
    if len(data) < 10 or len(y.unique()) < 2:
        return None, None, data

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42, stratify=y
    )
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    coefficients = pd.DataFrame({
        'feature': features,
        'coefficient': model.coef_[0]
    })
    return accuracy, coefficients, data

@st.cache_data
def build_sentiment_summary(daily_metrics):
    summary = daily_metrics.groupby('sentiment_group').agg(
        avg_daily_pnl=('daily_pnl', 'mean'),
        median_daily_pnl=('daily_pnl', 'median'),
        avg_win_rate=('win_rate', 'mean'),
        avg_trades=('num_trades', 'mean'),
        avg_long_short_ratio=('long_short_ratio', 'mean'),
        worst_daily_loss=('daily_pnl', 'min')
    ).reset_index()
    return summary

@st.cache_data
def build_behavior_summary(df2):
    behavior = df2.groupby('sentiment_group').agg(
        total_trades=('date', 'size'),
        avg_trade_size=('Size USD', 'mean'),
        long_trades=('is_long', 'sum'),
        short_trades=('is_short', 'sum'),
        unique_accounts=('Account', 'nunique')
    ).reset_index()
    behavior['long_short_ratio'] = behavior['long_trades'] / behavior['short_trades'].replace(0, np.nan)
    return behavior

st.title('Trader Performance vs Market Sentiment')

df1, df2 = load_data()
daily_metrics, account_stats = prepare_metrics(df1, df2)
account_stats, cluster_summary = fit_cluster_labels(account_stats)
sentiment_summary = build_sentiment_summary(daily_metrics)
behavior_summary = build_behavior_summary(df2)
model_accuracy, model_coeffs, prediction_data = train_predictive_model(daily_metrics)

st.sidebar.header('Dashboard navigation')
page = st.sidebar.selectbox('Choose a page', ['Overview', 'Sentiment analysis', 'Trader segments', 'Model & predictions'])

if page == 'Overview':
    st.subheader('Overview')
    total_trades = len(df2)
    total_accounts = account_stats['Account'].nunique()
    avg_win_rate = df2['is_win'].mean()
    avg_daily_pnl = daily_metrics['daily_pnl'].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Total trades', f'{total_trades:,}')
    col2.metric('Unique accounts', f'{total_accounts:,}')
    col3.metric('Overall win rate', f'{avg_win_rate:.2%}')
    col4.metric('Avg daily PnL', f'${avg_daily_pnl:,.0f}')

    st.markdown('### Daily sentiment performance')
    fig = px.bar(
        sentiment_summary,
        x='sentiment_group',
        y='avg_daily_pnl',
        text='avg_daily_pnl',
        labels={'sentiment_group': 'Sentiment', 'avg_daily_pnl': 'Avg daily PnL'},
        title='Average daily PnL by sentiment'
    )
    fig.update_traces(texttemplate='$%{text:.0f}', textposition='outside')
    fig.update_layout(yaxis_tickprefix='$', uniformtext_minsize=8, uniformtext_mode='hide')
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.box(
        daily_metrics,
        x='sentiment_group',
        y='daily_pnl',
        labels={'sentiment_group': 'Sentiment', 'daily_pnl': 'Daily PnL'},
        title='Daily PnL distribution by sentiment'
    )
    st.plotly_chart(fig2, use_container_width=True)

    with st.expander('Show daily metrics table'):
        st.dataframe(daily_metrics.sort_values('date', ascending=False).head(20))

elif page == 'Sentiment analysis':
    st.subheader('Sentiment-based performance and behavior')
    st.write('Compare Fear vs Greed day performance and trader behavior.')

    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(
            px.bar(sentiment_summary, x='sentiment_group', y='avg_trades', text='avg_trades',
                   labels={'sentiment_group': 'Sentiment', 'avg_trades': 'Avg trades per day'},
                   title='Average trades per day by sentiment'), use_container_width=True
        )
    with col2:
        st.plotly_chart(
            px.bar(sentiment_summary, x='sentiment_group', y='avg_win_rate', text='avg_win_rate',
                   labels={'sentiment_group': 'Sentiment', 'avg_win_rate': 'Avg win rate'},
                   title='Win rate by sentiment').update_layout(yaxis_tickformat='.0%'),
            use_container_width=True
        )

    st.write('Behavior summary by sentiment:')
    st.dataframe(behavior_summary)

elif page == 'Trader segments':
    st.subheader('Trader behavioral archetypes')
    st.write('Clusters are built from each trader’s trade count, win rate, trade size and long-share.')
    st.dataframe(cluster_summary)

    cluster_select = st.selectbox('Select cluster to inspect', sorted(account_stats['behavior_cluster'].unique()))
    st.write(f'### Accounts in cluster {cluster_select}')
    st.dataframe(account_stats[account_stats['behavior_cluster'] == cluster_select].sort_values('trades', ascending=False).head(20))

    fig3 = px.scatter(
        account_stats,
        x='trades',
        y='win_rate',
        color='behavior_cluster',
        size='avg_trade_size',
        hover_data=['Account', 'total_pnl', 'long_share'],
        labels={'trades': 'Total trades', 'win_rate': 'Win rate'},
        title='Trader clusters: trade frequency, win rate, size'
    )
    st.plotly_chart(fig3, use_container_width=True)

else:
    st.subheader('Predictive model for next-day profitability')
    if model_accuracy is None:
        st.warning('Not enough daily data to train a predictive model.')
    else:
        st.markdown('This model predicts whether the next day will be profitable based on current daily behavior and sentiment features.')
        st.metric('Model accuracy', f'{model_accuracy:.2%}')
        st.dataframe(model_coeffs)
        st.write('Feature coefficients from the logistic regression model:')

        st.write('Training data sample:')
        st.dataframe(prediction_data[['date', 'num_trades', 'win_rate', 'long_short_ratio', 'avg_trade_size', 'next_day_pnl', 'next_day_profit']].head(10))

        fig4 = px.bar(
            sentiment_summary,
            x='sentiment_group',
            y='avg_daily_pnl',
            color='sentiment_group',
            title='Sentiment average daily PnL'
        )
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown('### Strategy rules of thumb')
    st.write('- During Fear days, favor traders with stronger win rates and balanced long/short exposure. Fear days show higher average PnL but also higher trade activity.')
    st.write('- During Greed days, reduce exposure to high-frequency and inconsistent traders. Greed days have more downside tail risk in this dataset.')
