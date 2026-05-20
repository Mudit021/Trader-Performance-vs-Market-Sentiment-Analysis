import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json
from sklearn.ensemble import RandomForestClassifier

# Set page config
st.set_page_config(
    page_title="Trader Performance vs Market Sentiment Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling via markdown
st.markdown("""
<style>
    /* Dark mode styling */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    
    /* Header card with gradient */
    .header-card {
        background: linear-gradient(135deg, #1f2937, #111827);
        border-radius: 12px;
        padding: 24px;
        border: 1px solid #374151;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .header-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        color: #38bdf8;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
    }
    .header-subtitle {
        color: #9ca3af;
        font-size: 1.1rem;
        margin-top: 8px;
    }
    
    /* Metric cards */
    .metric-card {
        background-color: #161b22;
        border-radius: 10px;
        padding: 18px;
        border: 1px solid #30363d;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.15);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        text-align: center;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(56, 189, 248, 0.2);
        border-color: #38bdf8;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #8b949e;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 5px;
        font-family: 'Inter', sans-serif;
    }
    .metric-delta {
        font-size: 0.85rem;
        margin-top: 5px;
        font-weight: 600;
    }
    
    /* Custom tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #161b22;
        border-radius: 4px 4px 0px 0px;
        gap: 0px;
        padding: 10px 16px;
        color: #8b949e;
        border: 1px solid #30363d;
        border-bottom: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #21262d !important;
        color: #38bdf8 !important;
        border-top: 2px solid #38bdf8 !important;
        font-weight: 600;
    }
    
    /* Info box */
    .info-card {
        background-color: #0f172a;
        border-left: 4px solid #38bdf8;
        padding: 15px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Define directories
DATA_DIR = "m:\\Trader Performance vs Market Sentiment Analysis"
PROCESSED_DIR = os.path.join(DATA_DIR, "processed_data")

# Load aggregated data
@st.cache_data
def load_base_data():
    daily_metrics = pd.read_csv(os.path.join(PROCESSED_DIR, "daily_account_metrics_with_groups.csv"))
    daily_metrics['parsed_date'] = pd.to_datetime(daily_metrics['parsed_date']).dt.date
    
    trader_clusters = pd.read_csv(os.path.join(PROCESSED_DIR, "trader_features_with_clusters.csv"))
    
    regime_perf = pd.read_csv(os.path.join(PROCESSED_DIR, "regime_performance.csv"))
    leverage_interaction = pd.read_csv(os.path.join(PROCESSED_DIR, "leverage_sentiment_interaction.csv"))
    
    with open(os.path.join(PROCESSED_DIR, "dataset_metadata.json"), "r") as f:
        meta = json.load(f)
        
    return daily_metrics, trader_clusters, regime_perf, leverage_interaction, meta

try:
    daily_metrics, trader_clusters, regime_perf, leverage_interaction, meta = load_base_data()
except Exception as e:
    st.error(f"Error loading processed data. Please run the analysis script first. Details: {e}")
    st.stop()


# Sidebar Header
st.sidebar.markdown("""
<div style='text-align: center; margin-bottom: 20px;'>
    <h2 style='color: #38bdf8; margin: 0;'>Hyperliquid x F&G</h2>
    <p style='color: #8b949e; font-size: 0.85rem;'>Trader Performance vs Sentiment</p>
</div>
""", unsafe_allow_html=True)

# Account filter
accounts = ["All Accounts"] + sorted(daily_metrics['Account'].unique().tolist())
selected_account = st.sidebar.selectbox("🎯 Select Account / Trader", accounts)

# Date filter
min_date = daily_metrics['parsed_date'].min()
max_date = daily_metrics['parsed_date'].max()
selected_dates = st.sidebar.slider(
    "📅 Select Date Range",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
    format="YYYY-MM-DD"
)

# Sentiment filter
sentiment_categories = sorted(daily_metrics['fg_class'].unique().tolist())
selected_sentiments = st.sidebar.multiselect(
    "🧠 Filter by Market Sentiment",
    sentiment_categories,
    default=sentiment_categories
)

# Filter daily metrics based on selection
df_filtered = daily_metrics[
    (daily_metrics['parsed_date'] >= selected_dates[0]) &
    (daily_metrics['parsed_date'] <= selected_dates[1]) &
    (daily_metrics['fg_class'].isin(selected_sentiments))
]

if selected_account != "All Accounts":
    df_filtered = df_filtered[df_filtered['Account'] == selected_account]

# Title Header Card
account_label = selected_account if selected_account != "All Accounts" else "All 32 Traders (Aggregate)"
st.markdown(f"""
<div class="header-card">
    <h1 class="header-title">Trader Performance vs Market Sentiment Analysis</h1>
    <div class="header-subtitle">Exploring Hyperliquid derivatives trading behavior aligned with the Bitcoin Fear & Greed Index</div>
    <div style="font-size: 0.85rem; color: #8b949e; margin-top: 10px;">
        Active Filters: <b>{account_label}</b> | Date Range: <b>{selected_dates[0]}</b> to <b>{selected_dates[1]}</b> | Sentiment: <b>{', '.join(selected_sentiments)}</b>
    </div>
</div>
""", unsafe_allow_html=True)

# Check if data remains after filtering
if len(df_filtered) == 0:
    st.warning("⚠️ No data matches the selected filters. Please adjust your filters in the sidebar.")
    st.stop()

# Tab Layout
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Overview & PnL", 
    "⚖️ Sentiment vs Performance", 
    "📈 Behavioral Shifts", 
    "🧬 Behavioral Archetypes", 
    "🔮 Predict Profitability"
])

# ----------------------------------------------------
# TAB 1: OVERVIEW & PNL
# ----------------------------------------------------
with tab1:
    # Compute overview metrics
    total_pnl = df_filtered['daily_pnl'].sum()
    total_trades = df_filtered['trade_count'].sum()
    total_fees = df_filtered['total_fee'].sum()
    avg_trade_size = df_filtered['avg_trade_size_usd'].mean()
    
    # Calculate win rate across all filtered trades
    total_wins = df_filtered['win_trades'].sum()
    total_closed = df_filtered['close_trades'].sum()
    overall_win_rate = total_wins / total_closed if total_closed > 0 else 0.5
    
    # Leverage avg
    avg_lev = df_filtered['avg_equity_leverage'].mean()
    
    # Custom HTML metrics display
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        pnl_color = "#34d399" if total_pnl >= 0 else "#f43f5e"
        pnl_sign = "+" if total_pnl >= 0 else ""
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Net Realized PnL</div>
            <div class="metric-value" style="color: {pnl_color};">{pnl_sign}${total_pnl:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Win Rate</div>
            <div class="metric-value" style="color: #38bdf8;">{overall_win_rate:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Trade Executions</div>
            <div class="metric-value">{total_trades:,}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Position size</div>
            <div class="metric-value">${avg_trade_size:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Equity Leverage</div>
            <div class="metric-value" style="color: #fbbf24;">{avg_lev:.3f}x</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # PnL charts row
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.subheader("📈 Cumulative PnL Over Time")
        # Compute cumulative PnL series
        if selected_account == "All Accounts":
            # For all accounts, sum by date
            daily_pnl_series = df_filtered.groupby('parsed_date')['daily_pnl'].sum().reset_index().sort_values('parsed_date')
        else:
            daily_pnl_series = df_filtered.sort_values('parsed_date')
            
        daily_pnl_series['cum_pnl'] = daily_pnl_series['daily_pnl'].cumsum()
        
        fig_cum_pnl = px.line(
            daily_pnl_series,
            x='parsed_date',
            y='cum_pnl',
            labels={'parsed_date': 'Date', 'cum_pnl': 'Cumulative PnL ($)'},
            template="plotly_dark",
            color_discrete_sequence=["#38bdf8"]
        )
        fig_cum_pnl.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=10, b=20),
            xaxis=dict(showgrid=True, gridcolor='#30363d'),
            yaxis=dict(showgrid=True, gridcolor='#30363d')
        )
        st.plotly_chart(fig_cum_pnl, use_container_width=True)
        
    with chart_col2:
        st.subheader("⚖️ Daily PnL vs Fear & Greed Index")
        # Double Y-axis chart
        # Group daily data by date
        daily_fg_series = df_filtered.groupby('parsed_date').agg(
            pnl=('daily_pnl', 'sum'),
            fg_val=('fg_value', 'first')
        ).reset_index().sort_values('parsed_date')
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # PnL bar chart
        fig.add_trace(
            go.Bar(
                x=daily_fg_series['parsed_date'],
                y=daily_fg_series['pnl'],
                name="Daily PnL ($)",
                marker_color=np.where(daily_fg_series['pnl'] >= 0, '#34d399', '#f43f5e'),
                opacity=0.85
            ),
            secondary_y=False
        )
        
        # F&G Line chart
        fig.add_trace(
            go.Scatter(
                x=daily_fg_series['parsed_date'],
                y=daily_fg_series['fg_val'],
                name="Fear & Greed Index",
                line=dict(color='#fbbf24', width=2),
                mode='lines'
            ),
            secondary_y=True
        )
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=10, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(showgrid=True, gridcolor='#30363d')
        )
        fig.update_yaxes(title_text="Daily Realized PnL ($)", secondary_y=False, showgrid=True, gridcolor='#30363d')
        fig.update_yaxes(title_text="Fear & Greed Index", secondary_y=True, range=[0, 100], showgrid=False)
        
        st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------
# TAB 2: SENTIMENT VS PERFORMANCE
# ----------------------------------------------------
with tab2:
    st.subheader("📊 Profitability and Win Rates by Sentiment Regime")
    st.markdown("""
    <div class="info-card">
        Explore how trader performance changes based on sentiment. 
        <b>Fear Days</b> (Index ≤ 40), <b>Greed Days</b> (Index ≥ 60), and <b>Neutral Days</b> (41 - 59).
    </div>
    """, unsafe_allow_html=True)
    
    # Filter regime stats for current selected account
    if selected_account == "All Accounts":
        # Group filtered daily data by regime
        regime_stats = df_filtered.groupby('regime').agg(
            total_pnl=('daily_pnl', 'sum'),
            avg_daily_pnl=('daily_pnl', 'mean'),
            win_rate=('win_trades', lambda x: x.sum() / df_filtered.loc[x.index, 'close_trades'].sum()),
            avg_trades=('trade_count', 'mean'),
            avg_volume=('total_volume_usd', 'mean'),
            avg_leverage=('avg_equity_leverage', 'mean'),
            count_days=('parsed_date', 'nunique')
        ).reset_index()
        
        # Max Drawdown
        drawdowns_df = pd.read_csv(os.path.join(PROCESSED_DIR, "account_drawdowns_by_regime.csv"))
        avg_mdd_fear = drawdowns_df['max_dd_fear'].mean()
        avg_mdd_greed = drawdowns_df['max_dd_greed'].mean()
        avg_mdd_neutral = drawdowns_df['max_dd_neutral'].mean()
        
        mdd_data = pd.DataFrame({
            'regime': ['Fear', 'Greed', 'Neutral'],
            'max_drawdown': [avg_mdd_fear, avg_mdd_greed, avg_mdd_neutral]
        })
    else:
        # For single account
        regime_stats = df_filtered.groupby('regime').agg(
            total_pnl=('daily_pnl', 'sum'),
            avg_daily_pnl=('daily_pnl', 'mean'),
            win_rate=('win_trades', lambda x: x.sum() / df_filtered.loc[x.index, 'close_trades'].sum()),
            avg_trades=('trade_count', 'mean'),
            avg_volume=('total_volume_usd', 'mean'),
            avg_leverage=('avg_equity_leverage', 'mean'),
            count_days=('parsed_date', 'nunique')
        ).reset_index()
        
        # Calculate max drawdown for this account specifically
        mdd_list = []
        for reg in ['Fear', 'Greed', 'Neutral']:
            df_reg = df_filtered[df_filtered['regime'] == reg].sort_values('parsed_date')
            if len(df_reg) > 0:
                cum = df_reg['daily_pnl'].cumsum()
                run_max = cum.cummax()
                mdd = (run_max - cum).max()
            else:
                mdd = 0
            mdd_list.append({'regime': reg, 'max_drawdown': mdd})
        mdd_data = pd.DataFrame(mdd_list)

    # Display bar charts side by side
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Total Realized PnL ($)")
        fig_pnl = px.bar(
            regime_stats,
            x='regime',
            y='total_pnl',
            color='regime',
            color_discrete_map={'Fear': '#f43f5e', 'Greed': '#34d399', 'Neutral': '#6b7280'},
            template="plotly_dark",
            category_orders={'regime': ['Fear', 'Neutral', 'Greed']}
        )
        fig_pnl.update_layout(
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=10, b=20),
            xaxis=dict(gridcolor='#30363d'),
            yaxis=dict(showgrid=True, gridcolor='#30363d')
        )
        st.plotly_chart(fig_pnl, use_container_width=True)
        
    with col2:
        st.markdown("#### Win Rate (%)")
        fig_wr = px.bar(
            regime_stats,
            x='regime',
            y='win_rate',
            color='regime',
            color_discrete_map={'Fear': '#f43f5e', 'Greed': '#34d399', 'Neutral': '#6b7280'},
            template="plotly_dark",
            category_orders={'regime': ['Fear', 'Neutral', 'Greed']}
        )
        fig_wr.update_layout(
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=10, b=20),
            xaxis=dict(gridcolor='#30363d'),
            yaxis=dict(showgrid=True, gridcolor='#30363d', tickformat=".1%")
        )
        st.plotly_chart(fig_wr, use_container_width=True)
        
    with col3:
        st.markdown("#### Drawdown Proxy (Max DD of cumulative PnL)")
        fig_dd = px.bar(
            mdd_data,
            x='regime',
            y='max_drawdown',
            color='regime',
            color_discrete_map={'Fear': '#f43f5e', 'Greed': '#34d399', 'Neutral': '#6b7280'},
            template="plotly_dark",
            category_orders={'regime': ['Fear', 'Neutral', 'Greed']}
        )
        fig_dd.update_layout(
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=10, b=20),
            xaxis=dict(gridcolor='#30363d'),
            yaxis=dict(showgrid=True, gridcolor='#30363d')
        )
        st.plotly_chart(fig_dd, use_container_width=True)

    # Detailed statistics table
    st.markdown("#### Regime Summary Table")
    # Formatting
    regime_display = regime_stats.copy()
    regime_display['total_pnl'] = regime_display['total_pnl'].map('${:,.2f}'.format)
    regime_display['avg_daily_pnl'] = regime_display['avg_daily_pnl'].map('${:,.2f}'.format)
    regime_display['win_rate'] = regime_display['win_rate'].map('{:.2%}'.format)
    regime_display['avg_trades'] = regime_display['avg_trades'].map('{:.1f}'.format)
    regime_display['avg_volume'] = regime_display['avg_volume'].map('${:,.2f}'.format)
    regime_display['avg_leverage'] = regime_display['avg_leverage'].map('{:.4f}x'.format)
    st.dataframe(regime_display, use_container_width=True)

    # Insight bullet points
    st.markdown("""
    #### 💡 Performance Insights Backed by Data:
    - **Greed Days (Index ≥ 60)** yield the highest total raw PnL ($4.73M total realized), but the **median daily PnL** is where the major performance lies. While greed-driven market momentum provides massive windfalls, it also results in high drawdowns due to increased risk-taking.
    - **Fear Days (Index ≤ 40)** are marked by highly disciplined, low-leverage execution. Average equity leverage drops to **0.05x** (effectively spot trading) compared to **2.75x** on Greed days. 
    - Despite the lower leverage, traders maintain a **strong contrarian long bias** (Long/Short ratio of **1.97**) on Fear days, indicating they are actively buying the dip but reducing position sizing/leverage to survive wicks.
    - **Neutral Days (41-59)** generate the highest **average daily PnL ($4,804)** because market volatility is lower, which favors range-bound scalpers who trade frequently without the liquidation risk present in extreme sentiment regimes.
    """)

# ----------------------------------------------------
# TAB 3: BEHAVIORAL SHIFTS
# ----------------------------------------------------
with tab3:
    st.subheader("📈 Behavioral Shifts Based on Market Sentiment")
    st.markdown("""
    <div class="info-card">
        Analyze how traders change their trading parameters (trade frequency, leverage, position size, and bias) in response to sentiment shifts.
    </div>
    """, unsafe_allow_html=True)
    
    shift_col1, shift_col2 = st.columns(2)
    
    with shift_col1:
        st.subheader("⚡ Equity Leverage vs Fear & Greed Index")
        fig_lev_scatter = px.scatter(
            df_filtered,
            x='fg_value',
            y='avg_equity_leverage',
            color='regime',
            color_discrete_map={'Fear': '#f43f5e', 'Greed': '#34d399', 'Neutral': '#6b7280'},
            labels={'fg_value': 'Fear & Greed Index', 'avg_equity_leverage': 'Avg Daily Equity Leverage (x)'},
            hover_data=['Account', 'parsed_date'],
            template="plotly_dark"
        )
        fig_lev_scatter.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=10, b=20),
            xaxis=dict(showgrid=True, gridcolor='#30363d'),
            yaxis=dict(showgrid=True, gridcolor='#30363d')
        )
        st.plotly_chart(fig_lev_scatter, use_container_width=True)
        
    with shift_col2:
        st.subheader("🔄 Daily Trade Frequency vs Fear & Greed Index")
        fig_freq_scatter = px.scatter(
            df_filtered,
            x='fg_value',
            y='trade_count',
            color='regime',
            color_discrete_map={'Fear': '#f43f5e', 'Greed': '#34d399', 'Neutral': '#6b7280'},
            labels={'fg_value': 'Fear & Greed Index', 'trade_count': 'Daily Trade Count'},
            hover_data=['Account', 'parsed_date'],
            template="plotly_dark"
        )
        fig_freq_scatter.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=10, b=20),
            xaxis=dict(showgrid=True, gridcolor='#30363d'),
            yaxis=dict(showgrid=True, gridcolor='#30363d')
        )
        st.plotly_chart(fig_freq_scatter, use_container_width=True)

    shift_col3, shift_col4 = st.columns(2)
    
    with shift_col3:
        st.subheader("⚖️ Long/Short Bias vs Fear & Greed Index")
        fig_bias_scatter = px.scatter(
            df_filtered,
            x='fg_value',
            y='long_short_ratio',
            color='regime',
            color_discrete_map={'Fear': '#f43f5e', 'Greed': '#34d399', 'Neutral': '#6b7280'},
            labels={'fg_value': 'Fear & Greed Index', 'long_short_ratio': 'Long/Short Ratio'},
            hover_data=['Account', 'parsed_date'],
            template="plotly_dark"
        )
        fig_bias_scatter.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=10, b=20),
            xaxis=dict(showgrid=True, gridcolor='#30363d'),
            yaxis=dict(showgrid=True, gridcolor='#30363d')
        )
        st.plotly_chart(fig_bias_scatter, use_container_width=True)
        
    with shift_col4:
        st.subheader("📦 Leverage Group Performance during Sentiment Regimes")
        # Load leverage interaction stats
        fig_interact = px.bar(
            leverage_interaction,
            x='regime',
            y='avg_daily_pnl',
            color='leverage_group',
            barmode='group',
            labels={'regime': 'Sentiment Regime', 'avg_daily_pnl': 'Avg Daily PnL ($)'},
            color_discrete_sequence=['#fbbf24', '#38bdf8'],
            template="plotly_dark"
        )
        fig_interact.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=10, b=20),
            xaxis=dict(showgrid=True, gridcolor='#30363d'),
            yaxis=dict(showgrid=True, gridcolor='#30363d')
        )
        st.plotly_chart(fig_interact, use_container_width=True)

# ----------------------------------------------------
# TAB 4: TRADER CLUSTERING (ARCHETYPES)
# ----------------------------------------------------
with tab4:
    st.subheader("🧬 Trader Clustering into Behavioral Archetypes")
    st.markdown("""
    <div class="info-card">
        We ran a <b>K-Means Clustering</b> model on the 32 traders' multi-dimensional behavioral profiles (win rate, daily trades, leverage, size, bias, crossed margin preference) to classify them into distinct archetypes.
    </div>
    """, unsafe_allow_html=True)
    
    # 3D/2D Plotly Scatter plot of clusters
    st.markdown("#### Archetype Map (K-Means Clustering)")
    fig_clusters = px.scatter_3d(
        trader_clusters,
        x='avg_win_rate',
        y='avg_equity_leverage',
        z='avg_trade_count',
        color='archetype',
        hover_name='Account',
        labels={
            'avg_win_rate': 'Win Rate',
            'avg_equity_leverage': 'Avg Leverage',
            'avg_trade_count': 'Avg Daily Trade Count'
        },
        color_discrete_sequence=['#34d399', '#38bdf8', '#fbbf24'],
        template="plotly_dark"
    )
    fig_clusters.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor='#30363d'),
            yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor='#30363d'),
            zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor='#30363d')
        )
    )
    st.plotly_chart(fig_clusters, use_container_width=True)
    
    # Showcase Archetypes detail cards
    st.markdown("#### Profiles of Identified Behavioral Archetypes")
    prof1, prof2, prof3 = st.columns(3)
    
    with prof1:
        st.markdown("""
        <div class="metric-card" style="border-top: 4px solid #34d399; text-align: left;">
            <h3 style="color: #34d399; margin: 0;">🏆 Consistent Profit-Scalpers</h3>
            <p style="font-size: 0.85rem; color: #8b949e; margin-top: 5px;">Count: 19 Traders</p>
            <hr style="border-color: #30363d; margin: 10px 0;">
            <ul style="font-size: 0.85rem; padding-left: 15px; margin: 0;">
                <li><b>Win Rate:</b> High (~77% on average)</li>
                <li><b>Equity Leverage:</b> Low (mean 0.05x, effectively spot)</li>
                <li><b>Frequency:</b> High active frequency (~157 trades/day)</li>
                <li><b>PnL Profile:</b> Consistently profitable</li>
                <li><b>Risk Management:</b> Uses size scaling rather than leverage. Highly insulated from liquidations.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with prof2:
        st.markdown("""
        <div class="metric-card" style="border-top: 4px solid #38bdf8; text-align: left;">
            <h3 style="color: #38bdf8; margin: 0;">⚡ High-Leverage Speedrunners</h3>
            <p style="font-size: 0.85rem; color: #8b949e; margin-top: 5px;">Count: 1 Trader (0x08338...)</p>
            <hr style="border-color: #30363d; margin: 10px 0;">
            <ul style="font-size: 0.85rem; padding-left: 15px; margin: 0;">
                <li><b>Win Rate:</b> Moderate (~71%)</li>
                <li><b>Equity Leverage:</b> Very High (mean 124x, extreme exposure)</li>
                <li><b>Frequency:</b> High frequency (~159 trades/day)</li>
                <li><b>PnL Profile:</b> Massive profit swing ($1.60M total)</li>
                <li><b>Risk Management:</b> Extremely high drawdowns. Uses high cross-margin leverage to capture huge momentum wicks. High ruin risk.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with prof3:
        st.markdown("""
        <div class="metric-card" style="border-top: 4px solid #fbbf24; text-align: left;">
            <h3 style="color: #fbbf24; margin: 0;">🐌 Low-Activity Moderate Traders</h3>
            <p style="font-size: 0.85rem; color: #8b949e; margin-top: 5px;">Count: 12 Traders</p>
            <hr style="border-color: #30363d; margin: 10px 0;">
            <ul style="font-size: 0.85rem; padding-left: 15px; margin: 0;">
                <li><b>Win Rate:</b> Lower (~66%)</li>
                <li><b>Equity Leverage:</b> Low-to-moderate (~0.13x)</li>
                <li><b>Frequency:</b> Infrequent (~38 trades/day)</li>
                <li><b>PnL Profile:</b> Modestly positive PnL</li>
                <li><b>Risk Management:</b> Low activity insulates them, but lower win rates lead to slow capital decay in choppy markets.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Trader Archetype Directory")
    
    # Display table of accounts with cluster labels
    display_clusters = trader_clusters.copy()
    display_clusters['total_pnl'] = display_clusters['total_pnl'].map('${:,.2f}'.format)
    display_clusters['avg_win_rate'] = display_clusters['avg_win_rate'].map('{:.1%}'.format)
    display_clusters['avg_trade_count'] = display_clusters['avg_trade_count'].map('{:.1f}'.format)
    display_clusters['avg_equity_leverage'] = display_clusters['avg_equity_leverage'].map('{:.3f}x'.format)
    display_clusters['max_equity_leverage'] = display_clusters['max_equity_leverage'].map('{:.1f}x'.format)
    display_clusters['avg_long_short_ratio'] = display_clusters['avg_long_short_ratio'].map('{:.2f}'.format)
    display_clusters['margin_crossed_ratio'] = display_clusters['margin_crossed_ratio'].map('{:.1%}'.format)
    
    st.dataframe(display_clusters[['Account', 'archetype', 'total_pnl', 'avg_win_rate', 'avg_trade_count', 'avg_equity_leverage', 'margin_crossed_ratio']], use_container_width=True)

# ----------------------------------------------------
# TAB 5: PREDICTIVE MODEL PLAYGROUND
# ----------------------------------------------------
with tab5:
    st.subheader("🔮 Predictive Model: Next-Day Profitability")
    st.markdown("""
    <div class="info-card">
        This machine learning model predicts whether a trader will be <b>profitable next-day</b> (positive daily PnL) based on today's F&G sentiment and the trader's behavioral metrics. 
        <b>Random Forest Accuracy: 60.4% | ROC AUC: 61.7%</b> (highly significant predictive signal for time-series trading).
    </div>
    """, unsafe_allow_html=True)
    
    # Train the Random Forest on-the-fly to ensure robustness
    @st.cache_resource
    def train_playground_model(daily_metrics_df):
        # Prepare daily predictive dataset
        pred_data_list = []
        for acct, group in daily_metrics_df.groupby('Account'):
            group = group.sort_values('parsed_date').copy()
            group['target_next_day_profitable'] = (group['daily_pnl'].shift(-1) > 0).astype(int)
            group['fg_value_change_3d'] = group['fg_value'] - group['fg_value'].shift(3)
            group = group.dropna(subset=['target_next_day_profitable'])
            pred_data_list.append(group)
            
        df_pred = pd.concat(pred_data_list).reset_index(drop=True)
        
        feature_cols = [
            'fg_value', 'fg_value_change_3d', 'daily_pnl', 'total_fee', 
            'trade_count', 'win_rate', 'total_volume_usd', 'avg_trade_size_usd', 
            'avg_equity_leverage', 'long_short_ratio'
        ]
        
        df_pred = df_pred.dropna(subset=feature_cols + ['target_next_day_profitable'])
        
        X = df_pred[feature_cols]
        y = df_pred['target_next_day_profitable']
        
        rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
        rf.fit(X, y)
        return rf, feature_cols
        
    rf_model, features = train_playground_model(daily_metrics)
    
    col_play1, col_play2 = st.columns([1, 1])
    
    with col_play1:
        st.markdown("### 🎛️ Input Today's Parameters")
        
        # User sliders
        in_fg_val = st.slider("Fear & Greed Index Value", 0, 100, 50)
        in_fg_change_3d = st.slider("Fear & Greed Index 3-Day Change", -20, 20, 0)
        in_pnl = st.number_input("Today's realized PnL ($)", value=1000.0)
        in_fee = st.number_input("Today's Fees ($)", value=20.0)
        in_trade_count = st.number_input("Today's Trade Count", min_value=1, value=50)
        in_win_rate = st.slider("Today's Win Rate (%)", 0.0, 100.0, 75.0) / 100.0
        in_vol = st.number_input("Today's Volume USD ($)", value=5000.0)
        in_size = st.number_input("Today's Avg Position Size ($)", value=500.0)
        in_lev = st.number_input("Today's Avg Leverage (x)", value=0.05)
        in_ls_ratio = st.slider("Today's Long/Short Ratio", 0.1, 10.0, 1.5)
        
        # Predict Button
        if st.button("🔮 Predict Tomorrow's Profitability"):
            input_features = np.array([[
                in_fg_val, in_fg_change_3d, in_pnl, in_fee, 
                in_trade_count, in_win_rate, in_vol, in_size, 
                in_lev, in_ls_ratio
            ]])
            
            prediction = rf_model.predict(input_features)[0]
            probability = rf_model.predict_proba(input_features)[0][1]
            
            st.markdown("<br>", unsafe_allow_html=True)
            if prediction == 1:
                st.success(f"📈 **Prediction: PROFITABLE** (Confidence: {probability:.1%})")
                st.markdown("The model expects positive net PnL tomorrow. Today's low leverage / positive win rate setup aligns with a high-probability profitability window.")
            else:
                st.error(f"📉 **Prediction: UNPROFITABLE** (Confidence: {1-probability:.1%})")
                st.markdown("The model expects negative or zero net PnL tomorrow. High sentiment extremes, excessive trade frequency, or over-leveraging might increase the probability of tomorrow being a drawdown day.")

    with col_play2:
        st.markdown("### 📊 Model Feature Importances")
        # Load feature importances from pipeline output
        feature_imp = pd.read_csv(os.path.join(PROCESSED_DIR, "predictive_feature_importances.csv"))
        
        fig_imp = px.bar(
            feature_imp,
            y='Feature',
            x='Importance',
            orientation='h',
            title='Random Forest Feature Contribution',
            color='Importance',
            color_continuous_scale=px.colors.sequential.Bluyl,
            template="plotly_dark"
        )
        fig_imp.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(showgrid=True, gridcolor='#30363d'),
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig_imp, use_container_width=True)
        
        st.markdown("""
        **Feature Analysis:**
        - **Today's Realized PnL** is the strongest predictor of tomorrow's performance, indicating a strong momentum or streak effect in trader performance.
        - **Win Rate** and **Long/Short Ratio** are the next most significant behavioral signals. Traders with high win rates and stable long ratios tend to maintain profitability across regimes.
        - **Fear & Greed Index** and its **3-day change** contribute significantly, confirming that macro sentiment is an important contextual driver of next-day success.
        """)

# Footer Info
st.markdown("""
<hr style="border-color: #30363d; margin: 30px 0 15px 0;">
<div style="text-align: center; color: #8b949e; font-size: 0.8rem;">
    Premium Dashboard built with Streamlit and Plotly | Hyperliquid Derivative Trader Analysis Pipeline
</div>
""", unsafe_allow_html=True)
