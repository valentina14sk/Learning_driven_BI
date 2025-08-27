"""
Main Streamlit dashboard for StockPulse - Interactive Stock Analysis Dashboard.
Provides comprehensive stock analysis with technical indicators and portfolio tracking.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io

# Import custom modules
from stocks import StockDataFetcher
from indicators import TechnicalIndicators

# Page configuration
st.set_page_config(
    page_title="StockPulse - Interactive BI Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        color: white;
        margin-bottom: 2rem;
        border-radius: 10px;
    }
    
    .kpi-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        border-left: 4px solid #1f77b4;
    }
    
    .positive {
        color: #2ca02c;
        font-weight: bold;
    }
    
    .negative {
        color: #d62728;
        font-weight: bold;
    }
    
    .neutral {
        color: #ff7f0e;
        font-weight: bold;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa, #e9ecef);
    }
    
    div[data-testid="metric-container"] {
        background-color: white;
        border: 1px solid #ddd;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


class StockPulseDashboard:
    def render_chatbot(self, ticker, metrics, signals, portfolio=None):
        import requests
        st.sidebar.markdown("---")
        st.sidebar.subheader("💬 AI Chatbot")
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []
        user_input = st.sidebar.chat_input("Ask about finance, indicators, or your portfolio...")
        if user_input:
            # Prepare context for Groq API
            context = f"Ticker: {ticker}\nPrice: {metrics.get('latest_price', 0):.2f}\nChange: {metrics.get('percentage_change', 0):.2f}%\nSignal: {signals.get('signal', 'N/A')}\nStrength: {signals.get('strength', 0)}\nReasons: {', '.join(signals.get('reasons', []))}"
            if portfolio:
                context += f"\nPortfolio P/L: {portfolio.get('profit_loss', 0):.2f}\nPortfolio P/L %: {portfolio.get('profit_loss_pct', 0):.2f}%"
            prompt = f"You are a helpful finance assistant for beginners.\nContext: {context}\nUser question: {user_input}\nAnswer in short, professional, beginner-friendly language."
            headers = {"Authorization": "Bearer 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", "Content-Type": "application/json"}
            payload = {"messages": [{"role": "user", "content": prompt}], "model": "llama-3.3-70b-versatile"}
            try:
                response = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers, timeout=15)
                if response.status_code == 200:
                    ai_reply = response.json()['choices'][0]['message']['content']
                else:
                    # Show status code and error message from API
                    try:
                        error_detail = response.json().get('error', response.text)
                    except Exception:
                        error_detail = response.text
                    ai_reply = f"API Error {response.status_code}: {error_detail}"
            except Exception as e:
                ai_reply = f"Request Exception: {e}"
            st.session_state['chat_history'].append(("user", user_input))
            st.session_state['chat_history'].append(("assistant", ai_reply))
        # Show chat history
        for role, msg in st.session_state['chat_history']:
            st.sidebar.chat_message(role).write(msg)
    """Main dashboard class for StockPulse application."""
    
    def __init__(self):
        self.stock_fetcher = StockDataFetcher()
        self.technical_indicators = TechnicalIndicators()
    
    def render_header(self):
        """Render the main dashboard header."""
        st.markdown("""
        <div class="main-header">
            <h1>📊 StockPulse – Interactive BI for Stock Analysis</h1>
            <p>Real-time Insights for Beginners & Investors</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render sidebar navigation and controls."""
        st.sidebar.title("🎛️ Dashboard Controls")
        
        # Navigation
        page = st.sidebar.selectbox(
            "📋 Navigate to:",
            ["Overview", "Technical Indicators", "Portfolio Tracker", "Multi-Stock Comparison"]
        )
        
        st.sidebar.markdown("---")
        
        # Stock input
        ticker = st.sidebar.text_input(
            "🎯 Stock Symbol:",
            value="AAPL",
            help="Enter stock ticker (e.g., AAPL, GOOGL, TSLA)"
        ).upper()
        
        # Time period
        period_options = {
            "1 Month": "1mo",
            "3 Months": "3mo", 
            "6 Months": "6mo",
            "1 Year": "1y",
            "2 Years": "2y",
            "5 Years": "5y",
            "Custom Range": "custom"
        }
        
        period_selection = st.sidebar.selectbox(
            "📅 Time Period:",
            list(period_options.keys()),
            index=3
        )
        
        start_date, end_date = None, None
        
        if period_selection == "Custom Range":
            col1, col2 = st.sidebar.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=datetime.now() - timedelta(days=365)
                )
            with col2:
                end_date = st.date_input(
                    "End Date",
                    value=datetime.now()
                )
        
        return page, ticker, period_options[period_selection], start_date, end_date
    
    def create_kpi_cards(self, metrics: dict, stock_info: dict):
        """Create KPI cards showing key metrics."""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            price_color = "positive" if metrics.get('percentage_change', 0) >= 0 else "negative"
            st.metric(
                label="💰 Latest Price",
                value=f"${metrics.get('latest_price', 0):.2f}",
                delta=f"{metrics.get('percentage_change', 0):.2f}%"
            )
        
        with col2:
            st.metric(
                label="📊 Average Volume",
                value=f"{metrics.get('avg_volume', 0):,.0f}",
                help="Average daily trading volume"
            )
        
        with col3:
            st.metric(
                label="🔝 52W High",
                value=f"${metrics.get('high_52w', 0):.2f}",
                help="Highest price in 52 weeks"
            )
        
        with col4:
            st.metric(
                label="🔻 52W Low", 
                value=f"${metrics.get('low_52w', 0):.2f}",
                help="Lowest price in 52 weeks"
            )
        
        # Additional metrics row
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            st.metric(
                label="🏢 Market Cap",
                value=f"${stock_info.get('market_cap', 0)/1e9:.1f}B",
                help="Market capitalization in billions"
            )
        
        with col6:
            st.metric(
                label="📈 P/E Ratio",
                value=f"{stock_info.get('pe_ratio', 0):.1f}" if stock_info.get('pe_ratio') else "N/A"
            )
        
        with col7:
            st.metric(
                label="📊 Volatility",
                value=f"{metrics.get('volatility', 0):.1f}%",
                help="Annualized volatility"
            )
        
        with col8:
            dividend = stock_info.get('dividend_yield', 0)
            st.metric(
                label="💎 Dividend Yield",
                value=f"{dividend*100:.2f}%" if dividend else "N/A"
            )
    
    def create_price_chart(self, data: pd.DataFrame, ticker: str):
        """Create interactive price chart with moving averages."""
        fig = go.Figure()
        
        # Price line
        fig.add_trace(go.Scatter(
            x=data['date'],
            y=data['close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#1f77b4', width=2),
            hovertemplate='<b>Date:</b> %{x}<br><b>Price:</b> $%{y:.2f}<extra></extra>'
        ))
        
        # Moving averages
        if 'sma_20' in data.columns:
            fig.add_trace(go.Scatter(
                x=data['date'],
                y=data['sma_20'],
                mode='lines',
                name='SMA 20',
                line=dict(color='#ff7f0e', width=1, dash='dash'),
                hovertemplate='<b>SMA 20:</b> $%{y:.2f}<extra></extra>'
            ))
        
        if 'sma_50' in data.columns:
            fig.add_trace(go.Scatter(
                x=data['date'],
                y=data['sma_50'],
                mode='lines',
                name='SMA 50',
                line=dict(color='#2ca02c', width=1, dash='dot'),
                hovertemplate='<b>SMA 50:</b> $%{y:.2f}<extra></extra>'
            ))
        
        fig.update_layout(
            title=f'📈 {ticker} Price Trend with Moving Averages',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            height=500,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
    
    def create_candlestick_chart(self, data: pd.DataFrame, ticker: str):
        """Create interactive candlestick chart."""
        fig = go.Figure(data=go.Candlestick(
            x=data['date'],
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name=ticker,
            increasing_line_color='#2ca02c',
            decreasing_line_color='#d62728'
        ))
        
        fig.update_layout(
            title=f'🕯️ {ticker} Candlestick Chart',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            height=500,
            xaxis_rangeslider_visible=False
        )
        
        return fig
    
    def create_volume_chart(self, data: pd.DataFrame, ticker: str):
        """Create volume bar chart."""
        # Color bars based on price movement
        colors = ['#2ca02c' if data['close'].iloc[i] >= data['open'].iloc[i] 
                 else '#d62728' for i in range(len(data))]
        
        fig = go.Figure(data=go.Bar(
            x=data['date'],
            y=data['volume'],
            name='Volume',
            marker_color=colors,
            hovertemplate='<b>Date:</b> %{x}<br><b>Volume:</b> %{y:,.0f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'📊 {ticker} Trading Volume',
            xaxis_title='Date',
            yaxis_title='Volume',
            height=400,
            showlegend=False
        )
        
        return fig
    
    def create_technical_indicators_charts(self, data: pd.DataFrame, ticker: str):
        """Create technical indicators charts."""
        # RSI Chart
        rsi_fig = go.Figure()
        
        if 'rsi' in data.columns:
            rsi_fig.add_trace(go.Scatter(
                x=data['date'],
                y=data['rsi'],
                mode='lines',
                name='RSI',
                line=dict(color='#9467bd', width=2)
            ))
            
            # Add overbought/oversold lines
            rsi_fig.add_hline(y=70, line_dash="dash", line_color="red", 
                             annotation_text="Overbought (70)")
            rsi_fig.add_hline(y=30, line_dash="dash", line_color="green", 
                             annotation_text="Oversold (30)")
            
            rsi_fig.update_layout(
                title=f'📊 {ticker} RSI (Relative Strength Index)',
                xaxis_title='Date',
                yaxis_title='RSI',
                height=400,
                yaxis=dict(range=[0, 100])
            )
        
        # MACD Chart
        macd_fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('MACD Line & Signal', 'MACD Histogram'),
            row_heights=[0.7, 0.3]
        )
        
        if all(col in data.columns for col in ['macd', 'macd_signal', 'macd_histogram']):
            # MACD and Signal lines
            macd_fig.add_trace(
                go.Scatter(x=data['date'], y=data['macd'], name='MACD', 
                          line=dict(color='#1f77b4')),
                row=1, col=1
            )
            macd_fig.add_trace(
                go.Scatter(x=data['date'], y=data['macd_signal'], name='Signal', 
                          line=dict(color='#ff7f0e')),
                row=1, col=1
            )
            
            # MACD Histogram
            colors = ['green' if x >= 0 else 'red' for x in data['macd_histogram']]
            macd_fig.add_trace(
                go.Bar(x=data['date'], y=data['macd_histogram'], name='Histogram',
                       marker_color=colors),
                row=2, col=1
            )
        
        macd_fig.update_layout(
            title=f'📈 {ticker} MACD (Moving Average Convergence Divergence)',
            height=500
        )
        
        # Bollinger Bands Chart
        bb_fig = go.Figure()
        
        if all(col in data.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']):
            bb_fig.add_trace(go.Scatter(
                x=data['date'], y=data['bb_upper'], name='Upper Band',
                line=dict(color='red', width=1), fill=None
            ))
            
            bb_fig.add_trace(go.Scatter(
                x=data['date'], y=data['bb_lower'], name='Lower Band',
                line=dict(color='red', width=1), fill='tonexty', fillcolor='rgba(255,0,0,0.1)'
            ))
            
            bb_fig.add_trace(go.Scatter(
                x=data['date'], y=data['bb_middle'], name='Middle Band (SMA)',
                line=dict(color='blue', width=1)
            ))
            
            bb_fig.add_trace(go.Scatter(
                x=data['date'], y=data['close'], name='Close Price',
                line=dict(color='black', width=2)
            ))
        
        bb_fig.update_layout(
            title=f'📊 {ticker} Bollinger Bands',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            height=500
        )
        
        return rsi_fig, macd_fig, bb_fig
    
    def render_overview_page(self, data: pd.DataFrame, ticker: str, stock_info: dict, metrics: dict):
        """Render the main overview page."""
        st.subheader(f"📊 Overview - {stock_info.get('company_name', ticker)}")
        
        # Company info
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**Sector:** {stock_info.get('sector', 'N/A')}")
            st.write(f"**Industry:** {stock_info.get('industry', 'N/A')}")
        
        with col2:
            # Trading signals
            signals = self.technical_indicators.get_trading_signals(data)
            signal_color = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}
            st.write(f"**Signal:** {signal_color.get(signals['signal'], '⚪')} {signals['signal']}")
            st.write(f"**Strength:** {signals['strength']}/100")
        
        st.markdown("---")
        
        # KPI Cards
        self.create_kpi_cards(metrics, stock_info)
        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            price_chart = self.create_price_chart(data, ticker)
            st.plotly_chart(price_chart, use_container_width=True)
        
        with col2:
            candlestick_chart = self.create_candlestick_chart(data, ticker)
            st.plotly_chart(candlestick_chart, use_container_width=True)
            # Storytelling for candlestick chart
            last_close = data['close'].iloc[-1]
            prev_close = data['close'].iloc[-2] if len(data) > 1 else last_close
            trend = "uptrend" if last_close > prev_close else "downtrend" if last_close < prev_close else "sideways"
            st.info(f"Candlestick shows recent volatility with a strong {trend}.")
        
        # Volume chart
        volume_chart = self.create_volume_chart(data, ticker)
        st.plotly_chart(volume_chart, use_container_width=True)
        
        # Trading signals details
        if signals['reasons']:
            st.subheader("🎯 Trading Analysis")
            for reason in signals['reasons']:
                st.write(f"• {reason}")
        # Storytelling for KPI cards
        pct_change = metrics.get('percentage_change', 0)
        if pct_change > 0:
            st.success(f"Price rose by {pct_change:.2f}%, indicating strong buying interest.")
        elif pct_change < 0:
            st.error(f"Price fell by {abs(pct_change):.2f}%, indicating selling pressure.")
        else:
            st.info("Price remained stable, showing neutral sentiment.")
        # Overall summary
        summary = f"{ticker} is currently trading at ${metrics.get('latest_price', 0):.2f}. "
        if trend == "uptrend":
            summary += "The market shows bullish momentum with buyers in control."
        elif trend == "downtrend":
            summary += "The market is experiencing a bearish phase with sellers dominating."
        else:
            summary += "The market is moving sideways, indicating indecision."
        st.info(summary)
    
    def render_technical_indicators_page(self, data: pd.DataFrame, ticker: str):
        """Render technical indicators page."""
        st.subheader(f"📊 Technical Indicators - {ticker}")
        
        if data.empty:
            st.error("No data available for technical analysis")
            return
        
        # Create indicator charts
        rsi_fig, macd_fig, bb_fig = self.create_technical_indicators_charts(data, ticker)
        
        # Display charts in tabs
        tab1, tab2, tab3 = st.tabs(["RSI", "MACD", "Bollinger Bands"])
        
        with tab1:
            st.plotly_chart(rsi_fig, use_container_width=True)
            st.info("RSI above 70 indicates overbought conditions, below 30 indicates oversold conditions.")
            # Storytelling for RSI
            latest_rsi = data['rsi'].iloc[-1] if 'rsi' in data.columns else None
            if latest_rsi is not None:
                if latest_rsi > 70:
                    st.error("RSI is above 70: The stock may be overbought and due for a pullback.")
                elif latest_rsi < 30:
                    st.success("RSI is below 30: The stock may be oversold and could rebound.")
                else:
                    st.info("RSI is in the neutral zone, indicating balanced momentum.")
        
        with tab2:
            st.plotly_chart(macd_fig, use_container_width=True)
            st.info("MACD line crossing above signal line suggests bullish momentum, crossing below suggests bearish momentum.")
            # Storytelling for MACD
            if 'macd' in data.columns and 'macd_signal' in data.columns:
                macd_val = data['macd'].iloc[-1]
                macd_signal = data['macd_signal'].iloc[-1]
                if macd_val > macd_signal:
                    st.success("MACD is above the signal line: Momentum is shifting bullish.")
                elif macd_val < macd_signal:
                    st.error("MACD is below the signal line: Momentum is shifting bearish.")
                else:
                    st.info("MACD and signal line are equal: No clear momentum.")
        
        with tab3:
            st.plotly_chart(bb_fig, use_container_width=True)
            st.info("Price touching upper band may indicate overbought conditions, touching lower band may indicate oversold conditions.")
            # Storytelling for Moving Average and Bollinger Bands
            if 'sma_20' in data.columns and 'close' in data.columns:
                latest_close = data['close'].iloc[-1]
                latest_sma20 = data['sma_20'].iloc[-1]
                if latest_close > latest_sma20:
                    st.success("The 20-day moving average is below the price: Bullish trend.")
                elif latest_close < latest_sma20:
                    st.error("The 20-day moving average is above the price: Bearish trend.")
                else:
                    st.info("Price is at the 20-day moving average: Neutral trend.")
        
        # Current indicator values
        if len(data) > 0:
            latest = data.iloc[-1]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                rsi_val = latest.get('rsi', 0)
                rsi_status = "Overbought" if rsi_val > 70 else "Oversold" if rsi_val < 30 else "Neutral"
                st.metric("Current RSI", f"{rsi_val:.1f}", rsi_status)
            
            with col2:
                macd_val = latest.get('macd', 0)
                signal_val = latest.get('macd_signal', 0)
                macd_status = "Bullish" if macd_val > signal_val else "Bearish"
                st.metric("MACD", f"{macd_val:.3f}", macd_status)
            
            with col3:
                bb_position = "Above Upper" if latest.get('close', 0) > latest.get('bb_upper', 0) else \
                             "Below Lower" if latest.get('close', 0) < latest.get('bb_lower', 0) else "Within Bands"
                st.metric("Bollinger Position", bb_position)
            
            with col4:
                if 'stoch_k' in data.columns:
                    stoch_k = latest.get('stoch_k', 0)
                    stoch_status = "Overbought" if stoch_k > 80 else "Oversold" if stoch_k < 20 else "Neutral"
                    st.metric("Stochastic %K", f"{stoch_k:.1f}", stoch_status)
    
    def render_portfolio_page(self, data: pd.DataFrame, ticker: str, metrics: dict):
        """Render portfolio tracking page."""
        st.subheader("💼 Portfolio Tracker")
        
        if data.empty:
            st.error("No data available for portfolio calculations")
            return
        
        # Portfolio input
        col1, col2, col3 = st.columns(3)
        
        with col1:
            shares = st.number_input(
                f"Number of {ticker} shares:",
                min_value=0.0,
                value=100.0,
                step=1.0
            )
        
        with col2:
            avg_cost = st.number_input(
                "Average cost per share ($):",
                min_value=0.0,
                value=float(metrics.get('latest_price', 0)),
                step=0.01
            )
        
        with col3:
            current_price = metrics.get('latest_price', 0)
            st.metric("Current Price", f"${current_price:.2f}")
        
        if shares > 0 and avg_cost > 0:
            # Portfolio calculations
            total_cost = shares * avg_cost
            current_value = shares * current_price
            profit_loss = current_value - total_cost
            profit_loss_pct = (profit_loss / total_cost) * 100
            
            st.markdown("---")
            
            # Portfolio metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "💰 Total Investment",
                    f"${total_cost:,.2f}"
                )
            
            with col2:
                st.metric(
                    "📊 Current Value",
                    f"${current_value:,.2f}"
                )
            
            with col3:
                st.metric(
                    "📈 Profit/Loss",
                    f"${profit_loss:,.2f}",
                    f"{profit_loss_pct:+.2f}%"
                )
            
            with col4:
                daily_change = metrics.get('price_change', 0) * shares
                st.metric(
                    "📅 Today's P&L",
                    f"${daily_change:,.2f}"
                )
            # Storytelling for portfolio table
            if profit_loss_pct > 0:
                st.success(f"Your portfolio is up {profit_loss_pct:.2f}%. This is a bullish sign!")
            elif profit_loss_pct < 0:
                st.error(f"Your portfolio is down {abs(profit_loss_pct):.2f}%. This is a bearish sign.")
            else:
                st.info("Your portfolio is flat, showing neutral performance.")
            
            # Portfolio performance chart
            if 'cumulative_return' in data.columns:
                portfolio_value = total_cost * (1 + data['cumulative_return'])
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=data['date'],
                    y=portfolio_value,
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='#1f77b4', width=3),
                    fill='tonexty' if profit_loss >= 0 else None,
                    fillcolor='rgba(31, 119, 180, 0.1)'
                ))
                
                # Add cost basis line
                fig.add_hline(
                    y=total_cost,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text=f"Cost Basis: ${total_cost:,.0f}"
                )
                
                fig.update_layout(
                    title=f'💼 {ticker} Portfolio Performance',
                    xaxis_title='Date',
                    yaxis_title='Portfolio Value ($)',
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Risk metrics
        if 'daily_return' in data.columns and len(data) > 30:
            st.subheader("⚠️ Risk Analysis")
            
            returns = data['daily_return'].dropna()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
                st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
            
            with col2:
                max_drawdown = ((data['close'] / data['close'].expanding().max()) - 1).min() * 100
                st.metric("Max Drawdown", f"{max_drawdown:.1f}%")
            
            with col3:
                var_95 = returns.quantile(0.05) * np.sqrt(252) * 100
                st.metric("VaR (95%)", f"{var_95:.1f}%")
    
    def render_comparison_page(self):
        """Render multi-stock comparison page."""
        st.subheader("📊 Multi-Stock Comparison")
        
        # Stock input
        tickers_input = st.text_input(
            "Enter stock symbols (separated by commas):",
            value="AAPL,GOOGL,MSFT,TSLA",
            help="Example: AAPL,GOOGL,MSFT"
        )
        
        if tickers_input:
            tickers = [ticker.strip().upper() for ticker in tickers_input.split(',')]
            
            # Fetch data for all stocks
            stock_data = self.stock_fetcher.fetch_multiple_stocks(tickers, "1y")
            
            if stock_data:
                # Comparison chart
                fig = go.Figure()
                
                for ticker, data in stock_data.items():
                    if not data.empty:
                        # Normalize to percentage change from first day
                        normalized_prices = (data['close'] / data['close'].iloc[0] - 1) * 100
                        
                        fig.add_trace(go.Scatter(
                            x=data['date'],
                            y=normalized_prices,
                            mode='lines',
                            name=ticker,
                            line=dict(width=3),
                            hovertemplate=f'<b>{ticker}</b><br>Change: %{{y:.2f}}%<extra></extra>'
                        ))
                
                fig.update_layout(
                    title='📈 Stock Performance Comparison (% Change)',
                    xaxis_title='Date',
                    yaxis_title='Percentage Change (%)',
                    height=600,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Comparison table
                st.subheader("📊 Key Metrics Comparison")
                
                comparison_data = []
                for ticker, data in stock_data.items():
                    if not data.empty:
                        # Ensure returns are calculated for each stock
                        data = self.stock_fetcher.calculate_returns(data)
                        metrics = self.stock_fetcher.get_key_metrics(data)
                        stock_info = self.stock_fetcher.get_stock_info(ticker)
                        comparison_data.append({
                            'Symbol': ticker,
                            'Current Price': f"${metrics.get('latest_price', 0):.2f}",
                            'Change (%)': f"{metrics.get('percentage_change', 0):.2f}%",
                            '52W High': f"${metrics.get('high_52w', 0):.2f}",
                            '52W Low': f"${metrics.get('low_52w', 0):.2f}",
                            'Volatility': f"{metrics.get('volatility', 0):.1f}%",
                            'Market Cap': f"${stock_info.get('market_cap', 0)/1e9:.1f}B"
                        })
                
                if comparison_data:
                    df_comparison = pd.DataFrame(comparison_data)
                    st.dataframe(df_comparison, use_container_width=True)
    
    def render_data_export(self, data: pd.DataFrame, ticker: str):
        """Render data export section."""
        if not data.empty:
            st.sidebar.markdown("---")
            st.sidebar.subheader("💾 Data Export")
            
            # Prepare data for export
            export_data = data.copy()
            
            # Convert to CSV
            csv_buffer = io.StringIO()
            export_data.to_csv(csv_buffer, index=False)
            csv_string = csv_buffer.getvalue()
            
            # Download button
            st.sidebar.download_button(
                label=f"📥 Download {ticker} Data (CSV)",
                data=csv_string,
                file_name=f"{ticker}_stock_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
            # Storytelling for export
            st.sidebar.info("You can download this CSV to share or analyze offline.")
            
            st.sidebar.info(f"Data contains {len(export_data)} records with technical indicators")
    
    def run(self):
        """Main application runner."""
        # Render header
        self.render_header()
        # Render sidebar and get inputs
        page, ticker, period, start_date, end_date = self.render_sidebar()
        # Fetch stock data
        with st.spinner(f"Fetching data for {ticker}..."):
            if period == "custom" and start_date and end_date:
                data = self.stock_fetcher.fetch_stock_data(ticker, start_date=start_date, end_date=end_date)
            else:
                data = self.stock_fetcher.fetch_stock_data(ticker, period)
            stock_info = self.stock_fetcher.get_stock_info(ticker)
        if data.empty:
            st.error(f"Unable to fetch data for {ticker}. Please check the ticker symbol and try again.")
            return
        # Calculate indicators and metrics
        data = self.stock_fetcher.calculate_returns(data)
        data = self.technical_indicators.calculate_all_indicators(data)
        metrics = self.stock_fetcher.get_key_metrics(data)
        signals = self.technical_indicators.get_trading_signals(data)
        # Portfolio context for chatbot (if on portfolio page)
        portfolio = None
        if page == "Portfolio Tracker":
            shares = 100.0
            avg_cost = float(metrics.get('latest_price', 0))
            current_price = metrics.get('latest_price', 0)
            total_cost = shares * avg_cost
            current_value = shares * current_price
            profit_loss = current_value - total_cost
            profit_loss_pct = (profit_loss / total_cost) * 100 if total_cost else 0
            portfolio = {"profit_loss": profit_loss, "profit_loss_pct": profit_loss_pct}
        # Render chatbot in sidebar
        self.render_chatbot(ticker, metrics, signals, portfolio)
        # Render data export
        self.render_data_export(data, ticker)
        # Render selected page
        if page == "Overview":
            self.render_overview_page(data, ticker, stock_info, metrics)
        elif page == "Technical Indicators":
            self.render_technical_indicators_page(data, ticker)
        elif page == "Portfolio Tracker":
            self.render_portfolio_page(data, ticker, metrics)
        elif page == "Multi-Stock Comparison":
            self.render_comparison_page()


# Run the dashboard
if __name__ == "__main__":
    dashboard = StockPulseDashboard()
    dashboard.run()