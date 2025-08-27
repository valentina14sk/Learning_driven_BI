# StockPulse - Interactive Stock Analysis Dashboard

A comprehensive Streamlit dashboard for stock analysis with technical indicators, portfolio tracking, and multi-stock comparison.

## Features

- 📊 **Real-time Stock Data**: Fetch live stock data using Yahoo Finance
- 📈 **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages
- 💼 **Portfolio Tracking**: Track your investments and P&L
- 🔍 **Multi-Stock Comparison**: Compare multiple stocks side by side
- 📱 **Responsive Design**: Professional UI with interactive charts

## Quick Start

### Local Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd stockpulse
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the dashboard:
```bash
streamlit run dashboard.py
```

### Deploy on Streamlit Cloud (Free)

1. **Push to GitHub**:
   - Create a new GitHub repository
   - Push all files to the repository

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Connect your GitHub account
   - Select your repository
   - Set main file path: `dashboard.py`
   - Click "Deploy"

3. **One-Click Deployment**:
   - Streamlit Cloud will automatically detect `requirements.txt`
   - Your app will be live at: `https://your-app-name.streamlit.app`

## Project Structure

```
stockpulse/
├── dashboard.py          # Main Streamlit application
├── stocks.py            # Stock data fetching and processing
├── indicators.py        # Technical indicators calculations
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Usage

1. **Overview Page**: View stock price, key metrics, and trading signals
2. **Technical Indicators**: Analyze RSI, MACD, and Bollinger Bands
3. **Portfolio Tracker**: Monitor your investment performance
4. **Multi-Stock Comparison**: Compare multiple stocks performance

## Dependencies

- **Streamlit**: Web app framework
- **yfinance**: Yahoo Finance data fetching
- **pandas**: Data manipulation
- **plotly**: Interactive charts
- **matplotlib/seaborn**: Additional plotting
- **numpy**: Numerical computations

## Troubleshooting

If you encounter Python environment issues:

1. **Reset Environment**: Use your platform's environment reset option
2. **Fresh Install**: Create a new virtual environment
3. **Alternative Deployment**: Use Streamlit Cloud for hassle-free hosting

## License

MIT License - Feel free to use and modify for your projects.