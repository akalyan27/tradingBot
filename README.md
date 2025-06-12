# Financial News Sentiment Analyzer & Trading Bot

## Overview
This project presents an intelligent trading bot powered by financial news sentiment analysis. It leverages estimates sentiment from recent market news, dynamically influencing trading decisions
and position sizing, with strategies rigorously backtested and integrated for live market operation.

## Features
* **Unsupervised Sentiment Analysis:** Utilizes unsupervised learning techniques to estimate the sentiment of recent financial news, providing crucial insights for the bot's trading logic.
* **Data-Driven Trading Logic:** Integrates estimated sentiment directly into the trading algorithm to drive and optimize buy/sell decisions.
* **Robust Backtesting Framework:** Implements comprehensive backtesting strategies using historical data from Yahoo Finance to validate and refine performance before live deployment.
* **Live Market Integration:** Seamlessly connects the trading bot with the Alpaca API for real-time market testing and automated execution of trading strategies.

## Technologies Used
* Python
* Pytorch
* Yahoo Finance API (for historical data)
* Alpaca API (for live trading)
* (Potentially other libraries like Pandas, NumPy, Scikit-learn for data manipulation and ML)

## Installation & Usage
(Add detailed instructions here on how to set up and run your project. This typically includes cloning the repository, creating a virtual environment, installing dependencies, configuring API keys, and running the main script.)

```bash
git clone https://github.com/akalyan27/tradingBot.git
cd tradingBot
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
# Configure your Alpaca API keys (e.g., via environment variables)
python tradeBot.py 
