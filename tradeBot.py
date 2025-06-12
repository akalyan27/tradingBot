from lumibot.brokers import Alpaca
from lumibot.backtesting import BacktestingBroker, YahooDataBacktesting
from lumibot.strategies import Strategy
from lumibot.traders import Trader

from datetime import datetime 
from alpaca_trade_api import REST 
from timedelta import Timedelta 
from sentiment_analysis import estimate_sentiment
from lumibot.entities import Asset

API_KEY = "PKFYP978QVCAFL61QZHI" 
API_SECRET = "3VrzeTfepwllnrjwxbQe1CalWbcUJ4jPRVLOuBDN" 
BASE_URL = "https://paper-api.alpaca.markets"

ALPACA_CREDS = {
    "API_KEY":API_KEY, 
    "API_SECRET": API_SECRET, 
    "PAPER": True
}


class MyStrategy(Strategy): 
    def initialize(self, symbol:str="SPY", cash_at_risk:float=.5): 
        self.symbol = symbol
        self.sleeptime = "24H" 
        self.last_trade = None 
        self.cash_at_risk = cash_at_risk
        self.api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)

    def position_sizing(self): 
        cash = self.get_cash() 
        last_price = self.get_last_price(self.symbol)
        quantity = round(cash * self.cash_at_risk / last_price,0)
        return cash, last_price, quantity

    def get_dates(self): 
        today = self.get_datetime()
        three_days_prior = today - Timedelta(days=3)
        return today.strftime('%Y-%m-%d'), three_days_prior.strftime('%Y-%m-%d')

    def get_sentiment(self): 
        today, three_days_prior = self.get_dates()
        news = self.api.get_news(symbol=self.symbol, 
                                 start=three_days_prior, 
                                 end=today) 
        news = [ev.__dict__["_raw"]["headline"] for ev in news]
        probability, sentiment = estimate_sentiment(news)
        return probability, sentiment 

    def on_trading_iteration(self):
        cash, last_price, quantity = self.position_sizing() 
        probability, sentiment = self.get_sentiment()

        if cash > last_price: 
            if sentiment == "positive" and probability > .999: 
                if self.last_trade == "sell": 
                    self.sell_all() 
                order = self.create_order(
                    self.symbol, 
                    quantity, 
                    "buy", 
                    type="bracket", 
                    take_profit_price=last_price*1.20, 
                    stop_loss_price=last_price*.95
                )
                self.submit_order(order) 
                self.last_trade = "buy"
            elif sentiment == "negative" and probability > .999: 
                if self.last_trade == "buy": 
                    self.sell_all() 
                order = self.create_order(
                    self.symbol, 
                    quantity, 
                    "sell", 
                    type="bracket", 
                    take_profit_price=last_price*.8, 
                    stop_loss_price=last_price*1.05
                )
                self.submit_order(order) 
                self.last_trade = "sell"

# Pick the dates that you want to start and end your backtest
start_date = datetime(2025, 1, 1)
end_date = datetime(2025, 5, 31)


broker = Alpaca(ALPACA_CREDS) 
strategy = MyStrategy(name='mlstrat', broker=broker, 
                    parameters={"symbol":"SPY", 
                                "cash_at_risk":.5})
result = MyStrategy.run_backtest(
    YahooDataBacktesting, 
    start_date, 
    end_date, 
    parameters={"symbol":"SPY", "cash_at_risk":.5}
)

# broker = Alpaca(ALPACA_CREDS)
# strategy = MyStrategy(name = 'MLBot', broker = broker, parameters = {"symbol":"SPY", "cash_at_risk":.5})
# benchmark_asset = Asset(symbol = "SPY", asset_type = "stock")

# strategy.backtest(
#     YahooDataBacktesting,
#     start_date,
#     end_date,
#     benchmark_asset = benchmark_asset
# )


# trader = Trader(
#     strategies=MyStrategy(name = 'mlstrat', broker = broker, parameters={}),
#     paper=False,  # Paper mode is irrelevant for backtesting
#     backtest=True,
#     start_date = start_date,
#     end_date = end_date
# )
# trader.run()

