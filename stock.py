"""Stock class representing individual stocks in the market."""
from typing import List, Dict, Tuple
import random
import numpy as np


class Stock:
    """Represents a stock in the market with price history and trading volume."""
    def __init__(self, symbol: str, initial_price: float):
        self.symbol = symbol
        self.current_price = initial_price
        self.price_history = [initial_price]
        self.volume_history = [0]  # Trading volume history
        self.volatility = random.uniform(0.01, 0.03)  # Daily volatility
        self.market_sentiment = random.uniform(-0.2, 0.2)  # Market sentiment bias
        self.momentum = 0  # Price momentum
        self.order_book = {
            "bids": [],  # List of (price, quantity, agent_id) tuples for buy orders
            "asks": [],  # List of (price, quantity, agent_id) tuples for sell orders
        }

    def update_price(self, new_price: float):
        """Update the stock price and record in history."""
        # Calculate momentum as a weighted average of previous price changes
        if len(self.price_history) >= 2:
            self.momentum = 0.7 * self.momentum + 0.3 * (new_price - self.current_price) / self.current_price
        
        self.current_price = new_price
        self.price_history.append(new_price)

    def record_volume(self, volume: int):
        """Record trading volume for the current period."""
        self.volume_history.append(volume)
        
        # Update volatility based on trading volume
        if len(self.volume_history) > 2:
            avg_volume = sum(self.volume_history[-5:]) / min(5, len(self.volume_history))
            if volume > 2 * avg_volume:
                self.volatility = min(0.05, self.volatility * 1.2)  # Increase volatility on high volume
            elif volume < 0.5 * avg_volume:
                self.volatility = max(0.005, self.volatility * 0.9)  # Decrease volatility on low volume

    def clear_order_book(self):
        """Clear the order book for a new trading session."""
        self.order_book = {
            "bids": [],
            "asks": []
        }

    def add_buy_order(self, price: float, quantity: int, agent_id: int):
        """Add a buy order to the order book."""
        self.order_book["bids"].append((price, quantity, agent_id))

    def add_sell_order(self, price: float, quantity: int, agent_id: int):
        """Add a sell order to the order book."""
        self.order_book["asks"].append((price, quantity, agent_id))

    def match_orders(self) -> List[Dict]:
        """Match buy and sell orders and execute trades."""
        # Sort bids (highest price first) and asks (lowest price first)
        self.order_book["bids"].sort(key=lambda x: x[0], reverse=True)
        self.order_book["asks"].sort(key=lambda x: x[0])

        executed_trades = []

        while self.order_book["bids"] and self.order_book["asks"]:
            bid = self.order_book["bids"][0]
            ask = self.order_book["asks"][0]

            # If highest bid price is greater than or equal to lowest ask price
            if bid[0] >= ask[0]:
                bid_price, bid_qty, bid_agent = bid
                ask_price, ask_qty, ask_agent = ask

                # Determine trade quantity and price
                trade_qty = min(bid_qty, ask_qty)
                trade_price = (bid_price + ask_price) / 2  # Midpoint price

                # Record the trade
                trade = {
                    "buyer_id": bid_agent,
                    "seller_id": ask_agent,
                    "price": trade_price,
                    "quantity": trade_qty,
                    "symbol": self.symbol
                }
                executed_trades.append(trade)

                # Update order book
                if bid_qty > ask_qty:
                    # Partial fill for bid, complete fill for ask
                    self.order_book["bids"][0] = (bid_price, bid_qty - ask_qty, bid_agent)
                    self.order_book["asks"].pop(0)
                elif ask_qty > bid_qty:
                    # Complete fill for bid, partial fill for ask
                    self.order_book["asks"][0] = (ask_price, ask_qty - bid_qty, ask_agent)
                    self.order_book["bids"].pop(0)
                else:
                    # Complete fill for both
                    self.order_book["bids"].pop(0)
                    self.order_book["asks"].pop(0)

                # Update current price to the last trade price
                self.current_price = trade_price
                
                # Update market sentiment based on trade volume
                self.market_sentiment += 0.01 * trade_qty / 1000  # Positive sentiment with higher volume
            else:
                # No more matches possible
                break

        return executed_trades
        
    def generate_market_price_movement(self, event_impact=0):
        """Generate a realistic price movement based on market factors."""
        # Base random component (following random walk with drift)
        random_component = np.random.normal(0, self.volatility)
        
        # Add momentum effect
        momentum_effect = self.momentum * random.uniform(0.5, 1.5)
        
        # Add sentiment bias
        sentiment_effect = self.market_sentiment * random.uniform(0.01, 0.03)
        
        # Event impact (if any)
        event_component = event_impact * random.uniform(0.8, 1.2)
        
        # Calculate total price change
        price_change_pct = random_component + momentum_effect + sentiment_effect + event_component
        
        # Apply the price change
        new_price = self.current_price * (1 + price_change_pct)
        new_price = max(0.01, new_price)  # Ensure price doesn't go below 0.01
        
        return new_price