"""Stock market environment for the simulation."""
import random
import pandas as pd
from typing import Dict, List

from Agent import Agent
from BulletinBoardSystem import BulletinBoardSystem
from SimulationConfig import SimulationConfig
from stock import Stock


class StockMarket:
    """Represents the entire stock market environment."""
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.stocks = {}
        self.agents = []
        self.bbs = BulletinBoardSystem()
        self.current_day = 0
        self.trade_history = []

        # Initialize stocks
        for symbol, price in config.initial_stock_prices.items():
            self.stocks[symbol] = Stock(symbol, price)

        # Initialize agents
        for i in range(config.num_agents):
            # Randomly assign personality and initial cash
            personality = random.choice(config.personalities)
            initial_cash = random.uniform(config.min_initial_assets, config.max_initial_assets)
            self.agents.append(Agent(i, personality, initial_cash))

    def get_stock_prices(self) -> Dict[str, float]:
        """Get current prices for all stocks."""
        return {symbol: stock.current_price for symbol, stock in self.stocks.items()}

    def execute_trade(self, buyer_id: int, seller_id: int, symbol: str, quantity: int, price: float) -> bool:
        """Execute a trade between two agents."""
        buyer = self.agents[buyer_id]
        seller = self.agents[seller_id]

        # Calculate total cost and fees
        total_cost = quantity * price
        buyer_fee = max(min(quantity * self.config.transaction_fee_per_share, self.config.max_transaction_fee), self.config.min_transaction_fee)
        seller_fee = max(min(quantity * self.config.transaction_fee_per_share, self.config.max_transaction_fee), self.config.min_transaction_fee)

        # Check if buyer has enough cash
        if buyer.cash < total_cost + buyer_fee:
            return False

        # Check if seller has enough stock
        if symbol not in seller.stocks or seller.stocks[symbol] < quantity:
            return False

        # Execute trade
        buyer.cash -= (total_cost + buyer_fee)
        seller.cash += (total_cost - seller_fee)
        buyer.add_stock(symbol, quantity)
        seller.remove_stock(symbol, quantity)

        # Record trade
        trade_record = {
            "day": self.current_day,
            "buyer_id": buyer_id,
            "seller_id": seller_id,
            "symbol": symbol,
            "quantity": quantity,
            "price": price,
            "buyer_fee": buyer_fee,
            "seller_fee": seller_fee
        }
        self.trade_history.append(trade_record)
        buyer.transaction_history.append(trade_record)
        seller.transaction_history.append(trade_record)

        return True

    def process_order_matching(self):
        """Match orders for all stocks and execute trades."""
        total_matched_trades = 0
        
        for symbol, stock in self.stocks.items():
            trades = stock.match_orders()
            volume = 0
            matched_trades = 0

            for trade in trades:
                success = self.execute_trade(
                    trade["buyer_id"],
                    trade["seller_id"],
                    symbol,
                    trade["quantity"],
                    trade["price"]
                )
                if success:
                    volume += trade["quantity"]
                    matched_trades += 1

            # Record trading volume and clear order book
            stock.record_volume(volume)
            stock.clear_order_book()
            total_matched_trades += matched_trades
        
        return total_matched_trades

    def agent_status_report(self) -> pd.DataFrame:
        """Generate a DataFrame with agent status information."""
        stock_prices = self.get_stock_prices()
        data = []

        for agent in self.agents:
            stock_value = sum(qty * stock_prices[symbol] for symbol, qty in agent.stocks.items())
            total_assets = agent.cash + stock_value

            # Calculate loan information
            active_loans = [loan for loan in agent.loans if not loan["repaid"]]
            total_debt = sum(loan["amount"] for loan in active_loans)

            data.append({
                "agent_id": agent.agent_id,
                "personality": agent.personality,
                "cash": agent.cash,
                "stock_value": stock_value,
                "total_assets": total_assets,
                "debt": total_debt,
                "net_worth": total_assets - total_debt,
                "bankrupted": agent.bankrupted
            })

        return pd.DataFrame(data)

    def market_status_report(self) -> pd.DataFrame:
        """Generate a DataFrame with market status information."""
        data = []

        for symbol, stock in self.stocks.items():
            data.append({
                "symbol": symbol,
                "current_price": stock.current_price,
                "daily_volume": stock.volume_history[-1],
                "price_change": stock.price_history[-1] - stock.price_history[-2] if len(stock.price_history) > 1 else 0,
                "percent_change": ((stock.price_history[-1] / stock.price_history[-2]) - 1) * 100 if len(stock.price_history) > 1 else 0
            })

        return pd.DataFrame(data)

    def update_prices(self, day, config):
        """Update stock prices based on market conditions and news."""
        # Get market conditions and sector performance
        market_conditions = config.get_market_conditions(day)
        sector_performance = config.get_sector_performance(day)
        
        # Get today's event if any
        event = config.get_event_for_day(day)
        event_impact = event["impact"] if event["description"] else 0
        
        # Check for ongoing crisis
        crisis_impact = 0
        if config.ongoing_crisis:
            crisis_impact = config.ongoing_crisis["impact"]
            # Crisis impact diminishes over time
            days_ongoing = day - config.ongoing_crisis['start_day']
            crisis_impact *= max(0.3, 1.0 - (days_ongoing * 0.15))  # Impact reduces by 15% each day
        
        # Update each stock price
        for symbol, stock in self.stocks.items():
            # Base market movement (random walk with drift)
            base_movement = random.normalvariate(0.0001, stock.volatility)
            
            # Sector influence
            sector = config.stock_sectors[symbol]
            sector_influence = sector_performance[sector] / 100  # convert percentage to decimal
            
            # Market sentiment effect
            sentiment_effect = (market_conditions["sentiment"] - 0.5) * 0.002
            
            # Event impact (weighted by relevance to stock)
            stock_event_impact = 0
            if event["description"]:
                # Check if event specifically mentions this stock
                stock_name = config.company_info[symbol]["name"]
                relevance = 1.0
                if stock_name.lower() in event["description"].lower() or f"Stock {symbol}" in event["description"]:
                    relevance = 2.5  # Higher impact for directly mentioned stocks
                elif sector.lower() in event["description"].lower():
                    relevance = 1.5  # Higher impact for mentioned sectors
                
                stock_event_impact = event_impact * relevance / 100
            
            # Apply crisis impact with stock-specific sensitivity
            stock_crisis_impact = 0
            if crisis_impact != 0 and config.ongoing_crisis:
                crisis_description = config.ongoing_crisis.get("description", "").lower()
                # Technology stocks may be less affected by some crises
                if sector == "Technology" and "pandemic" in crisis_description:
                    crisis_sensitivity = 0.7  # Tech can be more resilient to pandemic
                # Financial stocks more affected by financial crises
                elif sector == "Financial" and "financial" in crisis_description:
                    crisis_sensitivity = 1.5  # Financial more affected by financial crisis
                else:
                    crisis_sensitivity = 1.0
                
                stock_crisis_impact = (crisis_impact * crisis_sensitivity) / 100
            
            # Company-specific random movement
            company_specific = random.normalvariate(0, 0.005)
            
            # Check for company-specific major events that occurred today
            company_event_impact = 0
            for symbol_check in ["A", "B"]:
                if symbol == symbol_check and random.random() < 0.01:  # 1% chance per day per stock
                    if hasattr(config, 'company_events') and symbol in config.company_events:
                        event = random.choice(config.company_events[symbol])
                        company_event_impact = event["impact"] / 100
            
            # Combine all factors
            total_movement = (
                base_movement + 
                sector_influence + 
                sentiment_effect +
                stock_event_impact + 
                stock_crisis_impact +
                company_specific + 
                company_event_impact
            )
            
            # Ensure movement doesn't cause extreme price changes
            total_movement = max(min(total_movement, 0.10), -0.10)  # Cap at ±10% daily movement
            
            # Calculate and update new price
            new_price = stock.current_price * (1 + total_movement)
            new_price = max(0.01, new_price)  # Ensure price doesn't go below $0.01
            
            # Update stock price 
            stock.update_price(new_price)
            
            # Update stock volatility based on market conditions
            if market_conditions["volatility"] == "high":
                stock.volatility = min(0.05, stock.volatility * 1.05)
            elif market_conditions["volatility"] == "low":
                stock.volatility = max(0.005, stock.volatility * 0.95)
