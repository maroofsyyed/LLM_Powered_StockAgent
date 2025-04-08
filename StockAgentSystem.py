"""Main system for running the StockAgent simulation."""
import random
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from tqdm import tqdm

from Agent import Agent
from SimulationConfig import SimulationConfig
from StockMarket import StockMarket
from LLMInterface import LLMInterface
from AgentPromptBuilder import AgentPromptBuilder


class StockAgentSystem:
    """Main system for running the StockAgent simulation."""
    def __init__(
        self,
        llm_interface: LLMInterface,
        num_agents: int = 10,
        days_to_simulate: int = 10
    ):
        self.config = SimulationConfig()
        self.config.num_agents = num_agents
        self.llm = llm_interface
        self.market = StockMarket(self.config)
        self.days_to_simulate = days_to_simulate
        self.prompt_builder = AgentPromptBuilder()

    def process_loan_decision(self, agent: Agent) -> Dict:
        """Process loan decision for an agent using LLM."""
        if agent.bankrupted:
            return {"loan": "no"}

        prompt = self.prompt_builder.build_loan_decision_prompt(agent, self.market, self.config)
        system_prompt = f"You are a {agent.personality} stock trader. Make decisions that reflect this personality."

        response = self.llm.generate_response(prompt, system_prompt)

        try:
            # Clean response to handle markdown formatting and other noise
            response = response.strip()
            # Remove markdown code block delimiters if present
            if "```json" in response:
                response = response.split("```json")[1]
            if "```" in response:
                response = response.split("```")[0]
            # Remove any other characters before or after JSON
            response = response.strip()
            
            # Add missing closing brace if needed
            if response.count('{') > response.count('}'):
                response += '}'
                
            # Parse response
            decision = json.loads(response)
            print(f"Agent {agent.agent_id} ({agent.personality}) loan decision: {decision}")
            return decision
        except json.JSONDecodeError as e:
            print(f"Error parsing loan decision response for Agent {agent.agent_id}: {response}")
            print(f"JSON error: {e}")
            return {"loan": "no"}

    def process_trading_decision(self, agent: Agent, trading_session: int) -> Dict:
        """Process trading decision for an agent using LLM."""
        if agent.bankrupted:
            return {"action_type": "hold"}

        prompt = self.prompt_builder.build_trading_decision_prompt(
            agent, self.market, trading_session, self.config
        )
        system_prompt = f"You are a {agent.personality} stock trader. Make decisions that reflect this personality."

        response = self.llm.generate_response(prompt, system_prompt)

        try:
            # Clean response to handle markdown formatting and other noise
            response = response.strip()
            # Remove markdown code block delimiters if present
            if "```json" in response:
                response = response.split("```json")[1]
            if "```" in response:
                response = response.split("```")[0]
            # Remove any other characters before or after JSON
            response = response.strip()
            
            # Add missing closing brace if needed
            if response.count('{') > response.count('}'):
                response += '}'
                
            # Parse response
            decision = json.loads(response)
            
            # Print trading decision for debugging
            print(f"Agent {agent.agent_id} ({agent.personality}) trading decision: {decision}")
            
            return decision
        except json.JSONDecodeError as e:
            print(f"Error parsing trading decision response for Agent {agent.agent_id}: {response}")
            print(f"JSON error: {e}")
            return {"action_type": "hold"}

    def process_future_action_prediction(self, agent: Agent) -> Dict:
        """Process future action prediction for an agent using LLM."""
        if agent.bankrupted:
            return {
                "buy_A": "no", "buy_B": "no", "buy_C": "no",
                "sell_A": "no", "sell_B": "no", "sell_C": "no",
                "loan": "no"
            }

        prompt = self.prompt_builder.build_future_action_prompt(agent, self.market)
        system_prompt = f"You are a {agent.personality} stock trader making predictions about your future actions."

        response = self.llm.generate_response(prompt, system_prompt)

        try:
            # Clean response to handle markdown formatting and other noise
            response = response.strip()
            # Remove markdown code block delimiters if present
            if "```json" in response:
                response = response.split("```json")[1]
            if "```" in response:
                response = response.split("```")[0]
            # Remove any other characters before or after JSON
            response = response.strip()
            
            # Add missing closing brace if needed
            if response.count('{') > response.count('}'):
                response += '}'
                
            # Parse response
            prediction = json.loads(response)
            return prediction
        except json.JSONDecodeError as e:
            print(f"Error parsing future action response for Agent {agent.agent_id}: {response}")
            print(f"JSON error: {e}")
            return {
                "buy_A": "no", "buy_B": "no", "buy_C": "no",
                "sell_A": "no", "sell_B": "no", "sell_C": "no",
                "loan": "no"
            }

    def process_bbs_message(self, agent: Agent) -> str:
        """Process BBS message for an agent using LLM."""
        if agent.bankrupted:
            return ""

        prompt = self.prompt_builder.build_bbs_message_prompt(agent, self.market)
        system_prompt = f"You are a {agent.personality} stock trader."

        response = self.llm.generate_response(prompt, system_prompt)
        return response

    def execute_trading_day(self):
        """Execute a full trading day in the simulation."""
        self.market.current_day += 1
        day = self.market.current_day
        print(f"\n===== Executing trading day {day} =====")
        
        # Update market conditions based on the day and events
        self.config.update_market_conditions(day)
        
        # Get market event for the day
        event = self.config.get_event_for_day(day)
        if event["description"]:
            print(f"Market Event: {event['description']} (Impact: {event['impact']})")
            
        # Update prices using our enhanced method that accounts for crises, events, etc.
        self.market.update_prices(day, self.config)

        # 1. Pre-Trading Procedures
        print("\n--- Pre-Trading Procedures ---")

        # 1.1 Check for interest payments
        if self.config.is_interest_payment_day(day):
            print("Processing interest payments...")
            for agent in self.market.agents:
                if not agent.bankrupted:
                    interest_paid = agent.process_interest_payments(day)
                    if interest_paid == -1:  # Indicates bankruptcy
                        agent.go_bankrupt(self.market.get_stock_prices())

        # 1.2 Check for loan repayments
        print("Processing loan repayments...")
        for agent in self.market.agents:
            if not agent.bankrupted:
                if not agent.process_loan_repayments(day):
                    # Agent couldn't repay loans, declare bankruptcy
                    agent.go_bankrupt(self.market.get_stock_prices())

        # 1.3 Process loan decisions
        print("Processing loan decisions...")
        for agent in self.market.agents:
            if not agent.bankrupted:
                loan_decision = self.process_loan_decision(agent)

                if loan_decision.get("loan") == "yes":
                    loan_type = loan_decision.get("loan_type", 1)
                    loan_amount = loan_decision.get("amount", 0)

                    # Validate loan type and amount
                    if loan_type not in [1, 2, 3]:
                        loan_type = 1

                    max_loan = agent.calculate_total_assets(self.market.get_stock_prices()) * 0.5
                    if loan_amount > max_loan:
                        loan_amount = max_loan

                    if loan_amount > 0:
                        agent.add_loan(
                            loan_amount,
                            loan_type,
                            day,
                            self.config.loan_rates[loan_type]
                        )
                        print(f"Agent {agent.agent_id} took a loan of ${loan_amount:.2f} (Type: {loan_type})")

        # 2. Trading Sessions (3 sessions per day)
        for session in range(1, 4):
            print(f"\n--- Trading Session {session} ---")
            
            # 2.1 Generate random order for agents to execute trades
            agent_order = list(range(len(self.market.agents)))
            random.shuffle(agent_order)

            # Trading stats for this session
            buys = 0
            sells = 0
            holds = 0
            market_maker_trades = 0

            # 2.2 Process trading decisions for each agent
            for agent_idx in agent_order:
                agent = self.market.agents[agent_idx]
                if not agent.bankrupted:
                    decision = self.process_trading_decision(agent, session)

                    if decision["action_type"] == "buy":
                        buys += 1
                        # Process buy order
                        stock_symbol = decision.get("stock")
                        amount = int(decision.get("amount", 0))
                        price = float(decision.get("price", 0))

                        if stock_symbol in self.market.stocks and amount > 0 and price > 0:
                            stock = self.market.stocks[stock_symbol]
                            
                            # Check if agent has enough cash for the purchase
                            total_cost = amount * price
                            transaction_fee = max(min(amount * self.config.transaction_fee_per_share, 
                                                 self.config.max_transaction_fee), 
                                             self.config.min_transaction_fee)
                            
                            if agent.cash >= (total_cost + transaction_fee):
                                # Execute direct market buy (market maker functionality)
                                # Use a mix of order book and direct market transactions
                                if random.random() < 0.5:  # 50% chance of using order book
                                    print(f"Agent {agent.agent_id} placed order to buy {amount} shares of Stock {stock_symbol} at ${price:.2f}")
                                    stock.add_buy_order(price, amount, agent.agent_id)
                                else:
                                    # Direct market purchase
                                    agent.cash -= (total_cost + transaction_fee)
                                    agent.add_stock(stock_symbol, amount)
                                    market_maker_trades += 1
                                    
                                    print(f"Agent {agent.agent_id} bought {amount} shares of Stock {stock_symbol} at ${price:.2f} from market maker")
                                    
                                    # Record the market maker trade
                                    trade_record = {
                                        "day": self.market.current_day,
                                        "buyer_id": agent.agent_id,
                                        "seller_id": -1,  # -1 indicates market maker
                                        "symbol": stock_symbol,
                                        "quantity": amount,
                                        "price": price,
                                        "buyer_fee": transaction_fee,
                                        "seller_fee": 0,
                                        "action": "buy"  # Add action type for transaction history filtering
                                    }
                                    self.market.trade_history.append(trade_record)
                                    agent.transaction_history.append(trade_record)
                                    
                                    # Update volume
                                    stock.record_volume(amount)
                            else:
                                # Not enough cash, try to place order at a lower price
                                if agent.cash > 0:
                                    adjusted_amount = min(amount, int(agent.cash / price * 0.95))  # 5% buffer for fees
                                    if adjusted_amount > 0:
                                        print(f"Agent {agent.agent_id} has insufficient funds. Placed reduced order to buy {adjusted_amount} shares of Stock {stock_symbol} at ${price:.2f}")
                                        stock.add_buy_order(price, adjusted_amount, agent.agent_id)

                    elif decision["action_type"] == "sell":
                        sells += 1
                        # Process sell order
                        stock_symbol = decision.get("stock")
                        amount = int(decision.get("amount", 0))
                        price = float(decision.get("price", 0))

                        if (stock_symbol in agent.stocks and
                            stock_symbol in self.market.stocks and
                            agent.stocks.get(stock_symbol, 0) >= amount and
                            amount > 0 and price > 0):

                            stock = self.market.stocks[stock_symbol]
                            
                            # Use a mix of order book and direct market transactions
                            if random.random() < 0.5:  # 50% chance of using order book
                                print(f"Agent {agent.agent_id} placed order to sell {amount} shares of Stock {stock_symbol} at ${price:.2f}")
                                stock.add_sell_order(price, amount, agent.agent_id)
                            else:
                                # Direct market sale
                                total_sale = amount * price
                                transaction_fee = max(min(amount * self.config.transaction_fee_per_share, 
                                                     self.config.max_transaction_fee), 
                                                 self.config.min_transaction_fee)
                                
                                agent.cash += (total_sale - transaction_fee)
                                agent.remove_stock(stock_symbol, amount)
                                market_maker_trades += 1
                                
                                print(f"Agent {agent.agent_id} sold {amount} shares of Stock {stock_symbol} at ${price:.2f} to market maker")
                                
                                # Record the market maker trade
                                trade_record = {
                                    "day": self.market.current_day,
                                    "buyer_id": -1,  # -1 indicates market maker
                                    "seller_id": agent.agent_id,
                                    "symbol": stock_symbol,
                                    "quantity": amount,
                                    "price": price,
                                    "buyer_fee": 0,
                                    "seller_fee": transaction_fee,
                                    "action": "sell"  # Add action type for transaction history filtering
                                }
                                self.market.trade_history.append(trade_record)
                                agent.transaction_history.append(trade_record)
                                
                                # Update volume
                                stock.record_volume(amount)
                    else:  # hold
                        holds += 1
                        print(f"Agent {agent.agent_id} decided to hold this session")

            # Print trading session summary
            print(f"\nSession {session} Summary: {buys} buys, {sells} sells, {holds} holds, {market_maker_trades} market maker trades")

            # 2.3 Match orders and execute trades
            print("\nMatching orders and executing trades...")
            num_trades = self.market.process_order_matching()
            print(f"Matched {num_trades} trades between agents")

        # 3. Post-Trading Procedures
        print("\n--- Post-Trading Procedures ---")

        # 3.1 Generate BBS messages
        print("Generating BBS messages...")
        for agent in self.market.agents:
            if not agent.bankrupted and random.random() < 0.3:  # 30% chance to post a message
                message = self.process_bbs_message(agent)
                if message:
                    self.market.bbs.post_message(day, agent.agent_id, message)
                    print(f"Agent {agent.agent_id} posted a message to the BBS")

        # 3.2 Future action predictions
        print("Generating future action predictions...")
        for agent in self.market.agents:
            if not agent.bankrupted:
                prediction = self.process_future_action_prediction(agent)
                print(f"Agent {agent.agent_id} future prediction: {prediction}")

        # 3.3 Collect daily status
        agent_status = self.collect_agent_status()
        market_status = self.collect_market_status()

        # Print day summary
        self.print_day_summary(day)

        return agent_status, market_status

    def collect_agent_status(self):
        """Collect agent status information for reporting."""
        agent_data = []
        stock_prices = self.market.get_stock_prices()
        
        for agent in self.market.agents:
            total_assets = agent.calculate_total_assets(stock_prices)
            net_worth = agent.calculate_net_worth(stock_prices)
            portfolio_value = agent.calculate_portfolio_value(stock_prices)
            total_debt = agent.calculate_total_debt()
            
            agent_data.append({
                "agent_id": agent.agent_id,
                "personality": agent.personality,
                "cash": agent.cash,
                "portfolio_value": portfolio_value,
                "total_assets": total_assets,
                "debt": total_debt,
                "net_worth": net_worth,
                "bankrupted": agent.bankrupted,
                "stocks_held": agent.stocks.copy() if agent.stocks else {}
            })
        
        return pd.DataFrame(agent_data)
        
    def collect_market_status(self):
        """Collect market status information for reporting."""
        market_data = []
        
        for symbol, stock in self.market.stocks.items():
            market_data.append({
                "symbol": symbol,
                "price": stock.current_price,
                "volume": stock.volume_history[-1] if stock.volume_history else 0,
                "buy_orders": len(stock.order_book["bids"]),
                "sell_orders": len(stock.order_book["asks"])
            })
            
        return pd.DataFrame(market_data)
        
    def print_day_summary(self, day):
        """Print a summary of the day's trading activity."""
        stock_prices = self.market.get_stock_prices()
        
        print("\n========== Day Summary ==========")
        print(f"Day {day} Trading Complete")
        
        # Print stock summary
        print("\nStock Prices:")
        for symbol, price in stock_prices.items():
            stock = self.market.stocks[symbol]
            volume = stock.volume_history[-1] if stock.volume_history else 0
            print(f"Stock {symbol}: ${price:.2f} (Volume: {volume} shares)")
            
        # Print agent summary
        print("\nAgent Summary:")
        for agent in self.market.agents:
            if agent.bankrupted:
                status = "BANKRUPT"
            else:
                net_worth = agent.calculate_net_worth(stock_prices)
                portfolio_value = agent.calculate_portfolio_value(stock_prices)
                status = f"NW: ${net_worth:.2f}, Cash: ${agent.cash:.2f}, Portfolio: ${portfolio_value:.2f}"
                
            print(f"Agent {agent.agent_id} ({agent.personality}): {status}")
            
        # Print number of trades for the day
        trades_today = [t for t in self.market.trade_history if t["day"] == day]
        print(f"\nTotal Trades Today: {len(trades_today)}")
        
        print("=================================\n")
        
    def run_simulation(self):
        """Run the full simulation for the specified number of days."""
        agent_statuses = []
        market_statuses = []

        try:
            for day in tqdm(range(self.days_to_simulate)):
                agent_status, market_status = self.execute_trading_day()
                agent_statuses.append(agent_status)
                market_statuses.append(market_status)

                # Save checkpoints periodically
                if day % 5 == 0:
                    self.save_checkpoint(f"checkpoint_day_{day}")
        except Exception as e:
            print(f"Error during simulation: {e}")
            # Save emergency checkpoint
            self.save_checkpoint("emergency_checkpoint")
            raise

        # Generate final reports and visualizations
        self.generate_reports(agent_statuses, market_statuses)
        return agent_statuses, market_statuses

    def save_checkpoint(self, filename: str):
        """Save a checkpoint of the current simulation state."""
        data = {
            "current_day": self.market.current_day,
            "stocks": {
                symbol: {
                    "price_history": stock.price_history,
                    "volume_history": stock.volume_history,
                    "current_price": stock.current_price
                } for symbol, stock in self.market.stocks.items()
            },
            "agents": [
                {
                    "agent_id": agent.agent_id,
                    "personality": agent.personality,
                    "cash": agent.cash,
                    "stocks": agent.stocks,
                    "loans": agent.loans,
                    "bankrupted": agent.bankrupted
                } for agent in self.market.agents
            ],
            "bbs_messages": self.market.bbs.daily_messages,
            "trade_history": self.market.trade_history
        }

        with open(f"{filename}.json", "w") as f:
            json.dump(data, f, indent=2)

    def load_checkpoint(self, filename: str):
        """Load a simulation from a checkpoint."""
        with open(f"{filename}.json", "r") as f:
            data = json.load(f)

        # Restore market day
        self.market.current_day = data["current_day"]

        # Restore stocks
        for symbol, stock_data in data["stocks"].items():
            if symbol in self.market.stocks:
                stock = self.market.stocks[symbol]
                stock.price_history = stock_data["price_history"]
                stock.volume_history = stock_data["volume_history"]
                stock.current_price = stock_data["current_price"]

        # Restore agents
        for i, agent_data in enumerate(data["agents"]):
            if i < len(self.market.agents):
                agent = self.market.agents[i]
                agent.cash = agent_data["cash"]
                agent.stocks = agent_data["stocks"]
                agent.loans = agent_data["loans"]
                agent.bankrupted = agent_data["bankrupted"]

        # Restore BBS messages
        self.market.bbs.daily_messages = data["bbs_messages"]

        # Restore trade history
        self.market.trade_history = data["trade_history"]

    def generate_reports(self, agent_statuses: List[pd.DataFrame], market_statuses: List[pd.DataFrame]):
        """Generate reports and visualizations from simulation data."""
        # 1. Agent Performance Report
        self.generate_agent_performance_report(agent_statuses)

        # 2. Market Performance Report
        self.generate_market_performance_report(market_statuses)

        # 3. Trading Activity Report
        self.generate_trading_activity_report()

        # 4. Save Final Results
        self.save_final_results(agent_statuses[-1], market_statuses[-1])

    def generate_agent_performance_report(self, agent_statuses: List[pd.DataFrame]):
        """Generate a report on agent performance."""
        # Extract net worth for each agent across all days
        agent_worth_history = {}
        agent_cash_history = {}
        agent_portfolio_history = {}
        agent_debt_history = {}
        
        for agent in self.market.agents:
            agent_worth_history[agent.agent_id] = []
            agent_cash_history[agent.agent_id] = []
            agent_portfolio_history[agent.agent_id] = []
            agent_debt_history[agent.agent_id] = []

        for day, status in enumerate(agent_statuses):
            for _, row in status.iterrows():
                agent_id = row["agent_id"]
                # Handle numpy types by converting to native Python types
                net_worth = float(row["net_worth"]) if hasattr(row["net_worth"], 'item') else row["net_worth"]
                cash = float(row["cash"]) if hasattr(row["cash"], 'item') else row["cash"]
                portfolio = float(row["portfolio_value"]) if hasattr(row["portfolio_value"], 'item') else row["portfolio_value"]
                debt = float(row["debt"]) if hasattr(row["debt"], 'item') else row["debt"]
                
                agent_worth_history[agent_id].append(net_worth)
                agent_cash_history[agent_id].append(cash)
                agent_portfolio_history[agent_id].append(portfolio)
                agent_debt_history[agent_id].append(debt)

        # Create multiple visualizations
        # 1. Main plot with all agents (can be crowded but shows overall picture)
        plt.figure(figsize=(15, 10))
        
        # Use a color map that distinguishes between different lines
        colors = plt.cm.tab20(np.linspace(0, 1, len(agent_worth_history)))
        
        for i, (agent_id, worth_history) in enumerate(agent_worth_history.items()):
            agent = self.market.agents[agent_id]
            plt.plot(worth_history, label=f"Agent {agent_id} ({agent.personality})", 
                     linewidth=2, color=colors[i])

        plt.title("All Agents: Net Worth Over Time", fontsize=16)
        plt.xlabel("Trading Day", fontsize=12)
        plt.ylabel("Net Worth ($)", fontsize=12)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("all_agents_performance.png")
        
        # 2. Plot by personality type (average performance of each type)
        personality_data = {}
        for agent_id, worth_history in agent_worth_history.items():
            agent = self.market.agents[agent_id]
            personality = agent.personality
            if personality not in personality_data:
                personality_data[personality] = []
            personality_data[personality].append(worth_history)
        
        plt.figure(figsize=(15, 8))
        
        for personality, histories in personality_data.items():
            # Average the performance of all agents with this personality
            avg_history = np.mean(histories, axis=0)
            plt.plot(avg_history, label=f"{personality}", linewidth=3)
        
        plt.title("Average Net Worth by Personality Type", fontsize=16)
        plt.xlabel("Trading Day", fontsize=12)
        plt.ylabel("Net Worth ($)", fontsize=12)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("personality_performance.png")
        
        # 3. Top 5 performing agents for clarity
        final_performances = [(agent_id, worth_history[-1]) for agent_id, worth_history in agent_worth_history.items()]
        top_performers = sorted(final_performances, key=lambda x: x[1], reverse=True)[:5]
        
        plt.figure(figsize=(15, 8))
        
        for agent_id, _ in top_performers:
            worth_history = agent_worth_history[agent_id]
            agent = self.market.agents[agent_id]
            plt.plot(worth_history, label=f"Agent {agent_id} ({agent.personality})", linewidth=3)
            
        plt.title("Top 5 Performing Agents: Net Worth Over Time", fontsize=16)
        plt.xlabel("Trading Day", fontsize=12)
        plt.ylabel("Net Worth ($)", fontsize=12)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("top_performers.png")
        
        # 4. NEW: Asset Allocation Over Time for Top Performers
        plt.figure(figsize=(20, 15))
        
        # Create a 3x2 grid for the top 5 agents (one plot will be empty)
        for idx, (agent_id, _) in enumerate(top_performers[:5]):
            agent = self.market.agents[agent_id]
            plt.subplot(3, 2, idx+1)
            
            # Stack plot with cash, portfolio, and debt
            days = range(len(agent_cash_history[agent_id]))
            plt.stackplot(days, 
                         [agent_cash_history[agent_id], agent_portfolio_history[agent_id]], 
                         labels=['Cash', 'Stock Portfolio'],
                         colors=['#66c2a5', '#fc8d62'])
            
            # Plot debt as a negative line
            plt.plot(days, [-d for d in agent_debt_history[agent_id]], 'r--', label='Debt', linewidth=2)
            
            # Plot net worth as a line
            plt.plot(days, agent_worth_history[agent_id], 'k-', label='Net Worth', linewidth=2)
            
            plt.title(f"Agent {agent_id} ({agent.personality}) Asset Allocation", fontsize=14)
            plt.xlabel("Trading Day", fontsize=10)
            plt.ylabel("Value ($)", fontsize=10)
            plt.grid(True)
            if idx == 0:  # Only add legend to the first subplot
                plt.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig("top_performers_asset_allocation.png")
        
        # 5. NEW: Net Worth Percentage Change
        plt.figure(figsize=(15, 8))
        
        # Calculate percentage change from initial net worth
        for i, (agent_id, _) in enumerate(top_performers[:5]):
            initial_worth = agent_worth_history[agent_id][0]
            pct_change = [((worth / initial_worth) - 1) * 100 for worth in agent_worth_history[agent_id]]
            agent = self.market.agents[agent_id]
            plt.plot(pct_change, label=f"Agent {agent_id} ({agent.personality})", linewidth=2)
        
        plt.title("Top 5 Agents: Net Worth Percentage Change", fontsize=16)
        plt.xlabel("Trading Day", fontsize=12)
        plt.ylabel("Percentage Change (%)", fontsize=12)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("top_performers_percentage_change.png")
        
        # 6. NEW: Portfolio Composition of Top Performers (Final Day)
        plt.figure(figsize=(15, 10))
        
        for i, (agent_id, _) in enumerate(top_performers[:5]):
            agent = self.market.agents[agent_id]
            
            # Get final day portfolio composition
            plt.subplot(2, 3, i+1)
            
            # Prepare data for pie chart
            labels = []
            sizes = []
            
            # Add cash
            labels.append('Cash')
            sizes.append(agent_cash_history[agent_id][-1])
            
            # Add stocks
            for symbol, quantity in agent.stocks.items():
                if quantity > 0:
                    stock_value = quantity * self.market.get_stock_prices()[symbol]
                    labels.append(f'Stock {symbol}')
                    sizes.append(stock_value)
            
            # Plot pie chart
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            plt.axis('equal')
            plt.title(f"Agent {agent_id} ({agent.personality}) Final Portfolio", fontsize=14)
        
        plt.tight_layout()
        plt.savefig("top_performers_portfolio_composition.png")
        
        # Calculate and report final performance metrics
        final_status = agent_statuses[-1].copy()  # Create a copy to avoid warnings
        
        # Calculate return as a Python float to avoid int64 issues
        initial_status = agent_statuses[0]
        for i, row in final_status.iterrows():
            agent_id = row["agent_id"]
            initial_net_worth = float(initial_status[initial_status["agent_id"] == agent_id]["net_worth"].values[0])
            final_net_worth = float(row["net_worth"])
            if initial_net_worth > 0:
                final_status.at[i, "total_return"] = (final_net_worth / initial_net_worth) - 1
            else:
                final_status.at[i, "total_return"] = 0.0

        # Group by personality type and calculate average performance
        personality_performance = final_status.groupby("personality").agg({
            "total_assets": "mean",
            "debt": "mean",
            "net_worth": "mean",
            "total_return": "mean",
            "bankrupted": "sum"
        }).reset_index()

        # Save to CSV
        final_status.to_csv("agent_final_performance.csv", index=False)
        personality_performance.to_csv("personality_performance.csv", index=False)
        
        # Print detailed results to console
        print("\n======= AGENT PERFORMANCE SUMMARY =======")
        print("\nTop 5 Performing Agents:")
        for i, (agent_id, final_worth) in enumerate(top_performers[:5], 1):
            agent = self.market.agents[agent_id]
            initial_worth = agent_worth_history[agent_id][0]
            pct_return = ((final_worth / initial_worth) - 1) * 100
            print(f"{i}. Agent {agent_id} ({agent.personality}): ${final_worth:.2f} - Return: {pct_return:.2f}%")
            
            # Print portfolio composition
            print(f"   Portfolio: ", end="")
            for symbol, quantity in agent.stocks.items():
                if quantity > 0:
                    stock_value = quantity * self.market.get_stock_prices()[symbol]
                    print(f"Stock {symbol}: {quantity} shares (${stock_value:.2f}), ", end="")
            print(f"Cash: ${agent.cash:.2f}")
        
        print("\nPersonality Type Average Performance:")
        for _, row in personality_performance.iterrows():
            print(f"{row['personality']}: Avg Net Worth ${row['net_worth']:.2f} - Avg Return: {row['total_return']*100:.2f}%")
        
        print("\nStock Performance Summary:")
        for symbol, stock in self.market.stocks.items():
            initial_price = stock.price_history[0]
            final_price = stock.price_history[-1]
            stock_return = ((final_price / initial_price) - 1) * 100
            print(f"Stock {symbol}: ${final_price:.2f} - Change: {stock_return:.2f}%")
        
    def generate_market_performance_report(self, market_statuses: List[pd.DataFrame]):
        """Generate a report on market performance."""
        # Extract price data for each stock
        stock_prices = {}
        for symbol in self.market.stocks.keys():
            stock_prices[symbol] = []

        for day, status in enumerate(market_statuses):
            for _, row in status.iterrows():
                symbol = row["symbol"]
                stock_prices[symbol].append(row["price"])  # Changed from current_price to price

        # Plot stock prices over time
        plt.figure(figsize=(12, 8))
        for symbol, prices in stock_prices.items():
            plt.plot(prices, label=f"Stock {symbol}")

        plt.title("Stock Prices Over Time")
        plt.xlabel("Trading Day")
        plt.ylabel("Price ($)")
        plt.legend()
        plt.grid(True)
        plt.savefig("stock_prices.png")

        # Calculate volatility and other metrics
        market_metrics = {}
        for symbol, prices in stock_prices.items():
            returns = [prices[i]/prices[i-1] - 1 for i in range(1, len(prices))]
            market_metrics[symbol] = {
                "starting_price": prices[0],
                "ending_price": prices[-1],
                "total_return": prices[-1]/prices[0] - 1,
                "volatility": np.std(returns) * np.sqrt(252),  # Annualized volatility
                "max_price": max(prices),
                "min_price": min(prices)
            }

        # Save to CSV
        pd.DataFrame(market_metrics).transpose().to_csv("market_metrics.csv")

    def generate_trading_activity_report(self):
        """Generate a report on trading activity."""
        # Create a DataFrame from trade history
        trades_df = pd.DataFrame(self.market.trade_history)

        if len(trades_df) == 0:
            print("No trades to report")
            return

        # Daily trading volume
        daily_volume = trades_df.groupby(["day", "symbol"])["quantity"].sum().unstack(fill_value=0)

        # Plot daily trading volume
        plt.figure(figsize=(12, 8))
        for symbol in daily_volume.columns:
            plt.plot(daily_volume.index, daily_volume[symbol], label=f"Stock {symbol}")

        plt.title("Daily Trading Volume")
        plt.xlabel("Trading Day")
        plt.ylabel("Number of Shares")
        plt.legend()
        plt.grid(True)
        plt.savefig("trading_volume.png")

        # Agent trading activity
        buyer_activity = trades_df.groupby("buyer_id")["quantity"].sum()
        seller_activity = trades_df.groupby("seller_id")["quantity"].sum()

        # Calculate net position for each agent
        net_position = pd.DataFrame({
            "bought": buyer_activity,
            "sold": seller_activity
        }).fillna(0)

        net_position["net"] = net_position["bought"] - net_position["sold"]

        # Save to CSV
        net_position.to_csv("agent_trading_activity.csv")

    def save_final_results(self, final_agent_status: pd.DataFrame, final_market_status: pd.DataFrame):
        """Save final simulation results."""
        # Convert all numeric types to native Python types to avoid JSON serialization issues
        final_agent_status_dict = final_agent_status.to_dict(orient='records')
        final_market_status_dict = final_market_status.to_dict(orient='records')
        
        # Convert all numpy types to native Python types
        agent_status_copy = []
        for agent in final_agent_status_dict:
            agent_copy = {}
            for key, value in agent.items():
                if hasattr(value, 'item'):  # Check if it's a numpy type
                    agent_copy[key] = value.item()  # Convert to native Python type
                else:
                    agent_copy[key] = value
            agent_status_copy.append(agent_copy)
            
        market_status_copy = []
        for market in final_market_status_dict:
            market_copy = {}
            for key, value in market.items():
                if hasattr(value, 'item'):  # Check if it's a numpy type
                    market_copy[key] = value.item()  # Convert to native Python type
                else:
                    market_copy[key] = value
            market_status_copy.append(market_copy)

        results = {
            "simulation_days": self.days_to_simulate,
            "num_agents": len(self.market.agents),
            "final_market_status": market_status_copy,
            "final_agent_status": agent_status_copy
        }

        with open("simulation_results.json", "w") as f:
            json.dump(results, f, indent=2)