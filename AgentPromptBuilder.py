"""Builds prompts for agent interactions with LLMs."""
from typing import Dict
import json
import numpy as np

from Agent import Agent
from StockMarket import StockMarket
from SimulationConfig import SimulationConfig


class AgentPromptBuilder:
    """Builds prompts for agent interactions."""

    @staticmethod
    def get_crisis_affected_sectors(crisis_description: str) -> str:
        """Identify sectors most affected by the current crisis."""
        crisis_description = crisis_description.lower()
        
        if "pandemic" in crisis_description:
            return "Travel, Hospitality, Entertainment (negative impact); Healthcare, Technology (mixed impact)"
        elif "financial" in crisis_description or "bankruptcy" in crisis_description:
            return "Banking, Insurance, Investment Services (severe negative impact)"
        elif "trade war" in crisis_description or "tariff" in crisis_description:
            return "Manufacturing, Technology Hardware, Consumer Goods (negative impact); Domestic Services (relatively protected)"
        elif "cyber" in crisis_description:
            return "Financial Services, Retail, Healthcare (negative impact); Cybersecurity (positive impact)"
        elif "conflict" in crisis_description or "war" in crisis_description:
            return "Energy, Defense (mixed impact); Tourism, International Trade (negative impact)"
        else:
            return "Various sectors depending on crisis specifics; defensive sectors like Utilities and Consumer Staples typically less affected"

    @staticmethod
    def get_crisis_strategy_recommendation(crisis_description: str, days_ongoing: int) -> str:
        """Provide strategy recommendations based on crisis type and duration."""
        crisis_description = crisis_description.lower()
        
        # Early crisis phase (high uncertainty)
        if days_ongoing < 2:
            if "pandemic" in crisis_description:
                return "Consider defensive positioning; evaluate stocks with remote work/digital capabilities"
            elif "financial" in crisis_description:
                return "Reduce exposure to financial sector; prioritize cash preservation"
            elif "trade war" in crisis_description or "tariff" in crisis_description:
                return "Evaluate supply chain exposure in portfolio companies; consider domestic-focused alternatives"
            elif "cyber" in crisis_description:
                return "Assess technology dependencies in holdings; consider cybersecurity sector opportunities"
            elif "conflict" in crisis_description or "war" in crisis_description:
                return "Monitor commodities exposure; assess energy supply disruption impacts"
            else:
                return "Maintain higher cash levels; delay major position changes until impact is clearer"
        
        # Mid-crisis phase (adaptation)
        elif days_ongoing < 4:
            if "pandemic" in crisis_description:
                return "Selective buying in quality companies adapting to changed environment"
            elif "financial" in crisis_description:
                return "Watch for stabilization signals; consider quality companies with strong balance sheets"
            elif "trade war" in crisis_description:
                return "Position in companies with adaptable supply chains and pricing power"
            elif "cyber" in crisis_description:
                return "Look for companies implementing stronger security measures; avoid those with ongoing vulnerabilities"
            elif "conflict" in crisis_description:
                return "Balance defensive positions with selective opportunities in affected sectors showing resilience"
            else:
                return "Begin selective positioning in high-quality affected names with long-term resilience"
        
        # Late crisis phase (recovery positioning)
        else:
            if "pandemic" in crisis_description:
                return "Consider recovery plays in quality affected sectors; maintain positions in digital winners"
            elif "financial" in crisis_description:
                return "Evaluate opportunities in stronger financial institutions; watch for policy support benefits"
            elif "trade war" in crisis_description:
                return "Look for companies successfully navigating new trade landscape; consider resolution beneficiaries"
            elif "cyber" in crisis_description:
                return "Focus on companies demonstrating enhanced security measures and operational recovery"
            elif "conflict" in crisis_description:
                return "Position for normalization while maintaining awareness of structural changes to affected markets"
            else:
                return "Begin transitioning to recovery positioning while maintaining risk management discipline"

    @staticmethod
    def build_loan_decision_prompt(agent: Agent, market: StockMarket, config: SimulationConfig) -> str:
        """Build a prompt for loan decision."""
        stock_prices = market.get_stock_prices()
        total_assets = agent.calculate_total_assets(stock_prices)

        # Get recent BBS messages
        recent_messages = market.bbs.get_recent_messages(market.current_day)
        bbs_text = "\n".join([f"Agent: {msg['agent_id']}: {msg['message']}" for msg in recent_messages])
        
        # Get market conditions
        market_conditions = ""
        if config.market_conditions["bull_market"]:
            market_conditions += "MARKET CONDITION: Bull market - prices trending upward overall. "
        elif config.market_conditions["bear_market"]:
            market_conditions += "MARKET CONDITION: Bear market - prices trending downward overall. "
        
        market_conditions += f"Volatility is {config.market_conditions['volatility']}. "
        
        if config.market_conditions["sector_rotation"]:
            market_conditions += f"There is sector rotation favoring {config.market_conditions['sector_rotation']} stocks. "

        # Construct stock information
        stock_info = []
        for symbol in market.stocks:
            company = config.company_info[symbol]
            stock_info.append(f"Stock {symbol} - {company['name']} ({company['sector']}): "
                              f"Growth rate: {company['growth_rate']*100:.1f}%, "
                              f"P/E: {company['pe_ratio']}, "
                              f"Dividend yield: {company['dividend_yield']*100:.1f}%")
        
        stock_info_text = "\n".join(stock_info)

        prompt = f"""
You are a stock trader with {agent.personality} personality. Your current information:
- Cash: ${agent.cash:.2f}
- Total Assets: ${total_assets:.2f}
- Current Stocks: {', '.join([f"{symbol}: {qty} shares" for symbol, qty in agent.stocks.items()])}
- Current Stock Prices: {', '.join([f"{symbol}: ${price:.2f}" for symbol, price in stock_prices.items()])}
- Trading Day: {market.current_day}

{market_conditions}

Company Information:
{stock_info_text}

Stock market background:
{config.get_event_for_day(market.current_day)}

Last day's bulletin board messages:
{bbs_text}

Loan information:
- 1-month loan: {config.loan_rates[1] * 100:.1f}% interest rate
- 2-month loan: {config.loan_rates[2] * 100:.1f}% interest rate
- 3-month loan: {config.loan_rates[3] * 100:.1f}% interest rate
- You can borrow up to 50% of your total assets

As a {agent.personality} trader, decide whether to take a loan. If yes, specify the loan type (1, 2, or 3 months) and amount.
Respond in valid JSON format:
{{"loan": "yes/no", "loan_type": integer (1, 2, or 3), "amount": float}}
        """
        return prompt

    @staticmethod
    def build_trading_decision_prompt(agent: Agent, market: StockMarket, session: int, config: SimulationConfig) -> str:
        """Construct a prompt for the agent's trading decision."""
        day = market.current_day
        
        # Get market conditions
        market_conditions = config.get_market_conditions(day)
        sector_performance = config.get_sector_performance(day)
        
        # Get market news and events
        news_items = config.get_news_for_day(day)
        
        # Calculate agent's current portfolio
        stock_summary = {}
        total_assets = agent.calculate_total_assets(market.get_stock_prices())
        cash_pct = round(agent.cash / total_assets * 100, 2)
        
        for symbol, quantity in agent.stocks.items():
            if quantity > 0:
                stock = market.stocks[symbol]
                holding_value = quantity * stock.current_price
                holding_pct = round(holding_value / total_assets * 100, 2)
                avg_price = agent.get_average_purchase_price(symbol)
                
                profit_loss = 0
                profit_loss_pct = 0
                if avg_price > 0:
                    profit_loss = round((stock.current_price - avg_price) * quantity, 2)
                    profit_loss_pct = round((stock.current_price - avg_price) / avg_price * 100, 2)
                
                stock_summary[symbol] = {
                    "quantity": quantity,
                    "current_price": stock.current_price,
                    "avg_price": avg_price,
                    "holding_value": holding_value,
                    "holding_pct": holding_pct,
                    "profit_loss": profit_loss,
                    "profit_loss_pct": profit_loss_pct
                }
        
        # Technical analysis indicators (simple models)
        technical_indicators = {}
        for symbol, stock in market.stocks.items():
            price_data = stock.price_history
            if len(price_data) >= 10:
                # Calculate moving averages
                ma5 = sum(price_data[-5:]) / 5
                ma10 = sum(price_data[-10:]) / 10
                
                # Simple trend determination
                current_price = stock.current_price
                price_5days_ago = price_data[-5] if len(price_data) >= 5 else price_data[0]
                
                trend = "neutral"
                if current_price > ma5 and ma5 > ma10:
                    trend = "bullish"
                elif current_price < ma5 and ma5 < ma10:
                    trend = "bearish"
                
                # Calculate volatility (standard deviation of returns)
                returns = [price_data[i]/price_data[i-1] - 1 for i in range(1, len(price_data))]
                volatility = round(np.std(returns) * 100, 2) if len(returns) > 0 else 0
                
                # Calculate price momentum (percentage change over last 5 days)
                momentum = round((current_price - price_5days_ago) / price_5days_ago * 100, 2)
                
                technical_indicators[symbol] = {
                    "ma5": round(ma5, 2),
                    "ma10": round(ma10, 2),
                    "trend": trend,
                    "volatility": volatility,
                    "momentum": momentum
                }
        
        # Now prepare the specific sections of the prompt
        # Market overview section
        market_overview = f"""MARKET OVERVIEW:
Market sentiment: {market_conditions['sentiment']:.2f} on a scale of 0-1
Market volatility: {market_conditions['volatility']}
Interest rate: {config.base_interest_rate:.2f}%
"""
        
        # Sector performance section
        sector_list = []
        for sector, performance in sector_performance.items():
            sector_list.append(f"{sector}: {performance:+.2f}%")
        
        sector_performance_section = "SECTOR PERFORMANCE:\n" + "\n".join(sector_list)
        
        # Market news and events
        news_section = "MARKET NEWS AND EVENTS:\n"
        for news_item in news_items:
            news_section += f"- {news_item}\n"
        
        # Crisis assessment section - only include if there's an ongoing crisis
        crisis_section = ""
        if hasattr(config, 'ongoing_crisis') and config.ongoing_crisis:
            crisis_info = config.ongoing_crisis
            days_ongoing = day - crisis_info['start_day']
            crisis_section = f"""
MAJOR CRISIS ASSESSMENT:
- Active Crisis: {crisis_info['description']}
- Duration: Day {days_ongoing + 1} of expected {crisis_info['duration']} days
- Remaining Impact: {'High' if days_ongoing < 2 else 'Moderate' if days_ongoing < 4 else 'Diminishing'}
- Sectors Most Affected: {AgentPromptBuilder.get_crisis_affected_sectors(crisis_info['description'])}
- Recommended Strategy: {AgentPromptBuilder.get_crisis_strategy_recommendation(crisis_info['description'], days_ongoing)}
"""
        
        # Portfolio summary
        portfolio_section = f"""YOUR PORTFOLIO SUMMARY:
Agent ID: {agent.agent_id}
Cash: ${agent.cash:.2f} ({cash_pct}% of portfolio)
Total assets: ${total_assets:.2f}
"""
        
        # Add stocks details
        if len(stock_summary) > 0:
            portfolio_section += "\nSTOCK HOLDINGS:\n"
            for symbol, details in stock_summary.items():
                portfolio_section += f"Stock {symbol}: {details['quantity']} shares @ ${details['current_price']:.2f} (Avg: ${details['avg_price']:.2f})\n"
                portfolio_section += f"   Value: ${details['holding_value']:.2f} ({details['holding_pct']}% of portfolio)\n"
                portfolio_section += f"   P/L: ${details['profit_loss']:+.2f} ({details['profit_loss_pct']:+.2f}%)\n"
        else:
            portfolio_section += "\nNo stocks currently held.\n"
        
        # Loan details
        if len(agent.loans) > 0:
            portfolio_section += "\nLOANS:\n"
            for i, loan in enumerate(agent.loans):
                loan_type = loan.get('type', 'Standard')  # Default to "Standard" if type is missing
                portfolio_section += f"Loan {i+1}: ${loan['amount']:.2f} at {loan['interest_rate']:.2f}% (Type: {loan_type})\n"
        
        # Technical analysis
        technical_section = "TECHNICAL ANALYSIS:\n"
        for symbol, indicators in technical_indicators.items():
            technical_section += f"Stock {symbol}:\n"
            technical_section += f"   Current price: ${market.stocks[symbol].current_price:.2f}\n"
            technical_section += f"   5-day MA: ${indicators['ma5']:.2f}, 10-day MA: ${indicators['ma10']:.2f}\n"
            technical_section += f"   Trend: {indicators['trend'].upper()}\n"
            technical_section += f"   Volatility: {indicators['volatility']}%\n"
            technical_section += f"   Momentum: {indicators['momentum']:+.2f}%\n"
            
            # Add volume information
            volume_today = market.stocks[symbol].volume_history[-1] if market.stocks[symbol].volume_history else 0
            avg_volume_5day = sum(market.stocks[symbol].volume_history[-5:]) / 5 if len(market.stocks[symbol].volume_history) >= 5 else 0
            
            technical_section += f"   Today's volume: {volume_today} shares\n"
            if avg_volume_5day > 0:
                vol_change = (volume_today / avg_volume_5day - 1) * 100
                technical_section += f"   Volume vs 5-day avg: {vol_change:+.2f}%\n"
        
        # Trading session info
        trading_session = f"""TRADING SESSION INFO:
Current trading day: {day}
Current session: {session} of 3
"""
        
        # Company fundamentals
        fundamentals_section = "COMPANY FUNDAMENTALS:\n"
        for symbol, stock in market.stocks.items():
            company_info = config.company_info[symbol]
            fundamentals_section += f"Stock {symbol} ({company_info['name']}):\n"
            fundamentals_section += f"   Sector: {config.stock_sectors[symbol]}\n"
            fundamentals_section += f"   P/E ratio: {company_info['pe_ratio']:.2f}\n"
            fundamentals_section += f"   Dividend yield: {company_info['dividend_yield']:.2f}%\n"
            fundamentals_section += f"   Business: {company_info.get('description', 'No description available')}\n"
        
        # Agent personality
        personality_section = f"""YOUR TRADING PERSONALITY:
{agent.personality}
"""
        
        # BBS insights
        bbs_section = "MARKET BULLETIN BOARD (BBS) INSIGHTS:\n"
        recent_messages = market.bbs.get_recent_messages(5)
        if recent_messages:
            for msg in recent_messages:
                # Safely access message keys with default values
                day = msg.get('day', 'Unknown Day')
                agent_id = msg.get('agent_id', 'Unknown Agent')
                message = msg.get('message', 'No message')
                bbs_section += f"Day {day}, Agent {agent_id}: {message}\n"
        else:
            bbs_section += "No recent messages on the bulletin board.\n"
        
        # Decision framework
        decision_framework = """DECISION FRAMEWORK:
Based on the information above, make a decision for this trading session.

Your response must be in JSON format with the following structure:
{
  "thought_process": "string", // Your detailed analysis and reasoning
  "action_type": "string", // Must be one of: "buy", "sell", or "hold"
  "stock": "string", // Required if action is "buy" or "sell" (stock symbol)
  "amount": number, // Required if action is "buy" or "sell" (number of shares)
  "price": number, // Required if action is "buy" or "sell" (price per share)
  "confidence": number // Your confidence level (0-1) in this decision
}

Consider all the information provided, current market conditions, your portfolio, global events, technical indicators, and your trading personality.
DO NOT include any additional text before or after the JSON response.
"""
        
        # Combine all sections
        full_prompt = f"""{personality_section}

{trading_session}

{market_overview}

{sector_performance_section}

{news_section}

{crisis_section}

{portfolio_section}

{technical_section}

{fundamentals_section}

{bbs_section}

{decision_framework}
"""
        
        return full_prompt

    @staticmethod
    def build_future_action_prompt(agent: Agent, market: StockMarket) -> str:
        """Build a prompt for predicting future actions."""
        stock_prices = market.get_stock_prices()

        prompt = f"""
Based on today's trading (Day {market.current_day}), estimate your likely actions for tomorrow:
- Current Cash: ${agent.cash:.2f}
- Current Stock Prices: {', '.join([f"{symbol}: ${price:.2f}" for symbol, price in stock_prices.items()])}
- Current Portfolio: {', '.join([f"{symbol}: {qty} shares" for symbol, qty in agent.stocks.items()])}

As a {agent.personality} trader, estimate whether you will buy/sell stocks tomorrow or take a loan.
Respond in valid JSON format:
{{"buy_A": "yes/no", "buy_B": "yes/no", "buy_C": "yes/no", "sell_A": "yes/no", "sell_B": "yes/no", "sell_C": "yes/no", "loan": "yes/no"}}
        """
        return prompt

    @staticmethod
    def build_bbs_message_prompt(agent: Agent, market: StockMarket) -> str:
        """Build a prompt for generating a BBS message."""
        stock_prices = market.get_stock_prices()
        
        prompt = f"""
As a stock trader with a {agent.personality} personality, briefly post your trading tips on the forum.
Current Stock Prices: {', '.join([f"{symbol}: ${price:.2f}" for symbol, price in stock_prices.items()])}

Keep your message concise (max 100 words) and focus on what you believe will happen to Stocks A and B
in the near future based on your analysis of market conditions, recent events, and stock performance.

Your post should reflect your {agent.personality} approach to investing and should be opinionated.
        """
        return prompt