"""Agent class representing trading agents in the stock market simulation."""
from typing import Dict, List, Tuple


class Agent:
    """Represents a trading agent with a personality, assets, and trading behavior."""
    def __init__(self, agent_id: int, personality: str, initial_cash: float):
        self.agent_id = agent_id
        self.personality = personality
        self.cash = initial_cash
        self.stocks = {}  # Stock symbol -> quantity
        self.loans = []  # List of (amount, duration, start_day, interest_rate) tuples
        self.transaction_history = []  # List of trades made
        self.bankrupted = False

    def calculate_total_assets(self, stock_prices: Dict[str, float]) -> float:
        """Calculate total assets (cash + stock value)."""
        if self.bankrupted:
            return 0

        stock_value = sum(qty * stock_prices[symbol] for symbol, qty in self.stocks.items())
        return self.cash + stock_value

    def calculate_portfolio_value(self, stock_prices: Dict[str, float]) -> float:
        """Calculate only the stock portfolio value (without cash)."""
        if self.bankrupted:
            return 0
            
        return sum(qty * stock_prices[symbol] for symbol, qty in self.stocks.items())

    def calculate_net_worth(self, stock_prices: Dict[str, float]) -> float:
        """Calculate net worth (total assets - debt)."""
        total_assets = self.calculate_total_assets(stock_prices)
        total_debt = self.calculate_total_debt()
        return total_assets - total_debt

    def calculate_total_debt(self) -> float:
        """Calculate the total outstanding debt."""
        return sum(loan["amount"] for loan in self.loans if not loan["repaid"])

    def get_average_purchase_price(self, symbol: str) -> float:
        """Calculate the average purchase price for a given stock symbol."""
        if symbol not in self.stocks or self.stocks[symbol] <= 0:
            return 0
            
        # Filter buy transactions for this symbol
        buy_transactions = [t for t in self.transaction_history 
                          if t.get("action") == "buy" and t.get("symbol") == symbol]
        
        if not buy_transactions:
            return 0
            
        # Calculate total cost and shares
        total_cost = sum(t["price"] * t["quantity"] for t in buy_transactions)
        total_shares = sum(t["quantity"] for t in buy_transactions)
        
        if total_shares == 0:
            return 0
            
        return total_cost / total_shares

    def add_stock(self, symbol: str, quantity: int):
        """Add stock to the agent's portfolio."""
        if symbol not in self.stocks:
            self.stocks[symbol] = 0
        self.stocks[symbol] += quantity

    def remove_stock(self, symbol: str, quantity: int) -> bool:
        """Remove stock from the agent's portfolio. Return False if insufficient quantity."""
        if symbol not in self.stocks or self.stocks[symbol] < quantity:
            return False
        self.stocks[symbol] -= quantity
        return True

    def add_loan(self, amount: float, duration: int, start_day: int, interest_rate: float):
        """Add a new loan to the agent."""
        self.loans.append({
            "amount": amount,
            "duration": duration,
            "start_day": start_day,
            "interest_rate": interest_rate,
            "repaid": False,
            "type": duration  # Use duration as the loan type (1=short, 2=medium, 3=long)
        })
        self.cash += amount

    def process_interest_payments(self, current_day: int) -> float:
        """Process interest payments for all outstanding loans. Returns total interest paid."""
        total_interest = 0

        for loan in self.loans:
            if loan["repaid"]:
                continue

            # Calculate interest for the period
            interest = loan["amount"] * loan["interest_rate"] / 12  # Monthly interest

            # Check if we have enough cash to pay interest
            if self.cash >= interest:
                self.cash -= interest
                total_interest += interest
            else:
                # Not enough cash to pay interest
                return -1  # Indicate bankruptcy

        return total_interest

    def process_loan_repayments(self, current_day: int) -> bool:
        """Process loan repayments for matured loans. Returns False if bankrupt."""
        for loan in self.loans:
            if loan["repaid"]:
                continue

            # Check if loan has matured
            if current_day - loan["start_day"] >= loan["duration"] * 22:  # Assuming 22 trading days per month
                # Repay loan principal
                if self.cash >= loan["amount"]:
                    self.cash -= loan["amount"]
                    loan["repaid"] = True
                else:
                    # Not enough cash to repay loan
                    return False  # Indicate bankruptcy

        return True

    def go_bankrupt(self, stock_prices: Dict[str, float]):
        """Process bankruptcy - liquidate all assets."""
        # Liquidate all stocks
        for symbol, qty in list(self.stocks.items()):
            self.cash += qty * stock_prices[symbol]
            self.stocks[symbol] = 0

        # Mark all loans as repaid (they will be written off)
        for loan in self.loans:
            loan["repaid"] = True

        # Set bankrupt flag
        self.bankrupted = True