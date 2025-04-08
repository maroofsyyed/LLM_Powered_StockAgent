"""Configuration for the stock trading simulation."""
import random

class SimulationConfig:
    """Configuration for the stock trading simulation."""
    def __init__(self):
        # Simulation parameters
        self.num_agents = 10  # Default number of agents
        self.trading_days = 264  # One year of trading days
        self.quarters = 4  # Number of quarters in a year
        self.days_per_quarter = self.trading_days // self.quarters

        # Transaction costs
        self.transaction_fee_per_share = 0.005  # Currency units per share
        self.min_transaction_fee = 1.0  # Minimum fee per transaction
        self.max_transaction_fee = 5.95  # Maximum fee per transaction

        # Interest rates
        self.deposit_rate = 0.0  # Current deposit rate
        self.loan_rates = {
            1: 0.027,  # 1-month loan rate (2.7%)
            2: 0.030,  # 2-month loan rate (3.0%)
            3: 0.033,  # 3-month loan rate (3.3%)
        }

        # Market conditions that can change during simulation
        self.market_conditions = {
            "bull_market": False,  # Whether we're in a bull market
            "bear_market": False,  # Whether we're in a bear market
            "volatility": "moderate",  # Can be "low", "moderate", "high"
            "sector_rotation": None,  # Which sectors are favored (None, "tech", "energy", etc.)
            "interest_rate": 4.0,  # Current interest rate
            "sentiment": 0.5  # Market sentiment (0.0-1.0)
        }

        # Company fundamentals
        self.company_info = {
            "A": {
                "name": "Alpha Technologies",
                "sector": "Technology",
                "market_cap": "35B",
                "growth_rate": 0.15,
                "pe_ratio": 22,
                "dividend_yield": 0.01,
                "description": "Leading provider of cloud computing and AI solutions"
            },
            "B": {
                "name": "Beta Financial Group",
                "sector": "Financial",
                "market_cap": "28B",
                "growth_rate": 0.08,
                "pe_ratio": 16,
                "dividend_yield": 0.025,
                "description": "Diversified financial services company with strong presence in banking and investment management"
            }
        }

        # Initial stock settings
        self.initial_stock_prices = {
            "A": 30.0,  # Technology sector
            "B": 45.0,  # Financial sector
        }

        # Stock sector mappings
        self.stock_sectors = {
            "A": "Technology",
            "B": "Financial",
        }

        # Agent personality types
        self.personalities = [
            "Conservative", 
            "Balanced", 
            "Aggressive", 
            "Value-Focused", 
            "Growth-Oriented", 
            "Momentum-Driven", 
            "Contrarian"
        ]
        
        # Special events schedule (trading day -> event)
        self.events = {
            1: {"description": "Market opens with cautious optimism", "impact": 0.1},
            3: {"description": "Federal Reserve hints at potential interest rate changes", "impact": -0.08},
            5: {"description": "Alpha Technologies (Stock A) announces new AI product launch", "impact": 0.18},
            7: {"description": "Employment data released showing stronger than expected job growth", "impact": 0.12},
            9: {"description": "Beta Financial (Stock B) reports quarterly earnings above expectations", "impact": 0.15},
            11: {"description": "Market volatility increases due to geopolitical tensions", "impact": -0.17},
            12: {"description": "Reports of new virus outbreak in Asia trigger health emergency concerns", "impact": -0.25},
            13: {"description": "Tech sector rally on positive industry outlook", "impact": 0.14},
            14: {"description": "Government announces new tariffs on imported technology components", "impact": -0.22},
            15: {"description": "Inflation data comes in higher than expected", "impact": -0.13},
            16: {"description": "Military conflict escalates in Eastern Europe, disrupting energy markets", "impact": -0.30},
            17: {"description": "Financial regulators announce new oversight measures for banks", "impact": -0.11},
            18: {"description": "International trade agreement reached, reducing previous tariffs", "impact": 0.24},
            19: {"description": "Alpha Technologies (Stock A) faces patent infringement lawsuit", "impact": -0.16},
            20: {"description": "Major cybersecurity breach reported across financial institutions", "impact": -0.28},
            21: {"description": "Market stabilizes after previous volatility", "impact": 0.09},
            22: {"description": "Beta Financial (Stock B) announces expansion into emerging markets", "impact": 0.16},
            23: {"description": "Central bank announces economic stimulus package", "impact": 0.27},
            24: {"description": "Supply chain disruptions reported due to natural disaster in manufacturing hub", "impact": -0.21},
            25: {"description": "Breakthrough in renewable energy technology boosts market sentiment", "impact": 0.19}
        }

        # Global crisis events that can be triggered randomly
        self.crisis_events = [
            {"description": "Global pandemic fears intensify as new variant emerges", "impact": -0.45, "duration": 5},
            {"description": "Major financial institution declares bankruptcy", "impact": -0.38, "duration": 3},
            {"description": "International trade war escalates with new round of tariffs", "impact": -0.32, "duration": 4},
            {"description": "Major cyberattack disrupts global payment systems", "impact": -0.29, "duration": 2},
            {"description": "Geopolitical conflict escalates in oil-producing region", "impact": -0.36, "duration": 3},
            {"description": "Central banks coordinate emergency interest rate cut", "impact": 0.34, "duration": 4},
            {"description": "Unexpected peace agreement in conflict zone", "impact": 0.28, "duration": 3},
            {"description": "Technological breakthrough announced in AI sector", "impact": 0.31, "duration": 3},
            {"description": "Major merger announced between global tech companies", "impact": 0.27, "duration": 2},
            {"description": "Government announces massive infrastructure spending plan", "impact": 0.33, "duration": 4}
        ]
        
        # Company-specific major events
        self.company_events = {
            "A": [
                {"description": "Alpha Technologies announces revolutionary quantum computing breakthrough", "impact": 0.40},
                {"description": "Alpha Technologies facing major class-action lawsuit over data privacy", "impact": -0.35},
                {"description": "Alpha Technologies unexpectedly announces layoffs of 15% of workforce", "impact": -0.30},
                {"description": "Alpha Technologies secures major government contract for cloud services", "impact": 0.32},
                {"description": "Alpha Technologies CEO resigns amid accounting irregularities", "impact": -0.38}
            ],
            "B": [
                {"description": "Beta Financial Group announces acquisition of major fintech startup", "impact": 0.36},
                {"description": "Beta Financial Group under investigation for regulatory compliance issues", "impact": -0.34},
                {"description": "Beta Financial Group reports significant exposure to defaulting loans", "impact": -0.38},
                {"description": "Beta Financial Group secures exclusive partnership with global payment network", "impact": 0.33},
                {"description": "Beta Financial Group announces higher than expected dividend increase", "impact": 0.25}
            ]
        }

        # Ongoing crisis - can be set during simulation
        self.ongoing_crisis = None
        
        # Market conditions baselines
        self.base_interest_rate = 3.5  # Base interest rate %

        # Agent asset range
        self.min_initial_assets = 100000
        self.max_initial_assets = 5000000

    def get_event_for_day(self, day: int) -> dict:
        """Get special event for a given trading day if one exists."""
        # If there's a specific event for this day, return it
        if day in self.events:
            return self.events[day]
            
        # Otherwise, generate some market commentary based on the day
        if day % 5 == 0:  # Every 5 days, provide some general market commentary
            commentaries = [
                "Markets trading with low volatility today",
                "Trading volume below average as investors await catalyst",
                "Market sentiment mixed with sector rotation ongoing",
                "Technical analysts note market approaching key resistance levels",
                "Institutional investors reported to be increasing positions",
                "Retail trading activity shows increased interest in technology stocks",
                "Market breadth improving with advancing stocks outnumbering declining",
                "Foreign markets showing correlation with domestic movements",
                "Options market indicates hedging activity increasing",
                "Short interest declining across major indices"
            ]
            return {"description": random.choice(commentaries), "impact": 0}
            
        return {"description": "", "impact": 0}  # No event for this day

    def is_financial_report_day(self, day: int) -> bool:
        """Check if this is a financial report release day."""
        return day in [12, 78, 144, 210]

    def is_interest_payment_day(self, day: int) -> bool:
        """Check if this is an interest payment day (last day of each month)."""
        return day in [22, 44, 66, 88, 110, 132, 154, 176, 198, 220, 242, 264]
        
    def get_sector_performance(self, day):
        """Get sector performance for a specific day."""
        # Generate sector performance based on market conditions and events
        event = self.get_event_for_day(day)
        
        # Default sector performance - small random changes
        performance = {
            "Technology": round(random.uniform(-1.0, 1.0), 2),
            "Financial": round(random.uniform(-0.7, 0.9), 2)
        }
        
        # If there's a market event, adjust sector performance accordingly
        if event["description"]:
            event_description = event["description"].lower()
            event_impact = event["impact"]
            
            if "technology" in event_description or "alpha" in event_description or "stock a" in event_description:
                performance["Technology"] += random.uniform(1.0, 3.0) * event_impact
            if "financial" in event_description or "beta" in event_description or "stock b" in event_description:
                performance["Financial"] += random.uniform(1.0, 2.8) * event_impact
            
            # General market events affect all sectors to some degree
            if "market" in event_description or "economy" in event_description:
                for sector in performance:
                    performance[sector] += random.uniform(0.5, 1.5) * event_impact
        
        # Add cyclical behavior to certain sectors
        day_of_week = day % 5  # 0 = Monday, 4 = Friday
        if day_of_week == 0:  # Monday effects
            performance["Technology"] += random.uniform(-0.5, 0.8)  # Tech often starts week volatile
        elif day_of_week == 4:  # Friday effects
            performance["Financial"] += random.uniform(-0.2, 0.4)  # Financial often steadier at week end
        
        # Financial tends to be less volatile but reacts to specific events
        if day % 30 == 15:  # Mid-month financial policy effects
            performance["Financial"] += random.uniform(-1.5, 1.5)
            
        return performance
        
    def get_news_for_day(self, day):
        """Generate market news for a specific day."""
        event = self.get_event_for_day(day)
        
        # Base news items that appear regardless of events
        base_news = [
            f"Trading volume across markets {'increased' if random.random() > 0.5 else 'decreased'} by {random.randint(3, 15)}% compared to yesterday.",
            f"Analysts project {'positive' if random.random() > 0.5 else 'cautious'} outlook for next quarter."
        ]
        
        # Check for ongoing crisis
        crisis_news = []
        if self.ongoing_crisis:
            crisis_news.append(f"ONGOING CRISIS: {self.ongoing_crisis['description']} continues to impact markets.")
            # Add follow-up commentary based on how long the crisis has been ongoing
            days_ongoing = day - self.ongoing_crisis['start_day']
            if days_ongoing == 1:
                crisis_news.append("Markets continue to assess the full impact of this development.")
            elif days_ongoing == 2:
                crisis_news.append("Volatility remains elevated as uncertainty persists.")
            elif days_ongoing == 3:
                crisis_news.append("Some analysts suggest markets may be beginning to price in the full impact.")
            elif days_ongoing >= 4:
                crisis_news.append("Signs of stabilization emerging despite ongoing concerns.")
            
            # Check if crisis is ending today
            if day - self.ongoing_crisis['start_day'] >= self.ongoing_crisis['duration']:
                crisis_news.append(f"Analysts suggest the impact of {self.ongoing_crisis['description'].lower()} is starting to fade.")
                self.ongoing_crisis = None
        
        # Randomly trigger a new crisis (if no ongoing crisis and with low probability)
        elif random.random() < 0.05:  # 5% chance of a new crisis each day
            new_crisis = random.choice(self.crisis_events)
            self.ongoing_crisis = {
                "description": new_crisis["description"],
                "impact": new_crisis["impact"],
                "duration": new_crisis["duration"],
                "start_day": day
            }
            crisis_news.append(f"BREAKING: {new_crisis['description']}. Markets react strongly.")
            
            # Add sector-specific impact
            if "pandemic" in new_crisis["description"].lower():
                crisis_news.append("Health and technology sectors show divergent reactions; tech stocks more resilient while travel and leisure face pressure.")
            elif "financial" in new_crisis["description"].lower():
                crisis_news.append("Banking stocks falling sharply; contagion fears spreading to other financial institutions.")
            elif "trade war" in new_crisis["description"].lower() or "tariff" in new_crisis["description"].lower():
                crisis_news.append("Companies with international supply chains facing significant uncertainty; domestic-focused firms outperforming.")
            elif "conflict" in new_crisis["description"].lower() or "war" in new_crisis["description"].lower():
                crisis_news.append("Defense contractors seeing increased interest; energy prices volatile on supply concerns.")
            elif "cyber" in new_crisis["description"].lower():
                crisis_news.append("Cybersecurity firms rallying while affected sectors face operational challenges and potential regulatory scrutiny.")
        
        # Company-specific random events (5% chance per company per day)
        company_specific_news = []
        for symbol in ["A", "B"]:
            if random.random() < 0.05:  # 5% chance of a major company event
                company_event = random.choice(self.company_events[symbol])
                company_specific_news.append(f"BREAKING: {company_event['description']}.")
                
                # Additional commentary based on the event
                if company_event["impact"] > 0:
                    company_specific_news.append(f"Stock {symbol} seeing strong buying interest on this positive development.")
                else:
                    company_specific_news.append(f"Stock {symbol} under pressure as investors reassess valuation implications.")
        
        # Sector-specific news
        sector_performance = self.get_sector_performance(day)
        
        sector_news = []
        if sector_performance["Technology"] > 1.0:
            sector_news.append("Tech sector showing strong momentum, driven by software and semiconductor stocks.")
        elif sector_performance["Technology"] < -1.0:
            sector_news.append("Tech stocks under pressure amid concerns over valuation and future growth.")
            
        if sector_performance["Financial"] > 1.0:
            sector_news.append("Financial stocks rally on positive earnings reports and regulatory approvals.")
        elif sector_performance["Financial"] < -1.0:
            sector_news.append("Financial stocks decline amid concerns over interest rates and regulatory challenges.")
            
        # Company-specific news (regular updates)
        company_news = []
        
        # Alpha Technologies news
        if day % 7 == 2:
            if random.random() > 0.6:
                company_news.append(f"Alpha Technologies (Stock A) reported increased adoption of their cloud solutions.")
            else:
                company_news.append(f"Alpha Technologies (Stock A) facing {'increased' if random.random() > 0.5 else 'decreased'} competition in AI market.")
        
        # Beta Financial news
        if day % 7 == 3:
            if random.random() > 0.6:
                company_news.append(f"Beta Financial Group (Stock B) secured a major contract with an international client, boosting order visibility.")
            else:
                company_news.append(f"Beta Financial Group (Stock B) faces {'increased' if random.random() > 0.5 else 'decreased'} competition in key markets.")
        
        # Economic indicators randomly appearing
        if random.random() < 0.15:  # 15% chance each day
            indicators = [
                f"Unemployment rate {'rises' if random.random() > 0.5 else 'falls'} to {round(random.uniform(3.0, 8.0), 1)}%, {'above' if random.random() > 0.5 else 'below'} economist expectations.",
                f"GDP growth for the quarter reported at {round(random.uniform(-2.0, 5.0), 1)}%, {'exceeding' if random.random() > 0.5 else 'missing'} consensus forecasts.",
                f"Consumer confidence index {'improves' if random.random() > 0.5 else 'declines'} to {random.randint(70, 120)} from previous reading.",
                f"Manufacturing PMI comes in at {round(random.uniform(45.0, 60.0), 1)}, {'expansion' if random.random() > 50 else 'contraction'} territory.",
                f"Retail sales {'increased' if random.random() > 0.5 else 'decreased'} by {round(random.uniform(-3.0, 3.0), 1)}% month-over-month.",
                f"Housing starts {'up' if random.random() > 0.5 else 'down'} {round(random.uniform(1.0, 15.0), 1)}% from previous period."
            ]
            economic_news = random.choice(indicators)
            company_news.append(economic_news)
        
        # Event-driven news
        event_news = []
        if event["description"]:
            event_news.append(event["description"])
            
            # Add potential follow-up commentary on significant events
            if abs(event["impact"]) > 0.15:
                commentaries = [
                    f"Analysts are {'bullish' if event['impact'] > 0 else 'bearish'} following the recent {event['description'].lower()}.",
                    f"Market participants are reassessing portfolios after {event['description'].lower()}.",
                    f"Trading volume has {'increased' if event['impact'] > 0 else 'decreased'} significantly following this development."
                ]
                event_news.append(random.choice(commentaries))
        
        # Combine all news items with priority to major events
        all_news = crisis_news + company_specific_news + event_news + sector_news + company_news + base_news
        
        # Ensure we don't return too many news items
        return all_news[:min(len(all_news), 7)]  # Return up to 7 news items

    def get_market_conditions(self, day):
        """Get the current market conditions."""
        # This returns the current market conditions dictionary
        # The day parameter is included for consistency with other methods
        return self.market_conditions

    def update_market_conditions(self, day):
        """Update market conditions based on the day and events."""
        # Market conditions evolve over time and are influenced by events
        event = self.get_event_for_day(day)
        
        # Randomize base interest rate changes (happens infrequently)
        if day % 30 == 0:  # Roughly monthly adjustment potential
            if random.random() < 0.3:  # 30% chance of rate change
                rate_change = random.choice([-0.25, -0.125, 0.125, 0.25])
                self.market_conditions["interest_rate"] = max(0.25, min(8.0, self.market_conditions["interest_rate"] + rate_change))
                print(f"Interest rate changed to {self.market_conditions['interest_rate']}%")
        
        # Update market sentiment (mean-reverting with some randomness)
        prev_sentiment = self.market_conditions["sentiment"]
        # Mean reversion toward neutral (0.5)
        reversion = 0.05 * (0.5 - prev_sentiment)
        # Random noise
        noise = random.uniform(-0.05, 0.05)
        # Event impact
        event_impact = 0
        if event["description"]:
            event_impact = event["impact"] * 0.1  # Scale impact to sentiment range
        
        # Combine all factors
        new_sentiment = prev_sentiment + reversion + noise + event_impact
        # Ensure sentiment stays in range [0,1]
        self.market_conditions["sentiment"] = max(0.1, min(0.9, new_sentiment))
        
        # Update volatility (also mean-reverting with randomness)
        prev_volatility = self.market_conditions["volatility"]
        # Mean reversion toward medium (moderate)
        if prev_volatility == "low":
            if random.random() < 0.15:
                self.market_conditions["volatility"] = "moderate"
        elif prev_volatility == "high":
            if random.random() < 0.2:
                self.market_conditions["volatility"] = "moderate"
        else:  # moderate
            if random.random() < 0.1:
                self.market_conditions["volatility"] = random.choice(["low", "high"])
        
        # Events can directly change volatility
        if event["description"] and "rate" in event["description"].lower():
            self.market_conditions["volatility"] = "high"