"""Main execution script for the StockAgent simulation."""
import json
from LLMInterface import LLMInterface
from StockAgentSystem import StockAgentSystem
import argparse
import random
from SimulationConfig import SimulationConfig
from StockMarket import StockMarket


def run_stockagent_simulation(
    api_key: str,
    provider: str = "openai",
    model: str = "gpt-3.5-turbo",  # Add model parameter
    num_agents: int = 20,
    days_to_simulate: int = 10,
    load_checkpoint: str = None
):
    """Run a complete StockAgent simulation.
    
    Args:
        api_key: The API key for the LLM provider
        provider: The LLM provider (openai, gemini, or deepseek)
        model: The model to use (e.g., "gpt-3.5-turbo" for OpenAI)
        num_agents: Number of agents in the simulation
        days_to_simulate: Number of trading days to simulate
        load_checkpoint: Optional path to a checkpoint file to resume from
    
    Returns:
        tuple: (system, agent_statuses, market_statuses)
    """
    # Initialize LLM interface with specified model
    llm = LLMInterface(provider=provider, api_key=api_key, model=model)

    # Initialize simulation system
    system = StockAgentSystem(
        llm_interface=llm,
        num_agents=num_agents,
        days_to_simulate=days_to_simulate
    )

    # Load checkpoint if provided
    if load_checkpoint:
        system.load_checkpoint(load_checkpoint)
        print(f"Loaded checkpoint: {load_checkpoint}")

    # Run simulation
    print(f"Starting simulation with {num_agents} agents for {days_to_simulate} days...")
    agent_statuses, market_statuses = system.run_simulation()

    print("Simulation completed successfully!")
    return system, agent_statuses, market_statuses


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Stock Trading Simulation")
    parser.add_argument("--api-key", type=str, help="API key for the LLM provider", required=False)
    parser.add_argument("--provider", type=str, default="openai", choices=["openai", "gemini", "deepseek"], help="LLM provider")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="Model to use (e.g., gpt-3.5-turbo for OpenAI)")
    parser.add_argument("--agents", type=int, default=10, help="Number of agents in the simulation")
    parser.add_argument("--days", type=int, default=20, help="Number of trading days to simulate")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file to resume from")
    parser.add_argument("--demo", action="store_true", help="Run in demo mode without LLM API calls")
    
    args = parser.parse_args()
    
    if args.demo:
        print("\n=== Running in DEMO mode (no API calls) ===")
        # In demo mode, we'll create a simple simulation system manually
        from SimulationConfig import SimulationConfig
        from StockMarket import StockMarket
        
        # Initialize simulation components
        config = SimulationConfig()
        config.trading_days = args.days
        config.num_agents = args.agents
        
        # Create market
        market = StockMarket(config)
        
        # Simulate market
        print(f"\nStarting demo simulation with {args.agents} agents for {args.days} days...")
        print("This simulation uses only stocks A (Alpha Technologies) and B (Beta Financial Group)")
        
        # Simple simulation loop
        for day in range(args.days):
            # Update prices with simple random movements
            for symbol, stock in market.stocks.items():
                direction = 1 if random.random() > 0.5 else -1
                movement = random.uniform(0.5, 2.0) * direction
                new_price = max(1.0, stock.current_price * (1 + movement/100))
                stock.update_price(new_price)
            
            # Update agent cash (simple interest)
            for agent in market.agents:
                agent.cash *= 1.0005  # Small daily interest
                
            if day % 5 == 0:
                print(f"Day {day}: Stock A: ${market.stocks['A'].current_price:.2f}, Stock B: ${market.stocks['B'].current_price:.2f}")
        
        print("\nDemo simulation completed!")
        print("For full simulation with AI agents, please provide an API key:")
        print("python Run.py --api-key YOUR_API_KEY --days 25 --agents 10")
        
    elif not args.api_key:
        print("\n=== ERROR: API key is required for full simulation ===")
        print("Usage: python Run.py --api-key YOUR_API_KEY [options]")
        print("For a simple demo without API calls: python Run.py --demo")
        
    else:
        system, agent_statuses, market_statuses = run_stockagent_simulation(
            api_key=args.api_key,
            provider=args.provider,
            model=args.model,
            num_agents=args.agents,
            days_to_simulate=args.days,
            load_checkpoint=args.checkpoint
        )
        
        # The simulation reports are generated automatically inside run_stockagent_simulation