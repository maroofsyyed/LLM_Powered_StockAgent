"""Main entry point for StockAgent simulation.

This script provides an easy way to run the stock agent simulation
with different parameters and configurations.
"""
import os
import json
import argparse

from Run import run_stockagent_simulation
from config import LLM_API_KEYS, DEFAULT_PROVIDER, DEFAULT_MODEL


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run StockAgent simulation")
    parser.add_argument(
        "--api-key", 
        type=str, 
        help="API key for the LLM provider (override the one in config.py)"
    )
    parser.add_argument(
        "--provider", 
        type=str, 
        default=DEFAULT_PROVIDER,
        choices=["openai", "gemini", "deepseek"], 
        help=f"LLM provider (default: {DEFAULT_PROVIDER})"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default=DEFAULT_MODEL, 
        help=f"Model to use (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "-a", "--agents",
        type=int,
        default=20,
        help="Number of trading agents"
    )
    parser.add_argument(
        "--days", 
        type=int, 
        default=5, 
        help="Number of trading days to simulate (default: 22)"
    )
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        help="Path to checkpoint file to resume from"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the application."""
    args = parse_arguments()
    
    # Get API key - prioritize command line arg, then environment variable, then config file
    api_key = args.api_key or os.environ.get("STOCKAGENT_API_KEY") or LLM_API_KEYS.get(args.provider)
    
    if not api_key:
        print(f"Error: No API key found for provider '{args.provider}'")
        print("Please either:")
        print("  1. Update the API key in config.py")
        print("  2. Set STOCKAGENT_API_KEY environment variable")
        print("  3. Provide API key via --api-key command line argument")
        return
    
    # Run simulation
    system, agent_statuses, market_statuses = run_stockagent_simulation(
        api_key=api_key,
        provider=args.provider,
        model=args.model,
        num_agents=args.agents,
        days_to_simulate=args.days,
        load_checkpoint=args.checkpoint
    )
    
    print("Simulation complete. Results available in:")
    print("  - agent_performance.png")
    print("  - stock_prices.png")
    print("  - trading_volume.png")
    print("  - agent_final_performance.csv")
    print("  - personality_performance.csv")
    print("  - market_metrics.csv")
    print("  - simulation_results.json")


if __name__ == "__main__":
    main()