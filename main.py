from pathlib import Path
from simulator.loader import MarketDataLoader
import yaml


def load_config(config_path: str) -> dict:
    """
    Loads configuration parameters from a YAML file.
    
    Args:
        config_path: Path to the .yaml file.
        
    Returns:
        dict: Parsed configuration dictionary.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    with open(path, "r") as file:
        return yaml.safe_load(file)


def main() -> None:
    # 1. Load parameters
    config = load_config("config.yaml")
    data_cfg = config.get("data", {})
    
    # 2. Initialize Data Loader
    loader = MarketDataLoader(
        tickers=data_cfg.get("tickers", []),
        start_date=data_cfg.get("start_date", ""),
        end_date=data_cfg.get("end_date", "")
    )
    
    # 3. Execute Data Fetching
    returns = loader.fetch_data()
    
    # Output check
    print(f"Data successfully loaded. Shape: {returns.shape}")
    print(returns.head())


if __name__ == "__main__":
    main()