from pathlib import Path
import yaml

from simulator.loader import MarketDataLoader
from simulator.logger import setup_logger

logger = setup_logger(__name__)

def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.is_file():
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    with open(path, "r") as file:
        return yaml.safe_load(file)

def main() -> None:
    logger.info("Initializing pipeline...")
    
    config = load_config("config.yaml")
    data_cfg = config.get("data", {})
    
    logger.info(f"Loading data for tickers: {data_cfg.get('tickers', [])}")
    
    loader = MarketDataLoader(
        tickers=data_cfg.get("tickers", []),
        start_date=data_cfg.get("start_date", ""),
        end_date=data_cfg.get("end_date", "")
    )
    
    try:
        returns = loader.fetch_data()
        logger.info(f"Data successfully fetched. Shape: {returns.shape}")
    except Exception as e:
        logger.error(f"Data fetching failed: {e}")
        raise

if __name__ == "__main__":
    main()