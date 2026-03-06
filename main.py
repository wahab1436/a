import os
import yaml
from monitoring.logging import setup_logging

def main():
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    logger = setup_logging(config['logging']['file'])
    logger.info("System Initialization Started")
    logger.info("Ready. Launch dashboard via: streamlit run dashboard/app.py")

if __name__ == "__main__":
    main()
