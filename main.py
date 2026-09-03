"""Entry point: runs and orchestrates the wildfire management system agents."""
from agents.collection.news_monitoring_agent import NewsMonitoringAgent

if __name__ == "__main__":
    # Run the news monitoring agent in a loop
    news_monitoring_agent = NewsMonitoringAgent("configs/news_config.yaml")
    news_monitoring_agent.run_forever()
