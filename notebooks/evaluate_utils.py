import numpy as np
from experts.markowitz import MarkowitzExpert

def evaluate_path(args):
    sim_stocks, config = args
    
    expert = MarkowitzExpert(
        risk_aversion=config["expert"]["risk_aversion"],
        rolling_window=config["expert"]["rolling_window"],
        max_weight=config["expert"]["max_weight"],
        annual_risk_free_rate=config["expert"]["annual_risk_free_rate"]
    )
    
    daily_rf = config["expert"]["annual_risk_free_rate"] / 252.0
    BPS_COST = 0.0005 # 5 basis points (0.05%) transaction cost
    
    # 1. Generate Expert Allocations
    weights_df = expert.generate_labels(sim_stocks)
    
    # 2. Align Returns and Inject Cash Asset
    returns_with_cash = sim_stocks.copy()
    returns_with_cash['Cash'] = daily_rf
    aligned_returns = returns_with_cash.loc[weights_df.index]
    
    # 3. Gross Daily Returns
    gross_daily_returns = (weights_df * aligned_returns).sum(axis=1)
    
    # 4. Daily Turnover
    turnover = weights_df.diff().abs().sum(axis=1) / 2.0
    turnover.iloc[0] = 0.0
    
    # 5. Net Daily Returns
    net_daily_returns = gross_daily_returns - (turnover * BPS_COST)
    
    # 6. Cumulative Returns
    gross_cum_return = (1 + gross_daily_returns).cumprod()
    net_cum_return = (1 + net_daily_returns).cumprod()
    
    # 7. Maximum Drawdown
    rolling_max = net_cum_return.cummax()
    drawdown = (net_cum_return - rolling_max) / rolling_max
    
    return {
        "Gross Return (x)": gross_cum_return.iloc[-1],
        "Net Return (x)": net_cum_return.iloc[-1],
        "Net Sharpe": np.sqrt(252) * (net_daily_returns.mean() - daily_rf) / net_daily_returns.std(),
        "Max Drawdown (%)": drawdown.min() * 100,
        "Daily Turnover (%)": turnover.mean() * 100
    }