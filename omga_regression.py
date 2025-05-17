import pandas as pd
from datetime import date
from backtests.factor_portfolio import FactorPortfolio
from apis.kfrench import KfApi

# Setup factor portfolio
portfolio = FactorPortfolio("omega_lag_one_window_avrg", ["S&P500", "S&P400", "S&P600"], 5, (date(2007, 1, 1), date(2023, 1, 1)))
returns = portfolio.monthly_returns
returns.index = pd.to_datetime(returns.index, format='%Y-%m').to_period('M')

# Load RF data and format
rf = KfApi().fetch_data_ff5().iloc[:, [0, -1]]
rf.columns = ['Date', 'RF']

# Step-by-step fix to avoid TypeError
rf['Date'] = pd.to_datetime(rf['Date'].astype(str), format='%Y%m')  # → now datetime
rf.set_index('Date', inplace=True)  # → now datetime index
rf.index = rf.index.to_period('M')  # → convert to period index safely

# Align and calculate excess returns
rf['RF'] = rf['RF'] / 100.0
rf = rf.loc[returns.index]
excess_returns = returns.subtract(rf['RF'], axis=0)

print(excess_returns)
