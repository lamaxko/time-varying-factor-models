import os
from datetime import date
from preprocessing.factor_portfolio import FactorPortfolio

# Directory to save portfolio data
save_dir = "sdf_portfolios"
os.makedirs(save_dir, exist_ok=True)  # Create dir if it doesn't exist

dates = (date(2007, 1, 1), date(2023, 1, 1))

indices = ["S&P500", "S&P400", "S&P600"]
portfolio = FactorPortfolio("omega_lag_one_window_avrg", indices, 5, dates)
portfolio.create_plots(r"output/all/", long=5, short=1, freq_premium="y", annual_labels=False, monthly_trend=True)
for index in indices:
    portfolio = FactorPortfolio("omega_lag_one_window_avrg", [index,], 5, dates)
    portfolio.create_plots(fr"output/{index}/", long=5, short=1, freq_premium="y", annual_labels=False, monthly_trend=True)
