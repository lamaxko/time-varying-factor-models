import os
import pandas as pd
from datetime import date
from backtests.factor_portfolio import FactorPortfolio

save_dir = "sdf_portfolios"
n_quantiles = 5
indices = ["S&P500", "S&P400", "S&P600"]
dates = (date(2007, 1, 1), date(2023, 1, 1))

portfolio = FactorPortfolio("omega_lag_one_window_avrg", indices, n_quantiles, dates)
# portfolio = FactorPortfolio("omega_lag_one", indices, n_quantiles, dates)
portfolio.create_plots(fr"output/all/", long=n_quantiles, short=1, freq_premium="y", annual_labels=False, monthly_trend=True)
portfolio.sanity_check()

