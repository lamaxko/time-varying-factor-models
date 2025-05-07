from factor_portfolio import FactorPortfolio
from datetime import date
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

def plot_all_portfolio_data(portfolio, factor, index):
    # Ensure output folder exists
    output_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(output_dir, exist_ok=True)

    # Color scheme: Light to dark blue
    blue_shades = ["#cce5ff", "#99ccff", "#66b2ff", "#3399ff", "#0073e6"]

    fig, axes = plt.subplots(4, 2, figsize=(18, 16))
    fig.suptitle(f"Factor: {factor} | Index: {index}", fontsize=20)

    # Plot configs
    plots = [
        (portfolio.pct_returns, "Daily Percentage Returns", "Return", axes[0, 0], "line"),
        (portfolio.cum_returns, "Cumulative Returns", "Cumulative Return", axes[0, 1], "line"),
        (portfolio.monthly_returns, "Monthly Returns", "Monthly Return", axes[1, 0], "line"),
        (portfolio.annual_returns, "Annual Returns", "Annual Return", axes[1, 1], "bar_grouped"),
        (portfolio.premium(1, 5, freq="d"), "Daily Premium (P1 - P5)", "Premium", axes[2, 0], "line"),
        (portfolio.premium(1, 5, freq="m"), "Monthly Premium (P1 - P5)", "Premium", axes[2, 1], "line"),
        (portfolio.premium(1, 5, freq="y"), "Annual Premium (P1 - P5)", "Premium", axes[3, 0], "bar_single"),
        (portfolio.premium(5, 1, freq="y"), "Annual Premium (P5 - P1)", "Premium", axes[3, 1], "bar_single"),
    ]

    for data, title, ylabel, ax, kind in plots:
        if isinstance(data, pd.DataFrame):
            if kind == "line":
                for i, col in enumerate(data.columns):
                    data[col].plot(ax=ax, label=f"Q{col}", color=blue_shades[i % 5])
            elif kind == "bar_grouped":
                width = 0.15
                x = np.arange(len(data.index))
                for i, col in enumerate(data.columns):
                    ax.bar(x + i * width, data[col].values, width=width, label=f"Q{col}", color=blue_shades[i % 5])
                ax.set_xticks(x + width * 2)
                ax.set_xticklabels([str(label) for label in data.index], rotation=45)
        else:  # Series
            if kind == "line":
                data.plot(ax=ax, label="P1 - P5", color="#004080")
            elif kind == "bar_single":
                data.plot(kind="bar", ax=ax, label=title.split(" ")[-1], color="#004080", width=0.6)
                ax.set_xticklabels([str(label) for label in data.index], rotation=45)

        ax.set_title(title, fontsize=14)
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    filename = f"{factor}_{index}_portfolio_plots.png".replace(" ", "_")
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close()


if __name__ == "__main__":
    dates = (date(2000, 1, 1), date(2023, 1, 1))
    for index in ["S&P500", "S&P400", "S&P600", "EUROSTOXX", "EUROSTOXX50", "EUROSTOXX600"]:
        for factor in ["MarketCap", "Vola", "Momentum12", "Momentum6", "ForwardEPS", "TrailingEPS"]:
            portfolio = FactorPortfolio(factor, index, 5, dates)
            print(f"Factor: {factor} Index: {index}")
            plot_all_portfolio_data(portfolio, factor, index)
