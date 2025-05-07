import pandas as pd
from datetime import date

from time_series import RetTimeSeries
from db_manager import DB

class FactorPortfolio:
    def __init__(self, factor: str, index: str, num_quantiles: int, date_range: tuple):
        """
        Represents a portfolio constructed from a factor and a certain index.

        Args:
            factor (str): Factor names choose from:
                MarketCap
                Vola tes
                Momentum12
                Momentum6
                ForwardEPS
                TrailingEPS 
            index (str): Indices to choose from:
                S&P500
                S&P400
                S&P600
                EUROSTOXX
                EUROSTOXX50
                EUROSTOXX600
            num_quantiles (int): The number of portfolios to sort stocks into.
            date_range (tuple): Datetime date objects, start and end date of portfolio. 
        """

        self.factor = factor
        self.index = index
        self.num_quantiles = num_quantiles
        self.start_date, self.end_date = date_range

        self.years = self._years_between()
        self.quantiles = self._quantiles()

        self.constituents = self._get_constituents()
        self.quantile_consts = self._get_quantile_consts()
        self.returns_raw = self._get_returns()

        self.pct_returns = self._calc_pct_returns()
        self.TS = RetTimeSeries(self.pct_returns)

        self.cum_returns = self.TS.cum_returns()
        self.monthly_returns = self.TS.monthly_returns()
        self.annual_returns = self.TS.annual_returns()

    def _years_between(self):
        start, end = sorted((self.start_date.year, self.end_date.year))
        return list(range(start, end + 1))

    def _quantiles(self):
        return range(1, self.num_quantiles + 1)

    def _get_constituents(self, factor_lag=1):
        constituents = {}
        for year in self.years:
            query = f"""
                WITH params AS (
                  SELECT
                    '{self.factor}'      AS factor_name,
                    '{self.index}'    AS index_name,
                     {year} AS year,
                     {year-factor_lag} AS year_lagged
                )
                SELECT
                  c.id_stock,
                  f.value     AS factor_value
                FROM
                  params
                JOIN
                  index_mapper AS im
                    ON im.index_name = params.index_name
                JOIN
                  constituents AS c
                    ON c.id_index = im.id_index
                   AND c.year     = params.year
                JOIN
                  factor_mapper AS fm
                    ON fm.factor_name = params.factor_name
                JOIN
                  factors AS f
                    ON f.id_stock  = c.id_stock
                   AND f.id_factor = fm.id_factor
                   AND f.year      = params.year_lagged;
            """
            const_year = DB.fetch(query, output="df")
            constituents[year] = const_year
        return constituents
    
    def _get_quantile_consts(self):
        quantile_consts = {}
        for year in self.years:
            df = self.constituents[year].copy()

            df['quantile'] = pd.qcut(
                df['factor_value'],
                q=self.num_quantiles,
                labels=False,
                duplicates='drop'
            ) + 1

            quantile_consts[year] = {
                q: grp['id_stock'].tolist()
                for q, grp in df.groupby('quantile', sort=True)
            }
        return quantile_consts

    def _get_returns(self):
        returns = {}
        for year in self.years:
            quantiles = {}
            for quantile in self.quantiles:
                values = ", ".join(f"({const})" for const in self.quantile_consts[year][quantile])
                query = f"""
                    WITH stock_list(id_stock) AS (
                      VALUES
                        {values}
                    )
                    SELECT
                      r.id_stock,
                      r.date,
                      r.return
                    FROM
                      stock_list AS sl
                    JOIN
                      returns AS r
                        INDEXED BY idx_returns_stock_date
                          ON r.id_stock = sl.id_stock
                    WHERE
                      r.date BETWEEN '{year}-01-01' AND '{year}-12-31'
                    ORDER BY
                      r.id_stock,
                      r.date;
                """
                df = DB.fetch(query, output="df")

                df['date'] = pd.to_datetime(df['date'])
                ret_year = df.pivot(index='date', columns='id_stock', values='return')
                quantiles[quantile] = ret_year / 100
            returns[year] = quantiles
        return returns
    
    def _calc_pct_returns(self):
        pct_returns = {}
        for quantile in self.quantiles:
            pct_returns_quantile = []
            for year in self.years:
                returns_df = self.returns_raw[year][quantile]

                initial_date = returns_df.index.min() - pd.Timedelta(days=1)
                returns_with_initial = returns_df.copy()
                returns_with_initial.loc[initial_date] = 0
                returns_with_initial = returns_with_initial.sort_index()

                growth_index = (1 + returns_with_initial).cumprod() * 100

                growth_index['total'] = growth_index.sum(axis=1)
                pct_returns_quantile.append(growth_index['total'].pct_change().drop(initial_date))

            pct_returns[quantile] = pd.concat(pct_returns_quantile).sort_index()
        return pd.DataFrame(pct_returns)

    def premium(self, long_quantile: int, short_quantile: int, freq="d"):
        """
        Calculate the factor premium (long - short = premium).

        Args:
            long_quantile (int): Number of quantile to go long in.
            short_quantile (int): Number of quantile to go short in.
            freq (str): Frequency of time series d, m and y.
        """
        if freq == "d":
            returns = self.pct_returns.copy()
        elif freq == "m":
            returns = self.monthly_returns.copy()
        elif freq in ("y", "a"):
            returns = self.annual_returns.copy()
        return returns[long_quantile] - returns[short_quantile]
        

if __name__ == "__main__":
    dates = (date(2000, 1, 1), date(2023, 1, 1))
    for index in ["S&P500", "S&P400", "S&P600", "EUROSTOXX", "EUROSTOXX50", "EUROSTOXX600"]:
        for factor in ["MarketCap", "Vola", "Momentum12", "Momentum6", "ForwardEPS", "TrailingEPS"]:
            print(f"Factor: {factor} Index: {index}")
            portfolio = FactorPortfolio(factor, index, 5, dates)
            print(portfolio.pct_returns)
            print(portfolio.cum_returns)
            print(portfolio.monthly_returns)
            print(portfolio.annual_returns)
            print(portfolio.premium(1, 5, freq="d"))
            print(portfolio.premium(1, 5, freq="m"))
            print(portfolio.premium(1, 5, freq="y"))
            print("-"*40)

