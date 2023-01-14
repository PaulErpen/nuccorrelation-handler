from typing import Dict, List
import pandas as pd
from scipy.stats import pearsonr
from sklearn import linear_model
import numpy as np
from sklearn.preprocessing import minmax_scale

ASSUMED_NUC_REACTOR_EFFICIENCY = 0.38


class CorrelationAnalysisHandler():
    min_year: int
    max_year: int
    common_countries: List[str]
    df_nuc_twh: pd.DataFrame
    df_nuc_primary_share: pd.DataFrame
    df_co2: pd.DataFrame
    df_primary: pd.DataFrame
    df_pop: pd.DataFrame

    def __init__(self,
                 min_year: int,
                 max_year: int,
                 common_countries: List[str],
                 df_nuc_twh: pd.DataFrame,
                 df_nuc_primary_share: pd.DataFrame,
                 df_co2: pd.DataFrame,
                 df_primary: pd.DataFrame,
                 df_pop: pd.DataFrame) -> None:
        self.min_year = min_year
        self.max_year = max_year
        self.common_countries = common_countries
        self.df_nuc_twh = df_nuc_twh
        self.df_nuc_primary_share = df_nuc_primary_share
        self.df_co2 = df_co2
        self.df_primary = df_primary
        self.df_pop = df_pop

    @classmethod
    def from_paths(cls,
                   df_bp_stats_path: str,
                   df_co2_path: str,
                   df_pop_path: str) -> "CorrelationAnalysisHandler":

        df_bp_stats_raw = pd.read_csv(df_bp_stats_path)

        df_nuc_temp = (df_bp_stats_raw
                       .pivot(index='Country', columns='Year', values='nuclear_twh')
                       .transpose())

        df_primary_temp = (df_bp_stats_raw
                           .pivot(index='Country', columns='Year', values='primary_ej')
                           .transpose())

        WORLD_BANK_DROP_COLS: List[str] = ["Country Code", "Indicator Name", "Indicator Code", "Unnamed: 66"]
        WORLD_BANK_RENAME_COUNTRIES: Dict[str, str] = {
            "Trinidad and Tobago": "Trinidad & Tobago",
            "Turkiye": "Turkey",
            "Venezuela, RB": "Venezuela",
            "World": "Total World"
        }

        df_co2_temp = (
            pd.read_csv(df_co2_path, skiprows=3)
            .drop(WORLD_BANK_DROP_COLS, axis="columns")
            .set_index("Country Name")
            .transpose()
        )
        df_co2_temp.index = df_co2_temp.index.astype("int64")
        df_co2_temp.rename(WORLD_BANK_RENAME_COUNTRIES, inplace=True, axis=1)

        df_pop_temp = (
            pd.read_csv(df_pop_path, skiprows=3)
            .drop(WORLD_BANK_DROP_COLS, axis="columns")
            .set_index("Country Name")
            .transpose()
        )
        df_pop_temp.index = df_pop_temp.index.astype("int64")
        df_pop_temp.rename(WORLD_BANK_RENAME_COUNTRIES, inplace=True, axis=1)

        years_co2 = [
            int(str(i)) for i in df_co2_temp[df_co2_temp.notna().sum(axis=1) != 0].index]
        min_year = min(years_co2)
        max_year = max(years_co2)

        df_co2_temp = df_co2_temp.loc[min_year:max_year]
        df_co2_temp = df_co2_temp[df_co2_temp.columns[df_co2_temp.isna(
        ).sum() == 0]]

        df_nuc_temp = df_nuc_temp.loc[min_year:max_year]
        df_nuc_temp = df_nuc_temp[df_nuc_temp.columns[df_nuc_temp.isna(
        ).sum() == 0]]

        df_primary_temp = df_primary_temp.loc[min_year:max_year]
        df_primary_temp = df_primary_temp[df_primary_temp.columns[df_primary_temp.isna(
        ).sum() == 0]]

        df_primary_temp = df_primary_temp.applymap(
            lambda x: to_twh("primary_ej", x))

        df_nuc_primary_share = (df_nuc_temp
                                / ASSUMED_NUC_REACTOR_EFFICIENCY
                                / df_primary_temp)

        df_pop_temp = df_pop_temp.loc[min_year:max_year]
        df_pop_temp = df_pop_temp[df_pop_temp.columns[df_pop_temp.isna(
        ).sum() == 0]]

        common_countries = sorted(list({str(s) for s in df_nuc_temp.columns}.intersection(
            {str(s) for s in df_co2_temp.columns})))

        return cls(
            min_year=min_year,
            max_year=max_year,
            common_countries=common_countries,
            df_nuc_twh=df_nuc_temp[common_countries],
            df_nuc_primary_share=df_nuc_primary_share[common_countries],
            df_co2=df_co2_temp[common_countries],
            df_primary=df_primary_temp[common_countries],
            df_pop=df_pop_temp[common_countries]
        )

    def get_pearson_corr_for_country_dfs(self, df_1: pd.DataFrame, df_2: pd.DataFrame) -> pd.Series:
        values = []
        for country in self.common_countries:
            stat, other = pearsonr(df_1[country], df_2[country])
            values.append(stat)
        return pd.Series(values, index=self.common_countries)

    def get_trend_coefs_by_country(self, y: pd.DataFrame) -> pd.Series:
        beta = []
        x = np.reshape([i for i in range(0, y.shape[0])], (-1, 1)) / y.shape[0]
        y_scale = minmax_scale(y)
        for country in self.common_countries:
            lin_mod = linear_model.LinearRegression()
            lin_mod.fit(x, y_scale[:,self.common_countries.index(country)])
            beta.append(lin_mod.coef_[0])
        return pd.Series(beta, index=self.common_countries)

    def get_aggregated_stats_by_country(self) -> pd.DataFrame:
        series_co2_nuc = self.get_pearson_corr_for_country_dfs(
            self.df_nuc_twh, self.df_co2)
        series_c02_nuc_prim = self.get_pearson_corr_for_country_dfs(
            self.df_co2, self.df_nuc_primary_share)
        series_nuc_nuc_prim = self.get_pearson_corr_for_country_dfs(
            self.df_nuc_twh, self.df_nuc_primary_share)
        series_beta_trends = self.get_trend_coefs_by_country(self.df_primary / self.df_pop)
        series_beta_nuc_trends = self.get_trend_coefs_by_country(self.df_nuc_primary_share)
        df_corrs = pd.concat(
            [
                series_co2_nuc,
                series_c02_nuc_prim,
                series_nuc_nuc_prim,
                series_beta_trends,
                series_beta_nuc_trends
            ],
            axis=1
        ).transpose()
        df_corrs["ind"] = ["COR: CO2 v. Nuclear in TWh", "COR: CO2 vs nuclear share in primary",
                           "COR: Nuclear in TWh vs share in primary", "Trend: Primary energy consumption",
                           "Trend: Nuclear energy share"]
        df_corrs = df_corrs.set_index("ind")
        return df_corrs


def to_twh(colname: str, value: float) -> float:
    if colname.endswith("_twh"):
        return value
    if colname.endswith("_ej"):
        return value * 277.778
    if colname.endswith("_pj"):
        return value * 277.778
    raise Exception(
        f"column with name \"{colname}\" cannot be converted to twh! no known conversion!")

