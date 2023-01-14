import unittest

from nuccorrelation.correlation_analysis_handler import CorrelationAnalysisHandler

COMMON_COUNTRIES = [
    'Argentina',
    'Belgium',
    'Brazil',
    'Bulgaria',
    'Canada',
    'China',
    'Finland',
    'France',
    'Germany',
    'Hungary',
    'India',
    'Japan',
    'Luxembourg',
    'Mexico',
    'Netherlands',
    'Pakistan',
    'Romania',
    'Russian Federation',
    'Slovenia',
    'South Africa',
    'Spain',
    'Sweden',
    'Switzerland',
    'Total World',
    'Ukraine',
    'United Kingdom'
]

PATH_BP_STATS = "../raw-data/bp-stats-review.csv"
PATH_CO2 = "../raw-data/co2_kt_world.csv"
PATH_POP = "../raw-data/world_pop.csv"


class CorrelationAnalysisHandlerTest(unittest.TestCase):
    def test_from_frames_min_year(self):
        handler = CorrelationAnalysisHandler.from_paths(
            PATH_BP_STATS,
            PATH_CO2,
            PATH_POP
        )
        self.assertEqual(
            handler.min_year,
            1990
        )

    def test_from_frames_max_year(self):
        handler = CorrelationAnalysisHandler.from_paths(
            PATH_BP_STATS,
            PATH_CO2,
            PATH_POP
        )
        self.assertEqual(
            handler.max_year,
            2019
        )

    def test_from_frames_countries(self):
        handler = CorrelationAnalysisHandler.from_paths(
            PATH_BP_STATS,
            PATH_CO2,
            PATH_POP
        )
        self.assertSequenceEqual(
            handler.common_countries,
            COMMON_COUNTRIES
        )

    def test_from_frames_df_nuc_columns(self):
        handler = CorrelationAnalysisHandler.from_paths(
            PATH_BP_STATS,
            PATH_CO2,
            PATH_POP
        )
        self.assertSequenceEqual(
            [str(s) for s in handler.df_nuc_twh.columns],
            COMMON_COUNTRIES
        )

    def test_from_frames_df_nuc_primary_share_columns(self):
        handler = CorrelationAnalysisHandler.from_paths(
            PATH_BP_STATS,
            PATH_CO2,
            PATH_POP
        )
        self.assertSequenceEqual(
            [str(s) for s in handler.df_nuc_primary_share.columns],
            COMMON_COUNTRIES
        )

    def test_from_frames_df_co2_columns(self):
        handler = CorrelationAnalysisHandler.from_paths(
            PATH_BP_STATS,
            PATH_CO2,
            PATH_POP
        )
        self.assertSequenceEqual(
            [str(s) for s in handler.df_co2.columns],
            COMMON_COUNTRIES
        )

    def test_from_frames_df_pop_columns(self):
        handler = CorrelationAnalysisHandler.from_paths(
            PATH_BP_STATS,
            PATH_CO2,
            PATH_POP
        )
        self.assertSequenceEqual(
            [str(s) for s in handler.df_pop.columns],
            COMMON_COUNTRIES
        )

    def test_from_frames_df_co2_shape(self):
        handler = CorrelationAnalysisHandler.from_paths(
            PATH_BP_STATS,
            PATH_CO2,
            PATH_POP
        )
        self.assertEqual(
            handler.df_co2.shape,
            (30, 26)
        )

    def test_from_frames_df_co2_nuc_prim_share(self):
        handler = CorrelationAnalysisHandler.from_paths(
            PATH_BP_STATS,
            PATH_CO2,
            PATH_POP
        )
        self.assertEqual(
            handler.df_nuc_primary_share.shape,
            (30, 26)
        )

    def test_from_frames_df_nuc_shape(self):
        handler = CorrelationAnalysisHandler.from_paths(
            PATH_BP_STATS,
            PATH_CO2,
            PATH_POP
        )
        self.assertEqual(
            handler.df_nuc_twh.shape,
            (30, 26)
        )

    def test_from_frames_df_prim_shape(self):
        handler = CorrelationAnalysisHandler.from_paths(
            PATH_BP_STATS,
            PATH_CO2,
            PATH_POP
        )
        self.assertEqual(
            handler.df_primary.shape,
            (30, 26)
        )

    def test_from_frames_df_pop_shape(self):
        handler = CorrelationAnalysisHandler.from_paths(
            PATH_BP_STATS,
            PATH_CO2,
            PATH_POP
        )
        self.assertEqual(
            handler.df_pop.shape,
            (30, 26)
        )

    def test_from_frames_df_no_nans_co2(self):
        handler = CorrelationAnalysisHandler.from_paths(
            PATH_BP_STATS,
            PATH_CO2,
            PATH_POP
        )
        self.assertTrue(
            handler.df_co2.isna().sum().sum() == 0
        )

    def test_from_frames_df_no_nans_nuc(self):
        handler = CorrelationAnalysisHandler.from_paths(
            PATH_BP_STATS,
            PATH_CO2,
            PATH_POP
        )
        self.assertTrue(
            handler.df_nuc_primary_share.isna().sum().sum() == 0
        )

    def test_from_frames_df_no_nans_nuc_primary(self):
        handler = CorrelationAnalysisHandler.from_paths(
            PATH_BP_STATS,
            PATH_CO2,
            PATH_POP
        )
        self.assertTrue(
            handler.df_nuc_twh.isna().sum().sum() == 0
        )

    def test_from_frames_df_no_nans_pop(self):
        handler = CorrelationAnalysisHandler.from_paths(
            PATH_BP_STATS,
            PATH_CO2,
            PATH_POP
        )
        self.assertTrue(
            handler.df_pop.isna().sum().sum() == 0
        )

    def test_get_pearson_corr_for_country_dfs(self):
        handler = CorrelationAnalysisHandler.from_paths(
            PATH_BP_STATS,
            PATH_CO2,
            PATH_POP
        )
        self.assertAlmostEqual(
            handler.get_pearson_corr_for_country_dfs(
                handler.df_co2, handler.df_nuc_twh)["Romania"],
            -0.863207115
        )

    def test_get_pearson_corr_for_country_dfs_index(self):
        handler = CorrelationAnalysisHandler.from_paths(
            PATH_BP_STATS,
            PATH_CO2,
            PATH_POP
        )
        self.assertCountEqual(
            handler.get_pearson_corr_for_country_dfs(
                handler.df_co2, handler.df_nuc_twh).index,
            COMMON_COUNTRIES
        )

    def test_get_aggregated_stats_by_country_shape(self):
        handler = CorrelationAnalysisHandler.from_paths(
            PATH_BP_STATS,
            PATH_CO2,
            PATH_POP
        )
        self.assertEqual(
            handler.get_aggregated_stats_by_country().shape,
            (5, 26)
        )

    def test_get_aggregated_stats_by_country_colnames(self):
        handler = CorrelationAnalysisHandler.from_paths(
            PATH_BP_STATS,
            PATH_CO2,
            PATH_POP
        )
        self.assertCountEqual(
            handler.get_aggregated_stats_by_country().columns,
            COMMON_COUNTRIES
        )

    def test_get_aggregated_stats_by_country_index(self):
        handler = CorrelationAnalysisHandler.from_paths(
            PATH_BP_STATS,
            PATH_CO2,
            PATH_POP
        )
        self.assertCountEqual(
            handler.get_aggregated_stats_by_country().index,
            ["COR: CO2 v. Nuclear in TWh", "COR: CO2 vs nuclear share in primary",
             "COR: Nuclear in TWh vs share in primary", "Trend: Primary energy consumption",
             "Trend: Nuclear energy share"]
        )

    def test_get_trend_coefs_by_country_shape(self):
        handler = CorrelationAnalysisHandler.from_paths(
            PATH_BP_STATS,
            PATH_CO2,
            PATH_POP
        )
        self.assertEqual(
            handler.get_trend_coefs_by_country(handler.df_primary).shape,
            (26, )
        )

    def test_get_trend_coefs_by_country_value(self):
        handler = CorrelationAnalysisHandler.from_paths(
            PATH_BP_STATS,
            PATH_CO2,
            PATH_POP
        )
        self.assertAlmostEqual(
            handler.get_trend_coefs_by_country(
                handler.df_primary / handler.df_pop)["Pakistan"],
            0.90482224
        )


if __name__ == '__main__':
    unittest.main()
