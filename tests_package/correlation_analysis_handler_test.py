import unittest
import numpy as np
from nuccorrelation.correlation_analysis_handler import CorrelationAnalysisHandler, euclidean_distance

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

COMMON_COUNTRIES_BEFORE_PREPROCESSING = [
    "Colombia",
    "Austria",
    "Papua New Guinea",
    "China",
    "Equatorial Guinea",
    "Philippines",
    "Germany",
    "Cyprus",
    "Azerbaijan",
    "Turkmenistan",
    "Kuwait",
    "Total World",
    "Trinidad & Tobago",
    "Italy",
    "Bahrain",
    "Peru",
    "Oman",
    "Croatia",
    "Belarus",
    "Mexico",
    "Thailand",
    "Ecuador",
    "Iceland",
    "Tunisia",
    "Japan",
    "Australia",
    "Algeria",
    "Chile",
    "Ireland",
    "Morocco",
    "Bangladesh",
    "Denmark",
    "Spain",
    "Madagascar",
    "Sudan",
    "Poland",
    "Lithuania",
    "Finland",
    "Zimbabwe",
    "Malaysia",
    "Greece",
    "Canada",
    "Saudi Arabia",
    "Kazakhstan",
    "Pakistan",
    "Hungary",
    "Singapore",
    "Argentina",
    "Brazil",
    "New Zealand",
    "Sweden",
    "Mongolia",
    "Curacao",
    "United Kingdom",
    "Belgium",
    "Chad",
    "Netherlands",
    "South Africa",
    "Serbia",
    "Turkey",
    "India",
    "Romania",
    "Slovenia",
    "New Caledonia",
    "Gabon",
    "Nigeria",
    "Sri Lanka",
    "Cuba",
    "Switzerland",
    "Bulgaria",
    "France",
    "Uzbekistan",
    "Libya",
    "North Macedonia",
    "Norway",
    "Vietnam",
    "Ukraine",
    "Mozambique",
    "Qatar",
    "Venezuela",
    "Myanmar",
    "Iraq",
    "Estonia",
    "Indonesia",
    "Israel",
    "Bolivia",
    "Zambia",
    "Angola",
    "South Sudan",
    "Luxembourg",
    "United Arab Emirates",
    "Russian Federation",
    "Portugal",
    "Latvia"
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
            (7, 26)
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
             "COR: Nuclear in TWh vs share in primary", "Trend: Primary energy production",
             "Trend: Nuclear energy share", "DTW: CO2 vs nuclear share in primary",
             "DTW: CO2 vs nuclear share in primary (Normalized)"]
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

    def test_get_dtw_distance_for_country_dfs(self):
        handler = CorrelationAnalysisHandler.from_paths(
            PATH_BP_STATS,
            PATH_CO2,
            PATH_POP
        )
        self.assertCountEqual(
            handler.get_dtw_distance_for_country_dfs(
                handler.df_co2, handler.df_nuc_primary_share).index,
            COMMON_COUNTRIES
        )
    
    def test_common_countries_before_preprocessing(self):
        handler = CorrelationAnalysisHandler.from_paths(
            PATH_BP_STATS,
            PATH_CO2,
            PATH_POP
        )
        self.maxDiff = None
        self.assertCountEqual(
            handler._common_countries_before_preprocessing,
            COMMON_COUNTRIES_BEFORE_PREPROCESSING
        )
    
    def test_co2_before_preprocessing_shape(self):
        handler = CorrelationAnalysisHandler.from_paths(
            PATH_BP_STATS,
            PATH_CO2,
            PATH_POP
        )
        self.maxDiff = None
        self.assertEqual(
            handler.df_co2_before_preprocessing.shape,
            (62, 94)
        )
    
    def test_co2_before_preprocessing_columns(self):
        handler = CorrelationAnalysisHandler.from_paths(
            PATH_BP_STATS,
            PATH_CO2,
            PATH_POP
        )
        self.maxDiff = None
        self.assertCountEqual(
            [str(s) for s in handler.df_co2_before_preprocessing.columns],
            COMMON_COUNTRIES_BEFORE_PREPROCESSING
        )


if __name__ == '__main__':
    unittest.main()
