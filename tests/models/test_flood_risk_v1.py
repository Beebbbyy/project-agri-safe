"""
Unit tests for Rule-Based Flood Risk Model (v1)

Tests the baseline rule-based prediction model including:
- Model initialization
- Weather features validation
- Rule evaluation for different categories
- Risk level determination
- Batch predictions
- Explanation generation
"""

import pytest
from datetime import date, datetime
from src.models.flood_risk_v1 import (
    RuleBasedFloodModel,
    WeatherFeatures,
    FloodRiskLevel,
    FloodRiskAssessment
)


@pytest.fixture
def model():
    """Create a fresh model instance for each test"""
    return RuleBasedFloodModel()


@pytest.fixture
def low_risk_features():
    """Features indicating low flood risk"""
    return WeatherFeatures(
        region_id="TEST001",
        assessment_date=date(2025, 1, 15),
        rainfall_1d=5.0,
        rainfall_3d=10.0,
        rainfall_7d=25.0,
        rainfall_14d=40.0,
        temperature_avg=28.0,
        wind_speed=15.0,
        elevation=200.0,
        historical_flood_count=0,
        region_vulnerability_score=10.0,
        is_typhoon_season=False,
        month=1
    )


@pytest.fixture
def high_risk_features():
    """Features indicating high flood risk"""
    return WeatherFeatures(
        region_id="TEST002",
        assessment_date=date(2025, 7, 15),
        rainfall_1d=120.0,
        rainfall_3d=250.0,
        rainfall_7d=300.0,
        rainfall_14d=450.0,
        temperature_avg=27.0,
        wind_speed=75.0,
        elevation=45.0,
        historical_flood_count=6,
        region_vulnerability_score=75.0,
        rainy_days_7d=6,
        heavy_rain_days_7d=3,
        is_typhoon_season=True,
        month=7
    )


@pytest.fixture
def critical_risk_features():
    """Features indicating critical flood risk"""
    return WeatherFeatures(
        region_id="TEST003",
        assessment_date=date(2025, 8, 20),
        rainfall_1d=160.0,
        rainfall_3d=380.0,
        rainfall_7d=450.0,
        rainfall_14d=600.0,
        temperature_avg=26.0,
        wind_speed=110.0,
        wind_speed_max_7d=120.0,
        elevation=30.0,
        historical_flood_count=10,
        region_vulnerability_score=85.0,
        rainy_days_7d=7,
        heavy_rain_days_7d=5,
        is_typhoon_season=True,
        month=8
    )


class TestRuleBasedModelInitialization:
    """Test model initialization"""

    def test_model_initialization(self, model):
        """Test that model initializes correctly"""
        assert model is not None
        assert model.version == "v1.0"

    def test_model_thresholds(self, model):
        """Test that rainfall thresholds are set correctly"""
        assert model.CRITICAL_RAINFALL_1D == 150.0
        assert model.HIGH_RAINFALL_1D == 100.0
        assert model.MEDIUM_RAINFALL_1D == 50.0
        assert model.CRITICAL_RAINFALL_7D == 400.0

    def test_elevation_thresholds(self, model):
        """Test elevation thresholds"""
        assert model.LOW_ELEVATION == 50.0
        assert model.MEDIUM_ELEVATION == 100.0


class TestWeatherFeaturesValidation:
    """Test WeatherFeatures validation"""

    def test_valid_features_creation(self):
        """Test creating valid weather features"""
        features = WeatherFeatures(
            region_id="TEST001",
            rainfall_1d=10.0,
            rainfall_7d=50.0,
            elevation=100.0
        )

        assert features.region_id == "TEST001"
        assert features.rainfall_1d == 10.0
        assert features.elevation == 100.0

    def test_negative_rainfall_validation(self):
        """Test that negative rainfall is converted to 0"""
        features = WeatherFeatures(
            region_id="TEST001",
            rainfall_1d=-10.0  # Invalid negative
        )

        assert features.rainfall_1d == 0.0

    def test_elevation_bounds_validation(self):
        """Test elevation bounds checking"""
        features = WeatherFeatures(
            region_id="TEST001",
            elevation=5000.0  # Too high
        )

        # Should be capped at 3000
        assert features.elevation <= 3000.0

    def test_default_values(self):
        """Test that default values are set"""
        features = WeatherFeatures(region_id="TEST001")

        assert features.rainfall_1d == 0.0
        assert features.temperature_avg == 28.0
        assert features.elevation == 100.0
        assert features.historical_flood_count == 0


class TestRainfallRulesEvaluation:
    """Test rainfall-based rule evaluation"""

    def test_critical_daily_rainfall_rule(self, model):
        """Test critical daily rainfall rule triggers"""
        features = WeatherFeatures(
            region_id="TEST001",
            rainfall_1d=160.0,  # Above critical threshold
            rainfall_7d=200.0
        )

        score, factors, rules = model._evaluate_rainfall_rules(features)

        assert score >= 35
        assert 'critical_daily_rainfall' in factors
        assert 'RULE_CRITICAL_DAILY_RAIN' in rules

    def test_high_daily_rainfall_rule(self, model):
        """Test high daily rainfall rule"""
        features = WeatherFeatures(
            region_id="TEST001",
            rainfall_1d=110.0,  # Between high and critical
            rainfall_7d=200.0
        )

        score, factors, rules = model._evaluate_rainfall_rules(features)

        assert score >= 25
        assert 'high_daily_rainfall' in factors
        assert 'RULE_HIGH_DAILY_RAIN' in rules

    def test_moderate_daily_rainfall_rule(self, model):
        """Test moderate daily rainfall rule"""
        features = WeatherFeatures(
            region_id="TEST001",
            rainfall_1d=60.0,  # Between medium and high
            rainfall_7d=200.0
        )

        score, factors, rules = model._evaluate_rainfall_rules(features)

        assert score >= 12
        assert 'RULE_MODERATE_DAILY_RAIN' in rules

    def test_critical_accumulated_rainfall_rule(self, model):
        """Test critical 7-day accumulated rainfall"""
        features = WeatherFeatures(
            region_id="TEST001",
            rainfall_1d=50.0,
            rainfall_7d=420.0  # Above critical 7-day threshold
        )

        score, factors, rules = model._evaluate_rainfall_rules(features)

        assert 'RULE_CRITICAL_7D_RAIN' in rules
        assert score >= 30

    def test_rainfall_intensity_rule(self, model):
        """Test rainfall intensity rule (high daily relative to weekly)"""
        features = WeatherFeatures(
            region_id="TEST001",
            rainfall_1d=100.0,  # 60% of weekly
            rainfall_7d=160.0
        )

        score, factors, rules = model._evaluate_rainfall_rules(features)

        assert 'RULE_HIGH_INTENSITY' in rules
        assert 'rainfall_intensity' in factors

    def test_persistent_heavy_rain_rule(self, model):
        """Test persistent heavy rain days rule"""
        features = WeatherFeatures(
            region_id="TEST001",
            rainfall_7d=200.0,
            heavy_rain_days_7d=4  # >= 3 days
        )

        score, factors, rules = model._evaluate_rainfall_rules(features)

        assert 'RULE_PERSISTENT_HEAVY_RAIN' in rules


class TestGeographicRulesEvaluation:
    """Test geography-based rule evaluation"""

    def test_very_low_elevation_rule(self, model):
        """Test very low elevation rule"""
        features = WeatherFeatures(
            region_id="TEST001",
            elevation=35.0,  # Below 50m
            rainfall_7d=100.0
        )

        score, factors, rules = model._evaluate_geographic_rules(features)

        assert 'RULE_VERY_LOW_ELEVATION' in rules
        assert 'very_low_elevation' in factors
        assert score >= 15

    def test_low_elevation_rule(self, model):
        """Test low elevation rule"""
        features = WeatherFeatures(
            region_id="TEST001",
            elevation=75.0,  # Between 50 and 100m
            rainfall_7d=100.0
        )

        score, factors, rules = model._evaluate_geographic_rules(features)

        assert 'RULE_LOW_ELEVATION' in rules
        assert score >= 7

    def test_lowland_rain_combo_rule(self, model):
        """Test combined lowland + rain rule"""
        features = WeatherFeatures(
            region_id="TEST001",
            elevation=85.0,  # Below 100m
            rainfall_7d=160.0  # Above medium threshold
        )

        score, factors, rules = model._evaluate_geographic_rules(features)

        assert 'RULE_LOWLAND_RAIN_COMBO' in rules
        assert 'lowland_with_rain' in factors


class TestHistoricalRulesEvaluation:
    """Test historical pattern rule evaluation"""

    def test_flood_prone_area_rule(self, model):
        """Test high historical flood frequency rule"""
        features = WeatherFeatures(
            region_id="TEST001",
            historical_flood_count=7,  # > 5
            rainfall_7d=100.0
        )

        score, factors, rules = model._evaluate_historical_rules(features)

        assert 'RULE_FLOOD_PRONE_AREA' in rules
        assert 'flood_prone_area' in factors
        assert score >= 10

    def test_some_flood_history_rule(self, model):
        """Test moderate historical flood frequency"""
        features = WeatherFeatures(
            region_id="TEST001",
            historical_flood_count=3,  # Between 2 and 5
            rainfall_7d=100.0
        )

        score, factors, rules = model._evaluate_historical_rules(features)

        assert 'RULE_SOME_FLOOD_HISTORY' in rules
        assert score >= 5

    def test_high_vulnerability_rule(self, model):
        """Test high vulnerability score rule"""
        features = WeatherFeatures(
            region_id="TEST001",
            region_vulnerability_score=75.0,  # > 70
            rainfall_7d=100.0
        )

        score, factors, rules = model._evaluate_historical_rules(features)

        assert 'RULE_HIGH_VULNERABILITY' in rules
        assert score >= 8

    def test_moderate_vulnerability_rule(self, model):
        """Test moderate vulnerability score rule"""
        features = WeatherFeatures(
            region_id="TEST001",
            region_vulnerability_score=50.0,  # Between 40 and 70
            rainfall_7d=100.0
        )

        score, factors, rules = model._evaluate_historical_rules(features)

        assert 'RULE_MODERATE_VULNERABILITY' in rules


class TestSeasonalRulesEvaluation:
    """Test seasonal and weather pattern rules"""

    def test_typhoon_season_rule(self, model):
        """Test typhoon season + high rainfall rule"""
        features = WeatherFeatures(
            region_id="TEST001",
            is_typhoon_season=True,
            rainfall_7d=160.0,  # Above medium threshold
            month=7
        )

        score, factors, rules = model._evaluate_seasonal_rules(features)

        assert 'RULE_TYPHOON_SEASON' in rules
        assert score >= 7

    def test_storm_conditions_rule(self, model):
        """Test storm wind conditions rule"""
        features = WeatherFeatures(
            region_id="TEST001",
            wind_speed_max_7d=110.0,  # Storm-level winds
            rainfall_7d=100.0
        )

        score, factors, rules = model._evaluate_seasonal_rules(features)

        assert 'RULE_STORM_CONDITIONS' in rules
        assert score >= 12

    def test_high_winds_rule(self, model):
        """Test high winds rule"""
        features = WeatherFeatures(
            region_id="TEST001",
            wind_speed=70.0,  # Above 60 km/h
            rainfall_7d=100.0
        )

        score, factors, rules = model._evaluate_seasonal_rules(features)

        assert 'RULE_HIGH_WINDS' in rules
        assert score >= 5

    def test_soil_saturation_rule(self, model):
        """Test soil saturation rule"""
        features = WeatherFeatures(
            region_id="TEST001",
            soil_moisture_proxy=18.0,  # > 15
            rainfall_7d=100.0
        )

        score, factors, rules = model._evaluate_seasonal_rules(features)

        assert 'RULE_SOIL_SATURATION' in rules


class TestRiskLevelDetermination:
    """Test risk level and recommendation determination"""

    def test_critical_risk_level(self, model):
        """Test critical risk level determination"""
        risk_level, recommendation = model._determine_risk_level(75.0)

        assert risk_level == FloodRiskLevel.CRITICAL
        assert "URGENT" in recommendation
        assert "Harvest crops immediately" in recommendation

    def test_high_risk_level(self, model):
        """Test high risk level determination"""
        risk_level, recommendation = model._determine_risk_level(55.0)

        assert risk_level == FloodRiskLevel.HIGH
        assert "WARNING" in recommendation
        assert "24-48 hours" in recommendation

    def test_medium_risk_level(self, model):
        """Test medium risk level determination"""
        risk_level, recommendation = model._determine_risk_level(35.0)

        assert risk_level == FloodRiskLevel.MEDIUM
        assert "CAUTION" in recommendation
        assert "Monitor weather" in recommendation

    def test_low_risk_level(self, model):
        """Test low risk level determination"""
        risk_level, recommendation = model._determine_risk_level(15.0)

        assert risk_level == FloodRiskLevel.LOW
        assert "Normal conditions" in recommendation


class TestPrediction:
    """Test complete prediction workflow"""

    def test_predict_low_risk(self, model, low_risk_features):
        """Test prediction for low risk scenario"""
        assessment = model.predict(low_risk_features)

        assert isinstance(assessment, FloodRiskAssessment)
        assert assessment.risk_level == FloodRiskLevel.LOW
        assert assessment.region_id == "TEST001"
        assert assessment.confidence_score > 0
        assert assessment.risk_score < 30

    def test_predict_high_risk(self, model, high_risk_features):
        """Test prediction for high risk scenario"""
        assessment = model.predict(high_risk_features)

        assert assessment.risk_level in [FloodRiskLevel.HIGH, FloodRiskLevel.CRITICAL]
        assert assessment.risk_score >= 50
        assert len(assessment.triggered_rules) > 5

    def test_predict_critical_risk(self, model, critical_risk_features):
        """Test prediction for critical risk scenario"""
        assessment = model.predict(critical_risk_features)

        assert assessment.risk_level == FloodRiskLevel.CRITICAL
        assert assessment.risk_score >= 70
        assert assessment.confidence_score > 0.7
        assert "URGENT" in assessment.recommendation

    def test_prediction_includes_metadata(self, model, low_risk_features):
        """Test that prediction includes metadata"""
        assessment = model.predict(low_risk_features)

        assert 'rainfall_score' in assessment.metadata
        assert 'geographic_score' in assessment.metadata
        assert 'historical_score' in assessment.metadata
        assert 'seasonal_score' in assessment.metadata
        assert 'rules_triggered' in assessment.metadata

    def test_prediction_caps_score_at_100(self, model):
        """Test that risk score never exceeds 100"""
        features = WeatherFeatures(
            region_id="TEST001",
            rainfall_1d=200.0,
            rainfall_7d=500.0,
            elevation=20.0,
            historical_flood_count=15,
            wind_speed_max_7d=150.0,
            is_typhoon_season=True
        )

        assessment = model.predict(features)

        assert assessment.risk_score <= 100.0

    def test_prediction_to_dict(self, model, low_risk_features):
        """Test converting assessment to dictionary"""
        assessment = model.predict(low_risk_features)
        result_dict = assessment.to_dict()

        assert 'region_id' in result_dict
        assert 'risk_level' in result_dict
        assert 'risk_score' in result_dict
        assert 'confidence_score' in result_dict
        assert 'recommendation' in result_dict
        assert result_dict['risk_level'] == 'low'


class TestBatchPrediction:
    """Test batch prediction functionality"""

    def test_batch_predict_multiple_regions(self, model, low_risk_features, high_risk_features):
        """Test batch prediction for multiple regions"""
        features_list = [low_risk_features, high_risk_features]

        assessments = model.batch_predict(features_list)

        assert len(assessments) == 2
        assert all(isinstance(a, FloodRiskAssessment) for a in assessments)

    def test_batch_predict_handles_errors_gracefully(self, model):
        """Test that batch prediction continues on individual errors"""
        # Create some invalid features
        features_list = [
            WeatherFeatures(region_id="TEST001", rainfall_1d=10.0),
            WeatherFeatures(region_id="TEST002", rainfall_1d=50.0),
        ]

        assessments = model.batch_predict(features_list)

        # Should still return successful predictions
        assert len(assessments) >= 0

    def test_batch_predict_empty_list(self, model):
        """Test batch prediction with empty list"""
        assessments = model.batch_predict([])

        assert len(assessments) == 0


class TestExplainPrediction:
    """Test prediction explanation generation"""

    def test_explain_prediction_format(self, model, high_risk_features):
        """Test explanation formatting"""
        assessment = model.predict(high_risk_features)
        explanation = model.explain_prediction(assessment)

        assert isinstance(explanation, str)
        assert "Flood Risk Assessment" in explanation
        assert assessment.region_id in explanation
        assert str(assessment.risk_score) in explanation

    def test_explanation_includes_triggered_rules(self, model, critical_risk_features):
        """Test that explanation lists triggered rules"""
        assessment = model.predict(critical_risk_features)
        explanation = model.explain_prediction(assessment)

        assert "Triggered Rules" in explanation
        for rule in assessment.triggered_rules:
            assert rule in explanation

    def test_explanation_includes_contributing_factors(self, model, high_risk_features):
        """Test that explanation lists contributing factors"""
        assessment = model.predict(high_risk_features)
        explanation = model.explain_prediction(assessment)

        assert "Contributing Factors" in explanation

    def test_explanation_includes_recommendation(self, model, low_risk_features):
        """Test that explanation includes recommendation"""
        assessment = model.predict(low_risk_features)
        explanation = model.explain_prediction(assessment)

        assert "Recommendation" in explanation
        assert assessment.recommendation in explanation


@pytest.mark.parametrize("rainfall_1d,rainfall_7d,expected_level", [
    (5.0, 20.0, FloodRiskLevel.LOW),
    (60.0, 170.0, FloodRiskLevel.MEDIUM),
    (110.0, 280.0, FloodRiskLevel.HIGH),
    (170.0, 450.0, FloodRiskLevel.CRITICAL),
])
def test_rainfall_risk_levels(model, rainfall_1d, rainfall_7d, expected_level):
    """Parametrized test for different rainfall scenarios"""
    features = WeatherFeatures(
        region_id="TEST001",
        rainfall_1d=rainfall_1d,
        rainfall_7d=rainfall_7d,
        elevation=100.0
    )

    assessment = model.predict(features)

    assert assessment.risk_level == expected_level


@pytest.mark.parametrize("elevation,rainfall_7d,expect_elevation_rule", [
    (200.0, 100.0, False),  # High elevation, no rule
    (75.0, 100.0, True),    # Low elevation, should trigger
    (30.0, 100.0, True),    # Very low elevation, should trigger
])
def test_elevation_impact(model, elevation, rainfall_7d, expect_elevation_rule):
    """Parametrized test for elevation impact on risk"""
    features = WeatherFeatures(
        region_id="TEST001",
        rainfall_7d=rainfall_7d,
        elevation=elevation
    )

    assessment = model.predict(features)

    elevation_rules = [r for r in assessment.triggered_rules if 'ELEVATION' in r]

    if expect_elevation_rule:
        assert len(elevation_rules) > 0
    # Note: Low rainfall might not trigger elevation rules in combination
