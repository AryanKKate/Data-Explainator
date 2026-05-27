import pandas as pd

from services.analysis_service import QueryAnalysisService


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_num": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            "feature_cat": ["A", "B"] * 10,
            "target": [10, 20, 15, 25, 30, 28, 32, 35, 40, 45, 42, 47, 50, 53, 57, 60, 62, 65, 68, 70],
        }
    )


def test_detect_intent_multi_type():
    intent = QueryAnalysisService.detect_intent(
        "predict target and recommend strategy with summary", ["feature_num", "feature_cat", "target"]
    )
    assert "predictive" in intent.analysis_types
    assert "prescriptive" in intent.analysis_types
    assert "descriptive" in intent.analysis_types
    assert intent.target_column == "target"


def test_analyze_predictive_completes():
    df = _sample_df()
    result = QueryAnalysisService.analyze(df, "predict target")
    assert result["predictive"]["status"] == "completed"
    assert "metrics" in result["predictive"]


def test_descriptive_always_available_on_fallback():
    df = _sample_df()
    result = QueryAnalysisService.analyze(df, "tell me about this dataset")
    assert result["detected_analysis_types"] == ["descriptive"]
    assert "descriptive" in result
