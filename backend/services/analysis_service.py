"""Query-driven analytics service for descriptive, diagnostic, predictive, and prescriptive analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from exceptions import DataValidationError


@dataclass
class QueryIntent:
    analysis_types: list[str]
    target_column: str | None


class QueryAnalysisService:
    """Performs multi-modal analysis from free-form user query."""

    ANALYSIS_KEYWORDS = {
        "descriptive": {"summary", "describe", "distribution", "overview", "stats", "profile"},
        "diagnostic": {"why", "reason", "correlation", "relationship", "root cause", "drivers"},
        "predictive": {"predict", "forecast", "estimate", "classification", "regression", "model"},
        "prescriptive": {"recommend", "optimize", "improve", "action", "strategy", "prescribe"},
    }

    @staticmethod
    def _normalize_query(query: str) -> str:
        return re.sub(r"\s+", " ", query.strip().lower())

    @classmethod
    def detect_intent(cls, query: str, columns: list[str]) -> QueryIntent:
        if not query or not isinstance(query, str):
            raise DataValidationError("Query must be a non-empty string")

        normalized = cls._normalize_query(query)
        matched: list[str] = []

        for analysis_type, keywords in cls.ANALYSIS_KEYWORDS.items():
            if any(keyword in normalized for keyword in keywords):
                matched.append(analysis_type)

        if not matched:
            matched = ["descriptive"]

        target = None
        for col in columns:
            if col.lower() in normalized:
                target = col
                break

        return QueryIntent(analysis_types=matched, target_column=target)

    @staticmethod
    def descriptive(df: pd.DataFrame) -> dict[str, Any]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        return {
            "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
            "missing_values": df.isna().sum().to_dict(),
            "numeric_summary": df[numeric_cols].describe(include="all").to_dict() if numeric_cols else {},
            "categorical_summary": df[categorical_cols].describe(include="all").to_dict() if categorical_cols else {},
        }

    @staticmethod
    def diagnostic(df: pd.DataFrame, target_column: str | None = None) -> dict[str, Any]:
        numeric_df = df.select_dtypes(include=[np.number])
        corr = numeric_df.corr(numeric_only=True)

        top_correlations = []
        if target_column and target_column in corr.columns:
            series = corr[target_column].drop(labels=[target_column], errors="ignore").dropna()
            top = series.abs().sort_values(ascending=False).head(5)
            top_correlations = [
                {"feature": idx, "correlation": float(series[idx])} for idx in top.index
            ]

        return {
            "correlation_matrix": corr.to_dict() if not corr.empty else {},
            "top_target_drivers": top_correlations,
        }

    @staticmethod
    def _infer_problem_type(y: pd.Series) -> str:
        if pd.api.types.is_numeric_dtype(y) and y.nunique(dropna=True) > 10:
            return "regression"
        return "classification"

    @classmethod
    def predictive(cls, df: pd.DataFrame, target_column: str | None = None) -> dict[str, Any]:
        if not target_column:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            target_column = numeric_cols[-1] if numeric_cols else None
        if not target_column or target_column not in df.columns:
            return {"status": "skipped", "reason": "No valid target column detected in query or dataset."}

        work_df = df.dropna(subset=[target_column]).copy()
        if work_df.shape[0] < 20:
            return {"status": "skipped", "reason": "Need at least 20 rows with target values for predictive modeling."}

        y = work_df[target_column]
        X = work_df.drop(columns=[target_column])
        X = X.loc[:, X.nunique(dropna=False) > 1]
        if X.empty:
            return {"status": "skipped", "reason": "No informative feature columns after preprocessing."}

        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = [c for c in X.columns if c not in numeric_features]

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
                ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
            ],
            remainder="drop",
        )

        problem_type = cls._infer_problem_type(y)

        if problem_type == "regression":
            model = LinearRegression()
        else:
            model = LogisticRegression(max_iter=1000)

        pipeline = Pipeline([("prep", preprocessor), ("model", model)])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        if problem_type == "regression":
            metrics = {"r2": float(r2_score(y_test, y_pred)), "mae": float(mean_absolute_error(y_test, y_pred))}
        else:
            metrics = {"accuracy": float(accuracy_score(y_test, y_pred)), "f1_weighted": float(f1_score(y_test, y_pred, average="weighted"))}

        return {
            "status": "completed",
            "problem_type": problem_type,
            "target_column": target_column,
            "train_rows": int(X_train.shape[0]),
            "test_rows": int(X_test.shape[0]),
            "metrics": metrics,
        }

    @staticmethod
    def prescriptive(df: pd.DataFrame, target_column: str | None = None) -> dict[str, Any]:
        recs: list[str] = []
        missing_ratio = (df.isna().sum() / max(len(df), 1)).sort_values(ascending=False)
        high_missing = missing_ratio[missing_ratio > 0.2]
        if not high_missing.empty:
            recs.append(f"Columns with >20% missing values should be imputed/dropped: {', '.join(high_missing.index[:5])}.")

        duplicates = int(df.duplicated().sum())
        if duplicates > 0:
            recs.append(f"Remove {duplicates} duplicate rows before final modeling.")

        if target_column and target_column in df.columns and pd.api.types.is_numeric_dtype(df[target_column]):
            q1, q3 = df[target_column].quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr > 0:
                outliers = int(((df[target_column] < q1 - 1.5 * iqr) | (df[target_column] > q3 + 1.5 * iqr)).sum())
                if outliers > 0:
                    recs.append(f"Target column '{target_column}' has {outliers} potential outliers; consider robust models or winsorization.")

        if not recs:
            recs.append("Data quality checks look healthy; proceed with cross-validation and model comparison.")

        return {"recommendations": recs}

    @classmethod
    def analyze(cls, df: pd.DataFrame, query: str) -> dict[str, Any]:
        intent = cls.detect_intent(query, list(df.columns))

        results: dict[str, Any] = {
            "query": query,
            "detected_analysis_types": intent.analysis_types,
            "detected_target_column": intent.target_column,
        }

        if "descriptive" in intent.analysis_types:
            results["descriptive"] = cls.descriptive(df)
        if "diagnostic" in intent.analysis_types:
            results["diagnostic"] = cls.diagnostic(df, intent.target_column)
        if "predictive" in intent.analysis_types:
            results["predictive"] = cls.predictive(df, intent.target_column)
        if "prescriptive" in intent.analysis_types:
            results["prescriptive"] = cls.prescriptive(df, intent.target_column)

        return results
