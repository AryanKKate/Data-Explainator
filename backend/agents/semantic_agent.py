import pandas as pd
import numpy as np
import re


class SemanticAgent:

    @staticmethod
    def analyze(df):

        semantic_schema = {}

        semantic_patterns = {

            "identifier": [
                "id",
                "uuid",
                "guid",
                "customer",
                "client",
                "user",
                "employee",
                "transaction",
                "invoice",
                "order",
                "house",
                "record"
            ],

            "location": [
                "city",
                "country",
                "state",
                "region",
                "district",
                "area",
                "address",
                "location",
                "locality",
                "zipcode",
                "postal"
            ],

            "currency": [
                "price",
                "salary",
                "income",
                "cost",
                "expense",
                "revenue",
                "profit",
                "maintenance",
                "rent",
                "amount",
                "fee",
                "payment",
                "budget"
            ],

            "distance": [
                "distance",
                "km",
                "kilometer",
                "mile",
                "meter",
                "radius"
            ],

            "area": [
                "sqft",
                "square",
                "acre",
                "hectare",
                "area",
                "plot"
            ],

            "percentage": [
                "percent",
                "percentage",
                "ratio",
                "rate",
                "growth",
                "margin"
            ],

            "datetime": [
                "date",
                "time",
                "timestamp",
                "year",
                "month",
                "day",
                "created",
                "updated",
                "joined",
                "dob",
                "birth"
            ]
        }

        boolean_tokens = {

            "yes",
            "no",
            "true",
            "false",
            "y",
            "n",
            "0",
            "1"
        }

        unit_patterns = {

            r"₹|\$|€|usd|inr|eur": "currency",

            r"km|kilometer|mile|meter": "distance",

            r"sq.?ft|sq.?m|acre|hectare": "area",

            r"%": "percentage"
        }

        type_mapping = {

            "identifier": "identifier",

            "location": "categorical",

            "currency": "numeric",

            "distance": "numeric",

            "area": "numeric",

            "percentage": "numeric",

            "datetime": "datetime"
        }

        priority_order = [

            "identifier",
            "currency",
            "distance",
            "area",
            "percentage",
            "location",
            "datetime"
        ]

        for col in df.columns:

            series = df[col]

            values = (

                series
                .dropna()
                .astype(str)
                .head(30)
            )

            joined = (

                " ".join(values)
                .lower()
            )

            col_lower = col.lower()

            info = {

                "type": "unknown",

                "semantic": "unknown",

                "unit": None,

                "nullable": (

                    series
                    .isnull()
                    .sum() > 0
                ),

                "confidence": 0
            }

            numeric_series = pd.to_numeric(

                series,
                errors="coerce"
            )

            numeric_ratio = (

                numeric_series
                .notna()
                .mean()
            )

            unique_ratio = (

                series
                .nunique(dropna=True)

                /

                max(len(series), 1)
            )

            if numeric_ratio > 0.7:

                info["type"] = "numeric"

                if unique_ratio < 0.1:

                    info["semantic"] = "ordinal"

                elif unique_ratio < 0.4:

                    info["semantic"] = "categorical"

                else:

                    info["semantic"] = "continuous"

                info["confidence"] = 0.9

            possible_date = (

                any(
                    keyword in col_lower
                    for keyword in semantic_patterns["datetime"]
                )

                or

                values.str.contains(
                    r"\d{1,4}[/-]\d{1,2}[/-]\d{1,4}",
                    regex=True
                ).mean() > 0.3
            )

            if possible_date:

                parsed_dates = pd.to_datetime(

                    series,
                    errors="coerce",
                    format="mixed"
                )

                date_ratio = (

                    parsed_dates
                    .notna()
                    .mean()
                )

            else:

                date_ratio = 0

            if date_ratio > 0.8:

                info["type"] = "datetime"

                info["semantic"] = "datetime"

                info["confidence"] = 0.95

            cleaned_tokens = set(

                joined
                .replace(",", " ")
                .split()
            )

            if (

                len(cleaned_tokens) > 0

                and

                cleaned_tokens.issubset(
                    boolean_tokens
                )
            ):

                info["type"] = "boolean"

                info["semantic"] = "binary"

                info["confidence"] = 0.95

            for pattern, meaning in unit_patterns.items():

                if re.search(
                    pattern,
                    joined
                ):

                    info["semantic"] = meaning

                    info["unit"] = pattern

                    info["confidence"] = max(
                        info["confidence"],
                        0.95
                    )

                    info["type"] = type_mapping.get(
                        meaning,
                        info["type"]
                    )

            for semantic in priority_order:

                keywords = semantic_patterns[semantic]
                
                tokens = re.split(
                r"[_\s\-]+",
                col_lower
            )
                if any(
                    keyword in tokens
                    for keyword in keywords
                ):

                    info["semantic"] = semantic

                    info["confidence"] = max(
                        info["confidence"],
                        0.95
                    )

                    info["type"] = type_mapping.get(
                        semantic,
                        info["type"]
                    )

                    break

            if info["semantic"] == "identifier":

                if unique_ratio > 0.9:

                    info["type"] = "identifier"

                    info["confidence"] = 0.98

            semantic_schema[col] = info


        return semantic_schema