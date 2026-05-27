import pandas as pd
import numpy as np
import re


class SemanticAgent:

    @staticmethod
    def analyze(df):

        semantic_schema={}

        semantic_patterns={

            "identifier":[
                "id",
                "uuid",
                "customer",
                "user",
                "order",
                "transaction",
                "house"
            ],

            "location":[
                "city",
                "country",
                "state",
                "region",
                "locality",
                "district",
                "address"
            ],

            "area":[
                "area",
                "sqft",
                "square",
                "acre"
            ],

            "currency":[
                "price",
                "salary",
                "income",
                "cost",
                "expense",
                "maintenance",
                "revenue",
                "profit"
            ],

            "distance":[
                "distance",
                "km",
                "mile",
                "meter"
            ],

            "percentage":[
                "percent",
                "ratio",
                "rate"
            ],

            "datetime":[
                "date",
                "time",
                "day",
                "month",
                "year",
                "timestamp",
                "created",
                "updated"
            ]
        }


        boolean_tokens={

            "yes",
            "no",
            "true",
            "false",
            "y",
            "n",
            "0",
            "1"
        }


        unit_patterns={

            r"km":"distance",
            r"mile":"distance",
            r"meter":"distance",

            r"sq.?ft":"area",
            r"sq.?m":"area",
            r"acre":"area",

            r"₹|\$|€":"currency",

            r"%":"percentage"
        }


        for col in df.columns:

            values=(

                df[col]
                .dropna()
                .astype(str)
                .head(30)

            )


            joined=(

                " ".join(values)
                .lower()

            )


            col_lower=col.lower()


            info={

                "type":"unknown",

                "semantic":"unknown",

                "unit":None,

                "nullable":

                df[col]
                .isnull()
                .sum()>0,

                "confidence":0
            }


            numeric_ratio=(

                pd.to_numeric(

                    df[col],
                    errors="coerce"

                )

                .notna()

                .mean()

            )


            if numeric_ratio>0.7:

                info["type"]="numeric"

                unique_ratio=(

                    df[col]
                    .nunique()

                    /

                    len(df)

                )

                if unique_ratio<0.1:

                    info["semantic"]="ordinal"

                elif unique_ratio<0.4:

                    info["semantic"]="categorical"

                else:

                    info["semantic"]="continuous"

                info["confidence"]=0.9


            possible_date=(

                any(

                    keyword in col_lower

                    for keyword in

                    semantic_patterns["datetime"]

                )

                or

                values.str.contains(

                    r"\d{1,4}[/-]\d{1,2}[/-]\d{1,4}",

                    regex=True

                ).mean()>0.3

            )


            if possible_date:

                date_ratio=(

                    pd.to_datetime(

                        df[col],

                        errors="coerce",
                        format="mixed"

                    )

                    .notna()

                    .mean()

                )

            else:

                date_ratio=0


            if date_ratio>0.8:

                info["type"]="datetime"

                info["semantic"]="timestamp"

                info["confidence"]=0.95


            if (

                set(

                    joined.split()

                )

                .issubset(

                    boolean_tokens

                )

            ):

                info["type"]="boolean"

                info["semantic"]="binary"

                info["confidence"]=0.95


            for pattern,meaning in unit_patterns.items():

                if re.search(

                    pattern,
                    joined

                ):

                    info["semantic"]=meaning

                    info["unit"]=pattern

                    info["confidence"]=0.95


            priority_order=[

                "identifier",
                "currency",
                "distance",
                "area",
                "percentage",
                "location",
                "datetime"

            ]


            type_mapping={

                "identifier":"identifier",

                "currency":"numeric",

                "distance":"numeric",

                "area":"numeric",

                "percentage":"numeric",

                "location":"categorical",

                "datetime":"datetime"
            }


            for semantic in priority_order:

                keywords=(

                    semantic_patterns[
                        semantic
                    ]

                )

                tokens=col.lower().split("_")

                if any(
                    word==token
                    for token in tokens
                    for word in date_keywords
                ):

                    info["semantic"]=semantic

                    info["confidence"]=0.95

                    info["type"]=(
                        type_mapping.get(
                            semantic,
                            info["type"]
                        )
                    )

                    break


            semantic_schema[
                col
            ]=info


        return semantic_schema