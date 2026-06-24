class InsightRanker:

    @staticmethod
    def rank(insights):

        ranked = sorted(

            insights,

            key=lambda x:
            x.get("score", 0),

            reverse=True

        )

        for rank, item in enumerate(

            ranked,

            start=1

        ):

            item["rank"] = rank

        return ranked