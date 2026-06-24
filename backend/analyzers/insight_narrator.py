class InsightNarrator:

    @staticmethod
    def narrate(insights):

        narratives = []

        for item in insights[:10]:

            if item["type"] == "relationship":

                narratives.append(

                    f"{item['headline']} "
                    f"shows {item['evidence']}."

                )

            elif item["type"] == "driver":

                narratives.append(

                    f"{item['headline']} "
                    f"is among the strongest "
                    f"drivers of model predictions."

                )

            elif item["type"] == "segment":

                narratives.append(

                    f"{item['headline']} "
                    f"achieves {item['evidence']}."

                )

            elif item["type"] == "quality":

                narratives.append(

                    f"Data quality remains high "
                    f"at {item['evidence']}."

                )

            elif item["type"] == "model":

                narratives.append(

                    f"The model achieved "
                    f"{item['evidence']}."

                )

            elif item["type"] == "trust":

                narratives.append(

                    f"Prediction trust score is "
                    f"{item['evidence']}."

                )

        return narratives