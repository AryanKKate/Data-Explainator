from analyzers.statistical_analyzer import StatisticalAnalyzer
from analyzers.segmentation_analyzer import SegmentationAnalyzer
from analyzers.opportunity_analyzer import OpportunityAnalyzer
from analyzers.trust_score_analyzer import TrustScoreAnalyzer
from analyzers.data_quality_analyzer import DataQualityAnalyzer
from analyzers.insight_translator import InsightTranslator
from analyzers.observation_generator import ObservationGenerator    


class BusinessIntelligenceAgent:
    
    @staticmethod
    def generate(

        engineered_data,
        target,
        training_results,
        explainability_report

    ):

        stats = StatisticalAnalyzer.analyze(

            engineered_data,
            target

        )

        segments = SegmentationAnalyzer.analyze(

            engineered_data,
            target

        )

        trust = (

            TrustScoreAnalyzer.analyze(

                training_results

            )

        )

        opportunities = (

            OpportunityAnalyzer.analyze(

                explainability_report,

                stats,

                trust

            )

        )

        translated_insights = (

            InsightTranslator.analyze(

                opportunities,

                explainability_report,

                stats

            )

        )

        observations = (

            ObservationGenerator.analyze(

                target=target,

                stats=stats,

                segments=segments,

                explainability_report=
                explainability_report,

                trust_score=
                trust["trust_score"]

            )

        )

        

        quality = (

            DataQualityAnalyzer.analyze(

                engineered_data

            )

        )

        summary = {

            "top_driver":

            explainability_report
            .get(
                "top_drivers",
                [None]
            )[0],

            "top_opportunity":

            opportunities[0]["feature"]

            if opportunities

            else None,

            "best_segment":

            segments.get(
                "best_segment",
                None
            )

        }
        print(translated_insights)
        print("========================================")
        print(segments)
        return {

            "statistical_findings":
            stats,

            "segments":
            segments,

            "opportunity_ranking":
            opportunities,

            "opportunities":
            opportunities,

            "translated_insights":
            translated_insights,

            "ranked_insights":
            translated_insights,
            
            "business_observations":
            observations,

            "trust_score":
            trust,

            "data_quality":
            quality,

            "summary":
            summary,

            "target":
            target,

        }