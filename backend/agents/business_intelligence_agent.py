from analyzers.statistical_analyzer import StatisticalAnalyzer
from analyzers.segmentation_analyzer import SegmentationAnalyzer
from analyzers.opportunity_analyzer import OpportunityAnalyzer
from analyzers.trust_score_analyzer import TrustScoreAnalyzer
from analyzers.data_quality_analyzer import DataQualityAnalyzer


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

        

        quality = (

            DataQualityAnalyzer.analyze(

                engineered_data

            )

        )

        return {

            "statistical_findings":
            stats,

            "segments":
            segments,

            "opportunity_ranking":
            opportunities,

            "trust_score":
            trust,

            "data_quality":
            quality

        }