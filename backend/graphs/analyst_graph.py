from typing import TypedDict
import pandas as pd

from services.clean_data import DataCleaner
from services.profile_data import DataProfiler

from langgraph.graph import StateGraph,END

from agents.intent_agent import IntentAgent
from agents.planning_agent import PlanningAgent
from agents.feature_agent import FeatureAgent
from agents.schema_agent import SchemaAgent
from agents.validation_agent import ValidationAgent
from agents.model_selection import ModelSelectionAgent
from agents.training_agent import TrainingAgent
from agents.semantic_agent import SemanticAgent
from agents.explainability_agent import ExplainabilityAgent
from agents.insight_agent import InsightAgent
from agents.recommendation_agent import RecommendationAgent
from agents.executive_summary_agent import ExecutiveSummaryAgent
from agents.visualization_agent import VisualizationAgent
from agents.business_intelligence_agent import BusinessIntelligenceAgent


class GraphState(TypedDict):

    data: object

    profile: dict

    schema: dict

    semantic_schema: dict

    user_query: str

    task_type: str

    target_column: str

    plan: dict

    engineered_data: object

    validation_report: dict

    model_plan: dict

    training_results: dict

    explanation: str

    explainability_report: dict

    insights: dict

    recommendations: list

    executive_summary: str

    visualizations: dict

    business_intelligence: dict


def clean_node(state):

    cleaned = DataCleaner.clean(
        state["data"]
    )

    return {

        "data": cleaned

    }


def profile_node(state):

    profile = DataProfiler.profile(
        state["data"]
    )

    return {

        "profile": profile

    }


def schema_node(state):

    columns = list(

        state["data"].columns

    )

    schema = SchemaAgent.analyze(
        columns
    )

    return {

        "schema": schema

    }


def semantic_node(state):

    semantic = SemanticAgent.analyze(

        state["data"]

    )

    return {

        "semantic_schema": semantic

    }


def intent_node(state):

    result = IntentAgent.detect(

        query=state["user_query"],

        columns=state["data"].columns

    )

    return {

        "task_type":
        result["task_type"],

        "target_column":
        result["target"]

    }


def planning_node(state):

    plan = PlanningAgent.create_plan(

        state["task_type"]

    )

    return {

        "plan": plan

    }


def model_selection_node(state):

    config = ModelSelectionAgent.select(

        intent=state["task_type"],

        schema=state["semantic_schema"],

        df=state["data"],

        target=state["target_column"]

    )

    return {

        "model_plan": config

    }


def feature_node(state):

    target = state[
        "target_column"
    ]

    engineered = FeatureAgent.process(

        df=state["data"],

        steps=state[
            "model_plan"
        ]["steps"],

        schema=state["schema"],

        semantic_schema=state[
            "semantic_schema"
        ],

        target=target

    )

    return {

        "engineered_data": engineered

    }


def validation_node(state):

    report = ValidationAgent.validate(

        state["engineered_data"]

    )

    return {

        "validation_report":
        report

    }


def training_node(state):

    result = TrainingAgent.train(

        df=state[
            "engineered_data"
        ],

        target=state[
            "target_column"
        ],

        model_plan=state[
            "model_plan"
        ]

    )

    return {

        "training_results":
        result

    }

def explainability_node(state):

    report = (

        ExplainabilityAgent.generate(

            state[
                "training_results"
            ]

        )

    )

    return {

        "explainability_report":
        report

    }

def insight_node(state):

    insights = (

        InsightAgent.generate(

            training_results=
            state["training_results"],

            explainability_report=
            state["explainability_report"],

            business_intelligence=
            state["business_intelligence"],

            target=
            state["target_column"]

        )

    )

    return {

        "insights":
        insights

    }


def recommendation_node(state):

    recommendations = (

        RecommendationAgent.generate(

            task=
            state["task_type"],

            target=
            state["target_column"],

            training_results=
            state["training_results"],

            explainability_report=
            state["explainability_report"],

            business_intelligence=
            state["business_intelligence"],

            query=
            state["user_query"],

            insights=
            state["insights"]


        )

    )

    return {

        "recommendations":
        recommendations

    }


def executive_summary_node(state):

    summary = ExecutiveSummaryAgent.generate(

        target=state["target_column"],

        task=state["task_type"],

        training_results=state["training_results"],

        insights=state["insights"],

        recommendations=state["recommendations"]

    )

    return {

        "executive_summary":
        summary

    }

def visualization_node(state):

    visuals = (

        VisualizationAgent.generate(

            training_results=
            state["training_results"],

            explainability_report=
            state["explainability_report"]

        )

    )

    return {

        "visualizations":
        visuals

    }


def business_intelligence_node(state):

    result = BusinessIntelligenceAgent.generate(

        engineered_data=
        state["engineered_data"],

        target=
        state["target_column"],

        training_results=
        state["training_results"],

        explainability_report=
        state["explainability_report"]


    )

    return {

        "business_intelligence":
        result

    }



workflow = StateGraph(
    GraphState
)


workflow.add_node(
    "clean",
    clean_node
)

workflow.add_node(
    "profile",
    profile_node
)

workflow.add_node(
    "schema",
    schema_node
)

workflow.add_node(
    "semantic",
    semantic_node
)

workflow.add_node(
    "intent",
    intent_node
)

workflow.add_node(
    "planning",
    planning_node
)

workflow.add_node(
    "model_selection",
    model_selection_node
)

workflow.add_node(
    "feature",
    feature_node
)

workflow.add_node(
    "validation",
    validation_node
)

workflow.add_node(
    "training",
    training_node
)

workflow.add_node(
    "explainability",
    explainability_node
)

workflow.add_node(
    "insight",
    insight_node
)

workflow.add_node(
    "recommendation",
    recommendation_node
)

workflow.add_node(
    "executive_summary",
    executive_summary_node
)

workflow.add_node(
    "visualization",
    visualization_node
)

workflow.add_node(
    "business_intelligence",
    business_intelligence_node
)

workflow.set_entry_point(
    "clean"
)


workflow.add_edge(
    "clean",
    "profile"
)

workflow.add_edge(
    "profile",
    "schema"
)

workflow.add_edge(
    "schema",
    "semantic"
)

workflow.add_edge(
    "semantic",
    "intent"
)

workflow.add_edge(
    "intent",
    "planning"
)

workflow.add_edge(
    "planning",
    "model_selection"
)

workflow.add_edge(
    "model_selection",
    "feature"
)

workflow.add_edge(
    "feature",
    "validation"
)

workflow.add_edge(
    "validation",
    "training"
)


workflow.add_edge(
    "training",
    "explainability"
)

workflow.add_edge(
    "explainability",
    "business_intelligence"
)

workflow.add_edge(
    "business_intelligence",
    "visualization"
)

workflow.add_edge(
    "visualization",
    "insight"
)


workflow.add_edge(
    "insight",
    "recommendation"
)

workflow.add_edge(
    "recommendation",
    "executive_summary"
)

workflow.add_edge(
    "executive_summary",
    END
)


app = workflow.compile()