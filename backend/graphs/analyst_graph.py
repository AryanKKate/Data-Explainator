from typing import TypedDict
import pandas as pd

from typing import TypedDict
import pandas as pd
from services.clean_data import DataCleaner
from services.profile_data import DataProfiler
from langgraph.graph import StateGraph
from agents.intent_agent import IntentAgent
from agents.planning_agent import PlanningAgent
from agents.feature_agent import FeatureAgent
from agents.schema_agent import SchemaAgent
from agents.validation_agent import ValidationAgent
from agents.model_selection import ModelSelectionAgent
from agents.training_agent import TrainingAgent
from agents.semantic_agent import SemanticAgent

class GraphState(TypedDict):

    data: object

    profile: dict

    schema: dict

    semantic_schema:dict

    user_query: str

    task_type: str

    target_column: str

    plan: dict

    engineered_data: object

    validation_report: dict

    model_plan:dict

    training_results: dict

##Define Nodes

def profile_node(state):

    profile=DataProfiler.profile(
        state["data"]
    )

    return{
        "profile" : profile
    }



def clean_node(state):

    cleaned=DataCleaner.clean(
        state["data"]
    )

    return {"data" : cleaned}

def intent_node(state):

    result = IntentAgent.detect(
        state["user_query"]
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


def schema_node(state):

    columns=list(

        state["data"].columns

    )

    schema=SchemaAgent.analyze(
        columns
    )

    return{
        "schema":schema
    }

def semantic_node(state):

    semantic=(

        SemanticAgent.analyze(

            state["data"]

        )

    )

    return{

        "semantic_schema":

        semantic

    }

def validation_node(state):

    report = (
        ValidationAgent.validate(
            state["engineered_data"]
        )
    )

    return {
        "validation_report":
        report
    }

def feature_node(state):


    target=(
        state[
        "target_column"
        ]
    )

    steps=(
        state[
        "plan"
        ]["steps"]
    )


    engineered=(

    FeatureAgent.process(

    state["data"],
    steps,
    state["schema"],
    state["semantic_schema"],
    target

)

    )

    return{
        "engineered_data" : engineered
    }

def model_selection_node(state):

    result = (

        ModelSelectionAgent.select(

            state["task_type"],
            state["schema"],
            state["engineered_data"],
            state["target_column"]

        )

    )

    return {

        "model_plan": result

    }

def training_node(state):

    result=(

        TrainingAgent.train(

            state[
                "engineered_data"
            ],

            state[
                "target_column"
            ],

            state[
                "model_plan"
            ]

        )

    )

    return {

        "training_results":
        result
    }

workflow=StateGraph(
GraphState
)

##Nodes

workflow.add_node(
"clean",
clean_node
)

workflow.add_node(
"profile",
profile_node
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
"feature",
feature_node
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

"validation",

validation_node

)

workflow.add_node(
    "model_selection",
    model_selection_node
)

workflow.add_node(
    "training",
    training_node
)

workflow.set_entry_point(
"clean"
)

## Edges

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
    "feature"
)

workflow.add_edge(
    "feature",
    "validation"
)

workflow.add_edge(
    "validation",
    "model_selection"
)

workflow.add_edge(
    "model_selection",
    "training"
)

workflow.set_finish_point(
    "training"
)
app=workflow.compile()