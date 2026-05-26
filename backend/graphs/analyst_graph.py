from typing import TypedDict
import pandas as pd

from typing import TypedDict
import pandas as pd
from agents.intent_agent import IntentAgent
from agents.planning_agent import PlanningAgent
from agents.feature_agent import FeatureAgent

class GraphState(TypedDict):

    data: pd.DataFrame

    profile: dict

    user_query: str

    task_type: str

    target_column: str

    plan: dict

    engineered_data: pd.DataFrame

    insights: str

from services.profile_data import DataProfiler

def profile_node(state):

    profile=DataProfiler.profile(
        state["data"]
    )

    state["profile"]=profile

    return state

from services.clean_data import DataCleaner

def clean_node(state):

    cleaned=DataCleaner.clean(
        state["data"]
    )

    state["data"]=cleaned

    return state

def intent_node(state):

    result=IntentAgent.detect(

        state["user_query"]

    )

    state["task_type"]=(
        result["task_type"]
    )

    state["target_column"]=(
        result["target"]
    )

    return state
def planning_node(state):

    plan=PlanningAgent.create_plan(

        state["task_type"]

    )

    state["plan"]=plan

    return state

def feature_node(state):

    steps=state["plan"]["steps"]

    engineered=FeatureAgent.process(

        state["data"],
        steps
    )

    state["engineered_data"]=(
        engineered
    )

    return state
from langgraph.graph import StateGraph

workflow=StateGraph(
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

workflow.set_entry_point(
"clean"
)

workflow.add_edge(
"clean",
"profile"
)

workflow.add_edge(
"profile",
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

app=workflow.compile()