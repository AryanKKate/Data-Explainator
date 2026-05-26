from typing import TypedDict
import pandas as pd

class GraphState(TypedDict):

    data:pd.DataFrame

    profile:dict

    insights:str
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

workflow.set_entry_point(
"clean"
)

workflow.add_edge(
"clean",
"profile"
)

app=workflow.compile()