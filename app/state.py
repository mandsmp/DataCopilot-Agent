from typing import TypedDict, List, Dict, Any
import pandas as pd

class AgentState(TypedDict):
    question: str
    dataframe: pd.DataFrame  
    dataframe_summary: str
    plan: dict
    tool_result: Any
    analysis_output: Any
    final_answer: str
    iterations: int