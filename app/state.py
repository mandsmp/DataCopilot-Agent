from typing import TypedDict, List, Dict, Any

class AgentState(TypedDict):
    question: str
    dataframe_summary: str
    plan: Dict[str, Any]
    tool_result: Any
    final_answer: str
    iterations: int