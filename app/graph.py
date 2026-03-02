import json
from langgraph.graph import StateGraph, END
from llm import get_llm
from state import AgentState
import tools
from pydantic import BaseModel
from typing import Optional, Literal


class Plan(BaseModel):
    action: Literal["summarize", "correlation", "regression"]
    col1: Optional[str] = None
    col2: Optional[str] = None
    target: Optional[str] = None
    feature: Optional[str] = None


base_llm = get_llm()

planner_llm = base_llm.with_structured_output(Plan)
responder_llm = base_llm

# -------------------
# Planner Node
# -------------------
def planner_node(state: AgentState):
    print(">> Planner executado")

    prompt = f"""
    Você é um analista de dados.

    Pergunta: {state['question']}
    Dados disponíveis: {state['dataframe_summary']}

    Escolha UMA ação:
    - summarize
    - correlation
    - regression
    """

    result = planner_llm.invoke(prompt)

    plan = result.parsed if hasattr(result, "parsed") else result

    return {
        "plan": plan.model_dump()
    }

# -------------------
# Tool Node
# -------------------
def tool_node(state: AgentState):
    print(">> Tool node executado")

    plan = state["plan"]
    action = plan["action"]

    if action == "summarize":
        result = tools.summarize_dataframe()

    elif action == "correlation":
        result = tools.correlation_matrix(
            col1=plan.get("col1"),
            col2=plan.get("col2")
        )

    elif action == "regression":
        result = tools.run_linear_regression(
            target=plan.get("target"),
            feature=plan.get("feature")
        )

    else:
        result = "Erro: ação não reconhecida."

    state["tool_result"] = result
    return state
# -------------------
# Responder Node
# -------------------
def responder_node(state: AgentState):
    print(">> Responder executado")
    print("DEBUG TOOL RESULT:", state["tool_result"])

    prompt = f"""
    Você é um analista de dados sênior.

    Pergunta:
    {state['question']}

    Resultado da análise:
    {state['tool_result']}

    Use EXPLICITAMENTE o valor numérico retornado.
    Não explique conceitos genéricos.
    Interprete o número e diga o que ele significa para o negócio.
    Seja direto e objetivo.
    """

    response = responder_llm.invoke(prompt)

    state["final_answer"] = response.content
    return state


# -------------------
# Build Graph
# -------------------
graph = StateGraph(AgentState)

graph.add_node("planner", planner_node)
graph.add_node("tool", tool_node)
graph.add_node("responder", responder_node)

graph.set_entry_point("planner")

graph.add_edge("planner", "tool")
graph.add_edge("tool", "responder")
graph.add_edge("responder", END)

app = graph.compile()