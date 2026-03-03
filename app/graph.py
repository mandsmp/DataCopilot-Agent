import json
from langgraph.graph import StateGraph, END
from app.llm import get_llm
from app.state import AgentState
import app.tools as tools
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

    Pergunta do usuário:
    {state['question']}

    Colunas disponíveis no dataframe:
    {list(state['dataframe'].columns)}

    Escolha UMA ação:
    - summarize
    - correlation
    - regression

    Se for correlation:
    - preencha col1 e col2

    Se for regression:
    - preencha target (variável dependente Y)
    - preencha feature (variável independente X)

    Retorne apenas os campos do modelo estruturado.
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
    df = state["dataframe"]  
    action = plan["action"]

    if action == "summarize":
        result = tools.summarize_dataframe(df)

    elif action == "correlation":
        result = tools.correlation_matrix(
            df,
            col1=plan.get("col1"),
            col2=plan.get("col2")
        )

    elif action == "regression":
        result = tools.run_linear_regression(
            df,
            target=plan.get("target"),
            feature=plan.get("feature")
        )

    else:
        result = {"error": "Ação não reconhecida."}

    state["analysis_output"] = result

    if isinstance(result, dict) and "metrics" in result:
        state["tool_result"] = result["metrics"]
    else:
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