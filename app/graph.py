import json
from langgraph.graph import StateGraph, END
from llm import get_llm
from state import AgentState
import tools

llm = get_llm()

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

    Se correlation, use parâmetros:
      col1, col2

    Se regression, use parâmetros:
      target, feature

    Retorne APENAS JSON válido no formato:

    {{
      "action": "...",
      "parameters": {{...}}
    }}
    """

    response = llm.invoke(prompt)

    print("RAW LLM RESPONSE:", response.content)

    try:
        cleaned = response.content.strip()
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
        plan = json.loads(cleaned)
    except Exception as e:
        print("JSON ERROR:", e)
        plan = {"action": "summarize", "parameters": {}}

    state["plan"] = plan
    return state


# -------------------
# Tool Node
# -------------------
def tool_node(state: AgentState):
    print(">> Tool node executado")

    plan = state["plan"]
    action = plan.get("action")
    params = plan.get("parameters", {})

    if action == "summarize":
        result = tools.summarize_dataframe()

    elif action == "correlation":
        result = tools.correlation_matrix(
            col1=params.get("col1"),
            col2=params.get("col2")
        )

    elif action == "regression":
        result = tools.run_linear_regression(
            target=params.get("target"),
            feature=params.get("feature")
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

    response = llm.invoke(prompt)

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