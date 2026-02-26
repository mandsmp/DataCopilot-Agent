from dotenv import load_dotenv
load_dotenv()
from graph import app
import tools

initial_state = {
    "question": "Preço impacta vendas?",
    "dataframe_summary": tools.summarize_dataframe(),
    "plan": {},
    "tool_result": "",
    "final_answer": ""
}

result = app.invoke(initial_state)

print("\nRESPOSTA FINAL:\n")
print(result["final_answer"])