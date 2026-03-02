from dotenv import load_dotenv
load_dotenv()
from graph import app
import tools

initial_state = {
    "question": "Qual a correlação entre AveragePrice e TotalVolume?",
    "dataframe_summary": tools.summarize_dataframe(),
    "plan": {},
    "tool_result": "",
    "final_answer": ""
}

result = app.invoke(initial_state)

print("\nRESPOSTA FINAL:\n")
print(result["final_answer"])