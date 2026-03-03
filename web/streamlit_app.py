import streamlit as st
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.graph import app

st.set_page_config(page_title="DataCopilot", layout="wide")

st.title("📊 DataCopilot")
st.write("Faça perguntas sobre seus dados usando IA.")

uploaded_file = st.file_uploader("Envie seu arquivo CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Preview dos dados")
    st.dataframe(df.head())

    question = st.text_input("Digite sua pergunta sobre os dados")

    if question:
        with st.spinner("Analisando..."):
            initial_state = {
            "question": question,
            "dataframe": df,   
            "dataframe_summary": str(df.describe()),
            "plan": {},
            "tool_result": None,
            "final_answer": "",
            "analysis_output": None,
            "iterations": 0,
            }

            result = app.invoke(initial_state)

        st.subheader("Resposta")
        st.success(result["final_answer"])

        if "analysis_output" in result and result["analysis_output"]:

            analysis = result["analysis_output"]

            if "figure" in analysis:
                st.pyplot(analysis["figure"], use_container_width = False)
