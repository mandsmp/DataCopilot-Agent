import pandas as pd
from sklearn.linear_model import LinearRegression
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "sample.csv"

df = pd.read_csv(DATA_PATH)


def summarize_dataframe():
    return df.describe().to_string()

def correlation_matrix(col1: str, col2: str):
    if col1 not in df.columns:
        return f"Erro: coluna '{col1}' não encontrada."
    if col2 not in df.columns:
        return f"Erro: coluna '{col2}' não encontrada."
    
    corr = df[[col1, col2]].corr().iloc[0,1]
    return f"Correlação entre {col1} e {col2}: {corr}"

def run_linear_regression(target: str, feature: str):
    if target not in df.columns:
        return f"Erro: coluna '{target}' não encontrada."
    if feature not in df.columns:
        return f"Erro: coluna '{feature}' não encontrada."
    
    model = LinearRegression()
    X = df[[feature]]
    y = df[target]
    
    model.fit(X, y)
    
    coef = model.coef_[0]
    intercept = model.intercept_
    
    return f"Regressão: {target} = {intercept:.4f} + {coef:.4f} * {feature}"