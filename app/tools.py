import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

def summarize_dataframe(df):
    return df.describe().to_string()

def correlation_matrix(df, col1: str, col2: str):
    if col1 not in df.columns:
        return {"error": f"Coluna '{col1}' não encontrada."}
    if col2 not in df.columns:
        return {"error": f"Coluna '{col2}' não encontrada."}

    corr = df[[col1, col2]].corr().iloc[0, 1]

    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=col1, y=col2, ax=ax)
    ax.set_title(f"Correlação: {corr:.3f}")

    return {
        "analysis_type": "correlation",
        "metrics": {"correlation": corr},
        "figure": fig
    }

def run_linear_regression(df, target: str, feature: str):
    if target not in df.columns:
        return {"error": f"Coluna '{target}' não encontrada."}
    if feature not in df.columns:
        return {"error": f"Coluna '{feature}' não encontrada."}

    X = sm.add_constant(df[feature])
    y = df[target]

    model = sm.OLS(y, X).fit()

    fig, ax = plt.subplots()
    sns.regplot(data=df, x=feature, y=target, ax=ax)
    ax.set_title(f"R² = {model.rsquared:.3f}")

    return {
        "analysis_type": "regression",
        "metrics": {
            "intercept": model.params["const"],
            "coef": model.params[feature],
            "r2": model.rsquared,
            "pvalue": model.pvalues[feature],
        },
        "figure": fig
    }