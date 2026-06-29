import pandas as pd


def summarize_phases(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty:
        return pd.DataFrame()
    df = dataframe.copy()
    df["PhaseGroup"] = (df["Phase"] != df["Phase"].shift()).cumsum()
    rows = []
    for _, group in df.groupby("PhaseGroup", sort=True):
        start = group.iloc[0]
        end = group.iloc[-1]
        risks = set(group["Risk"])
        rows.append({
            "Phase": start["Phase"],
            "Start week": round(float(start["Week"]), 1),
            "End week": round(float(end["Week"]), 1),
            "End weight (kg)": round(float(end["Weight"]), 2),
            "End body fat (%)": round(float(end["BodyFat"]), 2),
            "Weight change (kg)": round(float(end["Weight"] - start["Weight"]), 2),
            "Fat change (kg)": round(float(end["FatMass"] - start["FatMass"]), 2),
            "Fat-free change (kg)": round(float(end["LeanMass"] - start["LeanMass"]), 2),
            "Highest risk": "High" if "High" in risks else "Moderate" if "Moderate" in risks else "Low",
        })
    return pd.DataFrame(rows)


def phase_endpoints(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty:
        return dataframe.copy()
    groups = (dataframe["Phase"] != dataframe["Phase"].shift()).cumsum()
    return dataframe.groupby(groups, sort=True).tail(1).reset_index(drop=True)
