import pandas as pd


def summarize_phases(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty:
        return pd.DataFrame()
    df = dataframe.copy()
    df["PhaseGroup"] = (df["Phase"] != df["Phase"].shift()).cumsum()
    rows = []
    for _, group in df.groupby("PhaseGroup", sort=True):
        start, end = group.iloc[0], group.iloc[-1]
        rows.append({
            "Phase": start["Phase"],
            "Start week": round(float(start["Week"]), 1),
            "End week": round(float(end["Week"]), 1),
            "End scale weight (kg)": round(float(end["Weight"]), 2),
            "End tissue body fat (%)": round(float(end["BodyFat"]), 2),
            "Scale change (kg)": round(float(end["Weight"] - start["Weight"]), 2),
            "Fat change (kg)": round(float(end["FatMass"] - start["FatMass"]), 2),
            "Stable FFM change (kg)": round(float(end["LeanMass"] - start["LeanMass"]), 2),
            "Transient change (kg)": round(float(end["ScaleTransient"] - start["ScaleTransient"]), 2),
        })
    return pd.DataFrame(rows)


def phase_endpoints(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty:
        return dataframe.copy()
    groups = (dataframe["Phase"] != dataframe["Phase"].shift()).cumsum()
    return dataframe.groupby(groups, sort=True).tail(1).reset_index(drop=True)
