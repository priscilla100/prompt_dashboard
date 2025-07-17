import plotly.express as px
import streamlit as st
import numpy as np
import os
import csv
import json
from collections import defaultdict
import pandas as pd
def create_model_approach_comparison_from_df(df: pd.DataFrame, selected_metric: str):
    # Try to infer the model and approach column names
    model_col = next((col for col in df.columns if 'model' in col.lower()), None)
    approach_col = next((col for col in df.columns if any(x in col.lower() for x in ['approach', 'method', 'shot', 'prompt'])), None)

    if model_col is None or approach_col is None:
        st.error("Model or Approach column not found in dataset.")
        return None

    grouped = df.groupby([model_col, approach_col])[selected_metric].mean().reset_index()
    grouped['Model + Approach'] = grouped[model_col] + " - " + grouped[approach_col]

    fig = px.bar(
        grouped,
        x="Model + Approach",
        y=selected_metric,
        color=model_col,
        barmode='group',
        title=f"{selected_metric} by Model and Approach"
    )
    fig.update_layout(xaxis_title="Model + Approach", yaxis_title=selected_metric)
    return fig


def create_distribution_from_df(df: pd.DataFrame, selected_metric: str):
    # Use same inference logic
    model_col = next((col for col in df.columns if 'model' in col.lower()), None)
    approach_col = next((col for col in df.columns if any(x in col.lower() for x in ['approach', 'method', 'shot', 'prompt'])), None)

    if approach_col is None:
        st.error("Approach column not found for distribution comparison.")
        return None

    fig = px.box(
        df,
        x=approach_col,
        y=selected_metric,
        color=model_col if model_col else approach_col,
        points="all",
        title=f"Distribution of {selected_metric} across Approaches"
    )
    fig.update_layout(xaxis_title="Approach", yaxis_title=selected_metric)
    return fig
