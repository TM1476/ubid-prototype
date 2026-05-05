import streamlit as st
import pandas as pd
from rapidfuzz import fuzz
import uuid
import networkx as nx
import matplotlib.pyplot as plt

st.set_page_config(page_title="UBID Insight Engine", layout="wide")

st.title("UBID Insight Engine")

# Load data
df = pd.read_csv("data/sample.csv")

# Clean text
def clean(x):
    if pd.isna(x):
        return ""
    return str(x).lower().replace(".", "").strip()

df["name_clean"] = df["name"].apply(clean)
df["address_clean"] = df["address"].apply(clean)

# Similarity score
def similarity(r1, r2):
    name = fuzz.ratio(r1["name_clean"], r2["name_clean"]) / 100
    addr = fuzz.ratio(r1["address_clean"], r2["address_clean"]) / 100
    gst = 1 if pd.notna(r1["gstin"]) and r1["gstin"] == r2["gstin"] else 0
    return 0.4 * name + 0.4 * addr + 0.2 * gst

# Clustering
def cluster(df):
    visited = set()
    clusters = []

    for i in range(len(df)):
        if i in visited:
            continue

        group = [i]
        visited.add(i)

        for j in range(i + 1, len(df)):
            if similarity(df.iloc[i], df.iloc[j]) >= 0.75:
                group.append(j)
                visited.add(j)

        clusters.append(group)

    return clusters

# Classification
def classify(group):
    acts = list(group["activity"])
    if "inspection" in acts or "renewal" in acts:
        return "Active"
    elif all(a == "none" for a in acts):
        return "Dormant"
    return "Closed"

# Risk detection
def detect(group):
    acts = list(group["activity"])
    if "inspection" not in acts:
        return "⚠ No Inspection"
    return "OK"

# Button
if st.button("Run UBID Engine"):

    clusters = cluster(df)

    mapping = {}
    for group in clusters:
        uid = str(uuid.uuid4())[:8]
        for idx in group:
            mapping[idx] = uid

    df["UBID"] = df.index.map(mapping)

    status = df.groupby("UBID").apply(classify).reset_index(name="Status")
    risk = df.groupby("UBID").apply(detect).reset_index(name="Risk")

    df = df.merge(status, on="UBID")
    df = df.merge(risk, on="UBID")

    # TABLE
    st.subheader("Unified Records")
    st.dataframe(df[["name", "address", "UBID", "Status", "Risk"]])

    # METRICS
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Businesses", df["UBID"].nunique())
    col2.metric("Active", sum(df["Status"] == "Active"))
    col3.metric("Risk Cases", sum(df["Risk"] != "OK"))

    # CHARTS
    col1, col2 = st.columns(2)
    with col1:
        st.bar_chart(df["Status"].value_counts())
    with col2:
        st.bar_chart(df["Risk"].value_counts())

    # GRAPH
    st.subheader("Business Graph")

    G = nx.Graph()

    for i, row in df.iterrows():
        G.add_node(f"{row['name']}_{i}")

    for ubid, group in df.groupby("UBID"):
        nodes = list(group.index)
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                G.add_edge(
                    f"{df.loc[nodes[i],'name']}_{nodes[i]}",
                    f"{df.loc[nodes[j],'name']}_{nodes[j]}"
                )

    fig, ax = plt.subplots()
    nx.draw(G, with_labels=True, node_size=2000, font_size=8)
    st.pyplot(fig)
