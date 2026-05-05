import streamlit as st
import pandas as pd
from rapidfuzz import fuzz
import uuid
import networkx as nx
import plotly.graph_objects as go

st.set_page_config(page_title="UBID Insight Engine", layout="wide")

st.title("🚀 UBID Insight Engine")
st.caption("AI-powered Business Intelligence Platform")

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded = st.file_uploader("Upload Dataset (CSV)", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
else:
    df = pd.read_csv("data/sample.csv")

# -----------------------------
# PREPROCESS
# -----------------------------
def clean(x):
    if pd.isna(x):
        return ""
    return str(x).lower().replace(".", "").strip()

df["name_clean"] = df["name"].apply(clean)
df["address_clean"] = df["address"].apply(clean)

# -----------------------------
# SIMILARITY
# -----------------------------
def similarity(r1, r2):
    name = fuzz.ratio(r1["name_clean"], r2["name_clean"]) / 100
    addr = fuzz.ratio(r1["address_clean"], r2["address_clean"]) / 100
    gst = 1 if pd.notna(r1["gstin"]) and r1["gstin"] == r2["gstin"] else 0
    return 0.4 * name + 0.4 * addr + 0.2 * gst

# -----------------------------
# CLUSTERING
# -----------------------------
def cluster(df):
    visited = set()
    clusters = []

    for i in range(len(df)):
        if i in visited:
            continue

        group = [i]
        visited.add(i)

        for j in range(i+1, len(df)):
            if similarity(df.iloc[i], df.iloc[j]) >= 0.75:
                group.append(j)
                visited.add(j)

        clusters.append(group)

    return clusters

# -----------------------------
# CLASSIFICATION
# -----------------------------
def classify(group):
    acts = list(group["activity"])
    if "inspection" in acts or "renewal" in acts:
        return "Active"
    elif all(a == "none" for a in acts):
        return "Dormant"
    return "Closed"

def risk(group):
    acts = list(group["activity"])
    if "inspection" not in acts:
        return "⚠ No Inspection"
    return "OK"

# -----------------------------
# RUN ENGINE
# -----------------------------
if st.button("🚀 Run UBID Engine"):

    clusters = cluster(df)

    mapping = {}
    for group in clusters:
        uid = str(uuid.uuid4())[:8]
        for idx in group:
            mapping[idx] = uid

    df["UBID"] = df.index.map(mapping)

    status_df = df.groupby("UBID").apply(classify).reset_index(name="Status")
    risk_df = df.groupby("UBID").apply(risk).reset_index(name="Risk")

    df = df.merge(status_df, on="UBID")
    df = df.merge(risk_df, on="UBID")

    # -----------------------------
    # FILTERS
    # -----------------------------
    st.sidebar.header("Filters")

    status_filter = st.sidebar.multiselect("Status", df["Status"].unique(), default=df["Status"].unique())
    risk_filter = st.sidebar.multiselect("Risk", df["Risk"].unique(), default=df["Risk"].unique())

    search = st.sidebar.text_input("Search Business")

    df_filtered = df[
        (df["Status"].isin(status_filter)) &
        (df["Risk"].isin(risk_filter))
    ]

    if search:
        df_filtered = df_filtered[df_filtered["name"].str.contains(search, case=False)]

    # -----------------------------
    # METRICS
    # -----------------------------
    col1, col2, col3 = st.columns(3)
    col1.metric("Businesses", df_filtered["UBID"].nunique())
    col2.metric("Active", sum(df_filtered["Status"] == "Active"))
    col3.metric("Risk Cases", sum(df_filtered["Risk"] != "OK"))

    # -----------------------------
    # TABLE
    # -----------------------------
    st.subheader("📊 Business Data")
    st.dataframe(df_filtered[["name", "address", "UBID", "Status", "Risk"]])

    # -----------------------------
    # CHARTS
    # -----------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Status Distribution")
        st.bar_chart(df_filtered["Status"].value_counts())

    with col2:
        st.subheader("Risk Distribution")
        st.bar_chart(df_filtered["Risk"].value_counts())

    # -----------------------------
    # INTERACTIVE GRAPH
    # -----------------------------
    st.subheader("🔗 Business Relationship Graph")

    G = nx.Graph()

    for i, row in df_filtered.iterrows():
        G.add_node(i, label=row["name"])

    for ubid, group in df_filtered.groupby("UBID"):
        nodes = list(group.index)
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                G.add_edge(nodes[i], nodes[j])

    pos = nx.spring_layout(G, seed=42)

    edge_x = []
    edge_y = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    node_x = []
    node_y = []
    text = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        text.append(df_filtered.loc[node, "name"])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=1),
        hoverinfo='none'
    ))

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=text,
        textposition="top center",
        marker=dict(size=12),
        hoverinfo='text'
    ))

    fig.update_layout(
        showlegend=False,
        height=500,
        margin=dict(l=0, r=0, t=0, b=0)
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # INSIGHTS PANEL
    # -----------------------------
    st.subheader("🧠 Insights")

    if sum(df_filtered["Risk"] != "OK") > 0:
        st.warning("Some businesses lack inspection activity.")
    else:
        st.success("All businesses have proper inspection records.")

    st.info("System identifies duplicates and assigns UBID automatically.")
