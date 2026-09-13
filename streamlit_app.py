import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch_geometric.nn import GCNConv

st.set_page_config(page_title="Anakin T-GNN Live Explorer", layout="wide")

st.markdown("""
<style>
h1 {font-size: 28px !important;}
h2 {font-size: 20px !important;}
h3 {font-size: 16px !important;}
p, label, .stMarkdown {font-size: 13px !important;}
[data-testid="stMetricValue"] {font-size: 20px !important;}
[data-testid="stMetricLabel"] {font-size: 11px !important;}
[data-testid="stSidebar"] * {font-size: 12px !important;}
</style>
""", unsafe_allow_html=True)

st.title("Psychological Trajectory")


class TemporalAnakinGNN(nn.Module):
    def __init__(self, num_node_features=3, hidden_dim=16, num_classes=5):
        super().__init__()
        self.conv = GCNConv(num_node_features, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, edge_weight, h):
        out = self.conv(x, edge_index, edge_weight)
        out = torch.relu(out)
        h = self.gru(out, h)
        pred = self.classifier(h)
        return pred, h


@st.cache_resource
def load_model_and_data():
    edges_df = pd.read_csv("data/temporal_edges.csv")
    node_states_df = pd.read_csv("data/node_states.csv")
    characters_df = pd.read_csv("data/characters.csv")

    model = TemporalAnakinGNN(
        num_node_features=3,
        hidden_dim=16,
        num_classes=5
    )

    checkpoint = torch.load(
        "model/anakin_tgnn.pth",
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(checkpoint)
    model.eval()

    return edges_df, node_states_df, characters_df, model


edges_df, node_states_df, characters_df, model = load_model_and_data()

eras = sorted(
    edges_df["Era_Step"].unique(),
    key=lambda x: int(str(x).replace("t", ""))
)

st.sidebar.header("Scenario Controls")

selected_era = st.sidebar.selectbox(
    "Era Snapshot",
    eras,
    index=min(9, len(eras) - 1)
)

stress_override = st.sidebar.slider(
    "Stress Score",
    0.0, 1.0, 0.5, 0.01
)

palpatine_override = st.sidebar.slider(
    "Palpatine Proximity",
    0.0, 1.0, 0.5, 0.01
)


# ---------------------------------------------------------
# Node mapping
# ---------------------------------------------------------

character_nodes = list(characters_df["Node_ID"])

for node in edges_df["Source"]:
    if node not in character_nodes:
        character_nodes.append(node)

for node in edges_df["Target"]:
    if node not in character_nodes:
        character_nodes.append(node)

for node in node_states_df["Node_ID"]:
    if node not in character_nodes:
        character_nodes.append(node)

all_nodes = character_nodes

node_to_idx = {
    node: i for i, node in enumerate(all_nodes)
}

num_nodes = len(all_nodes)
HIDDEN_DIM = 16


# ---------------------------------------------------------
# Base features
# ---------------------------------------------------------

base_features = torch.zeros(
    (num_nodes, 3),
    dtype=torch.float32
)

for _, row in characters_df.iterrows():
    node = row["Node_ID"]

    if node not in node_to_idx:
        continue

    value = row["Force_Sensitive"]

    if isinstance(value, str):
        value = value.strip().lower() in [
            "true", "1", "yes"
        ]
    else:
        value = bool(value)

    base_features[
        node_to_idx[node], 0
    ] = 1.0 if value else 0.0


def get_graph(era):
    era_edges = edges_df[
        edges_df["Era_Step"] == era
    ]

    if era_edges.empty:
        return (
            torch.empty((2, 0), dtype=torch.long),
            torch.empty(0, dtype=torch.float32)
        )

    sources = []
    targets = []
    weights = []

    for _, row in era_edges.iterrows():
        source = row["Source"]
        target = row["Target"]

        if source not in node_to_idx:
            continue

        if target not in node_to_idx:
            continue

        sources.append(node_to_idx[source])
        targets.append(node_to_idx[target])
        weights.append(float(row["Weight"]))

    if not sources:
        return (
            torch.empty((2, 0), dtype=torch.long),
            torch.empty(0, dtype=torch.float32)
        )

    return (
        torch.tensor(
            [sources, targets],
            dtype=torch.long
        ),
        torch.tensor(
            weights,
            dtype=torch.float32
        )
    )


def get_target_node(era):
    if era == "t14" and "Vader" in node_to_idx:
        return "Vader"

    if era == "t16" and "Anakin/Vader" in node_to_idx:
        return "Anakin/Vader"

    if "Anakin" in node_to_idx:
        return "Anakin"

    if "Vader" in node_to_idx:
        return "Vader"

    if "Anakin/Vader" in node_to_idx:
        return "Anakin/Vader"

    return None


# =========================================================
# MAIN
# =========================================================

col1, col2 = st.columns(2, gap="small")


# =========================================================
# SOCIAL GRAPH
# =========================================================

with col1:
    st.subheader(f"Social Graph — {selected_era}")

    era_edges = edges_df[
        edges_df["Era_Step"] == selected_era
    ]

    if era_edges.empty:
        st.warning("No edges available.")
    else:
        G = nx.from_pandas_edgelist(
            era_edges,
            source="Source",
            target="Target",
            edge_attr="Weight"
        )

        fig, ax = plt.subplots(
            figsize=(4.2, 3.2)
        )

        pos = nx.spring_layout(
            G,
            seed=42
        )

        colors = []

        for node in G.nodes():
            if node in [
                "Anakin",
                "Vader",
                "Anakin/Vader"
            ]:
                colors.append(
                    "red"
                    if palpatine_override > 0.6
                    else "orange"
                )
            elif node == "Palpatine":
                colors.append("purple")
            else:
                colors.append("skyblue")

        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color=colors,
            node_size=1100,
            font_size=7,
            font_weight="bold",
            ax=ax
        )

        ax.set_title(
            f"Social Network — {selected_era}",
            fontsize=10
        )

        st.pyplot(
            fig,
            clear_figure=True
        )


# =========================================================
# MENTAL STATE PCA
# =========================================================

with col2:
    st.subheader("Mental-State Trajectory")

    hidden_history = []
    mental_states = []
    state_classes = []
    era_labels = []

    selected_logits = None
    selected_state = None
    selected_node = None

    model.eval()

    with torch.no_grad():
        h = torch.zeros(
            (num_nodes, HIDDEN_DIM),
            dtype=torch.float32
        )

        for era in eras:
            era_state = node_states_df[
                node_states_df["Era_Step"] == era
            ]

            if era_state.empty:
                continue

            target_node = get_target_node(era)

            if target_node is None:
                continue

            target_idx = node_to_idx[target_node]

            edge_index, edge_weight = get_graph(era)

            x = base_features.clone()

            target_state = era_state[
                era_state["Node_ID"] == target_node
            ]

            if target_state.empty:
                target_state = era_state.iloc[[0]]

            row = target_state.iloc[0]

            x[target_idx, 1] = float(
                row["Stress_Score"]
            )

            x[target_idx, 2] = float(
                row["Palpatine_Proximity"]
            )

            # What-if values affect only selected era.
            if era == selected_era:
                x[target_idx, 1] = stress_override
                x[target_idx, 2] = palpatine_override

            pred, h = model(
                x,
                edge_index,
                edge_weight,
                h
            )

            hidden_history.append(
                h[target_idx]
                .detach()
                .cpu()
                .numpy()
            )

            mental_states.append(
                str(row["Mental_State"])
            )

            state_classes.append(
                int(row["State_Label"])
            )

            era_labels.append(era)

            if era == selected_era:
                selected_logits = pred[
                    target_idx
                ].clone()

                selected_state = str(
                    row["Mental_State"]
                )

                selected_node = target_node

    if len(hidden_history) >= 2:
        hidden_array = np.array(
            hidden_history
        )

        pca = PCA(
            n_components=2
        )

        reduced = pca.fit_transform(
            hidden_array
        )

        fig, ax = plt.subplots(
            figsize=(4.2, 3.2)
        )

        ax.plot(
            reduced[:, 0],
            reduced[:, 1],
            "--",
            color="gray",
            alpha=0.6,
            linewidth=1
        )

        ax.scatter(
            reduced[:, 0],
            reduced[:, 1],
            c=state_classes,
            cmap="coolwarm",
            s=80,
            edgecolors="black",
            zorder=2
        )

        for i, label in enumerate(
            mental_states
        ):
            ax.annotate(
                label,
                (
                    reduced[i, 0],
                    reduced[i, 1]
                ),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                fontweight="bold"
            )

        if selected_era in era_labels:
            selected_idx = era_labels.index(
                selected_era
            )

            ax.scatter(
                reduced[selected_idx, 0],
                reduced[selected_idx, 1],
                s=180,
                facecolors="none",
                edgecolors="red",
                linewidths=2,
                zorder=3
            )

        ax.set_title(
            "Mental-State Representation",
            fontsize=10
        )

        ax.set_xlabel(
            "PC1",
            fontsize=8
        )

        ax.set_ylabel(
            "PC2",
            fontsize=8
        )

        ax.tick_params(
            axis="both",
            labelsize=7
        )

        ax.grid(
            True,
            alpha=0.25
        )

        st.pyplot(
            fig,
            clear_figure=True
        )

        explained = pca.explained_variance_ratio_

        st.caption(
            f"PC1 {explained[0] * 100:.1f}% | "
            f"PC2 {explained[1] * 100:.1f}%"
        )

    else:
        st.info(
            "Not enough observations for PCA."
        )


# =========================================================
# MODEL ASSESSMENT
# =========================================================

st.markdown("---")

st.subheader(
    f"Model Assessment — {selected_era}"
)

if selected_logits is not None:
    probabilities = torch.softmax(
        selected_logits,
        dim=0
    ).cpu().numpy()

    predicted_class = int(
        np.argmax(probabilities)
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Character",
        selected_node
    )

    c2.metric(
        "Mental State",
        selected_state
    )

    c3.metric(
        "Predicted",
        predicted_class
    )

    c4.metric(
        "Stress",
        f"{stress_override:.2f}"
    )

    st.write("**Model Probabilities**")

    prob_df = pd.DataFrame({
        "State": np.arange(5),
        "Probability": np.round(
            probabilities,
            4
        )
    })

    st.bar_chart(
        prob_df.set_index("State"),
        height=180
    )

else:
    st.warning(
        "No prediction available."
    )
