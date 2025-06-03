# Streamlit NBA Player Style Match Tool with ML-style Cosine Similarity
import streamlit as st
import random
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.title("🏀 NBA Player Style Match")

# Define grouped and sorted trait list
all_traits = [
    "Shooter", "Scorer", "Efficient",
    "Playmaker", "Pass-First", "Creative Spark",
    "Defender", "Rim Protector", "Switchable",
    "Team-First", "Leader", "Hustler",
    "Versatile", "All-Rounder", "Unpredictable"
]

quiz_questions = {
    "At home I am...": {
        "the planner": ["Efficient", "Leader"],
        "the entertainer": ["Creative Spark", "Unpredictable"],
        "the chill one": ["Pass-First", "Shooter"]
    },
    "At work I am...": {
        "data-driven": ["Efficient", "All-Rounder"],
        "collaborative": ["Team-First", "Pass-First"],
        "innovative": ["Creative Spark", "Versatile"]
    },
    "In free time I am...": {
        "watching sports": ["Scorer", "Shooter"],
        "playing games": ["Unpredictable", "Versatile"],
        "volunteering": ["Hustler", "Team-First"]
    }
}

player_traits = {
    'Tyrese Haliburton': ['Shooter', 'Playmaker', 'Pass-First', 'Defender'],
    'Jayson Tatum': ['Scorer', 'Shooter', 'Playmaker', 'Hustler', 'All-Rounder', 'Team-First', 'Versatile'],
    'Stephen Curry': ['Shooter', 'Playmaker'],
    'Evan Mobley': ['Rim Protector', 'Hustler'],
    'Giannis Antetokounmpo': ['Scorer', 'Playmaker', 'Rim Protector', 'Hustler', 'All-Rounder', 'Team-First'],
    'Nikola Jokic': ['Scorer', 'Playmaker', 'Pass-First', 'Hustler', 'All-Rounder', 'Team-First'],
    'Jimmy Butler': ['Scorer', 'Playmaker', 'Defender', 'Hustler', 'All-Rounder', 'Team-First'],
    'LeBron James': ['Scorer', 'Playmaker', 'Hustler', 'All-Rounder', 'Team-First'],
    'Shai Gilgeous-Alexander': ['Scorer', 'Playmaker', 'Defender', 'Team-First'],
    'Anthony Edwards': ['Scorer', 'Shooter', 'Defender', 'Team-First']
}

player_stats = {
    'Tyrese Haliburton': {'PTS': 18.6, 'REB': 3.5, 'AST': 9.2},
    'Jayson Tatum': {'PTS': 26.8, 'REB': 8.7, 'AST': 6.0},
    'Stephen Curry': {'PTS': 24.5, 'REB': 4.4, 'AST': 6.0},
    'Evan Mobley': {'PTS': 18.5, 'REB': 9.3, 'AST': 3.2},
    'Giannis Antetokounmpo': {'PTS': 30.4, 'REB': 11.9, 'AST': 6.5},
    'Nikola Jokic': {'PTS': 25.9, 'REB': 12.2, 'AST': 9.4},
    'Jimmy Butler': {'PTS': 22.4, 'REB': 5.9, 'AST': 5.0},
    'LeBron James': {'PTS': 27.5, 'REB': 8.0, 'AST': 8.3},
    'Shai Gilgeous-Alexander': {'PTS': 31.4, 'REB': 5.5, 'AST': 6.2},
    'Anthony Edwards': {'PTS': 25.9, 'REB': 5.4, 'AST': 4.5}
}

st.markdown("### 👤 Tell us about yourself")
user_traits = []
quiz_answers = {}

for question, options in quiz_questions.items():
    choice = st.radio(question, list(options.keys()), key=question)
    quiz_answers[question] = choice
    user_traits.extend(options[choice])

ready = all(q in quiz_answers for q in quiz_questions)

if ready and st.button("🔍 See my NBA match"):
    user_vector = np.array([1 if trait in user_traits else 0 for trait in all_traits]).reshape(1, -1)
    player_matrix = np.array([
        [1 if trait in player_traits[player] else 0 for trait in all_traits]
        for player in player_traits
    ])

    similarities = cosine_similarity(user_vector, player_matrix)[0]
    matched_player = list(player_traits.keys())[np.argmax(similarities)]

    # --- Similarity Table ---
    top_indices = similarities.argsort()[::-1][:3]
    st.markdown("### 🔗 Top 3 Similar Players")
    sim_df = pd.DataFrame({
        'Player': [list(player_traits.keys())[i] for i in top_indices],
        'Similarity Score': [round(similarities[i], 2) for i in top_indices]
    })
    st.dataframe(sim_df)

    st.markdown(f"### 🏆 You match most with **{matched_player}**")

    # --- Trait Radar Chart ---
    categories = all_traits
    match_vector = np.array([1 if trait in player_traits[matched_player] else 0 for trait in all_traits])
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(r=match_vector, theta=categories, fill='toself', name=matched_player))
    fig_radar.update_layout(title="Trait Radar Chart", polar=dict(radialaxis=dict(visible=True)), showlegend=True)
    st.plotly_chart(fig_radar)

    # --- Stat Comparison Bar Graph ---
    st.markdown("### 📊 Stat Comparison")
    league_avg = {k: np.mean([player_stats[p][k] for p in player_stats]) for k in ['PTS', 'REB', 'AST']}
    bar_fig = go.Figure()
    for stat in ['PTS', 'REB', 'AST']:
        bar_fig.add_trace(go.Bar(name=matched_player, x=[stat], y=[player_stats[matched_player][stat]]))
        bar_fig.add_trace(go.Bar(name='NBA Avg', x=[stat], y=[league_avg[stat]]))
    bar_fig.update_layout(barmode='group', title="Matched Player vs NBA Average")
    st.plotly_chart(bar_fig)

    # --- Clustering with PCA ---
    st.markdown("### 🧭 NBA Style Clusters")
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(player_matrix)
    cluster_labels = kmeans.labels_

    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(np.vstack([user_vector, player_matrix]))
    user_coords = reduced_data[0]
    player_coords = reduced_data[1:]

    cluster_colors = ['#636EFA', '#EF553B', '#00CC96']
    fig = go.Figure()
    for i, name in enumerate(player_traits.keys()):
        fig.add_trace(go.Scatter(
            x=[player_coords[i][0]], y=[player_coords[i][1]],
            mode='markers+text', name=name,
            text=name, marker=dict(size=10, color=cluster_colors[cluster_labels[i]], opacity=0.6)
        ))

    fig.add_trace(go.Scatter(
        x=[user_coords[0]], y=[user_coords[1]],
        mode='markers+text', name='You',
        marker=dict(size=14, color='black', symbol='x'),
        text=['You'], textposition='top center'
    ))

    fig.update_layout(
        title="Your NBA Style in Cluster Space",
        xaxis_title="PCA Axis 1: Style Dimension",
        yaxis_title="PCA Axis 2: Role Dimension",
        height=500, showlegend=False
    )
    st.plotly_chart(fig)

    st.caption("🔵 = Scorer Cluster · 🔴 = Playmaker Cluster · 🟢 = Hybrid Cluster")

    # --- How it Works ---
    with st.expander("🔎 How this works"):
        st.markdown("""
        This app uses:
        - Your quiz answers to build a **trait vector**
        - **Cosine similarity** to find the most similar NBA player
        - **Radar chart** to compare traits
        - **Bar chart** for stat comparison (with league average)
        - **KMeans clustering + PCA** to visualize play styles in 2D
        """)
