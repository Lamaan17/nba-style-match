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
    "Shooter", "Scorer", "Efficient",  # Scoring
    "Playmaker", "Pass-First", "Creative Spark",  # Creation
    "Defender", "Rim Protector", "Switchable",  # Defense
    "Team-First", "Leader", "Hustler",  # Intangibles
    "Versatile", "All-Rounder", "Unpredictable"  # Adaptability
]

trait_descriptions = {
    "Shooter": "Strong from long-range or mid-range scoring",
    "Playmaker": "Creates opportunities for teammates",
    "Defender": "Good at preventing opponents from scoring",
    "Pass-First": "Prioritizes assists over own scoring",
    "Team-First": "Always puts team goals above personal stats",
    "Leader": "Vocal or emotional leader on the court",
    "Versatile": "Plays multiple positions or roles",
    "All-Rounder": "Balances multiple skills well",
    "Scorer": "Tends to score a lot",
    "Creative Spark": "Plays with flair, makes unexpected plays",
    "Rim Protector": "Excellent at blocking or altering shots",
    "Hustler": "High energy, goes after every play",
    "Switchable": "Can guard multiple positions",
    "Unpredictable": "Hard to read, spontaneous decisions",
    "Efficient": "Makes the most out of every possession"
}

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

# User quiz and trait vector
st.markdown("### 👤 Tell us about yourself")
user_traits = []
for question, options in quiz_questions.items():
    choice = st.radio(question, list(options.keys()), key=question)
    user_traits.extend(options[choice])

user_vector = np.array([1 if trait in user_traits else 0 for trait in all_traits]).reshape(1, -1)
player_matrix = np.array([
    [1 if trait in player_traits[player] else 0 for trait in all_traits]
    for player in player_traits
])

# Cosine similarity
similarities = cosine_similarity(user_vector, player_matrix)[0]
matched_player = list(player_traits.keys())[np.argmax(similarities)]

# Display result
st.markdown(f"### 🏆 You match most with **{matched_player}**")

# Trait radar chart
categories = all_traits
user_values = user_vector.flatten()
match_vector = np.array([1 if trait in player_traits[matched_player] else 0 for trait in all_traits])

fig_radar = go.Figure()
fig_radar.add_trace(go.Scatterpolar(r=user_values, theta=categories, fill='toself', name='You'))
fig_radar.add_trace(go.Scatterpolar(r=match_vector, theta=categories, fill='toself', name=matched_player))
fig_radar.update_layout(title="Trait Radar Chart", polar=dict(radialaxis=dict(visible=True)), showlegend=True)
st.plotly_chart(fig_radar)

# Stat comparison
st.markdown("### 📊 Stat Comparison")
user_stats = {'PTS': random.uniform(15, 30), 'REB': random.uniform(4, 10), 'AST': random.uniform(3, 9)}
bar_fig = go.Figure()
for stat in user_stats:
    bar_fig.add_trace(go.Bar(name='You', x=[stat], y=[user_stats[stat]]))
    bar_fig.add_trace(go.Bar(name=matched_player, x=[stat], y=[player_stats[matched_player][stat]]))
bar_fig.update_layout(barmode='group', title="Your Stats vs Matched Player")
st.plotly_chart(bar_fig)

# KMeans Clustering and PCA
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(player_matrix)

st.markdown("### 🧭 NBA Style Clusters")
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(np.vstack([user_vector, player_matrix]))
user_coords = reduced_data[0]
player_coords = reduced_data[1:]

cluster_labels = kmeans.labels_
cluster_colors = ['#636EFA', '#EF553B', '#00CC96']

fig = go.Figure()
for i, name in enumerate(player_traits.keys()):
    fig.add_trace(go.Scatter(
        x=[player_coords[i][0]], y=[player_coords[i][1]],
        mode='markers', name=name,
        marker=dict(size=10, color=cluster_colors[cluster_labels[i]], opacity=0.6),
        hovertext=name
    ))

fig.add_trace(go.Scatter(
    x=[user_coords[0]], y=[user_coords[1]],
    mode='markers+text', name='You',
    marker=dict(size=14, color='black', symbol='x'),
    text=['You'], textposition='top center'
))

fig.update_layout(
    title="Your NBA Style in Cluster Space",
    xaxis_title="PCA Component 1",
    yaxis_title="PCA Component 2",
    height=500,
    showlegend=False
)
st.plotly_chart(fig)

# How it works section
st.markdown("### 🧠 How this works")
st.markdown("""
This app uses:
- Your quiz answers to build a **trait vector**
- **Cosine similarity** to find the most similar NBA player
- **Radar chart** to compare traits
- **Bar chart** for stat comparison
- **KMeans clustering + PCA** to visualize play styles in 2D
""")