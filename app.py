# Streamlit NBA Player Style Match Tool with ML-style Cosine Similarity
import streamlit as st
import random
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

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

# You will need to define these before similarity logic
quiz_questions = {...}  # assume your existing questions here
player_traits = {...}  # assume your player trait dictionary
player_stats = {...}   # assume your player stat dictionary

user_traits = []
for q, options in quiz_questions.items():
    answer = st.radio(q, list(options.keys()), key=q)
    user_traits.extend(options[answer])

# Build user and player vectors
user_vector = np.array([1 if trait in user_traits else 0 for trait in all_traits]).reshape(1, -1)
data = []
for player, traits in player_traits.items():
    vector = [1 if trait in traits else 0 for trait in all_traits]
    data.append(vector)
player_matrix = np.array(data)

# Cosine similarity scores
similarities = cosine_similarity(user_vector, player_matrix)[0]

# Match players to scores
df_scores = pd.DataFrame({
    "Player": list(player_traits.keys()),
    "Score": similarities
}).sort_values(by="Score", ascending=False)

if st.button("🧠 Show My NBA Match"):
    best_match = df_scores.iloc[0]["Player"]
    second_match = df_scores.iloc[1]["Score"]
    match_gap = df_scores.iloc[0]["Score"] - second_match
    st.success(f"🎯 You match with **{best_match}**!")
    st.info(f"Match confidence: {match_gap:.2f} (difference from next closest)")

    st.markdown("### 🔍 Similarity Scores")
    st.dataframe(df_scores.head(5))

    # Trait radar chart: You vs average NBA player with hovertext
    user_trait_vector = np.array([1 if t in user_traits else 0 for t in all_traits])
    avg_player_vector = player_matrix.mean(axis=0)
    hover_text = [trait_descriptions.get(t, t) for t in all_traits]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=user_trait_vector,
        theta=all_traits,
        fill='toself',
        name='You',
        hovertext=hover_text,
        hoverinfo='text+name'
    ))
    fig.add_trace(go.Scatterpolar(
        r=avg_player_vector,
        theta=all_traits,
        fill='none',
        name='NBA Avg',
        line=dict(dash='dash'),
        hovertext=hover_text,
        hoverinfo='text+name'
    ))
    fig.update_layout(
        title="Trait Radar: You vs. NBA Player Average",
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=True
    )
    st.plotly_chart(fig)

    # Normalize stats for fair comparison
    stat_df = pd.DataFrame(player_stats).T.loc[df_scores["Player"].head(5)]
    stat_df_norm = pd.DataFrame(StandardScaler().fit_transform(stat_df), columns=stat_df.columns, index=stat_df.index)

    st.markdown("### 📊 Normalized Stat Comparison")
    for stat in stat_df_norm.columns:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=stat_df_norm.index, y=stat_df_norm[stat], name=stat, marker_color='indigo'))
        fig.update_layout(
            title=f"{stat} (Z-Score Normalized)",
            xaxis_title="Player",
            yaxis_title="Z-Score",
            template="plotly_dark",
            bargap=0.3,
            height=400
        )
        st.plotly_chart(fig)

    # Optional: Cluster insight using k-means
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(player_matrix)
    labels = kmeans.predict(user_vector)
    st.markdown(f"### 🧬 You belong to Player Cluster #{labels[0]+1}")

    # Explanation
    with st.expander("How this works"):
        st.write("""
        We convert your selected traits into a binary feature vector.
        Each NBA player has their own trait vector.
        We compute cosine similarity between your vector and each player's vector — the player with the highest similarity score is your best match.
        Then, we compare your trait profile to the average NBA player to see how you stand out.
        We also normalize stats for comparison and cluster players by style.
        """)

    with st.expander("🧩 Trait Descriptions"):
        for trait in user_traits:
            st.markdown(f"**{trait}**: {trait_descriptions.get(trait, 'N/A')}")
