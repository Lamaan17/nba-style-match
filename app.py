# Streamlit NBA Player Style Match Tool with ML-style Cosine Similarity
import streamlit as st
import random
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity

# Define trait list and player data
all_traits = [
    "Shooter", "Playmaker", "Defender", "Pass-First", "Team-First",
    "Leader", "Versatile", "All-Rounder", "Scorer", "Creative Spark",
    "Rim Protector", "Hustler", "Switchable", "Unpredictable", "Efficient"
]

player_traits = {
    "Tyrese Haliburton": ["Shooter", "Playmaker", "Defender", "Pass-First", "Team-First", "Leader", "Versatile", "All-Rounder"],
    "Jayson Tatum": ["Scorer", "Leader", "Versatile", "All-Rounder", "Creative Spark"],
    "Stephen Curry": ["Shooter", "Scorer", "Versatile", "All-Rounder"],
    "Evan Mobley": ["Rim Protector", "Hustler", "Versatile", "All-Rounder"],
    "Giannis Antetokounmpo": ["Slasher", "Scorer", "Hustler", "Leader", "Versatile", "All-Rounder", "Creative Spark"],
    "Draymond Green": ["Defender", "Hustler", "Switchable", "Team-First", "Versatile", "All-Rounder"],
    "Jordan Clarkson": ["Versatile", "All-Rounder"],
    "OG Anunoby": ["Defender", "Hustler", "Switchable", "Versatile"],
    "Ja Morant": ["Unpredictable", "Versatile", "All-Rounder", "Creative Spark"],
    "Luka Doncic": ["Scorer", "Hustler", "Unpredictable", "Leader", "Versatile", "All-Rounder", "Creative Spark"],
    "Jimmy Butler": ["Hustler", "Team-First", "Versatile", "All-Rounder", "Leader"]
}

player_stats = {
    "Tyrese Haliburton": {"PPG": 20.1, "AST": 10.2, "3P%": 40.2},
    "Jayson Tatum": {"PPG": 26.4, "AST": 4.5, "3P%": 37.1},
    "Stephen Curry": {"PPG": 29.7, "AST": 6.3, "3P%": 42.3},
    "Evan Mobley": {"PPG": 15.0, "AST": 2.3, "3P%": 26.0},
    "Giannis Antetokounmpo": {"PPG": 31.1, "AST": 5.7, "3P%": 28.0},
    "Draymond Green": {"PPG": 8.4, "AST": 6.8, "3P%": 31.2},
    "Jordan Clarkson": {"PPG": 17.2, "AST": 4.0, "3P%": 34.9},
    "OG Anunoby": {"PPG": 16.7, "AST": 2.0, "3P%": 38.6},
    "Ja Morant": {"PPG": 27.4, "AST": 7.8, "3P%": 34.0},
    "Luka Doncic": {"PPG": 32.4, "AST": 8.5, "3P%": 36.9},
    "Jimmy Butler": {"PPG": 21.2, "AST": 5.3, "3P%": 35.6}
}

quiz_questions = {
    "At home, I am…": {
        "The quiet one who fixes things": ["Efficient"],
        "The energy in the room": ["Slasher", "Hustler"],
        "The one always making a plan": ["Playmaker", "Leader"],
        "The unexpected wildcard": ["Unpredictable"]
    },
    "At work, I am…": {
        "Calm and strategic": ["Efficient", "All-Rounder"],
        "Always delivering": ["Scorer", "Team-First"],
        "The connector": ["Pass-First", "Playmaker"],
        "Running the show": ["Leader"]
    },
    "In my free time, I am…": {
        "Always learning": ["Creative Spark"],
        "Building something quietly": ["Efficient"],
        "With my crew": ["Team-First", "Versatile"],
        "Trying new things": ["Unpredictable"]
    }
}

st.set_page_config(page_title="NBA Player Style Match", layout="centered")
st.title("🏀 NBA Player Style Match Tool")
st.write("Pick the options that best describe you — we'll find your NBA-style twin using cosine similarity and trait scoring logic.")

user_traits = []
for q, options in quiz_questions.items():
    answer = st.radio(q, list(options.keys()), key=q)
    user_traits.extend(options[answer])

# Convert traits to binary vector
user_vector = np.array([1 if trait in user_traits else 0 for trait in all_traits]).reshape(1, -1)

# Create player vectors
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
    st.success(f"🎯 You match with **{best_match}**!")

    # Trait radar chart
    user_trait_vector = [1 if t in user_traits else 0 for t in all_traits]
    match_trait_vector = [1 if t in player_traits[best_match] else 0 for t in all_traits]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=user_trait_vector, theta=all_traits, fill='toself', name='You'))
    fig.add_trace(go.Scatterpolar(r=match_trait_vector, theta=all_traits, fill='toself', name=best_match))
    fig.update_layout(title="Trait Radar Comparison", polar=dict(radialaxis=dict(visible=True)), showlegend=True)
    st.plotly_chart(fig)

    # Stats comparison chart
    stat_df = pd.DataFrame(player_stats).T.loc[df_scores["Player"].head(5)]
    st.markdown("### 📊 Player Stat Comparison")
    for stat in stat_df.columns:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=stat_df.index, y=stat_df[stat], name=stat))
        fig.update_layout(title=f"{stat} Comparison", xaxis_title="Player", yaxis_title=stat)
        st.plotly_chart(fig)

    # Show similarity scores
    st.markdown("### 🔍 Similarity Scores")
    st.dataframe(df_scores.head(5))

    # Explanation
    with st.expander("How this works"):
        st.write("""
        We convert your selected traits into a binary feature vector.
        Each NBA player has their own trait vector.
        We compute cosine similarity between your vector and each player's vector — the player with the highest similarity score is your best match.
        This mimics feature-based recommendation systems.
        """)
