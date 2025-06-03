# Streamlit NBA Player Style Match Tool
import streamlit as st
import random

# Predefined player trait data
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

# Quiz questions and answers (mapped to traits)
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

# UI
st.set_page_config(page_title="NBA Player Style Match", layout="centered")
st.title("🏀 NBA Player Style Match Tool")
st.write("Pick the options that best describe you — we'll find your NBA-style twin using soft data + matching logic.")

# Collect user responses
user_traits = []
for q, options in quiz_questions.items():
    answer = st.radio(q, list(options.keys()), key=q)
    user_traits.extend(options[answer])

# When ready, show results
if st.button("🧠 Show My NBA Match"):
    st.subheader("Here's your NBA Style Twin:")

    # Scoring: Count overlapping traits
    match_scores = {}
    for player, traits in player_traits.items():
        score = len(set(user_traits) & set(traits))
        match_scores[player] = score

    # Best match (break ties randomly)
    best_score = max(match_scores.values())
    top_players = [p for p, s in match_scores.items() if s == best_score]
    best_match = random.choice(top_players)

    # Show result
    st.success(f"🎯 You match with **{best_match}**!")
    st.markdown("---")
    st.markdown("**Trait Match Breakdown:**")
    for trait in user_traits:
        st.write(f"- {trait}")

    st.markdown("---")
    st.info("This tool maps soft personality indicators to NBA player archetypes using feature similarity — a data-driven recommendation approach similar to customer profiling or persona modeling.")
