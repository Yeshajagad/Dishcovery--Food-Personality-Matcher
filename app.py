# importing all the librabries
# personality metadata
# streamlit ui
def answers_to_features(answers):
    return np.array(list(answers.values())).reshape(1,-1)

def save_user_profile(name, features , personality):
    profile = {
        "name" : name,
        "features" : [int(x) for x in features.tolist()[0]],
        "personality" : str(personality)
    }
    with open(user_file, "a") as f:
        f.write(json.dumps(profile) + "\n")

def load_all_users():
    if not os.path.exists(user_file):
        return []
    with open(user_file, "r") as f:
        return [json.loads(line) for line in f if line.strip()]

def find_matches(user_features , all_users , top_n = 3):
    matches = []
    for u in all_users:
        person_vec = np.array(u["features"])
        sim = cosine_similarity(user_features , [person_vec])[0][0]
        matches.append({
            "name" : u["name"],
            "personality": u["personality"],
            "compatibility" : round(sim * 100 , 2)
        })
    return sorted(matches , key = lambda x : x["compatibility"] , reverse=True)[:top_n]


import streamlit as st
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import pickle , os , json 
from sklearn.metrics.pairwise import cosine_similarity

# model loading and personality meta data
with open("model/trained_model.pkl" , "rb") as f:
    model = pickle.load(f)
with open("model/personality_types.json" , "r")  as f:
    personality_types = json.load(f)
with open("model/label_map.json" , "r") as f:
    label_map = json.load(f)
# for user data 
os.makedirs("data" , exist_ok=True)
user_file = "data/users.json"

# ui based on streamlit
st.set_page_config(page_title = "Food personality Matcher" , page_icon="🍕")
st.title(" Food Personality Matcher")

st.markdown("Answer few fun questions and discover your Food Personality along with the Food Buddies !!!")
name = st.text_input("Enter your name")
# questions

gender = st.selectbox("Select your gender" , ["female" , "male" , "other"])

questions = {
    "sweet_preference": st.slider("How much do you love sweets?", 1, 10, 5),
    "spice_tolerance": st.slider("How much spice can you handle?", 1, 10, 5),
    "texture_preference": st.slider("Do textures in food matter?", 1, 10, 5),
    "flavor_intensity": st.slider("Do you enjoy bold flavors?", 1, 10, 5),
    "period_cravings": 0,
    "stress_eating": st.slider("Do you eat more under stress?", 1, 10, 5),
    "gaming_snacks": 0,
    "hearty_meals": st.slider("Do you love hearty meals?", 1, 10, 5),
    "late_night": st.slider("How often do you eat late night?", 1, 10, 5),
    "stress_response": st.slider("Do you crave food when anxious?", 1, 10, 5),
    "celebration": st.slider("Do you celebrate with food?", 1, 10, 5),
    "weather_craving": st.slider("Do you crave food based on weather?", 1, 10, 5),
    "adventure_level": st.slider("How adventurous are you with new food?", 1, 10, 5),
    "cooking_preference": st.slider("Do you enjoy cooking yourself?", 1, 10, 5),
    "social_eating": st.slider("Do you enjoy eating socially?", 1, 10, 5),
    "quick_meal": st.slider("Do you prefer quick meals?", 1, 10, 5),
    "budget_priority": st.slider("Do you prioritize budget over taste?", 1, 10, 5),
    "instagram_behavior": st.slider("Do you post food on Instagram?", 1, 10, 5),
    "comfort_food": st.slider("Do you rely on comfort food?", 1, 10, 5),
    "food_risk": st.slider("Do you take risks in trying weird foods?", 1, 10, 5),
}

# gender-specific questions
if gender == "female":
    questions["period_cravings"] = st.select_slider(
        "How strong are your food cravings during periods?",
        options=[0,1,2,3,4,5],
        value=0
    )
elif gender == "male":
    questions["gaming_snacks"] = st.select_slider(
        "How often do you crave for snacks while gaming?",
        options=[0,1,2,3,4,5],
        value=0
    )
else:
    questions["period_cravings"] = 0
    questions["gaming_snacks"] = 0

if st.button("Get My Food Personality"):
    if not name:
        st.error("Plese enter you name before submitting")
    else:
        features = answers_to_features(questions)
        prediction = model.predict(features)[0]
        id_to_presonality = label_map["id_to_personality"]
        personality = id_to_presonality[str(prediction)]

        save_user_profile(name,features , personality)

        all_users = load_all_users()
        matches = [u for u in all_users if u["name"] != name]
        top_matches = find_matches(features , matches)

        st.subheader(f"🎉 {name}, your Food Personality is: **{personality}**")
        id_to_presonality = label_map["id_to_personality"]
        personality_to_id = label_map["personality_to_id"]

        prediction = model.predict(features)[0]
        personality = id_to_presonality[str(prediction)]

        # Show traits + description
        if personality in personality_types:
            st.write("**Traits:**")
            for trait in personality_types[personality]["traits"]:
                st.markdown(f"- {trait}")

            st.write("**Description:**")
            st.info(personality_types[personality]["description"])
        else:
            st.warning("No detailed info available for this personality yet.")

        # top matches
        if top_matches:
            st.subheader("Your TopFood Buddies")
            for m in top_matches:
                st.write(f"👉 {m['name']} ({m['personality']}) — Compatibility: {m['compatibility']}%")
        else:
            st.info("No matches yet  . You'r the first user! ")
