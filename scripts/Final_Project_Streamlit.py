import os
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# Load needed data computed in the Final_Project.py script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_jobs():
    path = os.path.join(BASE_DIR, '..', 'datasets', 'updated_jobs_grouped.csv')
    jobs = pd.read_csv(path)
    return jobs

jobs=load_jobs()

def load_skills():
    path = os.path.join(BASE_DIR, '..', 'datasets', 'Merged Jobs.xlsx')
    ESCO = pd.ExcelFile(path)
    skills_tab = pd.read_excel(ESCO, 'Merged Table')
    skills = skills_tab['skills_en.preferredLabel'].dropna().unique()
    return skills
skills=load_skills()

#Extracting unique domains of interest from the jobs dataset
student_domains= jobs['Domain of Interest'].dropna().unique()

# Setting up the UI configuration and style using streamlit
st.set_page_config(page_title="Career Path Recommender System", layout='wide')
st.title("Career Path Recommender System")

st.markdown("""<style>
    body {background-color: #e6f2ff;}
    .stApp {background-color: #e6f2ff;}
    </style>""", unsafe_allow_html=True)

#Define input parameters for the user to select from (domain and skills)
selected_domain = st.selectbox("Select a domain of interest", options=student_domains, index=None)
selected_skills = st.multiselect("Select up to 5 skills", options=skills, max_selections=5)
st.write("If you start typing, the system will give you suggestions matching your entry.") 

# Define a recommender function using cosine similarity and top 3 matches
def recommend_career(selected_skills, selected_domain, data=jobs):
    if not selected_skills or not selected_domain:
        return []
    
    #Remove spaces for multiworded skills to improve vectorization accuracy
    selected_skills = [skill.replace(' ', '') for skill in selected_skills]

    #Concatenate skills into a single string to make them vectorizable
    user_input = ",".join(selected_skills) + "," + selected_domain

    #Vectorize the job profiles and the student profile
    vectorizer = TfidfVectorizer()
    vec_jobs = vectorizer.fit_transform(jobs['Skill and Domain'])
    vec_user = vectorizer.transform([user_input])

    #Compute similarity score
    similarity_scores = cosine_similarity(vec_user, vec_jobs).flatten()
    top_matches = similarity_scores.argsort()[-3:][::-1]

    #Append 3 best matches to the list of recommendations
    recommendations = []
    for i in top_matches:
        job_row = jobs.iloc[i]
        recommendations.append({
            "occupation": job_row['Occupation'],
            "similarity": round(similarity_scores[i] * 100, 1),
            "description":job_row.get('Description')
        })

    return recommendations

# Display 3 top matches alongside with similarity score and job description to the user
if selected_skills and selected_domain:
    st.subheader("Top 3 Career Recommendations:")
    results = recommend_career(selected_skills, selected_domain)
    for i, rec in enumerate(results, 1):
        st.markdown(f"{i}. {rec['occupation']}")
        st.markdown(f"Similarity Score: {rec['similarity']}%")
        st.markdown(f"{rec['description']}", unsafe_allow_html=True)
        st.markdown("---")  


       







