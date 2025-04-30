# app.py
import streamlit as st
import joblib
import pandas as pd
import requests
def hybrid_recommend(input_titles, movies, tfidf_sim, svd_sim, sbert_sim, w1=0.2, w2=0.3, w3=0.5, top_n=5):
    indices = []

    # Find index of each input movie title
    for title in input_titles:
        if title not in movies['title'].values:
            raise ValueError(f"Movie '{title}' not found!")
        indices.append(movies[movies['title'] == title].index[0])
        
    # Combine similarity scores using weighted average
    sim_scores = sum(w1 * tfidf_sim[idx] + w2 * svd_sim[idx] + w3 * sbert_sim[idx] for idx in indices)
    sim_scores /= len(indices)  # average for multiple movies

     # Get top N most similar movies (excluding the input ones)
    movie_indices = sim_scores.argsort()[-top_n - len(indices):][::-1]
    # Retrieve recommended movies, excluding ones already in input
    recommended = movies.iloc[movie_indices]
    recommended = recommended[~recommended['title'].isin(input_titles)].head(top_n)  # remove input movies
    
    # after you compute `movie_indices`…
    top_idxs = movie_indices[:top_n]
    rec_df = movies.iloc[top_idxs].reset_index(drop=True)
    rec_df = rec_df[~rec_df['title'].isin(input_titles)].head(top_n)
    return recommended.reset_index(drop=True)

def get_poster_url(movie_id):
    api_key = "88109762c3cd8f3eafe36d3f3e740ab7"
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"
    data = requests.get(url).json()
    if data.get('poster_path'):
        return "https://image.tmdb.org/t/p/w500" + data['poster_path']
    return None

# 1) Load data & similarity matrices
@st.cache_data(show_spinner=False)
def load_components():
    # if you used all_components.pkl:
    comps = joblib.load('all_components.pkl')
    movies       = comps['movies']
    tfidf_sim    = comps['tfidf_sim']
    svd_sim      = comps['svd_sim']
    sbert_sim    = comps['sbert_sim']
    # if you want the vectorizers/models too:
    # tfidf_vec   = comps['tfidf_vectorizer']
    # svd_model    = comps['svd_model']
    return movies, tfidf_sim, svd_sim, sbert_sim

movies, tfidf_sim, svd_sim, sbert_sim = load_components()

# 2) App header
st.set_page_config(layout="wide")
# st.markdown(
#     "<h1 style='text-align: center;'>🎬  Movie Recommender</h1>",
#     unsafe_allow_html=True
# )



# then, instead of st.title(...), do:
st.markdown("""
<div style="
    background-color: ##3D3C3C;
    padding: 12px 12px;
    border-radius: 8px;
    margin-bottom: 12px;
">
    <h1 style="
        color: #3A586F;
        margin: 0;
        font-family: sans-serif;
    ">
    <h1 style='text-align: center;'>🎬  Movie Recommender</h1>
    </h1>
</div>
""", unsafe_allow_html=True)


# 3) Sidebar inputs
# with st.sidebar:
titles = movies['title'].tolist()
selected = st.multiselect("Pick your favorite(s):", titles, default=["Avatar"])

# movie picker
with st.expander("🔧 Configuration"):
    w1 = st.slider("TF-IDF weight", 0.0, 1.0, 0.2, 0.05)
    w2 = st.slider("SVD weight",    0.0, 1.0, 0.3, 0.05)
    w3 = st.slider("SBERT weight",  0.0, 1.0, 0.5, 0.05)


total = w1 + w2 + w3
if total > 0:
    w1, w2, w3 = w1/total, w2/total, w3/total
# number of recs
top_n = st.number_input("How many recs?", 1, 20, 5)

st.markdown("---")
if st.button("🔍 Recommend"):
    # … inside your `if st.button("🔍 Recommend"):` block …

    raw = hybrid_recommend(
        input_titles=selected,
        movies=movies,
        tfidf_sim=tfidf_sim,
        svd_sim=svd_sim,
        sbert_sim=sbert_sim,
        w1=w1, w2=w2, w3=w3,
        top_n=top_n
    )

    # If it came back as (indices, scores), grab the indices and build a DataFrame
    if isinstance(raw, tuple):
        idxs, scores = raw
        recs = movies.iloc[idxs][:top_n].reset_index(drop=True)
    else:
        recs = raw

    # Now recs is always a DataFrame
    cols = st.columns(5)
    for i, row in recs.iterrows():
        col = cols[i % 5]
        movie_id = row.get("movie_id") or row.get("id")  # whatever column holds the TMDB ID
        poster = get_poster_url(movie_id) if movie_id else None
        with col:
            if poster:
                st.image(poster, width=250)

            else:
                st.write("🖼️ No poster available")
            st.markdown(f"**{row['title']}** ")
            if 'genres' in row:
                st.caption(f"🎞️ {row['genres']}")

