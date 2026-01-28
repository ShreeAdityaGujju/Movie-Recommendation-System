# Movie Recommendation System: Hybrid Content‑Based Recommender
This Movie Recommendation System is a Streamlit application that recommends movies based on the textual content of film metadata. It uses a hybrid approach that combines TF–IDF, Latent Semantic Analysis (SVD), and Sentence‑BERT similarity to generate personalized recommendations. Users can select one or more favourite movies, adjust the weights of each similarity metric, choose the number of recommendations, and view poster images of the suggested films.

## Executive Summary
Conventional recommendation engines often rely on user ratings or collaborative filtering, which can suffer from cold‑start issues and require large amounts of user interaction. This project builds a content‑based recommender using movie descriptions, genre keywords, cast, and crew information from the TMDb 5000 dataset. By encoding the combined textual features with multiple NLP techniques, the system offers robust recommendations even when user ratings are scarce. The Streamlit front‑end allows users to interactively fine‑tune the hybrid weights and explore recommendations with poster images fetched from TMDb.

## Problem Statement
Finding relevant movies in a vast catalogue can be overwhelming, especially for new users without explicit ratings or watch history. Pure collaborative filtering cannot handle the cold‑start problem for unseen items. The objective of this project is to:
* Preprocess textual metadata (overview, genres, keywords, cast, crew) to create a unified representation for each movie
* Compute multiple similarity matrices: TF‑IDF, SVD (latent semantic analysis), and Sentence‑BERT embeddings to capture both word‑level and semantic similarity between films
* Blend these similarity scores using adjustable weights to produce a top‑N list of recommendations
* Provide an interactive UI where users select input titles, configure weights and number of recommendations, and view posters

## Methodology
1. Data Ingestion & Preprocessing
    * Load the TMDb 5000 movies and credits datasets and merge them on the title column
    * Select relevant columns (movie_id, title, overview, genres, keywords, cast, crew) and drop rows with missing values
    * Convert JSON‑like lists into plain text: extract names from genres, keywords, and cast lists; keep only the top three cast members; extract directors from the crew
    * Tokenize the overview and combine it with genres, keywords, and cast to form a single tags field; join tokens into a single lowercase string
2. Feature Engineering & Similarity Computation
    * TF–IDF: Convert the tags text into a term‑frequency matrix using TfidfVectorizer and compute cosine similarity
    * SVD (Latent Semantic Analysis): Apply TruncatedSVD to reduce TF‑IDF features to latent components and compute cosine similarity on the reduced matrix
    * Sentence‑BERT: Encode the tags with the all‑MiniLM‑L6‑v2 SentenceTransformer to capture contextual semantics, then compute cosine similarity
3. Hybrid Recommendation Algorithm
    * For a set of input movies selected by the user, locate their indices in the dataset
    * Compute a weighted average of the similarity scores from TF‑IDF, SVD and SBERT matrices using user‑defined weights ww11, w2 and w3
    * Sort the similarity scores, remove the input movies from the results, and return the top‑N recommendations along with similarity scores
4. Interactive Streamlit UI
    * Use Streamlit to create a sidebar where users can select favorite movies, set the number of recommendations, and adjust the weights for each similarity model via sliders
    * Fetch movie posters from the TMDb API for each recommended film; if a poster is unavailable, display a placeholder
    * Display the recommended movies in a grid with posters, titles ,and genre captions, enhancing the browsing experience

###  Dataset
The system uses the TMDb 5000 Movie Dataset, comprising two CSV files: tmdb_5000_movies.csv and tmdb_5000_credits.csv. These datasets provide metadata such as movie titles, overviews, genres, keywords, cast, and crew information. After preprocessing, the final dataset consists of a tags field for each movie containing a concatenated and cleaned representation of its textual features.s

## Key Features & Outputs
* Hybrid content‑based recommendations that combine lexical, late,nt, and contextual similarity.
* Dynamic weighting users can tune the influence of TF‑IDF, SVD, and SBERT models to personalise recommendations.
* Flexible input accepts one or more favourite movies, making it easy to build a profile from multiple examples.
* Interactive visualisation displays movie posters, titles, and genres for each recommendation, enhancing discoverability

### Limitations
* The model is purely content‑based and does not incorporate user ratings, watch history, or collaborative filtering, which could improve personalization.
* Recommendations depend on the quality and completeness of the TMDb metadata; missing or sparse descriptions can reduce accuracy.
* The TMDb API key is required to fetch posters; without it, poster images cannot be displayed.

## Future Work
* Incorporate collaborative filtering or matrix factorisation to blend user ratings with content‑based features.
* Expand metadata by including release dates, runtime, revenue, or user ratings, and exploring additional text fields (e.g., taglines).
* Implement evaluation metrics (precision, recall, MAP) and collect user feedback to improve the weighting scheme.
* Deploy as a web service or integrate with a larger recommendation platform, with caching mechanisms for TMDb poster retrieval.

