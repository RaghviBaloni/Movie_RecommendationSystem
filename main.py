from sentence_transformers import SentenceTransformer
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate
import pandas as pd

# Define paths
vector_db_path = 'movie_vectordb.pkl'  # Path to the vectorized database

# Load movie vectors
with open(vector_db_path, 'rb') as f:
    movie_embeddings = pickle.load(f)

# Initialize SBERT model (same one used for vectorization)
model = SentenceTransformer('paraphrase-MiniLM-L12-v2')

def recommend_movies(user_input, top_n=5):
    # Convert user input into an embedding
    input_embedding = model.encode(user_input, convert_to_numpy=True)
    
    similarities = []
    for title, data in movie_embeddings.items():
        movie_embedding = data['embedding']
        genre = data['metadata'].get('Genre', '').lower()
        imdb_rating = data['metadata'].get('IMDB_Rating', 0)
        
        # Check genre match (optional filter)
        if any(word in genre for word in user_input.lower().split()):
            sim_score = cosine_similarity([input_embedding], [movie_embedding])[0][0]
            similarities.append((title, sim_score, imdb_rating, data['metadata']))

    # Sort by similarity first, then IMDB rating in descending order
    similarities.sort(key=lambda x: (-x[1], -x[2]))
    
    # Select top N movies
    top_movies = similarities[:top_n]
    
    # Prepare data for tabular display
    table_data = [[m[0], m[3]['Genre'], m[2], m[3]['Director'], m[3]['Stars']] for m in top_movies]
    headers = ['Title', 'Genre', 'IMDB Rating', 'Director', 'Stars']

    #df = pd.DataFrame(table_data, columns=headers)
    #return df
    
    print(tabulate(table_data, headers=headers, tablefmt='grid'))

if __name__ == "__main__":
    user_input = input("What do you wanna watch today?: ")
    recommend_movies(user_input)
    #print(recommendations.to_string(index=False))
