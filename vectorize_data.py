from sentence_transformers import SentenceTransformer
import pickle
import numpy as np

# Define paths
preprocessed_data = 'preprocessed_movies.pkl'  
vector_db = 'movie_vectordb.pkl'  # Path to store the vectorized database

# Load preprocessed movie data
with open(preprocessed_data, 'rb') as f:
    movie_data = pickle.load(f)

# Initialize SBERT model
model = SentenceTransformer('paraphrase-MiniLM-L12-v2')

# Generate embeddings for each movie's overview
movie_embeddings = {}
for title, details in movie_data.items():
    overview = details.get('Overview', '')
    embedding = model.encode(overview, convert_to_numpy=True)  #embedding generation
    movie_embeddings[title] = {
        'embedding': embedding,
        'metadata': details  
    }

# Save the vectorized database
with open(vector_db, 'wb') as f:
    pickle.dump(movie_embeddings, f)

print(f"Movie vector database saved to {vector_db}")
