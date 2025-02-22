import pandas as pd
import pickle

# Define input and output file paths
csv_path = 'imdb_top_1000.csv'  # Update with the actual path to your dataset
output_path = 'preprocessed_movies.pkl'  # Update with the desired output file path

def preprocess_movie_data(csv_path, output_path):
    # Load the dataset
    df = pd.read_csv(csv_path)
    
    # Select required columns
    df = df[['Series_Title', 'Released_Year', 'Genre', 'IMDB_Rating', 'Overview', 'Meta_score', 
             'Star1', 'Star2', 'Star3', 'Star4', 'Director']]
    
    # Combine stars into a single column
    df['Stars'] = df[['Star1', 'Star2', 'Star3', 'Star4']].apply(lambda x: ', '.join(x.dropna()), axis=1)
    
    # Drop original star columns
    df.drop(columns=['Star1', 'Star2', 'Star3', 'Star4'], inplace=True)
    
    # General data cleaning steps
    df.dropna(inplace=True)  # Drop rows with missing values
    df.drop_duplicates(inplace=True)  # Remove duplicate rows if needed
    
    # Verify missing values
    print("Missing values per column:\n", df.isnull().sum())
    
    # Convert to dictionary format {title: {overview, genre, etc.}}
    movie_dict = df.set_index('Series_Title').to_dict(orient='index')
    
    # Save preprocessed data as a pickle file
    with open(output_path, 'wb') as f:
        pickle.dump(movie_dict, f)
    
    print(f"Preprocessed data saved to {output_path}")

if __name__ == "__main__":
    preprocess_movie_data(csv_path, output_path)
