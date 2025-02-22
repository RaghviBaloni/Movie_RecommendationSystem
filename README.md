# Movie Recommender based on user input

## Environment Setup
```sh
conda create --name recommender python=3.9 -y
conda activate recommender

pip install pandas
pip install sentence_transformers
pip install tabulate

```

## Data Preprocessing
Download the dataset from the link below:
https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows

Adjust the file locations 
```sh
python data_preprocessing.py
```

## Creating the Vector DB
```sh
python vectorize_data.py
```

## Executing the recommendation system
```sh
python main.py
```

