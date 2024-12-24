import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import nltk
from nltk.stem import PorterStemmer
import ast

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# OMDB API configuration
OMDB_API_KEY = "73f8c521"  # Get your API key from http://www.omdbapi.com/

class MovieRecommender:
    def __init__(self):
        self.movies_df = None
        self.similarity_matrix = None
        self.load_data()
        
    def load_data(self):
        # Load movies data
        movies = pd.read_csv('tmdb_5000_movies.csv')
        credits = pd.read_csv('tmdb_5000_credits.csv')
        
        # Merge dataframes
        movies = movies.merge(credits, on='title')
        
        # Select required columns
        movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
        
        # Clean data
        movies.dropna(inplace=True)
        
        # Process different columns
        movies['genres'] = movies['genres'].apply(self.convert)
        movies['keywords'] = movies['keywords'].apply(self.convert)
        movies['cast'] = movies['cast'].apply(self.convert_cast)
        movies['crew'] = movies['crew'].apply(self.fetch_director)
        movies['overview'] = movies['overview'].apply(lambda x: x.split())
        
        # Remove spaces
        for feature in ['cast', 'crew', 'genres', 'keywords']:
            movies[feature] = movies[feature].apply(self.remove_space)
        
        # Create tags
        movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
        
        # Create new dataframe with required columns
        self.movies_df = movies[['movie_id', 'title', 'tags']]
        
        # Process tags
        self.movies_df['tags'] = self.movies_df['tags'].apply(lambda x: " ".join(x))
        self.movies_df['tags'] = self.movies_df['tags'].apply(lambda x: x.lower())
        
        # Apply stemming
        ps = PorterStemmer()
        self.movies_df['tags'] = self.movies_df['tags'].apply(lambda x: " ".join([ps.stem(word) for word in x.split()]))
        
        # Create similarity matrix
        cv = CountVectorizer(max_features=5000, stop_words='english')
        vectors = cv.fit_transform(self.movies_df['tags']).toarray()
        self.similarity_matrix = cosine_similarity(vectors)

    @staticmethod
    def convert(text):
        L = []
        for i in ast.literal_eval(text):
            L.append(i['name'])
        return L

    @staticmethod
    def convert_cast(text):
        L = []
        counter = 0
        for i in ast.literal_eval(text):
            if counter < 3:
                L.append(i['name'])
            counter += 1
        return L

    @staticmethod
    def fetch_director(text):
        L = []
        for i in ast.literal_eval(text):
            if i['job'] == 'Director':
                L.append(i['name'])
                break
        return L

    @staticmethod
    def remove_space(L):
        return [i.replace(" ", "") for i in L]

    def get_movie_details(self, title):
        try:
            # Clean movie title for API request
            clean_title = title.split('(')[0].strip()
            
            # Make API request
            url = f"http://www.omdbapi.com/?t={clean_title}&apikey={OMDB_API_KEY}"
            response = requests.get(url)
            data = response.json()
            
            if data.get('Response') == 'True':
                return {
                    'Title': data.get('Title'),
                    'Year': data.get('Year'),
                    'Poster': data.get('Poster'),
                    'Plot': data.get('Plot'),
                    'Genre': data.get('Genre'),
                    'Director': data.get('Director'),
                    'Actors': data.get('Actors'),
                    'imdbRating': data.get('imdbRating')
                }
            return None
        except:
            return None

    def recommend_movies(self, movie_title):
        try:
            idx = self.movies_df[self.movies_df['title'] == movie_title].index[0]
            distances = sorted(list(enumerate(self.similarity_matrix[idx])), reverse=True, key=lambda x: x[1])
            recommended_movies = []
            
            for i in distances[1:6]:
                movie_title = self.movies_df.iloc[i[0]]['title']
                movie_details = self.get_movie_details(movie_title)
                if movie_details:
                    recommended_movies.append(movie_details)
            
            return recommended_movies
        except:
            return None

def display_movie_card(movie_details):
    if movie_details:
        col1, col2 = st.columns([1, 2])
        with col1:
            if movie_details['Poster'] != 'N/A':
                st.image(movie_details['Poster'], width=200)
            else:
                st.write("No poster available")
        
        with col2:
            st.subheader(f"{movie_details['Title']} ({movie_details['Year']})")
            st.write(f"**Genre:** {movie_details['Genre']}")
            st.write(f"**Director:** {movie_details['Director']}")
            st.write(f"**Cast:** {movie_details['Actors']}")
            if movie_details['imdbRating'] != 'N/A':
                st.write(f"**IMDb Rating:** ⭐ {movie_details['imdbRating']}/10")
            st.write(f"**Plot:** {movie_details['Plot']}")

def main():
    st.title('🎬 Movie Recommender System')
    
    # Initialize recommender
    recommender = MovieRecommender()
    
    # Create selection box for movies
    selected_movie = st.selectbox(
        'Select a movie you like:',
        recommender.movies_df['title'].values
    )
    
    # Add a recommendation button
    if st.button('Show Recommendations'):
        with st.spinner('Getting movie recommendations...'):
            st.write(f"### Selected Movie")
            selected_movie_details = recommender.get_movie_details(selected_movie)
            if selected_movie_details:
                display_movie_card(selected_movie_details)
            
            st.write("### Recommended Movies")
            recommendations = recommender.recommend_movies(selected_movie)
            
            if recommendations:
                for movie in recommendations:
                    display_movie_card(movie)
                    st.markdown("---")
            else:
                st.error("Sorry, couldn't generate recommendations for this movie.")
    
    # Add additional information
    st.sidebar.title("About")
    st.sidebar.info(
        "This movie recommender system uses content-based filtering to suggest movies "
        "similar to your selection. It analyzes movie features like genre, cast, crew, "
        "and plot to make recommendations."
    )
    
    # Add footer
    st.markdown("---")
    st.markdown(
        "Built with ❤️ using Streamlit | Data source: TMDB 5000 Movie Dataset"
    )

if __name__ == '__main__':
    main()