from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)
error_msg = ""

def load_data_and_models():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base_dir, 'holanuevo.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"El archivo de películas no se encuentra en: {csv_path}")
        
        models_dir = os.path.join(base_dir, 'models')
        modelss_dir = os.path.join(base_dir, 'modelss')
        
        required_files = [
            os.path.join(models_dir, 'cosine_sim.pkl'),
            os.path.join(models_dir, 'kmeans_model.pkl'),
            os.path.join(models_dir, 'movies_with_clusters.pkl'),
            os.path.join(models_dir, 'tfidf_vectorizer.pkl'),
            os.path.join(modelss_dir, 'music_cosine_sim.pkl'),
            os.path.join(modelss_dir, 'albums_with_clusters.pkl')
        ]
        
        for file in required_files:
            if not os.path.exists(file):
                raise FileNotFoundError(f"El archivo {file} no se encuentra. Por favor, ejecuta primero train_model.py")
            print(f"Archivo encontrado: {file}")

        movies_df = joblib.load(os.path.join(models_dir, 'movies_with_clusters.pkl'))
        albums_df = joblib.load(os.path.join(modelss_dir, 'albums_with_clusters.pkl'))
        
        cosine_sim = joblib.load(os.path.join(models_dir, 'cosine_sim.pkl'))
        tfidf_vectorizer = joblib.load(os.path.join(models_dir, 'tfidf_vectorizer.pkl'))
        music_cosine_sim = joblib.load(os.path.join(modelss_dir, 'music_cosine_sim.pkl'))
        kmeans = joblib.load(os.path.join(models_dir, 'kmeans_model.pkl'))
        
        return movies_df, albums_df, tfidf_vectorizer, cosine_sim, music_cosine_sim, kmeans
    except Exception as e:
        print(f"Error al cargar los datos y modelos: {str(e)}")
        return None, None, None, None, None, None

movies_df, albums_df, tfidf_vectorizer, cosine_sim, music_cosine_sim, kmeans = load_data_and_models()

@app.route('/')
def home():
    if movies_df is None or albums_df is None:
        return render_template('error.html', 
                             message="Error: No se pudieron cargar los datos o modelos. Por favor, asegúrate de que los archivos necesarios estén en el lugar correcto.")
    
    top_movies = movies_df.sort_values(['vote_average', 'vote_count'], ascending=[False, False])
    top_albums = albums_df.sort_values(['rating', 'playcount'], ascending=[False, False])
    
    return render_template('index.html', 
                         movies=top_movies[['title', 'genres', 'vote_average', 'vote_count']].head(100).to_dict('records'),
                         albums=top_albums[['title', 'artist', 'genres', 'rating', 'image_url']].head(100).to_dict('records'))

@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    if movies_df is None or cosine_sim is None:
        return jsonify({"error": "El modelo no está inicializado correctamente"}), 500
    try:
        movie_title = request.json['movie']
        method = request.json.get('method', 'cosine')

        movie_matches = movies_df[movies_df['title'] == movie_title]
        if movie_matches.empty:
            return jsonify({"error": f"No se encontró la película: {movie_title}"}), 404
        
        idx = movie_matches.index[0]

        if method == 'cosine':
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
            movie_indices = [i[0] for i in sim_scores]
        else:
            movie_cluster = movies_df.iloc[idx]['cluster']
            similar_movies = movies_df[(movies_df['cluster'] == movie_cluster) & (movies_df['title'] != movie_title)]
            movie_indices = similar_movies.head(10).index

        recommendations = movies_df.iloc[movie_indices][
            ['title', 'genres', 'vote_average', 'vote_count', 'overview', 'poster_path']
        ].to_dict('records')

        for rec in recommendations:
            rec['poster_path'] = f"https://image.tmdb.org/t/p/w500{rec['poster_path']}" if pd.notna(rec['poster_path']) else None

        return jsonify(recommendations)

    except Exception as e:
        print("Error en /get_recommendations:", str(e))
        return jsonify({"error": f"Error al obtener recomendaciones: {str(e)}"}), 500

@app.route('/get_music_recommendations', methods=['POST'])
def get_music_recommendations():
    if albums_df is None or music_cosine_sim is None:
        return jsonify({"error": "El modelo de música no está inicializado correctamente"}), 500
    try:
        album_title = request.json['album']

        album_matches = albums_df[albums_df['title'] == album_title]
        if album_matches.empty:
            return jsonify({"error": f"No se encontró el álbum: {album_title}"}), 404
        
        idx = album_matches.index[0]
        sim_scores = list(enumerate(music_cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
        album_indices = [i[0] for i in sim_scores]

        recommendations = albums_df.iloc[album_indices][
            ['title', 'artist', 'genres', 'rating', 'image_url', 'description']
        ].to_dict('records')

        return jsonify(recommendations)

    except Exception as e:
        print("Error en /get_music_recommendations:", str(e))
        return jsonify({"error": f"Error al obtener recomendaciones de música: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
