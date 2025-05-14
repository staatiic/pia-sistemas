import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.sparse import csr_matrix
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import joblib
import os
import logging
from typing import Tuple, Dict
from joblib import Parallel, delayed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    logger.info("Descargando recursos NLTK...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

    def preprocess_text(text: str) -> str:
        """Preprocesamiento avanzado del texto."""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()

        tokens = [stemmer.stem(lemmatizer.lemmatize(token)) 
                 for token in tokens 
                 if token not in stop_words and token.isalnum()]
        
        return ' '.join(tokens)

    def preprocess_text_parallel(texts):
        return Parallel(n_jobs=-1)(delayed(preprocess_text)(text) for text in texts)

    def prepare_features(movies_df: pd.DataFrame) -> Tuple[csr_matrix, TfidfVectorizer]:
        """Prepara las características con enfoque en géneros y títulos."""
        logger.info("Preparando características...")
        
        # Dar más peso a géneros y títulos
        movies_df['combined_features'] = (
            movies_df['genres'].fillna('') + ' ' +
            movies_df['genres'].fillna('') + ' ' +  # Duplicar géneros para dar más peso
            movies_df['title'].fillna('') + ' ' +
            movies_df['overview'].fillna('')
        )
        
        movies_df['processed_features'] = preprocess_text_parallel(movies_df['combined_features'])
        
        tfidf = TfidfVectorizer(
            max_features=5000,     # Aumentar el número de características
            min_df=2,             # Palabras que aparecen en al menos 2 documentos
            max_df=0.95,          # Ignorar palabras que aparecen en más del 95% de los documentos
            ngram_range=(1, 2),   # Usar palabras individuales y bigramas
            stop_words='english'  # Usar stopwords en inglés
        )
        
        tfidf_matrix = tfidf.fit_transform(movies_df['processed_features'])
        
        logger.info(f"Matriz de características creada con forma: {tfidf_matrix.shape}")
        return tfidf_matrix, tfidf

    def find_optimal_clusters(tfidf_matrix: csr_matrix, max_clusters: int = 20) -> int:
        """Encuentra el número óptimo de clusters usando el método del codo y silhouette score."""
        logger.info("Buscando número óptimo de clusters...")
        
        silhouette_scores = []
        inertia_scores = []
        
        for n_clusters in range(2, max_clusters + 1):
            kmeans = KMeans(
                n_clusters=n_clusters, 
                random_state=17, 
                n_init=10,
                max_iter=300
            )
            cluster_labels = kmeans.fit_predict(tfidf_matrix)
            
            silhouette_avg = silhouette_score(tfidf_matrix, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            inertia_scores.append(kmeans.inertia_)
            
            logger.info(f"Clusters: {n_clusters}, Inertia: {kmeans.inertia_:.2f}")
        
        # Normalizar todas las métricas
        normalized_silhouette = np.array(silhouette_scores) / max(silhouette_scores)
        normalized_inertia = 1 - (np.array(inertia_scores) / max(inertia_scores))
        combined_scores = 0.7 * normalized_silhouette + 0.3 * normalized_inertia
        
        optimal_clusters = np.argmax(combined_scores) + 2
        
        logger.info(f"\nMejor número de clusters encontrado: {optimal_clusters}")
        
        return optimal_clusters

    def evaluate_model(tfidf_matrix: csr_matrix, cluster_labels: np.ndarray, cosine_sim: np.ndarray) -> Dict[str, float]:
        """Evalúa el modelo usando métricas básicas."""
        # Evaluar recomendaciones (usando una muestra más grande)
        sample_size = min(100, len(cosine_sim))  
        sample_indices = np.random.choice(len(cosine_sim), sample_size, replace=False)
        
        # Calcular diversidad promedio de recomendaciones
        diversity_scores = []
        for idx in sample_indices:
            # Obtener top 10 recomendaciones
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            movie_indices = [i[0] for i in sim_scores[1:11]]  
            
            # Calcular diversidad usando una métrica más robusta
            rec_similarities = cosine_sim[movie_indices][:, movie_indices]
            np.fill_diagonal(rec_similarities, 0)  
            diversity = 1 - np.mean(rec_similarities)
            diversity_scores.append(diversity)
        
        return {
            'avg_recommendation_diversity': np.mean(diversity_scores)
        }

    if not os.path.exists('models'):
        os.makedirs('models')

    logger.info("Cargando datos...")
    if not os.path.exists('holanuevo.csv'):
        raise FileNotFoundError("El archivo holanuevo.csv no existe en el directorio actual")
    
    movies_df = pd.read_csv('holanuevo.csv')
    logger.info(f"Datos cargados. Número de películas: {len(movies_df)}")

    logger.info("Filtrando películas...")
    movies_df = movies_df[
        (movies_df['vote_count'] >= 100) &  
        (movies_df['vote_average'] >= 5.5) &  
        (movies_df['overview'].notna()) &
        (movies_df['genres'].notna())  
    ]
    
    if len(movies_df) > 50000:
        movies_df = movies_df.nlargest(50000, 'vote_count')

    logger.info(f"Número de películas seleccionadas: {len(movies_df)}")

    tfidf_matrix, tfidf_vectorizer = prepare_features(movies_df)

    optimal_clusters = find_optimal_clusters(tfidf_matrix)
    logger.info(f"\nUsando {optimal_clusters} clusters para el modelo final")

    logger.info("Aplicando KMeans clustering...")
    kmeans = KMeans(
        n_clusters=optimal_clusters,
        random_state=17,
        n_init=10,
        max_iter=300,
        tol=1e-4,
        n_jobs=-1  # Usa todos los núcleos disponibles
    )
    movies_df['cluster'] = kmeans.fit_predict(tfidf_matrix)

    logger.info("Calculando matriz de similitud coseno...")
    cosine_sim = cosine_similarity(tfidf_matrix)
    logger.info(f"Dimensiones de la matriz de similitud: {cosine_sim.shape}")
    logger.info("Evaluando modelo...")
    metrics = evaluate_model(tfidf_matrix, movies_df['cluster'], cosine_sim)
    logger.info("\nMétricas del modelo:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")

    logger.info("Guardando modelos...")
    joblib.dump(tfidf_vectorizer, 'models/tfidf_vectorizer.pkl')
    joblib.dump(cosine_sim, 'models/cosine_sim.pkl')
    joblib.dump(kmeans, 'models/kmeans_model.pkl')
    joblib.dump(movies_df, 'models/movies_with_clusters.pkl')

    logger.info("¡Proceso completado exitosamente!")

except Exception as e:
    logger.error(f"Error durante el proceso: {str(e)}")
    raise