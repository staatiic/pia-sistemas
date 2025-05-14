import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.sparse import csr_matrix
import re
import joblib
import os
import logging
from typing import Tuple, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    def preprocess_text(text: str) -> str:
        """Preprocesamiento mejorado del texto para música."""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9\s\'-]', ' ', text)
        text = ' '.join(text.split())
        text = re.sub(r'\b\d+\b', '', text)
        
        return text

    def prepare_features(albums_df: pd.DataFrame) -> Tuple[csr_matrix, TfidfVectorizer]:
        """Prepara las características con enfoque en géneros y artistas."""
        logger.info("Preparando características...")
        
        # Dar más peso a géneros y artistas, y menos a la descripción
        albums_df['combined_features'] = (
            albums_df['genres'].fillna('') + ' ' +
            albums_df['genres'].fillna('') + ' ' +  
            albums_df['genres'].fillna('') + ' ' +  
            albums_df['artist'].fillna('') + ' ' +
            albums_df['artist'].fillna('') + ' ' +  
            albums_df['description'].fillna('')
        )
        
        # Aplicar preprocesamiento
        albums_df['processed_features'] = albums_df['combined_features'].apply(preprocess_text)
        
        # Crear vectorizador TF-IDF con parámetros optimizados para música
        tfidf = TfidfVectorizer(
            max_features=10000,     
            min_df=2,             
            max_df=0.90,          
            ngram_range=(1, 3),   
            stop_words='english',  
            sublinear_tf=True     
        )
        
        # Crear matriz de características
        tfidf_matrix = tfidf.fit_transform(albums_df['processed_features'])
        
        logger.info(f"Matriz de características creada con forma: {tfidf_matrix.shape}")
        return tfidf_matrix, tfidf

    def find_optimal_clusters(tfidf_matrix: csr_matrix, max_clusters: int = 20) -> int:
        """Encuentra el número óptimo de clusters usando el método del codo y silhouette score."""
        logger.info("Buscando número óptimo de clusters...")
        
        silhouette_scores = []
        inertia_scores = []
        
        # Probar con números de clusters más razonables
        for n_clusters in range(2, max_clusters + 1):
            kmeans = KMeans(
                n_clusters=n_clusters, 
                random_state=177, 
                n_init=20,        
                max_iter=5000,    
                tol=1e-5         
            )
            cluster_labels = kmeans.fit_predict(tfidf_matrix)
            
            # Calcular métricas
            silhouette_avg = silhouette_score(tfidf_matrix, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            inertia_scores.append(kmeans.inertia_)
            
            logger.info(f"Clusters: {n_clusters}, Inertia: {kmeans.inertia_:.2f}")
        
        # Normalizar todas las métricas
        normalized_silhouette = np.array(silhouette_scores) / max(silhouette_scores)
        normalized_inertia = 1 - (np.array(inertia_scores) / max(inertia_scores))
        combined_scores = 0.7 * normalized_silhouette + 0.3 * normalized_inertia
        
        optimal_clusters = np.argmax(combined_scores) + 2
        
        # Mostrar información sobre la selección
        logger.info(f"\nMejor número de clusters encontrado: {optimal_clusters}")
        
        return optimal_clusters

    def evaluate_model(tfidf_matrix: csr_matrix, cluster_labels: np.ndarray, cosine_sim: np.ndarray) -> Dict[str, float]:
        """Evalúa el modelo usando métricas básicas."""
        # Evaluar clustering
        silhouette_avg = silhouette_score(tfidf_matrix, cluster_labels)
        
        # Evaluar recomendaciones (usando una muestra más grande)
        sample_size = min(100, len(cosine_sim))  
        sample_indices = np.random.choice(len(cosine_sim), sample_size, replace=False)
        
        # Calcular diversidad promedio de recomendaciones
        diversity_scores = []
        for idx in sample_indices:
            # Obtener top 10 recomendaciones
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            album_indices = [i[0] for i in sim_scores[1:11]]  
            
            # Calcular diversidad usando una métrica más robusta
            rec_similarities = cosine_sim[album_indices][:, album_indices]
            np.fill_diagonal(rec_similarities, 0)  
            diversity = 1 - np.mean(rec_similarities)
            diversity_scores.append(diversity)
        
        return {
            'avg_recommendation_diversity': np.mean(diversity_scores)
        }

    if not os.path.exists('modelss'):
        os.makedirs('modelss')

    logger.info("Cargando datos de música...")
    if not os.path.exists('albums.csv'):
        raise FileNotFoundError("El archivo albums.csv no existe en el directorio actual")
    
    albums_df = pd.read_csv('albums.csv')
    logger.info(f"Datos cargados. Número de álbumes: {len(albums_df)}")

    albums_df['genres'] = albums_df['genres'].fillna('unknown')
    albums_df['artist'] = albums_df['artist'].fillna('unknown')
    albums_df['description'] = albums_df['description'].fillna('unknown')

    logger.info("Filtrando álbumes...")
    albums_df = albums_df[albums_df['description'].notna()]
    logger.info(f"Número de álbumes seleccionados: {len(albums_df)}")

    tfidf_matrix, tfidf_vectorizer = prepare_features(albums_df)

    optimal_clusters = find_optimal_clusters(tfidf_matrix)
    logger.info(f"\nUsando {optimal_clusters} clusters para el modelo final")

    logger.info("Aplicando KMeans clustering...")
    kmeans = KMeans(
        n_clusters=optimal_clusters,
        random_state=17,
        n_init=10,
        max_iter=300,
        tol=1e-4  
    )
    albums_df['cluster'] = kmeans.fit_predict(tfidf_matrix)

    logger.info("Calculando matriz de similitud coseno...")
    cosine_sim = cosine_similarity(tfidf_matrix)
    logger.info(f"Dimensiones de la matriz de similitud: {cosine_sim.shape}")

    logger.info("Evaluando modelo...")
    metrics = evaluate_model(tfidf_matrix, albums_df['cluster'], cosine_sim)
    logger.info("\nMétricas del modelo:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    logger.info("Guardando modelos...")
    joblib.dump(tfidf_vectorizer, 'modelss/music_tfidf_vectorizer.pkl')
    joblib.dump(cosine_sim, 'modelss/music_cosine_sim.pkl')
    joblib.dump(kmeans, 'modelss/music_kmeans_model.pkl')
    joblib.dump(albums_df, 'modelss/albums_with_clusters.pkl')

    logger.info("¡Proceso completado exitosamente!")

except Exception as e:
    logger.error(f"Error durante el proceso: {str(e)}")
    raise 