import requests
import pandas as pd
import os
import time
from sklearn.feature_extraction.text import TfidfVectorizer

API_KEY = "1b33b6902e864353325bb07c3e156729"  
BASE_URL = "http://ws.audioscrobbler.com/2.0/"

def get_top_albums(limit=200):
    """Obtiene los álbumes más populares de Last.fm."""
    params = {
        'method': 'chart.gettopartists',
        'api_key': API_KEY,
        'format': 'json',
        'limit': limit
    }
    
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    
    albums_data = []
    
    for artist in data['artists']['artist']:
        artist_name = artist['name']
        print(f"Obteniendo álbumes de {artist_name}...")
        
        params = {
            'method': 'artist.gettopalbums',
            'artist': artist_name,
            'api_key': API_KEY,
            'format': 'json',
            'limit': 10  
        }
        
        response = requests.get(BASE_URL, params=params)
        artist_data = response.json()
        
        if 'topalbums' in artist_data:
            for album in artist_data['topalbums']['album']:
                params = {
                    'method': 'album.getinfo',
                    'artist': artist_name,
                    'album': album['name'],
                    'api_key': API_KEY,
                    'format': 'json'
                }
                
                response = requests.get(BASE_URL, params=params)
                album_data = response.json()
                
                if 'album' in album_data:
                    album_info = album_data['album']
                    
                    genres = []
                    if 'tags' in album_info and 'tag' in album_info['tags']:
                        if isinstance(album_info['tags']['tag'], list):
                            genres = [tag['name'] for tag in album_info['tags']['tag']]
                        else:
                            genres = [album_info['tags']['tag']['name']]
                    
                    rating = 0
                    if 'rating' in album_info and 'value' in album_info['rating']:
                        try:
                            rating = float(album_info['rating']['value']) / 5.0
                        except:
                            rating = 0
                    
                    image_url = ""
                    if 'image' in album_info and len(album_info['image']) > 0:
                        image_url = album_info['image'][-1]['#text']
                    
                    album_dict = {
                        'title': album_info['name'],
                        'artist': artist_name,
                        'genres': ','.join(genres),
                        'description': album_info.get('wiki', {}).get('summary', ''),
                        'rating': rating,
                        'image_url': image_url,
                        'release_date': album_info.get('releasedate', ''),
                        'listeners': int(album_info.get('listeners', 0)),
                        'playcount': int(album_info.get('playcount', 0))
                    }
                    
                    albums_data.append(album_dict)
                    print(f"  - Añadido: {album_info['name']}")
                time.sleep(0.25)
    
    df = pd.DataFrame(albums_data)
    df.to_csv('albums.csv', index=False)
    print(f"\nSe han guardado {len(df)} álbumes en albums.csv")

if __name__ == "__main__":
    if not os.path.exists('albums.csv'):
        print("Obteniendo datos de Last.fm...")
        get_top_albums()
    else:
        print("El archivo albums.csv ya existe. Si deseas actualizar los datos, elimina el archivo actual.")

albums_df = pd.read_csv('albums.csv')
albums_df['genres'] = albums_df['genres'].fillna('unknown')
albums_df['artist'] = albums_df['artist'].fillna('unknown')
albums_df['description'] = albums_df['description'].fillna('unknown')

albums_df = albums_df[albums_df['description'].notna()]

albums_df = albums_df[
    (albums_df['rating'] >= 3.5) &  
    (albums_df['description'].notna())  
] 

tfidf = TfidfVectorizer(
    max_features=3000,
    min_df=1,
    max_df=0.9,
    ngram_range=(1, 1)
) 