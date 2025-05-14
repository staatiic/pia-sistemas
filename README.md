# Sistema de Recomendación de Películas y Música

Este es un sistema web avanzado que proporciona recomendaciones personalizadas tanto para películas como para música, utilizando técnicas de machine learning y procesamiento de lenguaje natural.

## Características Principales

### Sistema de Recomendación de Películas
- Recomendaciones basadas en similitud de contenido (coseno)
- Agrupamiento de películas similares usando KMeans
- Visualización de detalles completos de películas
- Interfaz moderna y responsiva
- Sistema de puntuación y votos integrado

### Sistema de Recomendación de Música
- Recomendaciones de álbumes basadas en similitud
- Visualización de artistas y géneros
- Sistema de calificación integrado
- Imágenes de portadas de álbumes
- Descripciones detalladas de álbumes

## Requisitos del Sistema

- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- Conexión a internet para cargar imágenes y datos

## Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/staatiic/pia-sistemas.git
cd pia-sistemas
```

2. Crear un entorno virtual (recomendado):
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. Instalar las dependencias:
```bash
pip install -r requirements.txt
```

## Preparación de Datos y Modelos

1. Descargar los datasets necesarios:
   - Para el dataset de películas (`holanuevo.csv`), descárgalo desde [Hugging Face](https://huggingface.co/datasets/ada-datadruids/full_tmdb_movies_dataset). Este archivo no está incluido en el repositorio debido a su gran tamaño (499MB).
   - El archivo `albums.csv` está incluido en el repositorio.

2. Coloca los archivos CSV en el directorio raíz del proyecto.

3. Entrenar los modelos:
```bash
python train_model.py
python train_music_model.py
```

## Ejecución de la Aplicación

1. Iniciar el servidor Flask:
```bash
python app.py
```

2. Abrir el navegador web y acceder a:
```
http://localhost:5000
```

## Tecnologías Utilizadas

- **Backend**:
  - Flask (framework web)
  - scikit-learn (machine learning)
  - pandas (procesamiento de datos)
  - numpy (cálculos numéricos)
  - NLTK (procesamiento de lenguaje natural)

- **Frontend**:
  - HTML5
  - CSS3
  - JavaScript
  - Bootstrap

## Estructura del Proyecto

```
pia-sistemas/
├── app.py                 # Aplicación principal Flask
├── train_model.py         # Entrenamiento del modelo de películas
├── train_music_model.py   # Entrenamiento del modelo de música
├── get_music_data.py      # Obtención de datos musicales
├── models/                # Modelos entrenados de películas
├── modelss/              # Modelos entrenados de música
├── static/               # Archivos estáticos (CSS, JS, imágenes)
├── templates/            # Plantillas HTML
├── requirements.txt      # Dependencias del proyecto
└── README.md            # Este archivo
```

## Cómo Funciona

### Sistema de Películas
1. Utiliza TF-IDF para vectorizar las descripciones de películas
2. Implementa KMeans para agrupar películas similares
3. Calcula similitud de coseno entre películas
4. Proporciona recomendaciones basadas en contenido y clusters

### Sistema de Música
1. Procesa información de álbumes y artistas
2. Calcula similitud entre álbumes
3. Genera recomendaciones personalizadas
4. Muestra información detallada de cada álbum
