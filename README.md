# Movie and Music Recommendation System

This is an advanced web system that provides personalized recommendations for both movies and music, using machine learning techniques and natural language processing.

## Main Features

### Movie Recommendation System
- Content-based recommendations using cosine similarity
- Movie clustering using KMeans
- Complete movie details visualization
- Modern and responsive interface
- Integrated rating and voting system

### Music Recommendation System
- Album similarity-based recommendations
- Artist and genre visualization
- Integrated rating system
- Album cover images
- Detailed album descriptions

## System Requirements

- Python 3.8 or higher
- pip (Python package manager)
- Internet connection for loading images and data

## Installation

1. Clone the repository:
```bash
git clone https://github.com/staatiic/pia-sistemas.git
cd pia-sistemas
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data and Model Preparation

1. Download required datasets:
   - For the movies dataset (`holanuevo.csv`), download it from [Hugging Face](https://huggingface.co/datasets/ada-datadruids/full_tmdb_movies_dataset). This file is not included in the repository due to its large size (499MB).
   - The `albums.csv` file is included in the repository.

2. Place the CSV files in the project root directory.

3. Train the models:
```bash
python train_model.py
python train_music_model.py
```

## Running the Application

1. Start the Flask server:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

## Technologies Used

- **Backend**:
  - Flask (web framework)
  - scikit-learn (machine learning)
  - pandas (data processing)
  - numpy (numerical computations)
  - NLTK (natural language processing)

- **Frontend**:
  - HTML5
  - CSS3
  - JavaScript
  - Bootstrap

## Project Structure

```
pia-sistemas/
├── app.py                 # Main Flask application
├── train_model.py         # Movie model training
├── train_music_model.py   # Music model training
├── get_music_data.py      # Music data retrieval
├── models/                # Trained movie models
├── modelss/              # Trained music models
├── static/               # Static files (CSS, JS, images)
├── templates/            # HTML templates
├── requirements.txt      # Project dependencies
└── README.md            # This file
```

## How It Works

### Movie System
1. Uses TF-IDF to vectorize movie descriptions
2. Implements KMeans for clustering similar movies
3. Calculates cosine similarity between movies
4. Provides recommendations based on content and clusters

### Music System
1. Processes album and artist information
2. Calculates similarity between albums
3. Generates personalized recommendations
4. Displays detailed information for each album
