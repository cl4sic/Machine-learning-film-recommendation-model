# -*- coding: utf-8 -*-
"""
================================================================================
PRÁCTICA: SISTEMA DE RECOMENDACIÓN BASADO EN MODELOS (FILTRO COLABORATIVO SVD)
================================================================================
Asignatura: Gestión de la Información
Universidad de Burgos (UBU) - Grado en Ingeniería Informática
Grupo de prácticas: Grupo 12 (Adrián Bello Sánchez y Alberto Cuervo Prieto)
Dataset: MovieLens 100k (ml-latest-small)

Módulo 3: Inclusión de perfiles de usuario y generación de recomendaciones Top-10.
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from surprise import Reader, Dataset, SVD
from surprise.model_selection import GridSearchCV

# Configuración visual
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11


def cargar_datos():
    """Carga los ficheros ratings.csv y movies.csv locales."""
    print(">>> 1. Cargando datos de MovieLens 100k...")
    ratings = pd.read_csv('ratings.csv')
    movies = pd.read_csv('movies.csv')
    print(f"    - Valoraciones: {len(ratings):,}")
    print(f"    - Usuarios únicos: {ratings['userId'].nunique():,}")
    print(f"    - Películas en catálogo: {movies['movieId'].nunique():,}")
    return ratings, movies


def crear_usuarios_equipo():
    """
    Crea dos usuarios representativos del equipo de prácticas (Grupo 12):
    - Adrián (userId 9991): Aficionado a Ciencia Ficción, Acción y Thriller.
    - Alberto (userId 9992): Aficionado a Animación, Familiar, Fantasía y Aventura.
    """
    print("\n>>> 2. Definiendo usuarios del equipo de prácticas...")
    adrian_ratings = [
        {'userId': 9991, 'movieId': 2571, 'rating': 5.0},   # Matrix
        {'userId': 9991, 'movieId': 79132, 'rating': 5.0},  # Inception
        {'userId': 9991, 'movieId': 109487, 'rating': 5.0}, # Interstellar
        {'userId': 9991, 'movieId': 296, 'rating': 4.5},    # Pulp Fiction
        {'userId': 9991, 'movieId': 589, 'rating': 4.5},    # Terminator 2
        {'userId': 9991, 'movieId': 260, 'rating': 5.0},    # Star Wars IV
        {'userId': 9991, 'movieId': 1196, 'rating': 5.0},   # Star Wars V
        {'userId': 9991, 'movieId': 367, 'rating': 1.5},    # The Mask
        {'userId': 9991, 'movieId': 357, 'rating': 1.5},    # Four Weddings
        {'userId': 9991, 'movieId': 597, 'rating': 1.0},    # Pretty Woman
    ]
    
    alberto_ratings = [
        {'userId': 9992, 'movieId': 1, 'rating': 5.0},      # Toy Story
        {'userId': 9992, 'movieId': 3114, 'rating': 5.0},   # Toy Story 2
        {'userId': 9992, 'movieId': 78499, 'rating': 5.0},  # Toy Story 3
        {'userId': 9992, 'movieId': 364, 'rating': 5.0},    # The Lion King
        {'userId': 9992, 'movieId': 595, 'rating': 4.5},    # Beauty and the Beast
        {'userId': 9992, 'movieId': 588, 'rating': 4.5},    # Aladdin
        {'userId': 9992, 'movieId': 6377, 'rating': 5.0},   # Finding Nemo
        {'userId': 9992, 'movieId': 589, 'rating': 1.0},    # Terminator 2
        {'userId': 9992, 'movieId': 1258, 'rating': 1.5},   # The Shining
        {'userId': 9992, 'movieId': 2571, 'rating': 1.5},   # Matrix
    ]
    
    return pd.DataFrame(adrian_ratings), pd.DataFrame(alberto_ratings)


def entrenar_evaluar_svd(ratings_df):
    """Optimiza los hiperparámetros de SVD con Grid Search CV y ajusta el mejor modelo."""
    print("\n>>> 3. Optimización de hiperparámetros con Grid Search CV...")
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
    
    param_grid = {
        'n_factors': [20, 50, 100],
        'n_epochs': [20, 30],
        'lr_all': [0.005, 0.01],
        'reg_all': [0.05, 0.1]
    }
    
    print("    - Ejecutando GridSearchCV (3-fold CV)...")
    gs = GridSearchCV(SVD, param_grid, measures=['RMSE', 'MAE'], cv=3, n_jobs=-1)
    gs.fit(data)
    
    print(f"      * Mejor RMSE: {gs.best_score['rmse']:.4f}")
    print(f"      * Mejor MAE:  {gs.best_score['mae']:.4f}")
    print(f"      * Mejores hiperparámetros: {gs.best_params['rmse']}")
    
    params = gs.best_params['rmse']
    modelo = SVD(n_factors=params['n_factors'], n_epochs=params['n_epochs'],
                 lr_all=params['lr_all'], reg_all=params['reg_all'], random_state=42)
    
    trainset = data.build_full_trainset()
    modelo.fit(trainset)
    print("    - Mejor modelo ajustado en todo el dataset de entrenamiento.")
    return modelo, trainset


def generar_top_10(modelo, user_id, ratings_df, movies_df, min_ratings=20):
    """Obtiene las 10 mejores recomendaciones para un usuario no vistas previamente."""
    vistas = set(ratings_df[ratings_df['userId'] == user_id]['movieId'])
    conteos = ratings_df['movieId'].value_counts().to_dict()
    
    candidatas = [m for m in movies_df['movieId'].unique() if m not in vistas]
    
    preds = []
    for mid in candidatas:
        votos = conteos.get(mid, 0)
        if votos >= min_ratings:
            pred = modelo.predict(uid=user_id, iid=mid)
            preds.append((mid, pred.est, votos))
            
    preds.sort(key=lambda x: x[1], reverse=True)
    top10 = preds[:10]
    
    filas = []
    for rank, (mid, est, votos) in enumerate(top10, 1):
        info = movies_df[movies_df['movieId'] == mid].iloc[0]
        filas.append({
            'Ranking': f"#{rank}",
            'Título': info['title'],
            'Géneros': info['genres'],
            'Nota Predicha': round(est, 2),
            'Votos': votos
        })
    return pd.DataFrame(filas)


def main():
    ratings, movies = cargar_datos()
    df_adrian, df_alberto = crear_usuarios_equipo()
    
    ratings_ampliado = pd.concat([ratings[['userId', 'movieId', 'rating']], df_adrian, df_alberto], ignore_index=True)
    modelo, trainset = entrenar_evaluar_svd(ratings_ampliado)
    
    print("\n" + "="*80)
    print("TOP 10 RECOMENDACIONES PARA ADRIÁN (userId: 9991)")
    print("="*80)
    top_adrian = generar_top_10(modelo, 9991, ratings_ampliado, movies)
    print(top_adrian.to_string(index=False))
    
    print("\n" + "="*80)
    print("TOP 10 RECOMENDACIONES PARA ALBERTO (userId: 9992)")
    print("="*80)
    top_alberto = generar_top_10(modelo, 9992, ratings_ampliado, movies)
    print(top_alberto.to_string(index=False))


if __name__ == '__main__':
    main()
