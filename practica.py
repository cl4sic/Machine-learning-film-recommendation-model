# -*- coding: utf-8 -*-
"""
================================================================================
PRÁCTICA: SISTEMA DE RECOMENDACIÓN BASADO EN MODELOS (FILTRO COLABORATIVO SVD)
================================================================================
Asignatura: Gestión de la Información
Universidad de Burgos (UBU) - Grado en Ingeniería Informática
Grupo de prácticas: Grupo 12 (Adrián Bello Sánchez y Alberto Cuervo Prieto)
Dataset: MovieLens 100k (ml-latest-small)

Módulo 2: Optimización de hiperparámetros con GridSearchCV y ajuste de SVD.
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


def entrenar_evaluar_svd(ratings_df):
    """Optimiza los hiperparámetros de SVD con Grid Search CV y ajusta el mejor modelo."""
    print("\n>>> 2. Optimización de hiperparámetros con Grid Search CV...")
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


def main():
    ratings, movies = cargar_datos()
    modelo, trainset = entrenar_evaluar_svd(ratings)
    print("\nFiltro SVD optimizado y entrenado correctamente.")


if __name__ == '__main__':
    main()
