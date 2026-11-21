# -*- coding: utf-8 -*-
"""
================================================================================
PRÁCTICA: SISTEMA DE RECOMENDACIÓN BASADO EN MODELOS (FILTRO COLABORATIVO SVD)
================================================================================
Asignatura: Gestión de la Información
Universidad de Burgos (UBU) - Grado en Ingeniería Informática
Grupo de prácticas: Grupo 12 (Adrián Bello Sánchez y Alberto Cuervo Prieto)
Dataset: MovieLens 100k (ml-latest-small)

Módulo 1: Carga, verificación y estructuración de datos para Surprise.
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import Reader, Dataset

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


def preparar_dataset_surprise(ratings_df):
    """Convierte el DataFrame de valoraciones a un Dataset de Surprise."""
    print("\n>>> Configurando Surprise Reader (escala 0.5 a 5.0)...")
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
    print("    - Dataset de Surprise instanciado correctamente.")
    return data, reader


def main():
    ratings, movies = cargar_datos()
    data, reader = preparar_dataset_surprise(ratings)
    print("\nInicialización de datos completada con éxito.")


if __name__ == '__main__':
    main()
