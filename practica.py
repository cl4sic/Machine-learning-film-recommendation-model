# -*- coding: utf-8 -*-
"""
================================================================================
PRÁCTICA: SISTEMA DE RECOMENDACIÓN BASADO EN MODELOS (FILTRO COLABORATIVO SVD)
================================================================================
Asignatura: Gestión de la Información
Universidad de Burgos (UBU) - Grado en Ingeniería Informática
Grupo de prácticas: Grupo 12 (Adrián Bello Sánchez y Alberto Cuervo Prieto)
Dataset: MovieLens 100k (ml-latest-small)

Este script implementa todos los requerimientos de la práctica:
1. Implementación básica del filtro colaborativo con SVD (Surprise) y evaluación.
2. Incorporación de los miembros del equipo y generación de las Top 10 recomendaciones.
3. Análisis de sesgos estimados en el modelo (bu y bi) y popularidad de películas.
4. Análisis en el espacio latente (similitud coseno y películas próximas a "Toy Story").
5. Clustering de usuarios (K-Means) para segmentación de clientes.
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import cross_validate, train_test_split, GridSearchCV

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# Configuración visual
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11


# ==============================================================================
# 1. CARGA DE DATOS
# ==============================================================================
def cargar_datos():
    """Carga los ficheros ratings.csv y movies.csv locales."""
    print(">>> 1. Cargando datos de MovieLens 100k...")
    ratings = pd.read_csv('ratings.csv')
    movies = pd.read_csv('movies.csv')
    print(f"    - Valoraciones: {len(ratings):,}")
    print(f"    - Usuarios únicos: {ratings['userId'].nunique():,}")
    print(f"    - Películas en catálogo: {movies['movieId'].nunique():,}")
    return ratings, movies


# ==============================================================================
# 2. DEFINICIÓN DE USUARIOS DEL EQUIPO DE PRÁCTICAS
# ==============================================================================
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


# ==============================================================================
# 3. ENTRENAMIENTO Y EVALUACIÓN BÁSICA DEL FILTRO SVD CON GRID SEARCH
# ==============================================================================
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
    
    # Entrenar en dataset completo
    trainset = data.build_full_trainset()
    modelo.fit(trainset)
    print("    - Mejor modelo ajustado en todo el dataset de entrenamiento.")
    return modelo, trainset


# ==============================================================================
# 4. RECOMENDACIONES PERSONALIZADAS TOP 10
# ==============================================================================
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


# ==============================================================================
# 5. ANÁLISIS DE SESGOS ESTIMADOS
# ==============================================================================
def analizar_sesgos(modelo, trainset, ratings_df, movies_df):
    """Analiza sesgos de usuarios (bu) y de películas (bi), respondiendo al guion."""
    print("\n>>> 4. Análisis de Sesgos Estimados (bu y bi)...")
    mu = trainset.global_mean
    bu = modelo.bu
    bi = modelo.bi
    print(f"    - Media global del modelo (μ): {mu:.4f}")
    
    # Sesgos de usuarios
    usuarios_sesgos = []
    for inner_u, bias in enumerate(bu):
        raw_u = trainset.to_raw_uid(inner_u)
        user_data = ratings_df[ratings_df['userId'] == raw_u]
        nota_media = user_data['rating'].mean()
        usuarios_sesgos.append({
            'userId': raw_u,
            'sesgo_bu': bias,
            'nota_media_real': nota_media,
            'votos': len(user_data)
        })
    df_u = pd.DataFrame(usuarios_sesgos)
    
    print("\n    [¿Qué usuarios tienden a puntuar más alto? - Top 3 Generosos]")
    print(df_u.sort_values(by='sesgo_bu', ascending=False).head(3).to_string(index=False))
    
    print("\n    [¿Qué usuarios tienden a puntuar más bajo? - Top 3 Críticos]")
    print(df_u.sort_values(by='sesgo_bu', ascending=True).head(3).to_string(index=False))
    
    # Películas más valoradas
    conteos = ratings_df['movieId'].value_counts()
    top_mas_votadas_ids = conteos.head(5).index
    print("\n    [¿Qué películas tienden a tener más ratings? - Top 5 Más Votadas]")
    for mid in top_mas_votadas_ids:
        title = movies_df[movies_df['movieId'] == mid]['title'].values[0]
        votos = conteos[mid]
        inner_i = trainset.to_inner_iid(mid)
        bias_i = bi[inner_i]
        print(f"      * {title}: {votos} valoraciones (Sesgo bi: {bias_i:+.2f})")

    # Histograma de sesgos
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].hist(bu, bins=30, color='#1f77b4', edgecolor='black', alpha=0.7)
    axes[0].axvline(0, color='red', linestyle='--', linewidth=1.5, label='Sesgo neutro (0)')
    axes[0].set_title('Distribución de sesgos de usuarios ($b_u$)')
    axes[0].set_xlabel('Valor del sesgo ($b_u$)')
    axes[0].set_ylabel('Frecuencia')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.5)

    axes[1].hist(bi, bins=30, color='#2ca02c', edgecolor='black', alpha=0.7)
    axes[1].axvline(0, color='red', linestyle='--', linewidth=1.5, label='Sesgo neutro (0)')
    axes[1].set_title('Distribución de sesgos de películas ($b_i$)')
    axes[1].set_xlabel('Valor del sesgo ($b_i$)')
    axes[1].set_ylabel('Frecuencia')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('sesgos_distribucion.png', dpi=300)
    plt.close()
    print("    - Gráfico guardado en 'sesgos_distribucion.png'.")


# ==============================================================================
# 6. ANÁLISIS EN EL ESPACIO LATENTE (TOY STORY)
# ==============================================================================
def analizar_espacio_latente(modelo, trainset, movies_df, ratings_df, titulo="Toy Story", top_k=10, min_ratings=20):
    """Encuentra las películas más próximas a una dada en el espacio latente usando similitud coseno."""
    print(f"\n>>> 5. Análisis en el espacio latente: Películas más próximas a '{titulo}'...")
    coincidencias = movies_df[movies_df['title'].str.contains(titulo, case=False, regex=False, na=False)]
    pelicula_ref = coincidencias.iloc[0]
    raw_id_ref = pelicula_ref['movieId']
    inner_id_ref = trainset.to_inner_iid(raw_id_ref)
    
    # Vector latente qi
    vec_ref = modelo.qi[inner_id_ref].reshape(1, -1)
    sims = cosine_similarity(vec_ref, modelo.qi)[0]
    indices_ordenados = np.argsort(sims)[::-1]
    
    conteos = ratings_df['movieId'].value_counts().to_dict()
    filas = []
    for idx in indices_ordenados:
        raw_id = trainset.to_raw_iid(idx)
        if raw_id == raw_id_ref:
            continue
        votos = conteos.get(raw_id, 0)
        if votos >= min_ratings:
            info = movies_df[movies_df['movieId'] == raw_id].iloc[0]
            filas.append({
                'Título': info['title'],
                'Géneros': info['genres'],
                'Similitud Coseno': round(sims[idx], 3),
                'Nº Ratings': votos
            })
            if len(filas) == top_k:
                break
    df_sim = pd.DataFrame(filas)
    print(df_sim.to_string(index=False))
    return df_sim


# ==============================================================================
# 7. CLUSTERING DE USUARIOS (SEGMENTACIÓN DE CLIENTES)
# ==============================================================================
def segmentar_usuarios(modelo, trainset, ratings_df, movies_df, k=4):
    """Aplica K-Means sobre la matriz de factores latentes de usuarios P."""
    print(f"\n>>> 6. Segmentación de usuarios mediante Clustering (K-Means con k={k})...")
    user_factors = modelo.pu
    km = KMeans(n_clusters=k, random_state=42, n_init=15)
    clusters = km.fit_predict(user_factors)
    
    user_cluster_map = {trainset.to_raw_uid(i): clusters[i] for i in range(trainset.n_users)}
    
    ratings_clustered = ratings_df.copy()
    ratings_clustered['cluster'] = ratings_clustered['userId'].map(user_cluster_map)
    ratings_clustered = ratings_clustered.dropna(subset=['cluster'])
    ratings_clustered['cluster'] = ratings_clustered['cluster'].astype(int)
    
    # Películas candidatas (min 25 votos para evitar ruido)
    conteos_votos = ratings_df['movieId'].value_counts()
    candidatas = conteos_votos[conteos_votos >= 25].index
    candidatas_inner = [trainset.to_inner_iid(mid) for mid in candidatas]
    qi_candidatas = modelo.qi[candidatas_inner]
    
    # Distribución global de géneros en notas >= 4.0
    merged = ratings_clustered.merge(movies_df, on='movieId')
    high_ratings = merged[merged['rating'] >= 4.0]
    gen_global = high_ratings['genres'].str.split('|').explode().value_counts(normalize=True)
    
    def obtener_etiqueta_segmento(generos):
        g_set = set(generos)
        if 'Animation' in g_set:
            return 'Animación / Familiar'
        elif 'Film-Noir' in g_set or 'War' in g_set:
            return 'Cine Clásico / Culto'
        elif 'Horror' in g_set or 'Crime' in g_set or 'Mystery' in g_set:
            return 'Suspense / Terror / Crimen'
        elif 'Adventure' in g_set or 'Action' in g_set or 'IMAX' in g_set:
            return 'Acción / Aventura'
        else:
            return ' / '.join(generos[:2])

    resumen_segmentos = {}
    etiquetas = {}
    for c in range(k):
        n_users = (clusters == c).sum()
        pct = (n_users / len(clusters)) * 100
        
        sub = high_ratings[high_ratings['cluster'] == c]
        gen_c = sub['genres'].str.split('|').explode().value_counts(normalize=True)
        lift = (gen_c / gen_global).dropna().sort_values(ascending=False)
        top_g = [g for g in lift.index if g != '(no genres listed)'][:3]
        
        centroide = km.cluster_centers_[c]
        afinidades = qi_candidatas.dot(centroide)
        top_indices = np.argsort(afinidades)[::-1][:3]
        top_m = [movies_df[movies_df['movieId'] == candidatas[i]]['title'].values[0] for i in top_indices]
        
        nombre_seg = obtener_etiqueta_segmento(top_g)
        etiquetas[c] = f"Segmento {c} ({nombre_seg})"
        resumen_segmentos[c] = {
            'n_users': n_users, 'pct': pct,
            'top_g': top_g, 'top_m': top_m
        }

    print("    === RESUMEN DE SEGMENTOS DE CLIENTES ===")
    for c in range(k):
        info = resumen_segmentos[c]
        print(f"    * {etiquetas[c]} - {info['n_users']} usuarios ({info['pct']:.1f}%):")
        print(f"        Géneros distintivos: {', '.join(info['top_g'])}")
        print(f"        Películas insignia:  {', '.join(info['top_m'])}")
        
    print(f"\n    Adrián (userId 9991) asignado a: {etiquetas[user_cluster_map[9991]]}")
    print(f"    Alberto (userId 9992) asignado a: {etiquetas[user_cluster_map[9992]]}")

    # Representación gráfica 2D con PCA
    pca = PCA(n_components=2, random_state=42)
    pu_2d = pca.fit_transform(user_factors)
    centroides_2d = pca.transform(km.cluster_centers_)
    
    plt.figure(figsize=(9, 6))
    colores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for c in range(k):
        mask = clusters == c
        plt.scatter(pu_2d[mask, 0], pu_2d[mask, 1], c=colores[c], label=etiquetas[c], alpha=0.6, s=40)
        
    plt.scatter(centroides_2d[:, 0], centroides_2d[:, 1], c='black', marker='X', s=130, label='Centroides')
    
    u_adrian = trainset.to_inner_uid(9991)
    u_alberto = trainset.to_inner_uid(9992)
    plt.scatter(pu_2d[u_adrian, 0], pu_2d[u_adrian, 1], c='gold', marker='*', s=250, edgecolors='black', label='Adrián (9991)')
    plt.scatter(pu_2d[u_alberto, 0], pu_2d[u_alberto, 1], c='cyan', marker='*', s=250, edgecolors='black', label='Alberto (9992)')
    
    plt.title('Segmentación de usuarios en el espacio latente (PCA 2D)')
    plt.xlabel('Componente principal 1')
    plt.ylabel('Componente principal 2')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('clusters_usuarios.png', dpi=300)
    plt.close()
    print("    - Gráfico guardado en 'clusters_usuarios.png'.")


# ==============================================================================
# FUNCIÓN PRINCIPAL
# ==============================================================================
def main():
    ratings, movies = cargar_datos()
    df_adrian, df_alberto = crear_usuarios_equipo()
    
    # Ampliar dataset con el equipo
    ratings_ampliado = pd.concat([ratings[['userId', 'movieId', 'rating']], df_adrian, df_alberto], ignore_index=True)
    
    # Entrenar modelo
    modelo, trainset = entrenar_evaluar_svd(ratings_ampliado)
    
    # Recomendaciones para Adrián y Alberto
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
    
    # Análisis de sesgos
    analizar_sesgos(modelo, trainset, ratings_ampliado, movies)
    
    # Espacio latente
    analizar_espacio_latente(modelo, trainset, movies, ratings_ampliado, titulo="Toy Story", top_k=10)
    
    # Clustering
    segmentar_usuarios(modelo, trainset, ratings_ampliado, movies, k=4)
    



if __name__ == '__main__':
    main()
