# Manual de Uso Rápido - Sistema de Recomendación SVD (MovieLens 100k)

**Asignatura:** Gestión de la Información  
**Titulación:** Grado en Ingeniería Informática — Universidad de Burgos (UBU)  
**Grupo de prácticas:** Grupo 12  
**Integrantes:** Adrián Bello Sánchez y Alberto Cuervo Prieto  

---

## 1. Descripción del Proyecto

Este proyecto implementa un **sistema de recomendación mediante filtrado colaborativo basado en modelos**, utilizando el algoritmo de factorización matricial de bajo rango **SVD** (*Singular Value Decomposition*) de la biblioteca `Surprise`, aplicado sobre el dataset de calificaciones **MovieLens 100k** (`ml-latest-small`).

El código resuelve las siguientes tareas prácticas:
1. **Entrenamiento y evaluación del filtro colaborativo SVD** con validación cruzada (5-Fold CV).
2. **Inclusión de nuevos usuarios del equipo de prácticas** (Adrián, `userId: 9991` y Alberto, `userId: 9992`) y cálculo de sus Top 10 recomendaciones personalizadas.
3. **Análisis de sesgos estimados** por el modelo: usuarios más críticos / benevolentes y popularidad de películas.
4. **Análisis en el espacio latente:** cálculo de películas más afines a una dada (*Toy Story*) mediante similitud coseno en la matriz $Q$.
5. **Clustering de usuarios:** segmentación de perfiles de clientes mediante **K-Means** sobre los factores latentes $P$.

---

## 2. Estructura de Ficheros

```text
gi26p-practica-fc-modelos-group-12/
│
├── practica.py                         # Script ejecutable en terminal con el flujo completo
├── practica_filtro_colaborativo.ipynb  # Jupyter Notebook interactivo con celdas y visualizaciones
├── requirements.txt                    # Dependencias y librerías necesarias
├── README.md                           # Memoria detallada con resultados y respuestas a las cuestiones
├── README.txt                          # Documentación oficial del dataset MovieLens
│
├── docs/                               # Carpeta de documentación auxiliar
│   ├── MANUAL_USO_RAPIDO.md            # Este manual de uso rápido
│   └── requirements.txt                # Copia de las dependencias del proyecto
│
├── dataset/                            # Dataset MovieLens 100k
│   ├── movies.csv                      # Identificadores, títulos y géneros de películas
│   ├── ratings.csv                     # Calificaciones de los usuarios (0.5 a 5.0)
│   ├── tags.csv                        # Etiquetas aplicadas por los usuarios
│   └── links.csv                       # Enlaces a IMDb y TMDb
│
├── movies.csv, ratings.csv, ...        # Enlaces/ficheros directos en la raíz para ejecución rápida
└── LICENSE                             # Licencia del proyecto
```

---

## 3. Requisitos del Sistema

* **Python:** Versión `3.8`, `3.9`, `3.10` o `3.11` recomendada (compatible con entornos estándar de Linux, macOS y Windows).
* **Gestor de paquetes:** `pip` actualizado.
* **Espacio en disco:** ~200 MB (incluyendo entorno virtual y librerías).

---

## 4. Instalación Paso a Paso

Sigue estos pasos en la terminal para configurar el entorno de ejecución:

### Paso 1: Clonar o situarse en la raíz del repositorio

```bash
cd gi26p-practica-fc-modelos-group-12
```

### Paso 2: Crear un entorno virtual (recomendado)

Es una buena práctica aislar las dependencias del proyecto en un entorno virtual:

* **En Linux / macOS:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

* **En Windows (CMD o PowerShell):**
  ```cmd
  python -m venv venv
  venv\Scripts\activate
  ```

### Paso 3: Instalar las dependencias

Ejecuta el siguiente comando para instalar las librerías necesarias especificadas en `requirements.txt`:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Nota sobre `scikit-surprise`:**  
> Esta librería contiene extensiones compiladas en C. Si estás en Linux, asegúrate de tener instalado el compilador C (`sudo apt install build-essential python3-dev` en Debian/Ubuntu). En Windows, si `pip` requiere compilar desde el código fuente, es conveniente tener instaladas las *Microsoft C++ Build Tools*.

---

## 5. Guía de Ejecución

Existen dos formas alternativas de ejecutar la práctica según tus preferencias:

### Opción A: Ejecución como Script de Consola (`practica.py`)

Esta es la forma más rápida y directa de obtener todas las salidas y cálculos en la terminal:

```bash
python practica.py
```

**Salidas generadas por el script:**
1. Carga de datos de MovieLens (número de valoraciones, usuarios y películas).
2. Incorporación de los perfiles de Adrián (`9991`) y Alberto (`9992`).
3. Métricas de validación cruzada 5-Fold CV (**RMSE** y **MAE**) del modelo SVD.
4. Tablas con las **Top 10 recomendaciones** para cada miembro del equipo.
5. Análisis de sesgos: usuarios con mayor sesgo positivo y negativo, y películas más votadas.
6. Tabla de las **10 películas más próximas a *Toy Story*** en el espacio latente.
7. Segmentación de usuarios en **4 clusters (K-Means)** con sus géneros y películas insignia.

---

### Opción B: Ejecución en Jupyter Notebook Interactivo (`practica_filtro_colaborativo.ipynb`)

El notebook contiene explicaciones detalladas paso a paso, visualizaciones gráficas y gráficos de dispersión/distribución.

1. **Iniciar el servidor Jupyter:**
   ```bash
   jupyter notebook
   ```
2. En la interfaz web que se abrirá en tu navegador, abre el archivo `practica_filtro_colaborativo.ipynb`.
3. Para ejecutar todo el documento de una vez:
   * Menú superior: **Kernel** $\rightarrow$ **Restart & Run All** (o **Cell** $\rightarrow$ **Run All**).
4. También puedes abrirlo directamente en editores como **VS Code** o **PyCharm** con el plugin de Jupyter.

---

## 6. Resumen de Parámetros del Modelo

| Parámetro | Valor | Descripción |
| :--- | :---: | :--- |
| **Algoritmo** | SVD | Descomposición en Valores Singulares regularizada |
| **Factores latentes (`n_factors`)** | 50 | Dimensión del espacio vectorial de usuarios y películas |
| **Épocas de entrenamiento (`n_epochs`)** | 20 | Iteraciones del descenso por gradiente estocástico (SGD) |
| **Tasa de aprendizaje (`lr_all`)** | 0.005 | Tamaño de paso para la actualización de pesos |
| **Regularización (`reg_all`)** | 0.02 | Penalización $L_2$ para evitar sobreajuste |
| **Validación cruzada** | 5-Fold | Particionado aleatorio con semilla fija (`random_state=42`) |
| **Número de clusters ($k$)** | 4 | Agrupamiento K-Means sobre la matriz latente de usuarios $P$ |

---

## 7. Solución de Problemas Comunes (Troubleshooting)

* **Error `FileNotFoundError: ratings.csv not found`:**  
  Asegúrate de ejecutar el comando `python practica.py` desde el directorio raíz del proyecto (`gi26p-practica-fc-modelos-group-12`), donde se encuentran los archivos `.csv`.
* **Error al instalar `scikit-surprise`:**  
  Si la instalación falla por falta de compilador C, puedes instalar una versión precompilada con:
  ```bash
  pip install --only-binary :all: scikit-surprise
  ```
  o utilizar `conda`:
  ```bash
  conda install -c conda-forge scikit-surprise
  ```
* **No aparece el entorno virtual en Jupyter Notebook:**  
  Si has creado el entorno virtual `venv`, registra el kernel con:
  ```bash
  python -m ipykernel install --user --name=venv-practica --display-name="Python (venv-practica)"
  ```
  y selecciónalo en el desplegable superior derecho del notebook.
