# 🎗️ MamaIA: Apoyo al tamizaje de cáncer de mama con aprendizaje automático supervisado

Aplicación web que clasifica masas mamarias como **benignas** o **malignas** a partir de medidas del núcleo celular obtenidas por biopsia por aspiración con aguja fina (BAAF). Compara tres algoritmos de aprendizaje supervisado (KNN, SVM y Random Forest) y despliega el mejor modelo en una app interactiva.



- 🌐 **App desplegada:** https://cancer-mama-ml.streamlit.app


---

## Datos del proyecto


|---|---|
| Universidad | Universidad Politécnica de Atlautla |
| Carrera | Lic. en Ing. en Tecnologías de la Información e Innovación Digital (Cuatrimestre 7) |
| Asignatura | Fundamentos de Inteligencia Artificial |
| Unidad / Tema | Unidad 3 · Tema 3.2.1 Aprendizaje Automático Supervisado |
| Docente | Mtro. Javier García Lara |
| Integrantes | Anahi Fuentes Avendaño  - Cesar Natanael Rosales Vallejo - Denise Estrada Franco |

---

##  Problematica

El cáncer de mama es uno de los más frecuentes y su detección temprana mejora de forma importante el pronóstico. El objetivo es construir un clasificador binario que apoye el tamizaje, estimando la probabilidad de que una masa sea maligna a partir de medidas citológicas.

- **Tipo de tarea:** clasificación binaria supervisada.
- **Clase positiva:** maligno (1). **Clase negativa:** benigno (0).
- **Métrica prioritaria:** *Recall* (sensibilidad), porque un falso negativo (no detectar un tumor maligno) es el error más costoso.

##  Dataset

- **Nombre:** Breast Cancer Wisconsin (Diagnostic).
- **Fuente:** UCI Machine Learning Repository / Kaggle (https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic).
- **Tamaño:** 569 muestras (212 malignas y 357 benignas).
- **Features usadas (10, valores medios del núcleo celular):** radio, textura, perímetro, área, suavidad, compacidad, concavidad, puntos cóncavos, simetría y dimensión fractal.
- **Target:** diagnóstico (0 = benigno, 1 = maligno).
- **División:** 455 muestras de entrenamiento (80 %) y 114 de prueba (20 %), con `random_state=42`.

##  Modelos que se compararon

Cada modelo se optimizó con búsqueda de hiperparámetros (GridSearchCV) y validación cruzada.

| Modelo | Mejores hiperparámetros |
|---|---|
| KNN | `n_neighbors=7`, `weights=uniform` |
| SVM | `C=10`, `kernel=rbf` |
| Random Forest | `n_estimators=100`, `max_depth=5`, `min_samples_leaf=3` |

### Resultados en el conjunto de prueba (114 muestras)

| Modelo | Accuracy | Precision | Recall | Especificidad | F1 | AUC-ROC |
|---|---|---|---|---|---|---|
| KNN | 0.9474 | 0.9091 | 0.9524 | 0.9444 | 0.9302 | 0.9747 |
| **SVM (ganador)** | **0.9737** | **0.9756** | 0.9524 | **0.9861** | **0.9639** | 0.9841 |
| Random Forest | 0.9386 | 0.9268 | 0.9048 | 0.9583 | 0.9157 | 0.9848 |

### Validación cruzada (media ± desviación estándar)

| Modelo | Accuracy | Recall | F1 | AUC |
|---|---|---|---|---|
| KNN | 0.937 ± 0.026 | 0.896 ± 0.053 | 0.914 ± 0.036 | 0.976 ± 0.017 |
| SVM | 0.944 ± 0.018 | 0.902 ± 0.044 | 0.923 ± 0.026 | 0.982 ± 0.009 |
| Random Forest | 0.939 ± 0.025 | 0.910 ± 0.046 | 0.917 ± 0.034 | 0.981 ± 0.013 |

### Análisis de sobreajuste (accuracy train vs. test)

| Modelo | Train | Test | Diferencia |
|---|---|---|---|
| KNN | 0.9495 | 0.9474 | 0.002 |
| SVM | 0.9736 | 0.9737 | 0.000 |
| Random Forest | 0.9802 | 0.9386 | 0.042 |

###  Modelo ganador: SVM (kernel RBF)

Se eligió SVM porque obtuvo la **mejor accuracy, precision, especificidad y F1** en prueba, empató con KNN en el mejor recall (95.2 %), mantuvo un AUC de 0.984 (prácticamente igual al de Random Forest) y **no muestra sobreajuste** (train ≈ test). Su matriz de confusión en prueba fue `[[71, 1], [2, 40]]`: solo 1 falso positivo y 2 falsos negativos.

##  Funcionalidades de la web

- **Evaluar muestra:** ingresa las medidas y obtiene la probabilidad de malignidad en un medidor; incluye casos de ejemplo (benigno típico, maligno típico y caso límite).
- **Análisis por lote:** clasificación de varias muestras a la vez.
- **Exploración de datos:** visualización del dataset.
- **Modelos y métricas:** comparación de los tres algoritmos, curvas ROC y curva de aprendizaje.
- **Umbral de decisión:** ajuste del umbral para equilibrar falsos positivos y falsos negativos.
- **Metodología y ética:** descripción del proceso y limitaciones.

##  Estructura del repositorio

```
├── app.py            # Aplicación Streamlit
├── config.py         # Configuración (textos, constantes)
├── entrenar.py       # Entrenamiento y comparación de modelos
├── datos.csv         # Dataset
├── metricas.json     # Métricas y resultados de los modelos
├── modelo.pkl        # Modelo ganador
├── modelos.pkl       # Modelos comparados
└── requirements.txt  # Dependencias
```

## Cómo ejecutarlo localmente

```bash
git clone https://github.com/deniseestrada84/cancer-mama.git
cd cancer-mama
pip install -r requirements.txt
streamlit run app.py
```

Para volver a entrenar los modelos (usa la semilla 42, por lo que los resultados son reproducibles):

```bash
python entrenar.py
```

**Entorno de entrenamiento:** Python 3.13.5 · scikit-learn 1.9.1 · semilla 42.

##  Consideraciones éticas

- El modelo es una herramienta de apoyo educativa y **no sustituye el diagnóstico médico**.
- El dataset es pequeño (569 casos) y de una sola fuente; el desempeño puede variar en otras poblaciones y equipos de medición.
- Un falso negativo puede retrasar un diagnóstico, por eso se prioriza el recall y se permite ajustar el umbral de decisión.

##  Uso de inteligencia artificial

Se utilizó IA generativa como apoyo para la generación de código base y la revisión de redacción. El código fue probado, comprendido y adaptado por el equipo. La declaración completa en formato APA 7 se incluye en el documento metodológico.

##  Referencias

- Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.
- Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3.ª ed.). O'Reilly Media.
- Russell, S., & Norvig, P. (2020). *Inteligencia Artificial: Un enfoque moderno* (4.ª ed.). Pearson Educación.
