"""MamaIA · Entrenamiento y evaluación de modelos (Fases 1 a 3 de la actividad).

Uso:
    python entrenar.py

Genera: datos.csv, modelos.pkl (KNN, SVM y Random Forest), modelo.pkl (el ganador) y metricas.json.
La aplicación (app.py) también ejecuta este script sola si faltan esos archivos.

Decisiones de metodología:
  * Pipeline (escalado + modelo): el escalador se ajusta solo con los datos de entrenamiento
    de cada pliegue, así no hay fuga de información hacia la validación.
  * 3 algoritmos supervisados (KNN, SVM y Random Forest), cada uno con GridSearchCV (5 pliegues, métrica F1).
  * Validación cruzada repetida 5x3 para estimar la media y la variabilidad de cada modelo.
  * El ganador se elige con la validación cruzada (nunca con el conjunto de prueba).
  * El conjunto de prueba (20 %, estratificado) se usa una sola vez, para el reporte final.
"""
import json
import os
import platform
import time
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_validate, learning_curve, train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import ARCHIVO_DATOS, RAIZ, SEED, VARIABLES, mapear_columnas, etiqueta_binaria

os.environ.setdefault("PYTHONWARNINGS", "ignore::FutureWarning")  # también para los procesos paralelos
warnings.filterwarnings("ignore", category=ConvergenceWarning)
# scikit-learn >= 1.9 avisa que SVC(probability=True) cambiará; en las versiones fijadas en
# requirements.txt sigue funcionando, así que se silencia solo ese aviso.
warnings.filterwarnings("ignore", category=FutureWarning, message=".*probability.*")


def _pipeline(estimador, escalar: bool) -> Pipeline:
    """Escala solo cuando el algoritmo lo necesita (distancias o gradientes)."""
    pasos = [("escalador", StandardScaler())] if escalar else []
    return Pipeline(pasos + [("modelo", estimador)])


def espacios_de_busqueda() -> dict:
    """Devuelve únicamente KNN, SVM y Random Forest con sus hiperparámetros."""
    return {
        "KNN": (
            _pipeline(KNeighborsClassifier(), True),
            {"modelo__n_neighbors": [3, 5, 7, 9, 11, 15],
             "modelo__weights": ["uniform", "distance"]}),
        "SVM": (
            _pipeline(SVC(probability=True, random_state=SEED), True),
            {"modelo__C": [0.1, 1, 10, 100],
             "modelo__kernel": ["linear", "rbf"]}),
        "Random Forest": (
            _pipeline(RandomForestClassifier(random_state=SEED), False),
            {"modelo__n_estimators": [100, 300],
             "modelo__max_depth": [3, 5, None],
             "modelo__min_samples_leaf": [1, 3]}),
    }


def cargar_datos():
    """Lee el CSV del proyecto y valida sus 10 variables y la etiqueta maligno.

    Se prioriza datos(1).csv (archivo entregado). Si no existe, se intenta datos.csv.
    La etiqueta queda normalizada como 1 = maligno y 0 = benigno.
    """
    candidatos = [ARCHIVO_DATOS, RAIZ / "datos.csv"]
    archivo = next((p for p in candidatos if p.exists()), None)
    if archivo is None:
        raise FileNotFoundError(
            "No se encontró el dataset. Coloca datos(1).csv junto a entrenar.py "
            "o guarda el archivo como datos.csv."
        )

    crudo = pd.read_csv(archivo)
    mapa = mapear_columnas(crudo.columns)
    faltan = [f for f in VARIABLES if f not in mapa.values()]
    if faltan:
        raise ValueError("Al CSV le faltan estas variables: " + ", ".join(faltan))

    X = crudo.rename(columns=mapa)[list(VARIABLES)].copy()
    for columna in X.columns:
        X[columna] = pd.to_numeric(X[columna], errors="coerce")
    if X.isna().any().any():
        columnas_malas = X.columns[X.isna().any()].tolist()
        raise ValueError("Hay valores vacíos o no numéricos en: " + ", ".join(columnas_malas))
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("El dataset contiene valores infinitos; corrige el CSV antes de entrenar.")

    etiqueta_col = next((c for c in crudo.columns if str(c).strip().lower() in
                         {"maligno", "diagnostico", "diagnóstico", "diagnosis", "clase"}), None)
    if etiqueta_col is None:
        raise ValueError("No se encontró la columna objetivo 'maligno' o 'diagnosis'.")
    y = etiqueta_binaria(crudo[etiqueta_col])
    if y is None:
        raise ValueError("La etiqueta debe usar 0/1, B/M o benigno/maligno.")
    y = y.astype(int).rename("maligno")
    if y.nunique() != 2:
        raise ValueError("La variable objetivo debe contener las dos clases: benigno y maligno.")
    if len(X) < 50:
        raise ValueError("El dataset contiene muy pocos registros para la partición y validación solicitadas.")
    return X, y


def _r(valor, dec=4):
    return round(float(valor), dec)


def calcular_metricas(y_real, pred, proba) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_real, pred).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "accuracy": _r((tp + tn) / (tp + tn + fp + fn)),
        "precision": _r(precision),
        "recall": _r(recall),
        "especificidad": _r(tn / (tn + fp)),
        "f1": _r(f1),
        "auc": _r(roc_auc_score(y_real, proba)),
        "cm": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def entrenar(verbose: bool = True) -> dict:
    log = print if verbose else (lambda *a, **k: None)

    # ---------------- Fase 1: datos ----------------
    X, y = cargar_datos()
    df = X.assign(maligno=y)
    df.to_csv(RAIZ / "datos.csv", index=False)
    columnas = list(X.columns)
    log(f"Datos: {df.shape[0]} registros, {len(columnas)} variables | "
        f"nulos={int(df.isnull().sum().sum())} | duplicados={int(df.duplicated().sum())} | "
        f"malignos={y.mean():.1%}")

    # ---------------- Fase 2: partición ----------------
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    log(f"Entrenamiento: {len(X_tr)} | Prueba: {len(X_te)}")

    # ---------------- Fase 3: modelos ----------------
    validacion_repetida = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    metricas_cv = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    resultados, entrenados = {}, {}

    for nombre, (base, malla) in espacios_de_busqueda().items():
        # Modelo con hiperparámetros por defecto (punto de comparación)
        defecto = clone(base).fit(X_tr, y_tr)
        met_defecto = calcular_metricas(y_te, defecto.predict(X_te), defecto.predict_proba(X_te)[:, 1])

        # Optimización con GridSearchCV
        t0 = time.time()
        busqueda = GridSearchCV(base, malla, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                                scoring="f1", n_jobs=-1).fit(X_tr, y_tr)
        segundos = time.time() - t0
        mejor = busqueda.best_estimator_

        # Estabilidad: validación cruzada repetida (5 pliegues x 3 repeticiones)
        cv = cross_validate(mejor, X_tr, y_tr, cv=validacion_repetida, scoring=metricas_cv, n_jobs=-1)
        cv_resumen = {m.replace("roc_auc", "auc"): {"media": _r(cv[f"test_{m}"].mean()),
                                                    "desv": _r(cv[f"test_{m}"].std())}
                      for m in metricas_cv}

        # Evaluación final en prueba (una sola vez)
        proba = mejor.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        fpr, tpr, _ = roc_curve(y_te, proba)

        proba_train = mejor.predict_proba(X_tr)[:, 1]
        pred_train = (proba_train >= 0.5).astype(int)
        metricas_train = calcular_metricas(y_tr, pred_train, proba_train)

        resultados[nombre] = {
            "params": {k.replace("modelo__", ""): str(v) for k, v in busqueda.best_params_.items()},
            "combinaciones": int(len(busqueda.cv_results_["params"])),
            "tiempo_busqueda_s": round(segundos, 1),
            "cv": cv_resumen,
            "acc_train": metricas_train["accuracy"],
            "train": metricas_train,
            "test": calcular_metricas(y_te, pred, proba),
            "base": {k: met_defecto[k] for k in ("accuracy", "recall", "f1", "auc")},
            "roc": {"fpr": [_r(v) for v in fpr], "tpr": [_r(v) for v in tpr]},
            "proba_test": [_r(v) for v in proba],
        }
        entrenados[nombre] = mejor
        log(f"{nombre:<22} F1 CV={cv_resumen['f1']['media']:.3f}±{cv_resumen['f1']['desv']:.3f} | "
            f"prueba: acc={resultados[nombre]['test']['accuracy']:.3f} "
            f"recall={resultados[nombre]['test']['recall']:.3f} auc={resultados[nombre]['test']['auc']:.3f} "
            f"| {segundos:.1f}s")

    # El ganador se elige con validación cruzada (F1 medio), desempatando por recall.
    ganador = max(resultados, key=lambda n: (resultados[n]["cv"]["f1"]["media"],
                                             resultados[n]["cv"]["recall"]["media"]))
    mejor_modelo = entrenados[ganador]
    log(f"\nGanador: {ganador}")

    # Importancia por permutación (cuánto cae el AUC al desordenar cada variable)
    perm = permutation_importance(mejor_modelo, X_te, y_te, scoring="roc_auc", n_repeats=30,
                                  random_state=SEED, n_jobs=-1)
    importancia = {c: {"media": _r(perm.importances_mean[i], 5), "desv": _r(perm.importances_std[i], 5)}
                   for i, c in enumerate(columnas)}

    # Curva de aprendizaje del ganador (diagnóstico de sobreajuste)
    tamanios, tr_sc, va_sc = learning_curve(
        mejor_modelo, X_tr, y_tr, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
        train_sizes=np.linspace(0.15, 1.0, 7), scoring="f1", n_jobs=-1, shuffle=True, random_state=SEED)
    curva = {"n": [int(v) for v in tamanios],
             "train": [_r(v) for v in tr_sc.mean(axis=1)], "val": [_r(v) for v in va_sc.mean(axis=1)],
             "train_desv": [_r(v) for v in tr_sc.std(axis=1)], "val_desv": [_r(v) for v in va_sc.std(axis=1)]}

    # Casos reales de prueba para la demostración (el modelo no los vio al entrenar)
    p_gan = np.array(resultados[ganador]["proba_test"])
    y_arr, idx = y_te.to_numpy(), X_te.index.to_numpy()

    def tipico(clase: int) -> int:
        correctos = [i for i in range(len(y_arr)) if y_arr[i] == clase and (p_gan[i] >= 0.5) == bool(clase)]
        if not correctos:
            # Evita detener el entrenamiento si el ganador no acierta ningún caso de una clase.
            candidatos = np.flatnonzero(y_arr == clase).tolist()
            if not candidatos:
                raise ValueError("No hay ejemplos de ambas clases en el conjunto de prueba.")
            objetivo = 1.0 if clase == 1 else 0.0
            return min(candidatos, key=lambda i: abs(float(p_gan[i]) - objetivo))
        correctos.sort(key=lambda i: p_gan[i])
        return correctos[len(correctos) // 2]

    def ejemplo(titulo, i):
        return {"titulo": titulo, "real": "Maligno" if y_arr[i] else "Benigno", "indice": int(idx[i]),
                "valores": {c: _r(X_te.iloc[i][c]) for c in columnas}}

    ejemplos = [ejemplo("🔵 Caso benigno típico", tipico(0)),
                ejemplo("🔴 Caso maligno típico", tipico(1)),
                ejemplo("🟡 Caso límite", int(np.argmin(np.abs(p_gan - 0.5))))]

    # ---------------- Exportación ----------------
    joblib.dump(entrenados, RAIZ / "modelos.pkl")
    joblib.dump(mejor_modelo, RAIZ / "modelo.pkl")

    meta = {
        "ganador": ganador, "features": columnas, "n": int(len(df)), "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)), "n_maligno": int(y.sum()),
        "rangos": {c: [float(X[c].min()), float(X[c].max()), float(X[c].median())] for c in columnas},
        "y_test": [int(v) for v in y_arr], "idx_test": [int(v) for v in idx],
        "archivo_origen": str(next((p.name for p in [ARCHIVO_DATOS, RAIZ / "datos.csv"] if p.exists()), "CSV del proyecto")),
        "modelos": resultados, "importancia": importancia, "curva_aprendizaje": curva,
        "ejemplos": ejemplos,
        "version": {"sklearn": sklearn.__version__, "python": platform.python_version(), "semilla": SEED,
                    "fecha": datetime.now().strftime("%Y-%m-%d %H:%M")},
    }
    (RAIZ / "metricas.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    log("Archivos generados: datos.csv, modelos.pkl, modelo.pkl, metricas.json")
    return meta


if __name__ == "__main__":
    entrenar()
