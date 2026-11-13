"""Configuración de MamaIA: variables, nombres y utilidades para el CSV."""
from pathlib import Path
import re
import unicodedata

RAIZ = Path(__file__).resolve().parent
SEED = 42

# Se prioriza el archivo entregado datos(1).csv; si no existe, se usa datos.csv.
ARCHIVO_DATOS = RAIZ / "datos(1).csv"
if not ARCHIVO_DATOS.exists():
    ARCHIVO_DATOS = RAIZ / "datos.csv"

COLOR_BENIGNO = "#0ea5e9"
COLOR_MALIGNO = "#be185d"

# Diez variables medias del conjunto Wisconsin Diagnostic.
VARIABLES = {
    "mean radius": {"nombre": "Radio medio", "decimales": 3, "ayuda": "Promedio de las distancias del centro al perímetro del núcleo."},
    "mean texture": {"nombre": "Textura media", "decimales": 3, "ayuda": "Variación local de intensidad de gris."},
    "mean perimeter": {"nombre": "Perímetro medio", "decimales": 3, "ayuda": "Perímetro medio del núcleo celular."},
    "mean area": {"nombre": "Área media", "decimales": 3, "ayuda": "Área media del núcleo celular."},
    "mean smoothness": {"nombre": "Suavidad media", "decimales": 5, "ayuda": "Variación local en la longitud del radio."},
    "mean compactness": {"nombre": "Compactación media", "decimales": 5, "ayuda": "Perímetro² / área − 1."},
    "mean concavity": {"nombre": "Concavidad media", "decimales": 5, "ayuda": "Severidad de las porciones cóncavas del contorno."},
    "mean concave points": {"nombre": "Puntos cóncavos medios", "decimales": 5, "ayuda": "Número de porciones cóncavas del contorno."},
    "mean symmetry": {"nombre": "Simetría media", "decimales": 5, "ayuda": "Simetría del núcleo celular."},
    "mean fractal dimension": {"nombre": "Dimensión fractal media", "decimales": 5, "ayuda": "Aproximación de la complejidad de la frontera."},
}

GRUPOS = {
    "Tamaño": ["mean radius", "mean perimeter", "mean area"],
    "Textura y forma": ["mean texture", "mean smoothness", "mean symmetry", "mean fractal dimension"],
    "Contorno": ["mean compactness", "mean concavity", "mean concave points"],
}
GRUPOS_UI = [
    ("📏 Tamaño del núcleo", ["mean radius", "mean perimeter", "mean area"]),
    ("🧩 Textura y forma", ["mean texture", "mean smoothness", "mean symmetry", "mean fractal dimension"]),
    ("〰️ Contorno", ["mean compactness", "mean concavity", "mean concave points"]),
]

ALIAS_ETIQUETA = {"maligno", "diagnostico", "diagnosis", "clase", "target", "label", "class"}


def limpiar(valor):
    """Normaliza nombres de columnas: minúsculas, sin acentos ni separadores."""
    s = unicodedata.normalize("NFKD", str(valor)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "", s.lower())


_ALIASES = {
    "mean radius": ["mean radius", "radius mean", "radius_mean", "radio medio", "radio"],
    "mean texture": ["mean texture", "texture mean", "texture_mean", "textura media", "textura"],
    "mean perimeter": ["mean perimeter", "perimeter mean", "perimeter_mean", "perimetro medio", "perimetro"],
    "mean area": ["mean area", "area mean", "area_mean", "area media", "area"],
    "mean smoothness": ["mean smoothness", "smoothness mean", "smoothness_mean", "suavidad media", "suavidad"],
    "mean compactness": ["mean compactness", "compactness mean", "compactness_mean", "compactacion media", "compactacion"],
    "mean concavity": ["mean concavity", "concavity mean", "concavity_mean", "concavidad media", "concavidad"],
    "mean concave points": ["mean concave points", "concave points mean", "concave_points_mean", "puntos concavos medios", "puntos concavos"],
    "mean symmetry": ["mean symmetry", "symmetry mean", "symmetry_mean", "simetria media", "simetria"],
    "mean fractal dimension": ["mean fractal dimension", "fractal dimension mean", "fractal_dimension_mean", "dimension fractal media", "dimension fractal"],
}
_ALIAS_MAP = {limpiar(alias): canon for canon, aliases in _ALIASES.items() for alias in aliases}


def mapear_columnas(columnas):
    """Devuelve {nombre_original: nombre_canónico} para las diez variables."""
    resultado = {}
    for columna in columnas:
        clave = limpiar(columna)
        if clave in _ALIAS_MAP:
            resultado[columna] = _ALIAS_MAP[clave]
    return resultado


def etiqueta_binaria(serie):
    """Convierte B/benigno a 0 y M/maligno a 1; acepta etiquetas 0/1."""
    import pandas as pd
    s = serie.copy()
    if pd.api.types.is_numeric_dtype(s):
        if s.isna().any():
            return None
        unicos = set(pd.Series(s.unique()).astype(float))
        if unicos.issubset({0.0, 1.0}):
            return s.astype(int)
        return None
    normal = s.astype(str).map(limpiar)
    mapa = {
        "b": 0, "benigno": 0, "benigna": 0, "benign": 0, "0": 0,
        "m": 1, "maligno": 1, "maligna": 1, "malignant": 1, "1": 1,
    }
    convertido = normal.map(mapa)
    if convertido.isna().any():
        return None
    return convertido.astype(int)
