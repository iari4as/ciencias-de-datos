import pandas as pd
import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class LimpiarComillas(BaseEstimator, TransformerMixin):
    """
    • Limpia comillas simples y espacios en headers y celdas.
    • Quita espacios internos SÓLO en valores numéricos.
    • Convierte strings numéricos a float (NaN si falla) cuando cast_numeric=True.
    """

    def __init__(self, cast_numeric: bool = True):
        self.cast_numeric = cast_numeric

    @staticmethod
    def _clean_cell(val, cast_numeric):
        if not isinstance(val, str):
            return val
        v = val.strip(" '")
        if re.fullmatch(r"[0-9\.,\s]+", v):
            v_num = v.replace(" ", "")
            if cast_numeric:
                if "," in v_num and "." not in v_num:
                    v_num = v_num.replace(",", ".")
                return pd.to_numeric(v_num, errors="coerce")
            return v_num
        return v

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Limpiar nombres de columnas
        cols_raw = X.columns.astype(str).str.strip(" '")
        seen, cols_clean = {}, []
        for c in cols_raw:
            cnt = seen.get(c, 0)
            cols_clean.append(f"{c}_{cnt}" if cnt else c)
            seen[c] = cnt + 1
        X.columns = cols_clean

        # Limpiar celdas en columnas object o string
        obj_cols = X.select_dtypes(include=["object", "string"]).columns
        X[obj_cols] = X[obj_cols].applymap(
            lambda v: self._clean_cell(v, self.cast_numeric)
        )
        return X

class ConvertirObjectAString(BaseEstimator, TransformerMixin):
    """
    Convierte columnas de tipo 'object' o 'string' a tipo 'string' nativo de pandas.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in X.select_dtypes(include=["object", "string"]).columns:
            X[col] = X[col].astype("string")
        return X

# Pipeline básico de limpieza
pipeline_red = Pipeline([
    ("convertir_a_str", ConvertirObjectAString()),
    ("strip", LimpiarComillas())
])

def crear_pipeline_completo(X):
    """
    Construye un Pipeline completo con:
    - Limpieza de comillas y conversión de objetos a string (pipeline_red)
    - Escalado de columnas numéricas
    - Codificación one-hot de columnas categóricas (salida sparse para ahorro de memoria)
    """
    # Identificar columnas categóricas y numéricas
    columnas_categoricas = X.select_dtypes(include=["object", "string"]).columns.tolist()
    columnas_numericas = X.select_dtypes(include=["number"]).columns.tolist()

    # ColumnTransformer con salida sparse
    preprocesador = ColumnTransformer([
        ("num", StandardScaler(), columnas_numericas),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32), columnas_categoricas)
    ], sparse_threshold=0.0)

    pipeline_completo = Pipeline([
        ("limpieza", pipeline_red),
        ("prepro", preprocesador)
    ])
    return pipeline_completo
