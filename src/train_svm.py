import pandas as pd
import re
import os
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# ============================================
# 1. Cargar dataset corregido
# ============================================

def load_dataset(path="data/data_spam.csv"):
    df = pd.read_csv(
        path,
        sep=",",
        usecols=[0, 1],
        names=["label", "text"],
        encoding="latin1",
        engine="python",
        on_bad_lines="skip"
    )
    # eliminar fila del encabezado original
    df = df.iloc[1:].reset_index(drop=True)
    return df


# ============================================
# 2. Limpieza básica del texto
# ============================================

def limpiar(texto):
    texto = str(texto).lower()
    texto = re.sub(r"[^a-zA-Z0-9áéíóúñ ]", " ", texto)
    return texto


# ============================================
# 3. Entrenar SVM calibrado (con probabilidades)
# ============================================

def entrenar_svm(df):

    df["text"] = df["text"].astype(str).apply(limpiar)

    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'],
        test_size=0.2, random_state=42,
        stratify=df['label']
    )

    # Pipeline TF-IDF + SVM calibrado
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.95,
            min_df=2
        )),
        ("svm", CalibratedClassifierCV(
            estimator=LinearSVC(class_weight='balanced'),
            cv=5,
            method='sigmoid'  # probabilidad suave
        ))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print("\n=========== RESULTADOS ===========")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["ham", "spam"],
        yticklabels=["ham", "spam"]
    )
    plt.title("Matriz de confusión - SVM Calibrado")
    plt.xlabel("Predicción")
    plt.ylabel("Verdadero")
    plt.show()

    return pipeline


# ============================================
# 4. Guardar modelo y vectorizador
# ============================================

def guardar_modelo(pipeline):
    os.makedirs("models", exist_ok=True)

    joblib.dump(pipeline.named_steps["svm"], "models/model.pkl")
    joblib.dump(pipeline.named_steps["tfidf"], "models/vectorizer.pkl")

    print("\nModelo guardado en: models/model.pkl")
    print("Vectorizador guardado en: models/vectorizer.pkl")


# ============================================
# 5. EJECUCIÓN DEL SCRIPT
# ============================================

if __name__ == "__main__":
    print("Cargando dataset...")
    df = load_dataset()

    print("Entrenando modelo SVM calibrado...")
    pipeline = entrenar_svm(df)

    print("Guardando modelo y vectorizador...")
    guardar_modelo(pipeline)

    print("\nProceso completado ✔")
