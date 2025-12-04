import os
import re
import joblib


class ClasificadorEmail:
    def __init__(self, email_texto):
        self.email_texto = email_texto
        self.modelo = None
        self.vectorizer = None

    # ---------------------------------------------------
    # 1. Cargar modelo y vectorizador desde /models/
    # ---------------------------------------------------
    def load_models(self):
        # Directorio base del proyecto (subir un nivel desde src/)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        model_path = os.path.join(base_dir, "models", "model.pkl")
        vectorizer_path = os.path.join(base_dir, "models", "vectorizer.pkl")

        print(f"Cargando modelo desde: {model_path}")
        print(f"Cargando vectorizador desde: {vectorizer_path}")

        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            print("ERROR: No se encontraron archivos de modelo o vectorizador.")
            return None, None

        try:
            self.modelo = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            print("Modelos cargados correctamente.")
        except Exception as e:
            print(f"ERROR cargando modelos: {e}")
            return None, None

        return self.modelo, self.vectorizer

    # ---------------------------------------------------
    # 2. Limpieza básica del texto
    # ---------------------------------------------------
    def limpiar(self, texto):
        texto = str(texto).lower()
        texto = re.sub(r"[^a-zA-Z0-9áéíóúñ ]", " ", texto)
        return texto

    # ---------------------------------------------------
    # 3. Clasificar correo y devolver etiqueta + probabilidad
    # ---------------------------------------------------
    def clasificar(self, email_texto):
        model, vectorizer = self.load_models()
        if model is None or vectorizer is None:
            print("ERROR: Modelos no cargados.")
            return None, None

        texto_limpio = self.limpiar(email_texto)

        try:
            # Vectorizar texto
            X = vectorizer.transform([texto_limpio])

            # Predicción: devuelve "ham" o "spam"
            prediction = model.predict(X)[0]

            prob_pred = None
            if hasattr(model, "predict_proba"):
                probas = model.predict_proba(X)[0]       # array de probs
                clases = list(model.classes_)            # p.ej. ['ham', 'spam']

                if prediction in clases:
                    idx = clases.index(prediction)       # índice de la clase predicha
                    prob_pred = float(probas[idx])       # probabilidad de ESA clase
                else:
                    # por seguridad, coger la prob máxima
                    prob_pred = float(max(probas))

            return prediction, prob_pred

        except Exception as e:
            print(f"ERROR clasificando: {e}")
            return None, None


# ---------------------------------------------------
# Prueba rápida desde consola
# ---------------------------------------------------
if __name__ == "__main__":
    texto = "Congratulations! You have won a FREE prize!"
    cls = ClasificadorEmail(texto)
    r, p = cls.clasificar(texto)
    print("Resultado:", r, "Probabilidad:", p)
