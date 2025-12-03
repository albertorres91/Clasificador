import pandas as pd
import os
import joblib
from deep_translator import GoogleTranslator


class ClasificadorEmail:
    def __init__(self, email_texto):
        self.email_texto = email_texto
        self.modelo = None
        self.vectorizer = None

    def load_models(self):
      # Obtener el directorio base del proyecto (un nivel arriba de src/)
      base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
      
      model_path = os.path.join(base_dir, 'models', 'model.pkl')
      vectorizer_path = os.path.join(base_dir, 'models', 'vectorizer.pkl')

      if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
        if not os.path.exists(vectorizer_path):
            print(f"Error: Vectorizer file not found at {vectorizer_path}")
        return None, None

      try:
        self.modelo = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)    
      except Exception as e:
        print(f"Error loading models: {e}")
        return None, None
      
      return self.modelo, self.vectorizer


    def clasificar(self, email_texto):
        # Implementar lógica de clasificación aquí
        model, vectorizer = self.load_models()
        if model is None or vectorizer is None:
            print("Error en el proceso, modelos no encontrados.")
            return None, None

        try:
            translated_text = GoogleTranslator(source='auto', target='en').translate(email_texto)
        except Exception as e:
            print(f"Error en la traducción: {e}")
            # Si falla la traducción, usar el texto original
            translated_text = email_texto

        try:
            input_tfidf = vectorizer.transform([translated_text])
            prediction = model.predict(input_tfidf)[0]
            proba = model.predict_proba(input_tfidf)[0]
            
            return prediction, proba[0]
        except Exception as e:
            print(f"Error en la clasificación: {e}")
            return None, None
    


# Ejemplo de uso
if __name__ == "__main__":
    email_texto = "AI and Learning to Code I've a few questions recently about AI and the impact it has on learning to code."
    clasificador = ClasificadorEmail(email_texto)
    resultado,probabilidad = clasificador.clasificar(email_texto)

    if resultado == 0:
        print("El correo NO es SPAM.","Probabilidad:", probabilidad)
    else:
        print("El correo SI es SPAM.","Probabilidad:", probabilidad)
