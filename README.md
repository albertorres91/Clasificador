<div align="center">

# **UNIVERSIDAD [Universidad Icesi]**  
## **Facultad de Ingeniería y Ciencias Aplicadas**  
### **MAestría en Inteligencia Artificial Aplicada**


# **CLASIFICADOR DE MENSAJES TIPO SMS (SPAM)**

## **Autores**
**Andrés Rodriguez**  
**Alberto Torres**
**David Osorio**
**Javier Yela**

</div>

Proyecto en Python para entrenar y desplegar un modelo de clasificación de correos electrónicos mediante Machine Learning y Streamlit.

Este repositorio incluye:

- Un modelo entrenado para clasificar mensajes como spam o no spam.
- Una aplicación web construida con Streamlit para pruebas de clasificación.
- Un script de entrenamiento del modelo.
- Un notebook para análisis exploratorio de datos (EDA).
- El dataset utilizado y los modelos ya entrenados.


## Estructura del repositorio

Clasificador/
│
├── data/
│   └── data_spam.csv                 # Dataset original (ham/spam)
│
├── models/
│   ├── model.pkl                     # Modelo entrenado (SVM balanceado)
│   └── vectorizer.pkl                # Vectorizador TF-IDF
│
├── src/
│   ├── app.py                        # Aplicación Streamlit
│   ├── ClasificadorEmail.py          # Lógica principal del clasificador
│   └── train_svm.py                  # Script de entrenamiento del modelo
│
├── images/                           # Evidencias de funcionamiento
│   ├── img1.png
│   ├── img2.png
│   └── img3.png
│
├── venv/                             # Entorno virtual (opcional, ignorado por git)
│
├── EDA_spam.ipynb                    # Notebook de análisis exploratorio
├── requirements.txt                  # Dependencias del proyecto
└── README.md                         # Documentación del proyecto


## Instalación

### 1. Clonar el repositorio
```bash
git clone https://github.com/jhyela/Clasificador.git
cd Clasificador
```
### 2. Clonar el repositorio
```bash
python -m venv venv
source venv/bin/activate         # Linux/Mac
venv\Scripts\activate            # Windows
```
### 3. Instalar dependencias
```bash
pip install -r requirements.txt

```
### 4 Ejecutar la aplicación
```bash
streamlit run src/app.py
```
Se abrirá la aplicación en
http://localhost:8501

### 5. Funcionamiento
1. El usuario ingresa el texto.

2. La aplicación carga el modelo (model.pkl) y el vectorizador (vectorizer.pkl).

3. El texto se transforma y se clasifica como spam o no spam.

4. Se muestra la predicción y la probabilidad asociada.

Ejmplos de clasificación

![Ejemplo de uso](images/img1.png)

![Ejemplo de uso](images/img2.png)

![Ejemplo de uso](images/img3.png)

### 6. Entrenar el modelo

El archivo train_svm.py:

1. Carga el dataset

2. Limpia el texto

3. Entrena un modelo SVM con class_weight='balanced'

4. Calibra el modelo para obtener probabilidades (CalibratedClassifierCV)

5. Guarda los modelos resultantes en /models/model.pkl y /models/vectorizer.pkl
```bash
python src/train_svm.py
```
Esto generará o actualizará los archivos:
```bash
models/model.pkl
models/vectorizer.pkl
```

### 8. Información del dataset

Este dataset corresponde al clásico SMS Spam Collection, que contiene:

ham → mensajes legítimos

spam → mensajes promocionales o fraudulentos

idioma inglés

formato SMS

El script train_svm.py corrige automáticamente el dataset, elimina columnas sobrantes y normaliza las etiquetas.

### 9. Clasificación individual
La aplicación permite ingresar un texto tipo SMS libremente.
El usuario escribe el texto, presiona "Clasificar", y la app devuelve:

Si es SPAM o NO SPAM

La probabilidad estimada por el modelo

### 10. Consluciones y consideraciones
El clasificador posee limitaciones evidentes debido al dataset de entrenamiento que se usó. El dataset está en inglés y contiene mensajes tipo SMS, esto es, cortos, informales y directos.

Si se desea integrar un idioma diferente, es necesario entrenar el modelo nuevamente. La función de traducción no resultó ser una alternativa útil ya que diferentes idiomas poseen diferentes estructuras y modos de escribir estafas. Esto se muestra en las imágenes de prueba del punto 5.
