import pandas

#1. Load dataset

df = pandas.read_csv('data/data_spam.csv', encoding='latin-1', usecols=['v1', 'v2'],delimiter=',')
print(df.head(50))

#2. Preprocess dataset
df = df.rename(columns={'v1': 'label', 'v2': 'text'})
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
print(df.head(50))

#3. Split dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

#4. Vectorize text data
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

#5. Train a classifier
from sklearn import svm

classifier = svm.SVC(kernel='linear',probability=True)
classifier.fit(X_train_vec, y_train)

#6. Evaluate the classifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
y_pred = classifier.predict(X_test_vec)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred))
print('Recall:', recall_score(y_test, y_pred))
print('F1 Score:', f1_score(y_test, y_pred))

#7. Save the trained model and vectorizer
import joblib
joblib.dump(classifier, 'models/model.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')   