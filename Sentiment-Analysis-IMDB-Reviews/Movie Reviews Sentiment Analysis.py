# Importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Importing dataset
dataset = pd.read_csv('IMDB Dataset.csv')

# Remove duplicate rows
dataset = dataset.drop_duplicates()

# Changing mapping of sentiment column
dataset['sentiment'] = dataset['sentiment'].map({'positive': 1, 'negative': 0})

# Convert text to lowercase
dataset['review'] = dataset['review'].str.lower()

# Cleaning reviews from special characters
dataset['review'] = dataset['review'].str.replace(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "")

# Split features into separate columns
X = dataset['review']
y = dataset['sentiment']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert reviews into vectors using TF-IDF representation
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train the model with SVM classifier
model = SVC()
model.fit(X_train_vectorized, y_train)

# Evaluate the model's performance on the test data
y_pred = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
