from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from data_prep import *


'''Converting human-readable data to machine-readable data'''
# max_features: using 1500 most occurring words as features for training our classifier.
vectorizer = CountVectorizer(max_features=3500,
                              ngram_range=(1, 2))  # n_gram isskirsto zodzius su abiem zodziu deriniais aplinkui

# fit_transform: converts text documents into corresponding numeric features.
X = vectorizer.fit_transform(all_reviews).toarray()
pos = len(pos_reviews)
neg = len(neg_reviews)
y = [1] * pos + [0] * neg  # sudarem outputo masyva


'''Splitting data into Training and Testing Sets'''
# divided data into 20% test set and 80% training set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


'''Training and testing the model'''
classifier = RandomForestClassifier(n_estimators=500, max_depth=100, min_samples_split= 100, random_state=0) # sumazint n_estimators
classifier.fit(X_train, y_train)

# predict the sentiment for the documents in our test
y_pred = classifier.predict(X_test)


'''Check the accuracy of the model'''
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

