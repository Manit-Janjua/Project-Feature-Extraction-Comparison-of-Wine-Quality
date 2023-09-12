from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import PCA
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time


data = pd.read_csv("winequality-white.csv", sep=';')
datay = data['quality']
datax = data.drop('quality', axis=1)
compare_results = []


def quality_mapping(quality):
    if quality <= 7:
        return 0  # not good
    elif quality > 7:
        return 1  # good
    else:
        return -1  # unknown


datay = data['quality'].map(quality_mapping)

X = datax
y = datay

pca = PCA(n_components=5)
X_pca = pca.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.25, random_state=42)

rf_classifier = RandomForestClassifier()

start_time = time.time()
rf_classifier.fit(X_train, y_train)
training_time = round(time.time()-start_time, 3)

print('****************************',
      "PCA - RandomForest Classifier", '****************************')
print("Training time taken: ", training_time, ' seconds')

start_time = time.time()
y_pred = rf_classifier.predict(X_test)
testing_time = round(time.time() - start_time, 3)

print("Testing time taken: ", testing_time, ' seconds')

accuracy = round(accuracy_score(y_test, y_pred)*100, 2)
precision = round(precision_score(y_test, y_pred)*100, 2)

print("Accuracy:", accuracy, ' %')
confusion_mat = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:\n", confusion_mat)
disp = ConfusionMatrixDisplay(
    confusion_matrix=confusion_mat, display_labels=['Not Good', 'Good'])

disp.plot()
plt.title("PCA - RandomForest Classifier")
plt.savefig('Project_SEP786_PCA_RF.png')

classification_rep = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_rep)

compare_results.append(
    ["PCA-randomforest", training_time, testing_time, accuracy, precision])

X = datax
y = datay

pca = PCA(n_components=5)
X_pca = pca.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.25, random_state=42)

knn_classifier = KNeighborsClassifier(n_neighbors=5)

start_time = time.time()

knn_classifier.fit(X_train, y_train)

training_time = round(time.time()-start_time, 3)

print('****************************', "PCA - KNeighbors classifier",
      '****************************',)
print("Training time taken: ", training_time, ' seconds')

start_time = time.time()

y_pred = knn_classifier.predict(X_test)

testing_time = round(time.time() - start_time, 3)

print("Testing time taken: ", testing_time, ' seconds')

accuracy = round(accuracy_score(y_test, y_pred)*100, 2)
precision = round(precision_score(y_test, y_pred)*100, 2)
print("Accuracy:", accuracy, ' %')

confusion_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", confusion_mat)
disp = ConfusionMatrixDisplay(
    confusion_matrix=confusion_mat, display_labels=['Not Good', 'Good'])
disp.plot()
plt.title("PCA - KNeighbors classifier")
plt.savefig('Project_SEP786_PCA_KN.png')

classification_rep = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_rep)

compare_results.append(
    ["PCA-knn_classifier", training_time, testing_time, accuracy, precision])

X = datax
y = datay

k_best = SelectKBest(score_func=chi2, k=5)
X_selected = k_best.fit_transform(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.25, random_state=42)

rf_classifier = RandomForestClassifier()

start_time = time.time()

rf_classifier.fit(X_train, y_train)

training_time = round(time.time()-start_time, 3)

print('****************************',
      "Feature Selection - RandomForest Classifier", '****************************')
print("Training time taken: ", training_time, ' seconds')

start_time = time.time()

y_pred = rf_classifier.predict(X_test)
testing_time = round(time.time() - start_time, 3)

print("Testing time taken: ", testing_time, ' seconds')

accuracy = round(accuracy_score(y_test, y_pred)*100, 2)
precision = round(precision_score(y_test, y_pred)*100, 2)
print("Accuracy:", accuracy, ' %')

confusion_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", confusion_mat)
disp = ConfusionMatrixDisplay(
    confusion_matrix=confusion_mat, display_labels=['Not Good', 'Good'])
disp.plot()
plt.title("Feature Selection - RandomForest Classifier")
plt.savefig('Project_SEP786_FS_RF.png')

classification_rep = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_rep)

compare_results.append(["FeatureSelect-randomforest",
                       training_time, testing_time, accuracy, precision])

X = datax
y = datay

k_best = SelectKBest(score_func=chi2, k=5)
X_selected = k_best.fit_transform(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.25, random_state=42)

knn_classifier = KNeighborsClassifier(n_neighbors=5)

start_time = time.time()
knn_classifier.fit(X_train, y_train)
training_time = round(time.time()-start_time, 3)

print('****************************',
      "Feature Selection - KNeighbors Classifier", '****************************')
print("Training time taken: ", training_time, ' seconds')

start_time = time.time()

y_pred = knn_classifier.predict(X_test)
testing_time = round(time.time() - start_time, 3)

print("Testing time taken: ", testing_time, ' seconds')

accuracy = round(accuracy_score(y_test, y_pred)*100, 2)
precision = round(precision_score(y_test, y_pred)*100, 2)
print("Accuracy:", accuracy, ' %')

confusion_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", confusion_mat)
disp = ConfusionMatrixDisplay(
    confusion_matrix=confusion_mat, display_labels=['Not Good', 'Good'])
disp.plot()
plt.title("Feature Selection - KNeighbors Classifier")
plt.savefig('Project_SEP786_FS_KN.png')

classification_rep = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_rep)
compare_results.append(["Featureselect-knn_classifier",
                       training_time, testing_time, accuracy, precision])

pprint(compare_results)
