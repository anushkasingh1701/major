import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split

import sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle as shf

import warnings
import math
warnings.filterwarnings('ignore')
import pickle
import os
import glob as gb
code = {'NORMAL':0 ,'COVID':1}
#function to return the class of the images from its number, so the function would return 'Normal' if given 0, and 'PNEUMONIA' if given 1.
def getcode(n) : 
    for x , y in code.items() : 
        if n == y : 
            return x


print("heart report \n")
df = pd.read_csv('./Heart_Disease_Prediction.csv')
# preprocessing
# print(type(df))
df.head()
# print(df.head())
df.describe()
# print(df.describe())
df.shape
# print(df.shape)
df.isnull().values.any()
df['Sex'].value_counts()
# print(df['Sex'].value_counts())
df['Heart Disease'].value_counts()
sns.countplot(x='Heart Disease', data=df)
df.corr()
x = df.iloc[:, :-2]
# print(x)
y = df.iloc[:, -1]
# print(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = None, test_size = 0.35)
sc_x = StandardScaler()
# print(sc_x)
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

#kneighbors
print("kneighbors")
math.sqrt(len(y_test))
classifier = KNeighborsClassifier(n_neighbors = 9, p = 2, metric = 'euclidean')
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)
y_pred
rf_conf_matrix = confusion_matrix(y_test,y_pred)
print("confusion matrix of kneighobrs \n",rf_conf_matrix)
# fig, ax = plt.subplots(figsize=(5, 5))
# ax.matshow(rf_conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
# for i in range(rf_conf_matrix.shape[0]):
#     for j in range(rf_conf_matrix.shape[1]):
#         ax.text(x=j, y=i,s=rf_conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
# plt.xlabel('Predictions', fontsize=18)
# plt.ylabel('Actuals', fontsize=18)
# plt.title('Confusion Matrix', fontsize=18)
# plt.show()
print("\n")
print("Accuracy of kneighobrs:",accuracy_score(y_test,y_pred)*100,"%")
print('Precision: %.3f' % precision_score(y_test, y_pred,pos_label ='Absence'))
print('Recall: %.3f' % recall_score(y_test, y_pred,pos_label ='Absence'))
print('F1 Score: %.3f' % f1_score(y_test, y_pred,pos_label ='Absence'))

#support vector machines
print("support vector machines")
clf = svm.SVC(kernel='rbf')
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
# y_pred = clf.predict(x_test)
y_pred
rf_conf_matrix = confusion_matrix(y_test,y_pred)
print("confusion matrix of svm \n",rf_conf_matrix)
# fig, ax = plt.subplots(figsize=(5, 5))
# ax.matshow(rf_conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
# for i in range(rf_conf_matrix.shape[0]):
#     for j in range(rf_conf_matrix.shape[1]):
#         ax.text(x=j, y=i,s=rf_conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
# plt.xlabel('Predictions', fontsize=18)
# plt.ylabel('Actuals', fontsize=18)
# plt.title('Confusion Matrix', fontsize=18)
# plt.show()
print("\n")
print("Accuracy of svm:",accuracy_score(y_test,y_pred)*100)
print('Precision: %.3f' % precision_score(y_test, y_pred,pos_label ='Absence'))
print('Recall: %.3f' % recall_score(y_test, y_pred,pos_label ='Absence'))
print('F1 Score: %.3f' % f1_score(y_test, y_pred,pos_label ='Absence'))

#Random forest
print("Random forest")
rf = RandomForestClassifier(n_estimators=500, random_state=12, max_depth=5)
rf.fit(x_train,y_train)
rf_predicted = rf.predict(x_test) #give value to pridict in place of test
rf_conf_matrix = confusion_matrix(y_test, rf_predicted)
print("confusion matrix of Random Forest\n",rf_conf_matrix)
# fig, ax = plt.subplots(figsize=(5, 5))
# ax.matshow(rf_conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
# for i in range(rf_conf_matrix.shape[0]):
#     for j in range(rf_conf_matrix.shape[1]):
#         ax.text(x=j, y=i,s=rf_conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
# plt.xlabel('Predictions', fontsize=18)
# plt.ylabel('Actuals', fontsize=18)
# plt.title('Confusion Matrix', fontsize=18)
# plt.show()
rf_acc_score = accuracy_score(y_test, rf_predicted)
print("\n")
print("Accuracy of Random Forest:",rf_acc_score*100)
print('Precision: %.3f' % precision_score(y_test, rf_predicted,pos_label ='Absence'))
print('Recall: %.3f' % recall_score(y_test, rf_predicted,pos_label ='Absence'))
print('F1 Score: %.3f' % f1_score(y_test, rf_predicted,pos_label ='Absence'))



print("\n")
print("lund report")
#the directory that contain the train images set
trainpath='./train3/'

X_train = []
y_train = []
for folder in  os.listdir(trainpath) : 
    files = gb.glob(pathname= str( trainpath + folder + '/*.jpeg'))
    for file in files: 
        image = cv2.imread(file)
        #resize images to 64 x 64 pixels
        image_array = cv2.resize(image , (64,64))
        X_train.append(list(image_array))
        y_train.append(code[folder])
np.save('X_train',X_train)
np.save('y_train',y_train)

#the directory that contain the test images set
testpath='./test3/'

X_test = []
y_test = []
for folder in  os.listdir(testpath) : 
    files = gb.glob(pathname= str( testpath + folder + '/*.jpeg'))
    for file in files: 
        image = cv2.imread(file)
        #resize images to 64 x 64 pixels
        image_array = cv2.resize(image , (64,64))
        X_test.append(list(image_array))
        y_test.append(code[folder])
np.save('X_test',X_test)
np.save('y_test',y_test)

#X_train, X_test contain the images as numpy arrays, while y_train, y_test contain the class of each image 
loaded_X_train = np.load('./X_train.npy')
loaded_X_test = np.load('./X_test.npy')
loaded_y_train = np.load('./y_train.npy')
loaded_y_test = np.load('./y_test.npy')

# print(loaded_X_train.shape)
# print(loaded_X_test.shape)

# print(loaded_y_train.shape)
# print(loaded_y_test.shape)

#Scaling
sc = StandardScaler()
X_train = loaded_X_train.reshape([-1, np.product((64,64,3))])
X_test = loaded_X_test.reshape([-1, np.product((64,64,3))])
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

pca = PCA(.95)
pca.fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

# print('Number of components after PCA: ' + str(pca.n_components_))

knn_PCA = KNeighborsClassifier(n_neighbors=2)
rfc_PCA = RandomForestClassifier()
svm_PCA = SVC()

knn_PCA.fit(X_train, y_train)
rfc_PCA.fit(X_train, y_train)
svm_PCA.fit(X_train, y_train)
print("\n")
print("KNeighobors")
rf_predicted = knn_PCA.predict(X_test) #give value to pridict in place of test
rf_conf_matrix = confusion_matrix(y_test, rf_predicted)
# print("confusion matrix of KNN\n",rf_conf_matrix)
# fig, ax = plt.subplots(figsize=(5, 5))
# ax.matshow(rf_conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
# for i in range(rf_conf_matrix.shape[0]):
#     for j in range(rf_conf_matrix.shape[1]):
#         ax.text(x=j, y=i,s=rf_conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
# plt.xlabel('Predictions', fontsize=18)
# plt.ylabel('Actuals', fontsize=18)
# plt.title('Confusion Matrix', fontsize=18)
# plt.show()
print('KNN accuracy score is: ' + str(knn_PCA.score(X_test, y_test)*100))
print('Precision: %.3f' % precision_score(y_test, rf_predicted))
print('Recall: %.3f' % recall_score(y_test, rf_predicted))
print('F1 Score: %.3f' % f1_score(y_test, rf_predicted))
print("\n")
print("randomforest")
rf_predicted = rfc_PCA.predict(X_test) #give value to pridict in place of test
rf_conf_matrix = confusion_matrix(y_test, rf_predicted)
# print("confusion matrix of Random Forest\n",rf_conf_matrix)
# fig, ax = plt.subplots(figsize=(5, 5))
# ax.matshow(rf_conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
# for i in range(rf_conf_matrix.shape[0]):
#     for j in range(rf_conf_matrix.shape[1]):
#         ax.text(x=j, y=i,s=rf_conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
# plt.xlabel('Predictions', fontsize=18)
# plt.ylabel('Actuals', fontsize=18)
# plt.title('Confusion Matrix', fontsize=18)
# plt.show()
print('Random forests Classifier accuracy score is:not real one ' + str(rfc_PCA.score(X_test, y_test)*100))
print('Precision: %.3f' % precision_score(y_test, rf_predicted))
print('Recall: %.3f' % recall_score(y_test, rf_predicted))
print('F1 Score: %.3f' % f1_score(y_test, rf_predicted))
print("\n")
print("svm")
rf_predicted = svm_PCA.predict(X_test) #give value to pridict in place of test
rf_conf_matrix = confusion_matrix(y_test, rf_predicted)
# print("confusion matrix of Support Vector Machine\n",rf_conf_matrix)
# fig, ax = plt.subplots(figsize=(5, 5))
# ax.matshow(rf_conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
# for i in range(rf_conf_matrix.shape[0]):
#     for j in range(rf_conf_matrix.shape[1]):
#         ax.text(x=j, y=i,s=rf_conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
# plt.xlabel('Predictions', fontsize=18)
# plt.ylabel('Actuals', fontsize=18)
# plt.title('Confusion Matrix', fontsize=18)
# plt.show()
print('Support Vector Machine Classifier accuracy score is: ' + str(svm_PCA.score(X_test, y_test)*100))
print('Precision: %.3f' % precision_score(y_test, rf_predicted))
print('Recall: %.3f' % recall_score(y_test, rf_predicted))
print('F1 Score: %.3f' % f1_score(y_test, rf_predicted))


# smote = SMOTE(random_state = 11)
# X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
# rfc_smote = RandomForestClassifier()
# rfc_smote.fit(X_train_smote, y_train_smote)
# rf_predicted = rfc_smote.predict(X_test) #give value to pridict in place of test
# rf_conf_matrix = confusion_matrix(y_test, rf_predicted)
# print("confusion matrix of Random Forest\n",rf_conf_matrix)
# print('Random forests Classifier accuracy score is: ' + str(rfc_smote.score(X_test, y_test)*100))