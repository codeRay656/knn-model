import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



df = pd.read_csv('gene_expression.csv')

print(df.head())

print(df.describe())



sns.scatterplot(x='Gene One', y ='Gene Two', hue = 'Cancer Present', data = df, alpha=0.7)

plt.savefig('gene_presence.png',dpi=300)





sns.scatterplot(x='Gene One',y='Gene Two',hue='Cancer Present',data=df)

plt.xlim(3,6)

plt.ylim(3,10)

plt.legend(loc=(1.1,0.5))

plt.savefig('expanded_fig.png')


import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

import pickle

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score





df = pd.read_csv('gene_expression.csv')

X = df.drop('Cancer Present',axis=1)

y = df['Cancer Present']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Training samples", len(X_train))

print("Test samples", len(X_test))

scaler = StandardScaler()



#scaled_X_train = scaler.fit_transform(X_train)

#scaled_X_test = scaler.transform(X_test)



knn_model = KNeighborsClassifier(n_neighbors=1)

knn_model.fit(X_train,y_train)



gene_model = 'knn_model.pkl'

pickle.dump(knn_model, open(gene_model,'wb'))

print("learned knn model created")



#evaluation

y_pred = knn_model.predict(X_test)

print(accuracy_score(y_test,y_pred))

print(confusion_matrix(y_test,y_pred))

import pickle





loaded_model = pickle.load(open('knn_model.pkl','rb'))

new_data = [[3,7.2]]

prediction = loaded_model.predict(new_data)

print("Prediction with new data: ")

print(prediction)



