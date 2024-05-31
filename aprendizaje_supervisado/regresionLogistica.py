import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

url = 'https://raw.githubusercontent.com/JairGuzman/datasets/main/dataset_par_RF.csv'
data=pd.read_csv(url, engine='python', index_col=0)
X = data.drop(columns="APROBO")
y = data["APROBO"]

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo
model = LogisticRegression()
model.fit(X_train, y_train)

# Hacer predicciones
y_pred = model.predict(X_test)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# matriz de confusión
from sklearn.metrics import confusion_matrix
mc = confusion_matrix(y_test, y_pred)
print('Matriz de confusión:', mc)

# show the confusion matrix as a image
plt.matshow(mc)
plt.title('Confusion Matrix')
plt.colorbar()
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# curva roc
from sklearn.metrics import roc_curve
y_pred_prob = model.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.show()

#precision
from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred)
print('Precision:', precision)



