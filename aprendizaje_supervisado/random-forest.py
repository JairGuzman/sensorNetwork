import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, cross_val_predict
from sklearn.metrics import confusion_matrix, roc_curve, precision_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint
import matplotlib.pyplot as plt

def load_data(url):
    """Carga los datos desde una URL y devuelve un DataFrame."""
    return pd.read_csv(url, engine='python', index_col=0)

def preprocess_data(data):
    """Separa las características y la variable objetivo, y normaliza las características."""
    X = data.iloc[:, 0:14].values
    y = data.iloc[:, 14].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def train_model(X, y):
    """Entrena un modelo de Random Forest con búsqueda aleatoria de hiperparámetros."""
    param_dist = {'n_estimators': randint(50, 500), 'max_depth': randint(1, 20)}
    rf = RandomForestClassifier()
    rf_random = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=3, cv=2)
    rf_random.fit(X, y)
    return rf_random.best_estimator_, rf_random.best_params_

def evaluate_model(model, X, y):
    """Evalúa el modelo usando validación cruzada y muestra varias métricas."""
    cv_scores = cross_val_score(model, X, y, cv=5)
    print('Cross-validation scores:', cv_scores)
    print('Mean cross-validation score:', cv_scores.mean())
    print('Accuracy:', cv_scores.mean())

    importances = model.feature_importances_
    plt.bar(range(len(importances)), importances)
    plt.show()

    y_pred = cross_val_predict(model, X, y, cv=5)
    cm = confusion_matrix(y, y_pred)
    print('Confusion Matrix:', cm)

    plt.matshow(cm)
    plt.title('Matriz de confusión')
    plt.colorbar()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='fuchsia')
    plt.show()

    y_pred_prob = cross_val_predict(model, X, y, cv=5, method='predict_proba')
    fpr, tpr, thresholds = roc_curve(y, y_pred_prob[:,1])
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Curva ROC')
    plt.show()

    precision = precision_score(y, y_pred)
    print('Precision:', precision)

    f_score = f1_score(y, y_pred)
    print('F1 Score:', f_score)

def main():
    url = 'https://raw.githubusercontent.com/JairGuzman/datasets/main/dataset_par_RF.csv'
    data = load_data(url)
    print("Total samples in dataset:", data.shape[0])
    print("Class distribution:\n", data.iloc[:, 14].value_counts())

    X, y = preprocess_data(data)
    best_model, best_params = train_model(X, y)
    print('Best parameters:', best_params)

    evaluate_model(best_model, X, y)

if __name__ == "__main__":
    main()