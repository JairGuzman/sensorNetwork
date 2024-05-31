import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# get the data
url = 'https://raw.githubusercontent.com/JairGuzman/datasets/main/dataset_par_RF.csv'
data=pd.read_csv(url, engine='python', index_col=0)

#split the data into features (x) and target variable (y)
X = data.iloc[:, 0:14].values
y = data.iloc[:, 14].values

#split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#cross validation
param_dist = {'n_estimators': randint(50, 500), 'max_depth': randint(1, 20)}  
#create a random forest classifier
rf = RandomForestClassifier()
#use random search to find the best hyperparameters
rf_random = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=3, cv=2)
#fit the model
rf_random.fit(X_train, y_train)

#create a varible and store the best hyperparameters
best_params = rf_random.best_estimator_

print('Best parameters:', rf_random.best_params_)

#make predictions
y_pred = best_params.predict(X_test)

#calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

#predict the probability of the target variable
y_pred_prob = best_params.predict_proba(X_test)
print('Probability:', y_pred_prob)

#predict the importance of the features
importances = best_params.feature_importances_
print('Importance:', importances)

#plot the importance of the features
import matplotlib.pyplot as plt
plt.bar(range(len(importances)), importances)
plt.show()

# confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:', cm)
# show confusion matrix as image
plt.matshow(cm)
plt.title('Confusion Matrix')
plt.colorbar()
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()


# curve roc
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:,1])
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

# precision
from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred)
print('Precision:', precision)

# F_score
from sklearn.metrics import f1_score
f_score = f1_score(y_test, y_pred)
print('F1 Score:', f_score)
