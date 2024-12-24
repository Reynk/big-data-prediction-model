import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

#package stat 
# Incarcare
data = pd.read_csv('Breast_cancer_data.csv')

# exploration 

# Privire de ansamblu asupra datelor
descriptive_stats = data.describe()

# Verificare pentru date goale
missing_values = data.isnull().sum()

# Verificam distributia variabilei/atribut ales ca tinta/dependent
target_distribution = data['diagnosis'].value_counts(normalize=True)

# Facem plot histogramei pentru fiecare variabila sa putem  
# vizualiza distributiile
data.hist(bins=15, figsize=(15, 10), layout=(3, 3))


# Asignam coloanele la variabilele noastre
X = data.iloc[:, :-1]  # toate coloanele exceptand ultima
y = data.iloc[:, -1]   # doar ultima coloana - diagnosis

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# try different modele:

##### logistic regression #####
lr_model = LogisticRegression(max_iter=500)

# Antrenam modelul
lr_model.fit(X_train, y_train)

# Modelul antrenat, este folosit acum pentru a prezice diagnosticul
# folosim datele de test, si le incarcam apoi pentru comparare
y_pred_lr = lr_model.predict(X_test)

# Comparam acum rezultatele prezicerii, cu datele noastre de test
lr_accuracy = accuracy_score(y_test, y_pred_lr)
print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")


##### support vector classifier #####
svc_model = SVC()

# Antrenam modelul
svc_model.fit(X_train, y_train)

# Modelul antrenat, este folosit acum pentru a prezice diagnosticul
# folosim datele de test, si le incarcam apoi pentru comparare
y_pred_svc = svc_model.predict(X_test)

# Comparam acum rezultatele prezicerii, cu datele noastre de test
sv_accuracy = accuracy_score(y_test, y_pred_svc)
print(f'Support Vector Classifier: {sv_accuracy:.4}')


###### random forest ######
rf_model = RandomForestClassifier(n_estimators=500)

# Antrenam modelul
rf_model.fit(X_train, y_train)

# Modelul antrenat, este folosit acum pentru a prezice diagnosticul
# folosim datele de test, si le incarcam apoi pentru comparare
y_pred_rf = rf_model.predict(X_test)

# Comparam acum rezultatele prezicerii, cu datele noastre de test
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")


##### xgboost #####
xgb_model = XGBClassifier(use_label_encoder=False,
                          eval_metric='logloss', 
                          n_estimators=500)

# Antrenam modelul
xgb_model.fit(X_train, y_train)

# Modelul antrenat, este folosit acum pentru a prezice diagnosticul
# folosim datele de test, si le incarcam apoi pentru comparare
y_pred_xgb = xgb_model.predict(X_test)

# Comparam acum rezultatele prezicerii, cu datele noastre de test
xgboost_accuracy = accuracy_score(y_test, y_pred_xgb)
print(f"XGboost Accuracy: {xgboost_accuracy:.4f}")


##### Continuam cu Random Forest deoarece are cea mai mare acuratete #####
#lipseste validarea
# Initializam modelul Random Forest
rf_model = RandomForestClassifier()

# Parametrii pentru cautarea hiperparametrilor
param_grid = {
    'n_estimators': [ 500],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Setam cautarea in grid
grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    scoring='accuracy',
    n_jobs=-1,
    cv=5,
    verbose=1
)

# Executam cautarea in grid pe datele de antrenament
grid_search.fit(X_train, y_train)


# Extragem cei mai buni parametri si cel mai bun scor obtinut
best_params = grid_search.best_params_
print(best_params)


# Initializam modelul ce cei mai buni parametri gasiti
rf_model = RandomForestClassifier(
    bootstrap=True, max_depth=None, min_samples_leaf=2, min_samples_split=2, n_estimators=500)


# Antrenam modelul cu hiperparametri pe setul de antrenament
history = rf_model.fit(X_train, y_train)

# Realizam predictia pe setul de testare
y_pred = rf_model.predict(X_test)


# Evaluam predictiile
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Importanta caracteristicilor in procesul de diagnosticare
feature_importances = rf_model.feature_importances_
plt.barh(X.columns, feature_importances)
plt.xlabel("Feature Importance")
plt.show()
