# Importations nécessaires
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# Chargement des données
df = pd.read_csv('database.csv')

# Suppression des colonnes inutiles
df.drop(['ID', 'Z_Revenue', 'Z_CostContact', 'Complain', 'Dt_Customer'], axis=1, inplace=True)

# Suppression des lignes avec valeurs manquantes pour 'Income'
dt_clear = df.dropna(subset=['Income'])

# Définition des variables explicatives et de la cible
y = dt_clear['Response']
X = dt_clear.drop('Response', axis=1)
X = pd.get_dummies(X, drop_first=True)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Encodage des variables catégorielles
# Remplacez 'Education' et 'Marital_Status' par les noms exacts de vos variables catégorielles :


# Séparation des données en jeu d'entraînement et jeu de test
X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(X, y_encoded, test_size=0.2, random_state=42)


# Modèle de régression logistique (baseline)
model = LogisticRegression()
model.fit(X_train, y_train_encoded)
print("=== Régression Logistique ===")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("Précision :", accuracy_score(y_test, y_pred))

# Définir la distribution des hyperparamètres pour RandomForest
param_dist = {
    'n_estimators': randint(50, 200),          # Nombre d'arbres
    'max_depth': [None, 10, 20, 30],           # Profondeur maximale
    'min_samples_split': randint(2, 10),       # Nombre min d'échantillons pour diviser un nœud
    'min_samples_leaf': randint(1, 10)         # Nombre min d'échantillons dans une feuille
}

# Initialisation du modèle RandomForest
rf = RandomForestClassifier(random_state=42)

# RandomizedSearchCV pour l'optimisation des hyperparamètres
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=50,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=2
)

# Entraînement avec RandomizedSearchCV
random_search.fit(X_train, y_train)

# Affichage des meilleurs hyperparamètres et de la précision
print("=== Optimisation RandomForest ===")
print("Meilleurs hyperparamètres :", random_search.best_params_)
print("Meilleure précision :", random_search.best_score_)

# Prédictions avec le meilleur modèle trouvé
y_pred_rf = random_search.best_estimator_.predict(X_test)
print(classification_report(y_test, y_pred_rf))
print("Précision RandomForest :", accuracy_score(y_test, y_pred_rf))

# Définir les paramètres à optimiser pour GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialiser GridSearchCV
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

# Entraîner avec GridSearchCV
grid_search.fit(X_train, y_train)

print("=== Optimisation RandomForest avec GridSearchCV ===")
print("Meilleurs hyperparamètres :", grid_search.best_params_)
print("Meilleure précision :", grid_search.best_score_)

y_pred_rf_gs = grid_search.best_estimator_.predict(X_test)
print(classification_report(y_test, y_pred_rf_gs))
print("Précision RandomForest avec GridSearchCV :", accuracy_score(y_test, y_pred_rf_gs))
# Exporter la base de données modifiée vers un fichier CSV
dt_clear.to_csv('new_data.csv', index=False)
print("Base exportée avec succès sous le nom 'new_data.csv'.")





def plot_feature_importance(rf, feature_names, top_n=10):
    # Obtenir les coefficients d'importance
    importance = rf.feature_importances_
    
    # Trouver les indices des n meilleures caractéristiques
    idx_top_pairs = np.argsort(importance)[::-1][:top_n]
    
    # Nommer les caractéristiques
    names = [feature_names[i] for i in idx_top_pairs]

    # Créer un graphique de barres
    plt.figure(figsize=(12, 8))
    bars = plt.bar(range(top_n), importance[idx_top_pairs], color='skyblue')
        
    # Ajouter les noms des caractéristiques comme étiquettes
    plt.xticks(range(top_n), names, rotation=45, ha='right')
    
    # Ajouter un titre et une légende
    plt.title(f'Importance des caractéristiques (Top {top_n})')
    plt.xlabel('Caractéristiques')
    plt.ylabel('Importance')
    
    # Afficher le graphique
    plt.tight_layout()
    plt.show()

# Utilisation du code
feature_names = dt_clear.columns.tolist()

# Appliquer la fonction à votre modèle RandomForestClassifier
plot_feature_importance(random_search.best_estimator_, feature_names)


# Si vous voulez utiliser toutes les caractéristiques :
# Remplacez 'target_column' par le nom réel de votre colonne cible
y = dt_clear['Income']
X = dt_clear.drop(['Income'], axis=1)

# Si vous voulez utiliser uniquement les top_n caractéristiques les plus importantes :
top_n_features = feature_names[:10]  # Par exemple, prendre les 10 premières caractéristiques
X = dt_clear[top_n_features]
y = dt_clear['Income']

# Encoder la variable cible

# Diviser les données en données d'entraînement et de test
X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Initialiser le modèle Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Entraîner le modèle
model.fit(X_train, y_train_encoded)

# Prédire sur les données de test
y_pred = model.predict(X_test)

# Évaluer le modèle
accuracy = accuracy_score(y_test_encoded, y_pred)
print(f"Score de précision : {accuracy:.3f}")
print("\nRapport de classification:")

# Afficher le rapport de classification
print("\nRapport de classification:")
print(classification_report(y_test_encoded, y_pred))

# Afficher la matrice de confusion
print("\nMatrice de confusion:")
print(confusion_matrix(y_test_encoded, y_pred))

# Affichage des informations sur le DataFrame
print(df.info())
print(df.describe())
print(df.head())
print(dt_clear.columns)
print(dt_clear.head())
print(dt_clear.info())

print(X.columns)
print(y.head())
print(X.dtypes)
print(y.dtype)     # Cela affichera le type de données de y
