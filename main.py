import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from scipy.stats import randint
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ipywidgets import interactive


df = pd.read_csv('database.csv')

#  colonnes inutiles
df.drop(['ID','Z_Revenue', 'Z_CostContact', 'Complain'], axis=1, inplace=True)

df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], errors='coerce')

# Suppression  valeurs manquantes pour 'Income'
dt_clear = df.dropna(subset=['Income'])


x = dt_clear.drop(columns=['Response'])
y = dt_clear['Response']

# Encodage des variables catégorielles
x = pd.get_dummies(x, columns=['Education', 'Marital_Status'], drop_first=True)

# Séparation des données en jeu d'entraînement et jeu de test
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Modèle de régression logistique 
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("=== Régression Logistique ===")
print(classification_report(y_test, y_pred))
print("Précision :", accuracy_score(y_test, y_pred))

param_dist = {
    'n_estimators': randint(50, 200),       
    'max_depth': [None, 10, 20, 30],          
    'min_samples_split': randint(2, 10),      
    'min_samples_leaf': randint(1, 10)       
}

# Initialisation du modèle RandomForest
rf = RandomForestClassifier(random_state=42)


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


print("=== Optimisation RandomForest ===")
print("Meilleurs hyperparamètres :", random_search.best_params_)
print("Meilleure précision :", random_search.best_score_)


y_pred_rf = random_search.best_estimator_.predict(X_test)
print(classification_report(y_test, y_pred_rf))
print("Précision RandomForest :", accuracy_score(y_test, y_pred_rf))


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


dt_clear.to_csv('new_data.csv', index=False)
"""print("Base exportée avec succès sous le nom 'new_data.csv'.")"""

# Affichage des informations sur le DataFrame
print(df.info())
print(df.describe())
print(df.head())
print(dt_clear.info())
print(dt_clear.head())

def plot_feature_importance(rf, feature_names, top_n=10):
  
    importance = rf.feature_importances_
    
    idx_top_pairs = np.argsort(importance)[::-1][:top_n]
    

    names = [feature_names[i] for i in idx_top_pairs]

    def update_graph(top_n):
        plt.figure(figsize=(12, 8))

        bars = plt.bar(range(top_n), importance[idx_top_pairs], color='skyblue')
        
        plt.xticks(range(top_n), names, rotation=45, ha='right')
        
        plt.title(f'Importance des caractéristiques (Top {top_n})')
        plt.xlabel('Caractéristiques')
        plt.ylabel('Importance')
        
        # Ajouter une légende pour les barres
        plt.legend([bars], ['Importance'])
        
        # Afficher le graphique
        plt.tight_layout()
        plt.show()
    
    plt.xticks(range(top_n), names, rotation=45, ha='right')
    
 
    plt.title(f'Importance des caractéristiques (Top {top_n})')
    plt.xlabel('Caractéristiques')
    plt.ylabel('Importance')
    
    plt.tight_layout()
    plt.show()


feature_names = dt_clear.columns.tolist()

# Appliquer la fonction à votre modèle RandomForestClassifier
interactive_plot = plot_feature_importance(random_search.best_estimator_, feature_names)
interactive_plot
