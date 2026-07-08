import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- 1. Charger le dataset généré précédemment ---
df = pd.read_csv('ppg_dataset.csv')

X = df[['variance', 'amplitude_max', 'energie', 'nb_changements_signe']]
y = df['label']

# --- 2. Séparer en train / test ---
# 80% pour entraîner le modèle, 20% pour évaluer sur des données jamais vues
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 3. Normalisation des features ---
# Important : variance, amplitude, energie n'ont pas la même échelle
# On centre/réduit pour que le modèle ne favorise pas une feature juste parce
# que ses valeurs sont numériquement plus grandes
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 4. Entraînement du modèle ---
modele = LogisticRegression(class_weight='balanced')  # compense le déséquilibre 480/270
modele.fit(X_train_scaled, y_train)

# --- 5. Évaluation ---
y_pred = modele.predict(X_test_scaled)

print(f"Précision (accuracy) : {accuracy_score(y_test, y_pred):.2%}")
print(f"\nMatrice de confusion :\n{confusion_matrix(y_test, y_pred)}")
print(f"\nRapport détaillé :\n{classification_report(y_test, y_pred)}")

# --- 6. Regarder l'importance de chaque feature ---
coefficients = pd.DataFrame({
    'feature': X.columns,
    'coefficient': modele.coef_[0]
}).sort_values('coefficient', key=abs, ascending=False)

print(f"\nImportance des features (valeur absolue = influence) :\n{coefficients}")
from sklearn.metrics import f1_score, precision_recall_curve
import numpy as np

# --- Récupérer les probabilités prédites (pas juste la classe finale) ---
y_proba = modele.predict_proba(X_test_scaled)[:, 1]  # probabilité de la classe 1 (corrompu)

# --- Tester plusieurs seuils et calculer le F1-score de la classe 1 pour chacun ---
seuils = np.arange(0.1, 0.9, 0.05)
meilleurs_resultats = []

for seuil in seuils:
    y_pred_seuil = (y_proba >= seuil).astype(int)
    f1 = f1_score(y_test, y_pred_seuil, pos_label=1)
    meilleurs_resultats.append((seuil, f1))

# --- Trouver le seuil optimal ---
meilleur_seuil, meilleur_f1 = max(meilleurs_resultats, key=lambda x: x[1])
print(f"\nMeilleur seuil trouvé : {meilleur_seuil:.2f} (F1-score classe 1 = {meilleur_f1:.3f})")

# --- Réévaluer le modèle avec ce nouveau seuil ---
y_pred_optimise = (y_proba >= meilleur_seuil).astype(int)

print(f"\n--- Résultats avec seuil optimisé ({meilleur_seuil:.2f}) ---")
print(f"Précision (accuracy) : {accuracy_score(y_test, y_pred_optimise):.2%}")
print(f"\nMatrice de confusion :\n{confusion_matrix(y_test, y_pred_optimise)}")
print(f"\nRapport détaillé :\n{classification_report(y_test, y_pred_optimise)}")
# ============================================================
# EXTRACTION DES PARAMÈTRES POUR DÉPLOIEMENT EMBARQUÉ (C++)
# ============================================================

print("\n" + "="*60)
print("PARAMÈTRES POUR L'IMPLÉMENTATION EN C++ (ESP32)")
print("="*60)

# --- Paramètres du scaler (normalisation) ---
print("\n--- Scaler (StandardScaler) ---")
print(f"Moyennes (mean_)  : {scaler.mean_}")
print(f"Écarts-types (scale_) : {scaler.scale_}")

# --- Paramètres du modèle (régression logistique) ---
print("\n--- Régression logistique ---")
print(f"Coefficients (poids) : {modele.coef_[0]}")
print(f"Intercept (biais)    : {modele.intercept_[0]}")

# --- Seuil optimisé qu'on a trouvé ---
print(f"\nSeuil de décision optimisé : {meilleur_seuil:.2f}")

# --- Format prêt à copier-coller en C++ ---
print("\n--- Version C++ (à copier dans le code Arduino) ---")
print(f"float scaler_mean[4] = {{{', '.join(f'{v:.6f}f' for v in scaler.mean_)}}};")
print(f"float scaler_scale[4] = {{{', '.join(f'{v:.6f}f' for v in scaler.scale_)}}};")
print(f"float model_coef[4] = {{{', '.join(f'{v:.6f}f' for v in modele.coef_[0])}}};")
print(f"float model_intercept = {modele.intercept_[0]:.6f}f;")
print(f"float seuil_decision = {meilleur_seuil:.2f}f;")