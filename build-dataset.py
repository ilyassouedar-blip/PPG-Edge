import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter
import pandas as pd

# ============================================================
# 1. FONCTIONS DE GÉNÉRATION DE SIGNAL (reprises de avant)
# ============================================================

def gaussienne(t, centre, amplitude, largeur):
    return amplitude * np.exp(-((t - centre) ** 2) / (2 * largeur ** 2))

def generer_battement(t_local):
    pic_systolique = gaussienne(t_local, centre=0.15, amplitude=1.0, largeur=0.035)
    pic_diastolique = gaussienne(t_local, centre=0.32, amplitude=0.35, largeur=0.04)
    return pic_systolique + pic_diastolique

def generer_signal_propre(fs, duree_totale, bpm_moyen=75):
    duree_battement_moyenne = 60 / bpm_moyen
    t_global = np.linspace(0, duree_totale, int(fs * duree_totale))
    signal = np.zeros_like(t_global)
    temps_courant = 0.0
    while temps_courant < duree_totale:
        duree_battement = duree_battement_moyenne * (1 + np.random.normal(0, 0.03))
        idx_debut = int(temps_courant * fs)
        idx_fin = int((temps_courant + duree_battement) * fs)
        idx_fin = min(idx_fin, len(t_global))
        if idx_fin <= idx_debut:
            break
        t_local = np.linspace(0, duree_battement, idx_fin - idx_debut)
        battement = generer_battement(t_local / duree_battement * 0.8)
        signal[idx_debut:idx_fin] = battement
        temps_courant += duree_battement
    return t_global, signal

def ajouter_bruit(signal_propre, t_global, niveau_bruit, freq_mouvement=0.3):
    """niveau_bruit : 0 = pas de bruit, 1 = bruit très fort"""
    bruit_mouvement = niveau_bruit * 0.5 * np.sin(2 * np.pi * freq_mouvement * t_global + np.random.rand())
    bruit_aleatoire = niveau_bruit * 0.15 * np.random.randn(len(t_global))
    derive = niveau_bruit * 0.4 * np.sin(2 * np.pi * 0.05 * t_global)
    return signal_propre + bruit_mouvement + bruit_aleatoire + derive

def filtrer_signal(signal, fs):
    """Butterworth puis Savitzky-Golay (filtrage combiné validé précédemment)"""
    nyquist = fs / 2
    b, a = butter(4, 5 / nyquist, btype='low')
    signal_butter = filtfilt(b, a, signal)
    signal_final = savgol_filter(signal_butter, window_length=15, polyorder=3)
    return signal_final

# ============================================================
# 2. EXTRACTION DE FEATURES SUR UNE FENÊTRE
# ============================================================

def extraire_features(fenetre):
    """Calcule des indicateurs numériques résumant une fenêtre de signal filtré."""
    variance = np.var(fenetre)
    amplitude_max = np.max(fenetre) - np.min(fenetre)
    energie = np.sum(fenetre ** 2)
    # Nombre de pics détectés (approximation simple : changements de signe de la dérivée)
    derivee = np.diff(fenetre)
    changements_signe = np.sum(np.diff(np.sign(derivee)) != 0)
    return [variance, amplitude_max, energie, changements_signe]

# ============================================================
# 3. GÉNÉRATION DU DATASET COMPLET
# ============================================================

fs = 100
duree_signal = 10       # secondes par signal généré
duree_fenetre = 2        # secondes par fenêtre (pour extraction de features)
n_signaux = 150          # nombre de signaux à générer (on varie le bruit)

X = []  # features
y = []  # labels (0 = exploitable, 1 = corrompu)

for i in range(n_signaux):
    # Niveau de bruit aléatoire entre 0 (propre) et 1.5 (très corrompu)
    niveau_bruit = np.random.uniform(0, 1.5)
    
    t_global, signal_propre = generer_signal_propre(fs, duree_signal)
    signal_bruite = ajouter_bruit(signal_propre, t_global, niveau_bruit)
    signal_filtre = filtrer_signal(signal_bruite, fs)
    
    # Label : on décide selon le niveau de bruit INJECTÉ à l'origine
    label = 0 if niveau_bruit < 0.6 else 1  # seuil de "corruption acceptable"
    
    # Découpage en fenêtres
    taille_fenetre = int(duree_fenetre * fs)
    n_fenetres = len(signal_filtre) // taille_fenetre
    
    for f in range(n_fenetres):
        debut = f * taille_fenetre
        fin = debut + taille_fenetre
        fenetre = signal_filtre[debut:fin]
        
        features = extraire_features(fenetre)
        X.append(features)
        y.append(label)

# --- Conversion en DataFrame pour inspection facile ---
colonnes = ['variance', 'amplitude_max', 'energie', 'nb_changements_signe']
df = pd.DataFrame(X, columns=colonnes)
df['label'] = y

print(f"Dataset généré : {len(df)} exemples")
print(f"Répartition des labels :\n{df['label'].value_counts()}")
print(f"\nAperçu des données :\n{df.head()}")

# --- Sauvegarde pour la suite ---
df.to_csv('ppg_dataset.csv', index=False)
print("\nDataset sauvegardé dans ppg_dataset.csv")