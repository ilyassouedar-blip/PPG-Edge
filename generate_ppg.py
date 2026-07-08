import numpy as np
import matplotlib.pyplot as plt

def gaussienne(t, centre, amplitude, largeur):
    """Une bosse gaussienne : amplitude * exp(-(t-centre)^2 / (2*largeur^2))"""
    return amplitude * np.exp(-((t - centre) ** 2) / (2 * largeur ** 2))

def generer_battement(t_local):
    """Génère un seul battement PPG (systolique + diastolique)."""
    pic_systolique = gaussienne(t_local, centre=0.15, amplitude=1.0, largeur=0.035)
    pic_diastolique = gaussienne(t_local, centre=0.32, amplitude=0.35, largeur=0.04)
    return pic_systolique + pic_diastolique

# --- Paramètres généraux ---
fs = 100                  # fréquence d'échantillonnage (Hz)
duree_totale = 10         # durée du signal complet (s)
bpm_moyen = 75             # fréquence cardiaque moyenne
duree_battement_moyenne = 60 / bpm_moyen  # durée moyenne d'un battement (s)

t_global = np.linspace(0, duree_totale, int(fs * duree_totale))
signal_propre = np.zeros_like(t_global)

# --- Construction du signal en enchaînant les battements ---
temps_courant = 0.0
while temps_courant < duree_totale:
    # Variabilité du rythme cardiaque : chaque battement dure un peu différemment
    duree_battement = duree_battement_moyenne * (1 + np.random.normal(0, 0.03))
    
    # Indices correspondant à ce battement dans le signal global
    idx_debut = int(temps_courant * fs)
    idx_fin = int((temps_courant + duree_battement) * fs)
    idx_fin = min(idx_fin, len(t_global))
    
    if idx_fin <= idx_debut:
        break
    
    # Temps local pour ce battement (recommence à 0 à chaque battement)
    t_local = np.linspace(0, duree_battement, idx_fin - idx_debut)
    battement = generer_battement(t_local / duree_battement * 0.8)  # on adapte l'échelle
    
    signal_propre[idx_debut:idx_fin] = battement
    temps_courant += duree_battement

# --- Ajout du bruit de mouvement (comme avant) ---
bruit_mouvement = 0.15 * np.sin(2 * np.pi * 0.3 * t_global + np.random.rand())
bruit_aleatoire = 0.03 * np.random.randn(len(t_global))
signal_bruite = signal_propre + bruit_mouvement + bruit_aleatoire

# --- Ajout de la dérive de ligne de base ---
derive = 0.2 * np.sin(2 * np.pi * 0.05 * t_global)
signal_derive = signal_bruite + derive

# --- Visualisation ---
fig, axs = plt.subplots(3, 1, figsize=(11, 10))

axs[0].plot(t_global, signal_propre, color='green')
axs[0].set_title(f"Signal PPG continu propre (~{bpm_moyen} bpm, {duree_totale}s)")

axs[1].plot(t_global, signal_bruite, color='orange')
axs[1].set_title("Signal PPG avec bruit de mouvement")

axs[2].plot(t_global, signal_derive, color='red')
axs[2].set_title("Signal PPG avec bruit + dérive de ligne de base")
axs[2].set_xlabel("Temps (s)")

plt.tight_layout()
plt.show()
from scipy.signal import butter, filtfilt, savgol_filter

# --- Filtre Butterworth passe-bas ---
# On veut garder les fréquences liées au rythme cardiaque (jusqu'à ~5 Hz)
# et couper le bruit haute fréquence
def filtre_butterworth(signal, fs, cutoff=5, ordre=4):
    nyquist = fs / 2
    normal_cutoff = cutoff / nyquist
    b, a = butter(ordre, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)  # filtfilt = filtrage sans déphasage

signal_filtre_butter = filtre_butterworth(signal_derive, fs)

# --- Filtre Savitzky-Golay ---
# Lisse le signal en gardant bien la forme des pics (contrairement à une simple moyenne mobile)
signal_filtre_savgol = savgol_filter(signal_derive, window_length=15, polyorder=3)
# --- Filtrage combiné : Butterworth d'abord, puis Savitzky-Golay ---
signal_filtre_combine = savgol_filter(signal_filtre_butter, window_length=15, polyorder=3)
# --- Visualisation comparative ---
fig, axs = plt.subplots(5, 1, figsize=(11, 10))

axs[0].plot(t_global, signal_propre, color='green')
axs[0].set_title("Signal propre (référence)")

axs[1].plot(t_global, signal_derive, color='red')
axs[1].set_title("Signal bruité + dérive (entrée du filtre)")

axs[2].plot(t_global, signal_filtre_butter, color='blue')
axs[2].set_title("Après filtre Butterworth passe-bas")

axs[3].plot(t_global, signal_filtre_savgol, color='purple')
axs[3].set_title("Après filtre Savitzky-Golay")
axs[3].set_xlabel("Temps (s)")

axs[4].plot(t_global, signal_filtre_combine, color='black')
axs[4].set_title("Filtrage combiné : Butterworth puis Savitzky-Golay")
axs[4].set_xlabel("Temps (s)")

plt.tight_layout()
plt.show()