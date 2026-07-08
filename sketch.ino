// ============================================================
// PPG-EDGE : Détection de qualité de signal PPG sur ESP32-S3
// ============================================================

#include <math.h>
#define LED_BUILTIN 2  // pin LED par défaut sur la plupart des ESP32

// --- Paramètres de simulation ---
const int FS = 100;                  // fréquence d'échantillonnage (Hz)
const float DUREE_FENETRE = 2.0;     // durée d'une fenêtre d'analyse (s)
const int TAILLE_FENETRE = FS * DUREE_FENETRE;  // 200 échantillons

// --- Paramètres du modèle (extraits de Python) ---
float scaler_mean[4]  = {0.117051f, 1.286160f, 57.503829f, 14.470000f};
float scaler_scale[4] = {0.076289f, 0.342193f, 44.658710f, 2.160810f};
float model_coef[4]   = {2.000772f, 2.934525f, 0.839123f, -0.518849f};
float model_intercept = 2.475668f;
float seuil_decision  = 0.45f;

// --- Buffer pour stocker une fenêtre de signal ---
float buffer_signal[TAILLE_FENETRE];
float buffer_filtre[TAILLE_FENETRE];
int index_buffer = 0;

// --- Variables de génération du signal (simulation) ---
float niveau_bruit = 1.2;   // on pourra faire varier ça pour tester différents scénarios
float temps_global = 0.0;
float bpm_moyen = 75.0;

// ============================================================
// GÉNÉRATION DU SIGNAL PPG (portage du code Python)
// ============================================================

float gaussienne(float t, float centre, float amplitude, float largeur) {
  return amplitude * exp(-pow(t - centre, 2) / (2 * pow(largeur, 2)));
}

float generer_echantillon(float t_global) {
  // Position dans le cycle cardiaque courant (0 à 0.8s, comme en Python)
  float duree_battement = 60.0 / bpm_moyen;
  float t_local = fmod(t_global, duree_battement) / duree_battement * 0.8;

  float pic_systolique = gaussienne(t_local, 0.15, 1.0, 0.035);
  float pic_diastolique = gaussienne(t_local, 0.32, 0.35, 0.04);
  float signal_propre = pic_systolique + pic_diastolique;

  // Bruit (simplifié par rapport à Python, mais même principe)
  float bruit_mouvement = niveau_bruit * 0.5 * sin(2 * PI * 0.3 * t_global);
  float bruit_aleatoire = niveau_bruit * 0.15 * ((float)random(-100, 100) / 100.0);
  float derive = niveau_bruit * 0.4 * sin(2 * PI * 0.05 * t_global);

  return signal_propre + bruit_mouvement + bruit_aleatoire + derive;
}

// ============================================================
// EXTRACTION DE FEATURES (portage du code Python)
// ============================================================
void filtrer_signal(float* signal_in, float* signal_out, int taille, int taille_fenetre_filtre) {
  for (int i = 0; i < taille; i++) {
    int debut = max(0, i - taille_fenetre_filtre / 2);
    int fin = min(taille - 1, i + taille_fenetre_filtre / 2);
    float somme = 0;
    int count = 0;
    for (int j = debut; j <= fin; j++) {
      somme += signal_in[j];
      count++;
    }
    signal_out[i] = somme / count;
  }
}
void extraire_features(float* fenetre, int taille, float* features_out) {
  // 1. Variance
  float somme = 0, somme_carre = 0;
  for (int i = 0; i < taille; i++) {
    somme += fenetre[i];
  }
  float moyenne = somme / taille;
  for (int i = 0; i < taille; i++) {
    somme_carre += pow(fenetre[i] - moyenne, 2);
  }
  float variance = somme_carre / taille;

  // 2. Amplitude max
  float val_min = fenetre[0], val_max = fenetre[0];
  for (int i = 1; i < taille; i++) {
    if (fenetre[i] < val_min) val_min = fenetre[i];
    if (fenetre[i] > val_max) val_max = fenetre[i];
  }
  float amplitude_max = val_max - val_min;

  // 3. Énergie
  float energie = 0;
  for (int i = 0; i < taille; i++) {
    energie += fenetre[i] * fenetre[i];
  }

  // 4. Nombre de changements de signe (de la dérivée)
  int changements_signe = 0;
  for (int i = 2; i < taille; i++) {
    float derivee_prec = fenetre[i-1] - fenetre[i-2];
    float derivee_courante = fenetre[i] - fenetre[i-1];
    if ((derivee_prec > 0 && derivee_courante < 0) || (derivee_prec < 0 && derivee_courante > 0)) {
      changements_signe++;
    }
  }

  features_out[0] = variance;
  features_out[1] = amplitude_max;
  features_out[2] = energie;
  features_out[3] = changements_signe;
}

// ============================================================
// PRÉDICTION (régression logistique)
// ============================================================

float sigmoide(float x) {
  return 1.0 / (1.0 + exp(-x));
}

int predire_qualite(float* features) {
  float features_normalisees[4];
  for (int i = 0; i < 4; i++) {
    features_normalisees[i] = (features[i] - scaler_mean[i]) / scaler_scale[i];
  }

  float z = model_intercept;
  for (int i = 0; i < 4; i++) {
    z += model_coef[i] * features_normalisees[i];
  }

  float probabilite = sigmoide(z);
  return (probabilite >= seuil_decision) ? 1 : 0;  // 1 = corrompu, 0 = exploitable
}

// ============================================================
// SETUP & LOOP
// ============================================================

void setup() {
  Serial.begin(115200);
  pinMode(LED_BUILTIN, OUTPUT);
  Serial.println("PPG-Edge : démarrage de l'analyse en temps réel...");
}

void loop() {
  // Générer un échantillon et le stocker dans le buffer
  float echantillon = generer_echantillon(temps_global);
  buffer_signal[index_buffer] = echantillon;
  index_buffer++;
  temps_global += 1.0 / FS;

  // Quand le buffer est plein (une fenêtre complète), on analyse
  if (index_buffer >= TAILLE_FENETRE) {
    float features[4];
    filtrer_signal(buffer_signal, buffer_filtre, TAILLE_FENETRE, 15);
    extraire_features(buffer_filtre, TAILLE_FENETRE, features);
    int resultat = predire_qualite(features);

    Serial.print("Variance=");
    Serial.print(features[0], 4);
    Serial.print(" | Amplitude=");
    Serial.print(features[1], 4);
    Serial.print(" | Energie=");
    Serial.print(features[2], 4);
    Serial.print(" | Changements=");
    Serial.print(features[3], 0);
    Serial.print(" => ");
    Serial.println(resultat == 0 ? "SIGNAL EXPLOITABLE" : "SIGNAL CORROMPU");

    digitalWrite(LED_BUILTIN, resultat == 1 ? HIGH : LOW);

    index_buffer = 0;
  }

  delay(1000 / FS);
}