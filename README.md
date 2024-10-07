# Projet de Mesure Automatisée de la Nasalité dans l'Audio

Ce projet permet de calculer un **indice de nasalité** sur des fichiers audio, en utilisant des techniques de reconnaissance automatique de la parole. Le script Bash **`main_script.sh`** permet de traiter des fichiers audio de plusieurs façons en fonction des paramètres fournis, notamment pour l'analyse et le découpage des séquences audio.

## Utilisation du Script Bash

Le script **`main_script.sh`** peut être exécuté de plusieurs manières en fonction de vos besoins. Voici les différentes options disponibles :

### 1. Exécution Standard

```bash
bash main_script.sh
```
Utilise les données enregistrées dans le répertoire ./data/audio_original.
Produit un indice de nasalité pour l'ensemble d'un fichier audio (non découpé).

### 2. Spécifier le Chemin des Données
```bash
bash main_script.sh --path=YOUR/PATH
```
Permet de spécifier un chemin personnalisé vers les données audio.
Calcule un score de nasalité pour chaque fichier audio dans le répertoire spécifié, sans découpage des fichiers.
#### Paramètre :
--path=YOUR/PATH : Remplacez YOUR/PATH par le chemin absolu ou relatif vers votre dossier contenant les fichiers audio.

### 3. Découper les Fichiers Audio en Séquences de 4 Secondes
```bash
bash main_script.sh --nb=5
```
Utilise les données du répertoire ./data/audio_original et les découpe en séquences de 4 secondes.
Le nombre fourni correspond au nombre de séquences de 4 secondes extraites de chaque fichier audio.
Le découpage est aléatoire, offrant une analyse plus fine des segments courts.
#### Paramètre :
--nb=5 : Remplacez 5 par le nombre de séquences de 4 secondes que vous souhaitez extraire de chaque fichier audio.

### 4. Spécifier un Chemin et Découper en Séquences de 4 Secondes
```bash
bash main_script.sh --path=YOUR/PATH --nb=5
```
Utilise les fichiers audio situés dans le répertoire YOUR/PATH.
Les fichiers sont ensuite découpés en séquences de 4 secondes, en fonction du nombre spécifié.
#### Paramètres :

### Structure du Projet
Le projet est structuré de la manière suivante :

```bash
mon_projet/
│
├── main_script.sh                # Script Bash principal pour exécuter le traitement
├── script/                       # Dossier contenant les scripts de traitement
│   ├── decoupe_audio_4s.praat    # Script Praat pour découper les fichiers audio en séquences de 4 secondes
│   ├── decoupe_audio.praat       # Script Praat pour le découpage des fichiers audio avec une fenêtre glissante de 50 ms
│   ├── get_representation_w2v.py # Script Python pour extraire des représentations vectorielles
│   ├── phonetisation.py          # Script Python pour la transcription phonétique
│   ├── ponderation_nasal.py      # Script Python pour appliquer une pondération sur les phonèmes nasaux
│   └── prediction.py             # Script Python pour faire des prédictions sur les données vectorielles
├── data/                         # Dossier contenant les données audio et les résultats
│   ├── audio_original/           # Fichiers audio d'origine
│   ├── audio_4s/                 # Fichiers audio découpés en séquences de 4 secondes
│   ├── audio_decoupe/            # Fichiers audio découpés après traitement
│   ├── csv/                      # Résultats des prédictions stockés sous forme de fichiers CSV
│   └── vectors/                  # Représentations vectorielles extraites
├── requirements.txt              # Fichier listant toutes les dépendances Python nécessaires au projet
├── nasalite_ponderee.txt         # Fichier de sortie contenant le score de nasalité pondéré des fichiers audio traités
└── README.md                     # Ce fichier d'information sur le projet
```

### Dépendances
Avant d'utiliser le script, assurez-vous d'avoir installé les dépendances nécessaires :
Python 3.7+ : Requis pour exécuter les scripts Python.
Praat : Requis pour le découpage des fichiers audio.
Bibliothèques Python : Pour installer les bibliothèques nécessaires, exécutez la commande suivante :

```bash
pip install -r requirements.txt
```
Le fichier requirements.txt devrait contenir les bibliothèques suivantes :

```
tensorflow
pandas
numpy
huggingface_hub
scipy
```

### Exemple d'Utilisation
Pour calculer l'indice de nasalité sur les fichiers audio situés dans ./data/audio_original, exécutez simplement :
```bash
bash main_script.sh
```

Pour spécifier un autre répertoire d'audio et découper les fichiers en segments de 4 secondes (par exemple 5 segments par fichier) :
```bash
bash main_script.sh --path=YOUR/PATH --nb=5
```

### Auteurs
Lila Kim : PhD Student at Université Sorbonne Nouvelle - Paris III.
Email : lila.kim@sorbonne_nouvelle.fr
