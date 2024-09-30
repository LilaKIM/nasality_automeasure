# Projet de Prédiction Automatisée de la nasalité

Ce projet utilise un modèle W2V2 et une suite de scripts pour effectuer des prédictions de la nasalité sur l'ensemble des productions de parole.

## Installation des bibliothèques


```python
pip install -r requirements.txt
```

## Exécution du script

Pour exécuter le projet, utilisez le script **Bash** qui orchestre toutes les étapes nécessaires pour traiter les fichiers audio, les phonetiser, et faire des prédictions. Voici la commande à exécuter :

```bash
bash main_script.sh
```

## main_script.sh
Le script main_script.sh exécute les différentes étapes suivantes :

- **Découpage Audio** : Utilise decoupe_audio.praat pour segmenter les fichiers audio. Les segments sont ensuite stockés dans audio_decoupe/.
- **Extraction de Représentations Vectorielles** : Utilise get_representation_w2v.py pour extraire les représentations vectorielles des fichiers audio situés dans audio_4s/ et audio_decoupe/.
- **Phonetisation** : Utilise le script phonetisation.py pour transcrire les segments d'audio en phonèmes.
- **Pondération des Nasales** : Utilise ponderation_nasal.py pour appliquer une pondération des probabilités de nasalité et du nombre de nasals présents dans un enregistrement.
- **Prédiction de l'indice de nasalité** : Utilise prediction.py pour faire des prédictions sur les données vectorielles situées dans vectors/ et stocke les résultats dans csv/.
