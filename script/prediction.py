import argparse
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from tensorflow import keras

from huggingface_hub import HfApi

api = HfApi()

def getCSV(resultat, path_output):
  """
  input : dict(result), str(path to csv)
  output : obj(df)
  """
 
  df = pd.DataFrame.from_dict(resultat, orient="index")
  df_trie = df.sort_values(by='tDebut', ascending=True)
#   print(df)
#   print(df_trie)
  return df_trie.to_csv(path_output, sep=",", encoding="utf8", index=False)


def proba2csv(y_pred, fileNames, loc):
    resultat = dict()
    for i in range(len(y_pred)):
        score = dict()
        score["locuteur"] = loc
        score["ind"] = fileNames[i].split("-")[1].split("_")[0]
        score["proba_nasal"] = 1-float(f"{y_pred[i][0]}")
        score["tDebut"] = fileNames[i].split("_")[-3]
        score["tFin"] = fileNames[i].split("_")[-2]
        score["duree"] = fileNames[i].split("_")[-1][:-4]
        score["filename"] = fileNames[i]
        resultat[i+1] = score

    return getCSV(resultat, f"./data/csv/{loc}.csv")


def charger_modele():
    from huggingface_hub import hf_hub_download
    from tensorflow import keras

    # Télécharger le modèle depuis Hugging Face Hub
    repo_id = "lilakim/w2v2_nasality_automeasure"
    filename = "fr_6c6v_lebenchmark_3corpus_allphon_4_010924_sigmoid_mean_0.000125.keras"

    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    model = keras.models.load_model(model_path)

    model.summary()

    # import keras
    # # model = keras.saving.load_model('lilakim/w2v2_nasality_automeasure')
    # # model = model.export("lilakim/w2v2_nasality_automeasure")
    # model = keras.saving.load_model("hf://lilakim/w2v2_nasality_automeasure")
    return model


def obtenir_fichiers(repertoire):
    # Obtenir tous les fichiers dans un répertoire donné
    repertoire = Path(repertoire)
    fichiers = [fichier for fichier in repertoire.iterdir() if fichier.is_file()]
    return fichiers


def get_predictions(model, fichiers):
    # Effectuer des prédictions sur les fichiers vectorisés
    for fichier in fichiers:
        test = pickle.load(open(fichier.resolve(), "rb"))
        feats_df_test = np.array([np.array(val) for val in test["encoded_audio"].values])

        if feats_df_test.size == 0:
            raise ValueError("feats_df_test est vide, veuillez vérifier les données d'entrée.")

        if np.isnan(feats_df_test).any():
            raise ValueError("feats_df_test contient des valeurs NaN, veuillez vérifier les données d'entrée.")

        if len(feats_df_test.shape) == 1:
            feats_df_test = np.expand_dims(feats_df_test, axis=0)

        # Vérifier la forme finale des données
        # print("Forme des données d'entrée pour la prédiction:", feats_df_test.shape)
        y_pred = model.predict(feats_df_test)

        proba2csv(y_pred, test["filename"], fichier.stem)


def main(args):
    # Charger le modèle avec le chemin fourni
    model = charger_modele()
    fichiers = obtenir_fichiers(args.repertoire)

    get_predictions(model, fichiers)


if __name__ == "__main__":
    # Définir l'analyseur d'arguments

    parser = argparse.ArgumentParser()

    # parser.add_argument('--modele', type=str, required=True, help="Chemin vers le fichier du modèle Keras à charger.")
    parser.add_argument('--repertoire', type=str, required=True, help="Chemin vers le répertoire contenant les fichiers de vecteurs.")

    args = parser.parse_args()

    main(args)