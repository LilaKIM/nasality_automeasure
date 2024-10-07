import argparse
import phonetisation as phon
import pandas as pd
from pathlib import Path
from speechbrain.inference.ASR import EncoderASR
import logging

logging.getLogger('speechbrain').setLevel(logging.ERROR)



def get_mean_nasal(loc, col="proba_nasal"):
    # Ouvrir un fichier csv pour calculer la moyenne des proba nasal 
    file = f"./data/csv/{loc}.csv"
    df = pd.read_csv(file)
    
    if col in df.columns:
        moyenne = df[col].mean()
        # print(f"La moyenne des probabilités de nasalité pour le locuteur '{loc}' est : {moyenne}")
        return moyenne
    else:
        return f"Erreur : La colonne '{col}' n'existe pas dans le fichier."
    

def get_nb_nasal_oral(phonemes):
    # Compter le nombre des nasales et orales
    nasal_info, oral_info = 0, 0
    for i in range(len(phonemes)):
        articulation = phon.phoneme_articulation.get(phonemes[i], 'Inconnu')
        if "nasal" in articulation:
            nasal_info += 1
        elif articulation != "Inconnu":
            oral_info += 1
    
    return (nasal_info, oral_info)


def transcription_phonemes(audio_path, asr_model):
    # Transcriptions orthographique et phonétique pour obtenir le nombre des nasales et orales
    text_to_phonemize = phon.transcribe_audio_speechbrain(audio_path, asr_model)
    print(f"Transcription orthographique : {text_to_phonemize}")

    # Transcription en API
    api_transcription = phon.transcribe_to_api(text_to_phonemize)
    print(f"Transcription en API : {api_transcription}")

    phonemes = phon.split_phonemes(api_transcription)

    nb_phonemes = get_nb_nasal_oral(phonemes)
    print(f"Nombre de nasales présentes dans l'audio : {nb_phonemes[0]}")
    print(f"Nombre d'orales présentes dans l'audio : {nb_phonemes[1]}")

    return nb_phonemes



    
def load_corpus(audio_dir: str, asr_model):
    corpus = pd.DataFrame({"filename_path": list(Path(audio_dir).rglob("*.mp3")) + list(Path(audio_dir).rglob("*.wav"))})
    corpus["filename"] = corpus["filename_path"].apply(lambda x: x.name)
    corpus["locuteur"] = corpus["filename"].apply(lambda x: Path(x).stem)

    corpus["filename_path"].apply(lambda x: print(x))
    corpus[["nb_nasal", "nb_oral"]] = corpus["filename_path"].apply(lambda x: pd.Series(transcription_phonemes(x, asr_model)))
    
    df = corpus
    df = df.dropna()
    return df
    

def charger_modele(chemin_modele):
    # charger le modèle Keras
    model = EncoderASR.from_hparams(source=f"speechbrain/{chemin_modele}", savedir=f"pretrained_models/{chemin_modele}")
    return model

def main(args):
    # Pondérer la moyenne et le nombre des nasales
    asr_model = charger_modele(args.modele)
    path_all_dir = args.audio_path
    df = load_corpus(path_all_dir, asr_model)
    # print(df)

    df["mean"] = df["locuteur"].apply(lambda x: get_mean_nasal(x, "proba_nasal"))
    df["mean_ponderee"] = df.apply(lambda row: row["mean"]/row["nb_nasal"], axis=1)
    
    # print(df)

    df = df.drop("filename_path", axis=1)
    df = df.drop("filename", axis=1)
    df = df.drop("mean", axis=1)

    df.to_csv("./nasalite_ponderee", sep=',', encoding='utf-8', index=False, header=True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()     

    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument('--modele', type=str, required=True, help="Chemin vers le fichier du modèle Keras à charger.")

    args = parser.parse_args()             

    main(args)