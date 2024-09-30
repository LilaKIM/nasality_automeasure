from pathlib import Path
import logging

import librosa
import torch
import torchaudio

import numpy as np
import pandas as pd

from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
)

from typing import Tuple

import argparse


# ---------------------------------------------------


def load_xlsr53_model(model_name: str, device: str = "auto") -> Tuple[Wav2Vec2ForCTC, Wav2Vec2Processor]:
    """
    Load the XLSR-53 Large model (i.e. processor and model).

    As the model is not fine-tuned, we need to manually build the processor

    XXX can the function be used to load other models

    Paramters
    ---------
    - device: where the model must be loaded (can either be "cpu" or "cuda")
    """

    import tempfile
    import json
    from transformers import Wav2Vec2FeatureExtractor
    from transformers import Wav2Vec2CTCTokenizer, HubertForCTC

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print (device)
    
    ofile = tempfile.NamedTemporaryFile("wt")
    json.dump({"[UNK]": "0", "[PAD]": "1"}, ofile)
    ofile.flush()

    tokenizer = Wav2Vec2CTCTokenizer(
        ofile.name, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16_000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )
    xlsr_processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )
    # xlsr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53").to(
    xlsr_model = Wav2Vec2ForCTC.from_pretrained(model_name).to(
    # xlsr_model = HubertForCTC.from_pretrained(model_name).to(
        device
    )
    print("Chargement du modèle", model_name)
    return xlsr_model, xlsr_processor


def get_xlsr_representation(
    speech_signal: np.ndarray,
    processor: Wav2Vec2Processor,
    model: Wav2Vec2ForCTC,
    device: str = "auto",
    pooling: str = "mean",
    what_hidden: int = 0
) -> np.ndarray:
    
    """
    Parameters
    ----------
    - device: on which device the audio signal should be loaded (either "cpu" or "cuda")
    - processor, model: the wav2vec model
    - pooling: wav2vec splits each second of the input signal into 49,000 segments and builds
               a representation for each of these segments. The pooling strategy defines how the
               representations of these segments will be aggregated to build a single vector
               representing the whole audio signal. Possible values are: "max" and "mean".
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # try:
    inputs = processor(speech_signal, sampling_rate=16_000, return_tensors="pt")

    # XXX use the same device as model and delete argument
    # XXX check the good practice with pytorch
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # print(what_hidden)
    with torch.no_grad():
        hidden_state = model(**inputs, output_hidden_states=True).hidden_states

        # hidden_state is a list of n+1 tensors: one for the output of the embeddings and one
        # for the output of each layer — here we we want the embeddings. # XXX check that
        # see documentation here: https://huggingface.co/transformers/v4.7.0/_modules/transformers/models/wav2vec2/modeling_wav2vec2.html
        #
        # With XLSR-53 there X layers.
        #
        # print("nb of hidden states :", len(hidden_state))
        if what_hidden==999: 
            what_hidden = -1
        embeddings = hidden_state[what_hidden]
        
        # embeddings is tensor of shape [batch_size, sequence_length, repr_size]
        #
        # batch size is always 1 — we are considering a single audio segment.
        # repr_size is XXX for XLSR-53
        # the encoder outputs representation at 49Hz --> sequence length is equal to 49 * number of seconds in signal

        # print(len(embeddings), embeddings, "_______________________________")
        # print(len(embeddings[0]), embeddings[0])
        # print(len(embeddings[0][0]))

        # XXX add a "none" strategy
        if pooling == "max":
            speech_representation = embeddings[0].max(axis=0).values
        elif pooling == "mean":
            speech_representation = embeddings[0].mean(axis=0)
        elif pooling == "sum":
            speech_representation = embeddings[0].sum(axis=0)
        else:
            raise Exception

        # print(speech_representation)

        
    return speech_representation.cpu().numpy()
    # except: print(len(speech_signal))


def load4xlsr(
    audio_filepath: str, frame_offset: int = None
) -> torch.TensorType:
    """
    Load an audio file and ensure that it can be used in XLS-R/wav2vec2

    To be used by XLS-R, wav files have to:
    - have a samping rate of 16_000Hz
    - use only one channel (mono file)

    Parameters:
    - audio_filepath, str --> path to the file
    
    """
   
    speech_array, sampling_rate = torchaudio.load(audio_filepath, format=audio_filepath.suffix.replace(".", ""))
    
    
    speech = ""
    
    
    speech_array = speech_array.numpy()
    
    
    if sampling_rate == 16000:
        # print ("16000")
        speech = librosa.to_mono(np.squeeze(speech_array))
    else:
        # print (sampling_rate, " passage à 16000")
        speech = librosa.to_mono(np.squeeze(speech_array))
        speech = librosa.resample(speech, orig_sr=sampling_rate, target_sr=16000)
    
    return speech     
    
    
    
def load_corpus(
    audio_dir: str,
):
    """
    Load a corpus from a directory. The directory must contain mp3 or wav files (with “mp3” or “wav” as extension).

    Parameters:
    - audio_dir: the name of the directory containing the files

   
    Returns:
    - a `DataFrame` with the following columns:
        - `filename_path` (Path)
        - `audio` (np.array)
    """
    
    corpus = pd.DataFrame({"filename_path": list(Path(audio_dir).rglob("*.mp3")) + list(Path(audio_dir).rglob("*.wav"))})
    corpus["filename"] = corpus["filename_path"].apply(lambda x: x.name)
    

    logging.info(f"loaded {corpus.shape[0]:,} files")

    logging.info(f"{corpus.shape[0]:,} files after filtering")
    
    # dataframe création de 2 nouvelles colonnes : audio = fenetre extraite et degradée + mono. encoded_audio = chargement de la représentation du fichier (vecteur) 
    # chargement de la fenetre, conversion en 16000hz, mono
    
    
    corpus["audio"] = corpus["filename_path"].apply(load4xlsr)
    
    df = corpus
   
    df = df.dropna()

    return df
    

def verifie_dossier(nom_dossier):
    # Crée un objet Path pour le dossier
    dossier = Path(nom_dossier)
    
    if not dossier.exists():
        dossier.mkdir()
        print(f"Le dossier '{nom_dossier}' a été créé.")
    # else:
        # print(f"Le dossier '{nom_dossier}' existe déjà.")


def create_representation(audio_path, model_name, what_hidden=0, path_results="./data/vectors/"):
    
    path_all_dir = Path(audio_path)
    liste_dossiers = [dossier for dossier in path_all_dir.iterdir() if dossier.is_dir()]

    for path in liste_dossiers:
        print ('Lancement de load_corpus....')
        df = load_corpus(path.resolve())
        # print(df)
        
        print ('Chargement du modèle et création de la représentation des audio....')
        xlsr_model, xlsr_processor = load_xlsr53_model(model_name)
        print("Encodage de l'audio à hidden state", what_hidden)
        df["encoded_audio"] = df["audio"].apply(lambda x: get_xlsr_representation(x, xlsr_processor, xlsr_model, what_hidden=what_hidden))

        
        df = df.drop("filename_path", axis=1)
        df = df.drop("audio", axis=1)
        
        # print(df)
        
        import pickle
        verifie_dossier(path_results)
        pickle.dump(df, open(f"{path_results}{path.name}_{str(what_hidden)}.pkl", "wb"))
        
        print ('Enregistrement du dataframe....')   
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()     

    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--hidden_state", type=int, required=False)

    args = parser.parse_args()             

    audio_path = args.audio_path
    print(audio_path)
    model_name = args.model_name
    what_hidden = args.hidden_state
    if not what_hidden: what_hidden = 0
    # 
    create_representation(audio_path, model_name, what_hidden)