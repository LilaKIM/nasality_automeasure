#!/bin/sh

script_praat="./script/decoupe_audio.praat"
dossier_vector="./data/vectors"

language="fr"
mode="lebenchmark"
model_name="LeBenchmark/wav2vec2-FR-3K-large"
# detecteur_nasalite="./model/fr_6c6v_lebenchmark_3corpus_allphon_4_010924_sigmoid_mean_0.000125.keras"

praat --run "$script_praat" "../data/audio_original/"
python ./script/get_representation_w2v.py --audio_path "./data/audio_decoupe" --model_name $model_name --hidden_state 4
python ./script/prediction.py --repertoire $dossier_vector
python ./script/ponderation_nasal.py --audio_path "./data/audio_original" --model "asr-wav2vec2-commonvoice-fr"