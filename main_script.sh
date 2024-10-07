#!/bin/sh

path=""
nb_extrait=""
language="fr"
dossier_vector="./data/vectors"
model_name="LeBenchmark/wav2vec2-FR-3K-large"

for arg in "$@"; do
    if [[ "$arg" == --path=* ]]; then
        path="${arg#--path=}"
    elif [[ "$arg" == --nb=* ]]; then
        nb_extrait="${arg#--nb=}"
    fi
done

# echo "$path" "$nb_extrait"

###########################################################
find ./data/audio_4s/ -type f -delete
find ./data/audio_decoupe/ -type f,d -delete
find ./data/csv/ -type f -delete
find ./data/vectors/ -type f -delete
###########################################################


if [[ -n "$path" && -n "$nb_extrait" ]]; then
    praat --run "./script/decoupe_audio_4s.praat" "$path" $nb_extrait
    praat --run "./script/decoupe_audio.praat" "../data/audio_4s/"
    path="./data/audio_4s"

elif [[ -n "$path" && -z "$nb_extrait" ]]; then
    praat --run "./script/decoupe_audio.praat" $path

elif [[ -n "$nb_extrait" && -z "$path" ]]; then
    path="./data/audio_original/"
    praat --run "./script/decoupe_audio_4s.praat" "$path" $nb_extrait
    praat --run "./script/decoupe_audio.praat" "../data/audio_4s/"

else
    path="./data/audio_original/"
    praat --run "./script/decoupe_audio.praat" $path
fi

python ./script/get_representation_w2v.py --audio_path "./data/audio_decoupe" --model_name $model_name --hidden_state 4
python ./script/prediction.py --repertoire $dossier_vector
python ./script/ponderation_nasal.py --audio_path $path --model "asr-wav2vec2-commonvoice-fr"
