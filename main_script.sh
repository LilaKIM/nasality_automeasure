#!/bin/sh

path=""
extrait=""
language="fr"
dossier_vector="./data/vectors"
model_name="LeBenchmark/wav2vec2-FR-3K-large"

for arg in "$@"; do
    if [[ "$arg" == --path=* ]]; then
        path="${arg#--path=}"
    elif [[ "$arg" == --cut=* ]]; then
        extrait="${arg#--cut=}"
    fi
done

echo "$path" "$extrait"

# ###########################################################
# find ./data/audio_4s/ -type f -delete
# find ./data/audio_decoupe/ -type f,d -delete
# find ./data/csv/ -type f -delete
# find ./data/vectors/ -type f -delete
# find ./data/audio_4s_temp/ -type f -delete
# ###########################################################


# # # if [[ -n "$path" && "$extrait" == "oui" ]]; then
path="./data/audio_original"
python "./script/decoupe_audio_4s_vad.py" --input_dir "$path"
praat --run "./script/decoupe_audio.praat" "../data/audio_4s/"
# path="./data/audio_4s_temp"

# elif [[ -n "$path" && -z "$extrait" ]]; then
#     praat --run "./script/decoupe_audio.praat" $path

# else
#     path="../data/audio_original/"
#     praat --run "./script/decoupe_audio.praat" $path
#     path="./data/audio_original/"
# fi

# python ./script/get_representation_w2v.py --audio_path "./data/audio_decoupe" --model_name $model_name --hidden_state 4
# python ./script/prediction.py --repertoire $dossier_vector --num_file 20
# python ./script/ponderation_nasal.py --audio_path $path --model "asr-wav2vec2-commonvoice-fr"


