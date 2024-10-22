import os
import torch
import argparse
import torchaudio
torch.set_num_threads(1)

def create_output_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def vad(path):
    """
    Utilise le modèle Silero pour détecter l'activation de la voix dans le fichier audio.
    Retourne les timestamps des zones de parole en secondes.
    """
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
    (get_speech_timestamps, _, read_audio, _, _) = utils

    wav = read_audio(path)
    speech_timestamps = get_speech_timestamps(
        wav,
        model,
        return_seconds=True,  # Return speech timestamps in seconds (default is samples)
    )

    return speech_timestamps

def split_wav(file_path, filename, output_dir, segment_length):
    """
    Coupe les segments audio détectés par la fonction VAD (zones de parole).
    Ne coupe que les zones de parole qui sont plus longues que segment_length (en secondes).
    """

    waveform, sample_rate = torchaudio.load(file_path)
    speech_timestamps = vad(file_path)

    segment_index = 0

    for speech in speech_timestamps:
        start_time, end_time = speech['start'], speech['end']
        # print(start_time, end_time, segment_length)
        
        if end_time - start_time >= segment_length:
            end_time = start_time + 4
            start_sample = int(start_time * sample_rate)
            end_sample = int((end_time) * sample_rate)

            segment = waveform[:, start_sample:end_sample]

            segment_filename = os.path.join(output_dir, f"{filename}-{segment_index}_{start_time}_{end_time}.wav")
            torchaudio.save(segment_filename, segment, sample_rate)

            segment_index += 1


def main(arg, segment_length=4, output_dir="./data/audio_4s"):
    """
    Lit tous les fichiers WAV d'un répertoire et applique la fonction split_wav à chacun.
    """

    create_output_dir(output_dir)

    for filename in os.listdir(arg.input_dir):
        if filename.endswith(".wav"):
            file_path = os.path.join(arg.input_dir, filename)
            print(f"Traitement du fichier : {file_path}")
            filename = os.path.splitext(filename)[0]
            # print(filename)

            split_wav(file_path, filename, output_dir, segment_length)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()     

    parser.add_argument("--input_dir", type=str, required=True)
    # parser.add_argument('--length', type=int, required=False)

    args = parser.parse_args()             

    main(args)