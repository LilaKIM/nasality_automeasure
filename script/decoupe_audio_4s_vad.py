import os
import torch
import torchaudio
import argparse


def create_output_dir(output_dir):
    """
    Crée un répertoire de sortie s'il n'existe pas.
    """
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
        return_seconds=True,  # Retourne les timestamps en secondes
    )

    return speech_timestamps


def concatenate_wav_files(file_path):
    """
    Concatène les segments détectés par VAD en une seule forme d'onde.
    """
    concatenated_waveform = None

    # Charger l'audio
    waveform, sample_rate = torchaudio.load(file_path)
    speech_timestamps = vad(file_path)

    # Concaténer les segments détectés par VAD
    for idx, speech in enumerate(speech_timestamps):
        start_time, end_time = speech['start'], speech['end']
        start_sample = int(start_time * sample_rate)
        end_sample = int((end_time) * sample_rate)
        segment = waveform[:, start_sample:end_sample]

        if idx == 0:
            concatenated_waveform = segment
        else:
            concatenated_waveform = torch.cat((concatenated_waveform, segment), dim=1)

    return concatenated_waveform, sample_rate


def split_wav(file_path, filename, output_dir, segment_length=4):
    """
    Découpe un fichier WAV en séquences de longueur spécifiée (par défaut, 4 secondes).
    """
    # Concaténer les segments détectés par VAD
    waveform, sample_rate = concatenate_wav_files(file_path)

    # Calculer le nombre d'échantillons pour un segment
    samples_per_segment = int(segment_length * sample_rate)
    total_samples = waveform.size(1)
    print(f"Total Samples: {total_samples}, Samples per Segment: {samples_per_segment}")

    segment_index = 0

    # Découper en segments de 4 secondes
    for start in range(0, total_samples, samples_per_segment):
        end = min(start + samples_per_segment, total_samples)
        segment = waveform[:, start:end]

        # Sauvegarder chaque segment dans un fichier distinct
        segment_filename = os.path.join(output_dir, f"{filename}_{segment_index}.wav")
        print(f"Segment sauvegardé : {segment_filename}")
        torchaudio.save(segment_filename, segment, sample_rate)

        segment_index += 1

    return torchaudio.save("concatenate_wav_files.wav", waveform, sample_rate)   


def main(arg, output_dir="./data/audio_4s"):
    """
    Lit tous les fichiers WAV d'un répertoire et applique la fonction split_wav à chacun.
    """
    create_output_dir(output_dir)

    # Parcourir tous les fichiers WAV du répertoire
    for filename in os.listdir(arg.input_dir):
        if filename.endswith(".wav"):
            file_path = os.path.join(arg.input_dir, filename)
            print(f"Traitement du fichier : {file_path}")
            filename = os.path.splitext(filename)[0]

            # Utiliser la longueur par défaut de 4 secondes si elle n'est pas spécifiée
            segment_length = arg.length if arg.length is not None else 4
            split_wav(file_path, filename, output_dir, segment_length)


if __name__ == "__main__":
    # Parser les arguments depuis la ligne de commande
    parser = argparse.ArgumentParser(description="Découpe les fichiers WAV en segments de 4 secondes.")

    parser.add_argument("--input_dir", type=str, required=True, help="Répertoire contenant les fichiers WAV.")
    parser.add_argument("--length", type=int, required=False, help="Longueur de chaque segment en secondes.")
    parser.add_argument("--output_dir", type=str, required=False)

    args = parser.parse_args()

    # Appeler la fonction principale
    main(args)
