import torch
import torchaudio
import subprocess

# from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor



# def transcribe_audio_fr(file_path):
#     # Charger le processeur et le modèle Wav2Vec2 pour le français
#     processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53-french")
#     model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53-french")

#     # Charger le fichier audio .wav et convertir en tenseur
#     waveform, sample_rate = torchaudio.load(file_path)

#     # Si l'échantillonnage est différent de celui attendu par le modèle (16000Hz), on le redimensionne
#     if sample_rate != 16000:
#         waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

#     # Préparer les données pour le modèle
#     inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)

#     # Effectuer la prédiction sans rétropropagation
#     with torch.no_grad():
#         logits = model(inputs.input_values).logits

#     # Décoder la prédiction en texte
#     predicted_ids = torch.argmax(logits, dim=-1)
#     transcription = processor.batch_decode(predicted_ids)[0]

#     return transcription


def transcribe_audio_speechbrain(file_path, model):
    file_path = str(file_path)
    transcription = model.transcribe_file(file_path)
    return transcription



def clean_transcription(phonetic_transcription):
    # Supprimer les symboles indésirables comme les accents ˈ et les tirets -
    cleaned_transcription = phonetic_transcription.replace('ˈ', '').replace('-', '')
    return cleaned_transcription

def transcribe_to_api(text):
    # transcription en API
    command = ['espeak', '-v', 'fr', '--ipa', '-q', text]
    result = subprocess.run(command, stdout=subprocess.PIPE, text=True)
    cleaned_transcription = clean_transcription(result.stdout.strip())
    return cleaned_transcription

def transcribe_to_sampa(text):
    # transcription en SAMPA
    command = ['espeak', '-v', 'fr', '-x', '-q', text]
    result = subprocess.run(command, stdout=subprocess.PIPE, text=True)
    return result.stdout.strip()

def split_phonemes(transcription):
    # Parcourir la transcription et extraire les phonèmes individuels
    phonemes = []
    i = 0
    while i < len(transcription):
        # Prendre deux caractères pour vérifier les phonèmes complexes
        if transcription[i:i+2] in phonemes_list:
            phonemes.append(transcription[i:i+2])
            i += 2
        else:
        # Sinon, prendre un seul caractère comme phonème
            phonemes.append(transcription[i])
            i += 1
    return phonemes

def get_articulation(phonemes):
    articulation_info = {}
    print(phonemes)
    for i in range(len(phonemes)):
        articulation = phoneme_articulation.get(phonemes[i], 'Inconnu')
        if articulation != "Inconnu":
            articulation_info[i] = (phonemes[i], articulation)
    print(articulation_info)
    exit()
    return articulation_info



##############################


phonemes_list = ['ɑ̃', 'ɔ̃', 'ɛ̃', 'œ̃', 'ʒ', 'ʁ', 'ɲ', 'ŋ', 'ʃ', 'e', 'a', 'i', 'o', 'u', 'y', 'ø', 'ə']

phoneme_articulation = {
    'p': 'occlusif bilabial sourd',
    'b': 'occlusif bilabial sonore',
    't': 'occlusif dental sourd',
    'd': 'occlusif dental sonore',
    'k': 'occlusif vélaire sourd',
    'g': 'occlusif vélaire sonore',
    'f': 'fricatif labiodental sourd',
    'v': 'fricatif labiodental sonore',
    's': 'fricatif alvéolaire sourd',
    'z': 'fricatif alvéolaire sonore',
    'ʃ': 'fricatif postalvéolaire sourd',
    'ʒ': 'fricatif postalvéolaire sonore',
    'm': 'nasal bilabial',
    'n': 'nasal alvéolaire',
    'ɲ': 'nasal palatal',
    'ŋ': 'nasal vélaire',
    'ʁ': 'fricatif uvulaire sonore',
    'l': 'latéral alvéolaire',
    'j': 'approximant palatal',
    'w': 'approximant vélaire',
    'ɑ̃': 'voyelle nasale ouverte postérieure',
    'ɛ̃': 'voyelle nasale moyenne ouverte antérieure',
    'ɔ̃': 'voyelle nasale moyenne fermée postérieure',
    'œ̃': 'voyelle nasale arrondie mi-ouverte',
    'a': 'voyelle ouverte antérieure non arrondie',
    'e': 'voyelle mi-fermée antérieure non arrondie',
    'i': 'voyelle fermée antérieure non arrondie',
    'o': 'voyelle mi-fermée postérieure arrondie',
    'u': 'voyelle fermée postérieure arrondie',
    'y': 'voyelle fermée antérieure arrondie',
    'ø': 'voyelle mi-fermée antérieure arrondie',
    'ə': 'schwa (voyelle centrale mi-fermée)'
}

##############################

    # # Transcription en SAMPA
    # sampa_transcription = transcribe_to_sampa(text_to_phonemize)
    # print(f"Transcription en SAMPA : {sampa_transcription}")

    # Obtenir le mode d'articulation des phonèmes
    # articulation = get_articulation(phonemes)
    # print("\nMode d'articulation des phonèmes :")
    # for info in articulation:
    #     print(articulation[info][0], ":", articulation[info][1])