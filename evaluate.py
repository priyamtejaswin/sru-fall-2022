import speech_recognition as sr
from transformers import pipeline
import os 
from nltk.translate.bleu_score import sentence_bleu

r = sr.Recognizer()

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")


from pydub import AudioSegment

def evaluate(test_audio_dir, test_result_dir):
    '''
        @param test_audio_dir : directory to test .wav files
        @param test_result_dir: directory to resulting translation scripts
    '''
    score = 0
    for fname in os.listdir(test_audio_dir):
        audio_path = os.path.join(test_audio_dir,fname)
        if '.wav' in fname:
            without_ext = fname.split('.wav')[0]
            audio = sr.AudioFile(audio_path)
        elif '.mp4' in fname:
            without_ext = fname.split('.mp4')[0]
            # convert mp4 to wav
            sound = AudioSegment.from_file(audio_path,format="mp4")
            sound.export(os.path.join(test_audio_dir,without_ext), format="wav")
            audio = sr.AudioFile(os.path.join(test_audio_dir,without_ext))
            
        ground_truth_file = os.path.join(test_result_dir,without_ext+'.txt')
        ground_truth = open(ground_truth_file,'r').readlines()[0]

        with audio as source:
            audio = r.record(source)
        phrase = r.recognize_google(audio)

        result = translator(phrase)[0]['translation_text']

        score += sentence_bleu(ground_truth.split(),result.split())

    return score/len(os.listdir(test_audio_dir))

