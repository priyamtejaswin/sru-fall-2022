import speech_recognition as sr
from transformers import pipeline
import os 
import sacrebleu 
from jiwer import wer

r = sr.Recognizer()

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-de")


from pydub import AudioSegment

def evaluate(test_audio_dir, test_tsv_file, wav_out):
    '''
        @param test_audio_dir : directory to test .wav files
        @param test_result_dir: file that has path	sentence	translation	client_id for all test clips
        @param wav_out: directory to output wav files if they're exported
    '''
    
    test_data = open(test_tsv_file,'r').readlines()[1:]
    
    transcription_score = 0
    translation_score = 0
    for data in test_data:
        splits = data.split("\t")
        clip_name, transcription, translation = splits[0],splits[1],splits[2]
        clip_path = os.path.join(test_audio_dir,clip_name)
        print(clip_name)
        if not os.path.exists(clip_path):
            print(clip_path, ' does not exist')
        #print(clip_name, transcription, translation)
        #print(clip_path)
        if '.wav' in clip_name:
            without_ext = clip_name.split('.wav')[0]
            audio = sr.AudioFile(clip_path)
        elif '.mp4' in clip_name:
            without_ext = clip_name.split('.mp4')[0]
            # convert mp4 to wav
            sound = AudioSegment.from_file(clip_path,format="mp4")
            sound.export(os.path.join(wav_out,without_ext+".wav"), format="wav")
            audio = sr.AudioFile(os.path.join(wav_out,without_ext+".wav"))
        elif '.mp3' in clip_name:
            #print('here')
            #print(clip_name)
            #print(clip_path)
            without_ext = clip_name.split('.mp3')[0]
            # convert mp4 to wav
            sound = AudioSegment.from_file(clip_path,format="mp3")
            sound.export(os.path.join(wav_out,without_ext+".wav"), format="wav")
            audio = sr.AudioFile(os.path.join(wav_out,without_ext+".wav"))

        with audio as source:
            audio = r.record(source)
        try:
            transcription_hypothesis = r.recognize_google(audio).lower()
            translation_hypothesis = (translator(transcription_hypothesis)[0]['translation_text']).lower()
         
            transcription_score += wer(transcription, transcription_hypothesis)
            translation_score += sacrebleu.sentence_bleu(translation_hypothesis, [translation],  smooth_method='exp').score
        except:
            transcription_score += 1
            translation_score += 0
    return transcription_score/len(test_data), translation_score/len(test_data)
