import azure.cognitiveservices.speech as speechsdk
import librosa

RATE = 16000


class MicrosoftSTT:

    def __init__(self, key, region):
        self.key = key
        self.region = region

    def speech_recognize_once_from_file(self, audio_path):
        speech_config = speechsdk.SpeechConfig(subscription=self.key, region=self.region)
        audio_config = speechsdk.audio.AudioConfig(filename=audio_path)
        speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

        result = speech_recognizer.recognize_once()

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return result.text
        elif result.reason == speechsdk.ResultReason.NoMatch:
            return "No speech could be recognized"
        elif result.reason == speechsdk.ResultReason.Canceled:
            return f"Speech Recognition canceled: {result.cancellation_details.reason}"
        else:
            return ""

    def recognize(self, audio_path):
        try:
            sound, sample_rate = librosa.load(audio_path, sr=RATE)
            librosa.output.write_wav(audio_path, sound, sample_rate)
            print(f"SR: {sample_rate}")
            return self.speech_recognize_once_from_file(audio_path)
        except KeyboardInterrupt:
            pass
