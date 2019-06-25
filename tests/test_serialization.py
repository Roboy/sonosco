from sonosco.models.deepspeech2_sonosco import DeepSpeech2
import inspect

d = DeepSpeech2()
# print(inspect.getsourcelines(d.__serialize__))
print(d.__serialize__())
# print("XD")