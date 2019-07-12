from sonosco.models import DeepSpeech2
from sonosco.decoders import GreedyDecoder
from sonosco.datasets.processor import AudioDataProcessor


# TODO: lazy loading
def load_models(config):
    models = {}
    for model_config in config:
        if model_config.get('script'):
            pass  # TODO add possibility to add custom load script, not necessaryly in this way
        else:
            model_dict = {}
            models[model_config['id']] = model_dict
            model = DeepSpeech2.load_model(model_config['path'])
            model_dict['model'] = model
            model_dict['decoder'] = GreedyDecoder(model.labels, blank_index=model.labels.index('_'))
            model_dict['processor'] = AudioDataProcessor(**model.audio_conf, normalize=True)
    return models
