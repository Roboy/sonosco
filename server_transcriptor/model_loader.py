from sonosco.models import DeepSpeech2
from sonosco.decoders import GreedyDecoder, BeamCTCDecoder
from sonosco.datasets.processor import AudioDataProcessor


# TODO: lazy loading
def load_models(config):
    models = {}

    for model_config in config:
        if model_config.get('external'):
            pass
        else:
            model_dict = dict()
            models[model_config["id"]] = model_dict
            model = DeepSpeech2.load_model(model_config["path"])
            model.eval()
            model_dict["model"] = model

            if model_config["decoder"] == "greedy":
                model_dict["decoder"] = GreedyDecoder(model.labels, blank_index=model.labels.index('_'))
            elif model_config["decoder"] == "beam":
                if model_config.get("language"):
                    lm_path = model_config["language"]
                else:
                    lm_path = None

                model_dict["decoder"] = BeamCTCDecoder(model.labels, lm_path=lm_path, blank_index=model.labels.index('_'))
            else:
                raise NotImplemented(f"Decoder {model_config['decoder']} not implemented.")

            model_dict["processor"] = AudioDataProcessor(**model.audio_conf, normalize=True)

    return models
