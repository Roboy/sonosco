from .microsoft import MicrosoftSTT
from sonosco.common.path_utils import parse_yaml


MICROSOFT_SECRET = "external/secret.yaml"


def create_external_model(model_id: str, sample_rate):
    if model_id == "microsoft":
        config = parse_yaml(MICROSOFT_SECRET)
        return MicrosoftSTT(config["key"], config["region"], sample_rate)
    else:
        raise NotImplemented(f"Model {model_id} not implemented.")
