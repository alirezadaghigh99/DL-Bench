import urllib.parse
from functools import partial
from os import path as osp
from typing import Dict, Optional
import requests
import torch
import torchvision.models
import examples.torch.common.models as custom_models
from examples.torch.classification.models.mobilenet_v2_32x32 import MobileNetV2For32x32
from examples.torch.common import restricted_pickle_module
from examples.torch.common.example_logger import logger
from nncf.definitions import CACHE_MODELS_PATH
from nncf.torch.checkpoint_loading import load_state
from nncf.torch.utils import safe_thread_call

def load_model(model, pretrained=True, num_classes=1000, model_params=None, weights_path: str=None) -> torch.nn.Module:
    """Implement a function called `load_model` that loads a machine learning model using PyTorch. The function should accept the model name, an optional boolean to indicate if pretrained weights should be used, the number of classes for the model, additional model parameters, and an optional path to custom weights. Depending on the model name, the function either loads a predefined model from `torchvision.models` or `custom_models`, or raises an exception if the model name is undefined. If `pretrained` is set to `False` and a weights path is provided, it loads the custom weights from the specified path, handling potential URL downloads and ensuring safe unpickling. The function returns the loaded model."""
    logger.info("Loading model: {}".format(model))
    if model_params is None:
        model_params = {}
    if model == "mobilenet_v2_32x32":
        load_model_fn = partial(MobileNetV2For32x32, num_classes=num_classes, **model_params)
    elif model in torchvision.models.__dict__:
        load_model_fn = partial(
            torchvision.models.__dict__[model], num_classes=num_classes, pretrained=pretrained, **model_params
        )
    elif model in custom_models.__dict__:
        load_model_fn = partial(
            custom_models.__dict__[model], num_classes=num_classes, pretrained=pretrained, **model_params
        )
    else:
        raise Exception("Undefined model name")
    loaded_model = safe_thread_call(load_model_fn)
    if not pretrained and weights_path is not None:
        if is_url(weights_path):
            weights_path = download_checkpoint(weights_path)
        sd = torch.load(weights_path, map_location="cpu", pickle_module=restricted_pickle_module)
        if MODEL_STATE_ATTR in sd:
            sd = sd[MODEL_STATE_ATTR]
        load_state(loaded_model, sd, is_resume=False)
    return loaded_model
MODEL_STATE_ATTR = 'state_dict'
COMPRESSION_STATE_ATTR = 'compression_state'

def load_resuming_checkpoint(resuming_checkpoint_path: str):
    if osp.isfile(resuming_checkpoint_path):
        logger.info("=> loading checkpoint '{}'".format(resuming_checkpoint_path))
        checkpoint = torch.load(resuming_checkpoint_path, map_location='cpu', pickle_module=restricted_pickle_module)
        return checkpoint
    raise FileNotFoundError("no checkpoint found at '{}'".format(resuming_checkpoint_path))

def extract_model_and_compression_states(resuming_checkpoint: Optional[Dict]=None):
    if resuming_checkpoint is None:
        return (None, None)
    compression_state = resuming_checkpoint.get(COMPRESSION_STATE_ATTR)
    model_state_dict = resuming_checkpoint.get(MODEL_STATE_ATTR)
    return (model_state_dict, compression_state)

def is_url(uri):
    """
    Checks if given URI is a URL
    :param uri: URI to check
    :return: True if URI is a URL, and False otherwise
    """
    try:
        parsed_url = urllib.parse.urlparse(uri)
        return parsed_url.scheme and parsed_url.netloc
    except:
        return False

def download_checkpoint(url):
    """
    Downloads a checkpoint by URL and returns the path where it was downloaded
    :param url: URL to download a checkpoint from
    :return: path where the checkpoint was downloaded
    """
    if not CACHE_MODELS_PATH.exists():
        CACHE_MODELS_PATH.mkdir(parents=True)
    download_path = CACHE_MODELS_PATH / url.split('/')[-1]
    if not download_path.exists():
        print('Downloading checkpoint ...')
        checkpoint = requests.get(url)
        with open(download_path, 'wb') as f:
            f.write(checkpoint.content)
    return str(download_path)
