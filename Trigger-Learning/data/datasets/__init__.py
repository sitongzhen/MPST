# encoding: utf-8

from .cuhk03 import CUHK03
from .dukemtmcreid import DukeMTMCreID, OCC_DukeMTMCreID
from .market1501 import Market1501
from .msmt17 import MSMT17
from .veri import VeRi
from .dataset_loader import ImageDataset

__factory = {
    'market': Market1501,
    'cuhk03': CUHK03,
    'duke': DukeMTMCreID,
    'msmt17': MSMT17,
    'veri': VeRi,
    'oc_duke': OCC_DukeMTMCreID,
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)
