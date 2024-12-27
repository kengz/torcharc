from torcharc.module import get, fn, fork, merge
from torcharc.validator.spec import build, Spec  # noqa
import torch


def register_nn(cls: type):
    """Register a module class in torch.nn"""
    # first check for conflict
    if hasattr(torch.nn, cls.__name__):
        raise ValueError(f"Module {cls.__name__} already exists in torch.nn")
    setattr(torch.nn, cls.__name__, cls)


# iterate over the classes in modules and register them in torch.nn
for module in [get, fn, fork, merge]:
    for cls in module.__dict__.values():
        if isinstance(cls, type):
            register_nn(cls)
