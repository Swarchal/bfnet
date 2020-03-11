from torch import nn


def set_dropout_(module):
    """
    Function to convert Dropout modules to training mode.
    This is useful as setting the entire model to training mode
    will also change BatchNorm layers etc.
    useage: (in place)
        >>> model.apply(set_dropout_)
    """
    Dropout = (nn.Dropout, nn.Dropout2d, nn.Dropout3d)
    if isinstance(module, Dropout):
        module.training = True
        module.train()


def unset_dropout_(module):
    """
    Function to convert Dropout modules to eval mode.
    This is useful as setting the entire model to eval mode
    will also change BatchNorm layers etc.
    useage: (in place)
        >>> model.apply(unset_dropout_)
    """
    Dropout = (nn.Dropout, nn.Dropout2d, nn.Dropout3d)
    if isinstance(module, Dropout):
        module.training = False
        module.train(False)


def init_weights_(module):
    """
    initialise weights according to FNet paper
    useage (in place)
        >>> model.apply(init_weights_)
    """
    if module.__class__.__name__.startswith("Conv"):
        module.weight.data.normal_(0.0, 0.02)
    elif module.__class__.__name__.find("BatchNorm") != -1:
        module.weight.data.normal_(1.0, 0.02)
        module.bias.data.fill_(0)

