def get_model(config):
    """ return given network
    """

    if config == 'ANN':
        from .ANN import ANN
        net = ANN()
    elif config == 'Multi_View_Net':
        from .Multi_View_Net import Multi_View_NN
        net = Multi_View_NN(768,7)

    else:
        raise NotImplementedError("the network name '{}' is not supported yet".format(config))

    return net