def build_denoiser(cfg):
    if cfg.ldm.get("factory"):
        from importlib import import_module
        module, name = cfg.ldm.factory.split(":")
        return getattr(import_module(module), name)(cfg)
    from .mcspnet import MCSPNet
    return MCSPNet(cfg)
