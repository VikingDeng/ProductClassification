class Registry:
    def __init__(self, name):
        self._name = name
        self._module_dict = {}

    def register(self, name=None):
        def _register(cls):
            key = name if name else cls.__name__
            if key in self._module_dict:
                raise KeyError(f"{key} is already registered")
            self._module_dict[key] = cls
            return cls
        return _register

    def build(self, cfg, **kwargs):
        if cfg is None: return None
        if isinstance(cfg, list): return [self.build(c, **kwargs) for c in cfg]
        if not isinstance(cfg, dict) or 'type' not in cfg:
            raise TypeError("Config must be a dict with type field")

        args = cfg.copy()
        obj_type = args.pop('type')
        cls = self._module_dict.get(obj_type)
        if cls is None:
            raise KeyError(f"{obj_type} not found in {self._name}")
        return cls(**args, **kwargs)

DATASETS = Registry("datasets")
TRANSFORMS = Registry("transforms")
BACKBONES = Registry("backbones")
HEADS = Registry("heads")
RUNNERS = Registry("runners")
OPTIMIZERS = Registry("optimizers")
