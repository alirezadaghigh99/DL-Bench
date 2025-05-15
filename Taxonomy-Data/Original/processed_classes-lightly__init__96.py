    def __init__(self, transform: torchvision.transforms.Compose):
        _deprecation_warning_collate_functions()
        super(BaseCollateFunction, self).__init__()
        self.transform = transform