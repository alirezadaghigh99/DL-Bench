    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img