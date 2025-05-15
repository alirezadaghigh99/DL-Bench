    def forward(self, image):
        return F.convert_image_dtype(image, self.dtype)