    def encode_images(self, image_dir=None, recursive: bool = False, num_enc_workers: int = cpu_count()):
        """
        Generate hashes for all images in a given directory of images.

        Args:
            image_dir: Path to the image directory.
            recursive: Optional, find images recursively in a nested image directory structure, set to False by default.
            num_enc_workers: Optional, number of cpu cores to use for multiprocessing encoding generation, set to number of CPUs in the system by default. 0 disables multiprocessing.

        Returns:
            dictionary: A dictionary that contains a mapping of filenames and corresponding 64 character hash string
                        such as {'Image1.jpg': 'hash_string1', 'Image2.jpg': 'hash_string2', ...}

        Example:
        ```
        from imagededup.methods import <hash-method>
        myencoder = <hash-method>()
        mapping = myencoder.encode_images('path/to/directory')
        ```
        """
        if not os.path.isdir(image_dir):
            raise ValueError('Please provide a valid directory path!')

        files = generate_files(image_dir, recursive)

        logger.info(f'Start: Calculating hashes...')

        hashes = parallelise(function=self.encode_image, data=files, verbose=self.verbose, num_workers=num_enc_workers)
        hash_initial_dict = dict(zip(generate_relative_names(image_dir, files), hashes))
        hash_dict = {
            k: v for k, v in hash_initial_dict.items() if v
        }  # To ignore None (returned if some probelm with image file)

        logger.info(f'End: Calculating hashes!')
        return hash_dict