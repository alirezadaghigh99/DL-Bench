import numpy as np

class Pointclouds:
    def __init__(self, points_list, normals_list=None, features_list=None):
        self.points_list = points_list
        self.normals_list = normals_list
        self.features_list = features_list
        self._points_padded = None
        self._normals_padded = None
        self._features_padded = None

    def _compute_padded(self, refresh: bool = False):
        if self._points_padded is not None and not refresh:
            return

        # Determine the maximum number of points in any point cloud
        max_num_points = max(len(points) for points in self.points_list)

        # Initialize padded arrays
        self._points_padded = np.zeros((len(self.points_list), max_num_points, 3))
        if self.normals_list is not None:
            self._normals_padded = np.zeros((len(self.normals_list), max_num_points, 3))
        if self.features_list is not None:
            feature_dim = len(self.features_list[0][0]) if self.features_list[0] else 0
            self._features_padded = np.zeros((len(self.features_list), max_num_points, feature_dim))

        # Fill the padded arrays
        for i, points in enumerate(self.points_list):
            num_points = len(points)
            self._points_padded[i, :num_points, :] = points

        if self.normals_list is not None:
            for i, normals in enumerate(self.normals_list):
                num_normals = len(normals)
                self._normals_padded[i, :num_normals, :] = normals

        if self.features_list is not None:
            for i, features in enumerate(self.features_list):
                num_features = len(features)
                self._features_padded[i, :num_features, :] = features

