import torch

class Pointclouds:
    _INTERNAL_TENSORS = [
        "_points_packed",
        "_points_padded",
        "_normals_packed",
        "_normals_padded",
        "_features_packed",
        "_features_padded",
        "_packed_to_cloud_idx",
        "_cloud_to_packed_first_idx",
        "_num_points_per_cloud",
        "_padded_to_packed_idx",
        "valid",
        "equisized",
    ]

    def __init__(self, points, normals=None, features=None) -> None:
        self.device = torch.device("cpu")
        self.equisized = False
        self.valid = None
        self._N = 0
        self._P = 0
        self._C = None
        self._points_list = None
        self._normals_list = None
        self._features_list = None
        self._num_points_per_cloud = None
        self._points_packed = None
        self._normals_packed = None
        self._features_packed = None
        self._packed_to_cloud_idx = None
        self._cloud_to_packed_first_idx = None
        self._points_padded = None
        self._normals_padded = None
        self._features_padded = None
        self._padded_to_packed_idx = None
        # initialization code...

    def padded_to_packed_idx(self):
        if self._padded_to_packed_idx is not None:
            return self._padded_to_packed_idx

        if self._points_padded is None or self._num_points_per_cloud is None:
            raise ValueError("Padded points or number of points per cloud not initialized.")

        num_clouds = len(self._num_points_per_cloud)
        max_points = self._points_padded.shape[1]  # Assuming shape is (N, P, D)

        idx_list = []
        for cloud_idx in range(num_clouds):
            num_points = self._num_points_per_cloud[cloud_idx]
            idx_list.append(torch.arange(num_points, device=self.device))

        self._padded_to_packed_idx = torch.cat(idx_list, dim=0)
        return self._padded_to_packed_idx