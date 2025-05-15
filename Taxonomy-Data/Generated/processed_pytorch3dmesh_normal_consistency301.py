import torch
from pytorch3d.structures import Meshes

def mesh_normal_consistency(meshes: Meshes) -> float:
    if len(meshes) == 0:
        return 0.0

    total_consistency = 0.0
    valid_mesh_count = 0

    for mesh in meshes:
        verts = mesh.verts_packed()
        faces = mesh.faces_packed()

        if faces.shape[0] == 0:
            continue

        # Compute face normals
        face_normals = torch.cross(
            verts[faces[:, 1]] - verts[faces[:, 0]],
            verts[faces[:, 2]] - verts[faces[:, 0]],
            dim=1
        )
        face_normals = face_normals / face_normals.norm(dim=1, keepdim=True)

        # Find neighboring faces
        edges = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], dim=0)
        edges, edge_indices = torch.sort(edges, dim=1)
        unique_edges, inverse_indices = torch.unique(edges, return_inverse=True, dim=0)

        # Group faces by shared edges
        face_groups = [[] for _ in range(unique_edges.shape[0])]
        for i, idx in enumerate(inverse_indices):
            face_groups[idx].append(i // 3)

        # Compute normal consistency
        consistency_sum = 0.0
        count = 0

        for group in face_groups:
            if len(group) < 2:
                continue
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    normal_i = face_normals[group[i]]
                    normal_j = face_normals[group[j]]
                    angle_cos = torch.clamp(torch.dot(normal_i, normal_j), -1.0, 1.0)
                    angle = torch.acos(angle_cos)
                    consistency_sum += angle
                    count += 1

        if count > 0:
            total_consistency += consistency_sum / count
            valid_mesh_count += 1

    if valid_mesh_count == 0:
        return 0.0

    return total_consistency / valid_mesh_count