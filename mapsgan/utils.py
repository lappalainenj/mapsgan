import torch

def relative_to_abs(rel_traj, start_pos):
    """Given the initial positions, computes the absolute trajectory from displacements.

    Args:
        rel_traj: Tensor of shape (seq_len, batch, 2). Displacements.
        start_pos: Tensor of shape (batch, 2). Initial positions.

    Returns:
        Tensor of shape (seq_len, batch, 2). Absolute trajectories.
    """
    rel_traj = rel_traj.permute(1, 0, 2)
    displacement = torch.cumsum(rel_traj, dim=1)
    start_pos = torch.unsqueeze(start_pos, dim=1)
    abs_traj = displacement + start_pos
    return abs_traj.permute(1, 0, 2)