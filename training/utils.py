# ================================
# 8. TRAINING UTILITIES
# ================================
import random
import torch
import torch.nn.functional as F


def create_mixed_batch(pain_loader, stress_loader):
    """
    Create mixed batches with both pain and stress samples
    """
    pain_iter = iter(pain_loader)
    stress_iter = iter(stress_loader)

    while True:
        try:
            # Get pain samples
            try:
                pain_batch = next(pain_iter)
            except StopIteration:
                pain_iter = iter(pain_loader)
                pain_batch = next(pain_iter)

            # Get stress samples
            try:
                stress_batch = next(stress_iter)
            except StopIteration:
                stress_iter = iter(stress_loader)
                stress_batch = next(stress_iter)

            pain_size = len(pain_batch['signal'])
            stress_size = len(stress_batch['signal'])

            # Create mixed batch
            mixed_signals = torch.cat(
                [pain_batch['signal'], stress_batch['signal']], dim=0)
            mixed_labels = torch.cat(
                [pain_batch['label'], stress_batch['label']], dim=0)
            mixed_tasks = [pain_batch['task']] * pain_size + \
                [stress_batch['task']] * stress_size

            yield {
                'signal': mixed_signals,
                'label': mixed_labels,
                'task': mixed_tasks
            }

        except StopIteration:
            break


def data_augmentation(x, noise_level=0.01, temporal_warping=True, scaling=True):
    """
    Apply data augmentation techniques to fNIRS signals

    Args:
        x: fNIRS signal (batch, channels, time)
        noise_level: Level of Gaussian noise to add
        temporal_warping: Whether to apply temporal warping
        scaling: Whether to apply amplitude scaling

    Returns:
        Augmented signal
    """
    batch_size, channels, time_steps = x.shape

    # Add noise
    if noise_level > 0:
        noise = torch.randn_like(x) * noise_level
        x = x + noise

    # Temporal warping
    if temporal_warping and random.random() > 0.5:
        warp_factor = random.uniform(0.8, 1.2)
        new_length = int(time_steps * warp_factor)
        x_warped = F.interpolate(
            x, size=new_length, mode='linear', align_corners=False)
        if new_length > time_steps:
            x = x_warped[:, :, :time_steps]
        else:
            padding = time_steps - new_length
            x = F.pad(x_warped, (0, padding), mode='replicate')

    # Amplitude scaling
    if scaling and random.random() > 0.5:
        scaling_factor = random.uniform(0.8, 1.2)
        x = x * scaling_factor

    return x
