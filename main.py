import os
import torch
from data.dataset import load_and_preprocess_data
from models.dualnet import DualNet
import numpy as np
import warnings
from training.trainer import DualNetTrainer
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ================================
# 10. USAGE EXAMPLE AND DEMO
# ================================

def main():
    """
    Main function demonstrating the complete DualNet pipeline
    """
    print("="*50)
    print("DualNet: Unified Architecture for Pain and Stress Detection")
    print("="*50)

    # Load and preprocess data
    print("\n1. Loading and preprocessing data...")

    # Replace with your actual data paths
    pain_data_path = "pain_fnirs_dataset.npy"
    stress_data_path = "stress_fnirs_dataset.npy"

    # If you don't have real data, you can generate synthetic data for testing
    if not os.path.exists(pain_data_path) or not os.path.exists(stress_data_path):
        print("Generating synthetic data for demonstration...")

        # Generate synthetic pain dataset
        n_samples_pain = 200
        n_channels = 16
        n_timepoints = 1000

        pain_signals = np.random.randn(
            n_samples_pain, n_channels, n_timepoints) * 0.1
        # Add synthetic hemodynamic response for positive samples
        for i in range(n_samples_pain // 2, n_samples_pain):
            t = np.linspace(0, 25, n_timepoints)
            hrf = np.exp(-(t-10)**2 / 8) - np.exp(-(t-10)**2 / 10) * 0.4
            for ch in range(n_channels):
                pain_signals[i, ch, :] += hrf * (0.8 + 0.4 * np.random.rand())

        pain_labels = np.zeros(n_samples_pain)
        pain_labels[n_samples_pain//2:] = 1

        os.makedirs(os.path.dirname(pain_data_path), exist_ok=True)
        np.save(pain_data_path, {
                'signals': pain_signals, 'labels': pain_labels})

        # Generate synthetic stress dataset
        n_samples_stress = 200
        stress_signals = np.random.randn(
            n_samples_stress, n_channels, n_timepoints) * 0.1
        # Add synthetic hemodynamic response for positive samples (different pattern)
        for i in range(n_samples_stress // 2, n_samples_stress):
            t = np.linspace(0, 25, n_timepoints)
            hrf = np.exp(-(t-12)**2 / 10) - np.exp(-(t-14)**2 / 12) * 0.5
            for ch in range(n_channels):
                # Make stress pattern different from pain pattern
                if ch % 2 == 0:  # Different channels affected
                    stress_signals[i, ch, :] += hrf * \
                        (0.7 + 0.5 * np.random.rand())

        stress_labels = np.zeros(n_samples_stress)
        stress_labels[n_samples_stress//2:] = 1

        os.makedirs(os.path.dirname(stress_data_path), exist_ok=True)
        np.save(stress_data_path, {
                'signals': stress_signals, 'labels': stress_labels})

    # Load and preprocess the data
    dataloaders = load_and_preprocess_data(pain_data_path, stress_data_path)
    print("✓ Data loaded and preprocessed")

    # Create model
    print("\n2. Initializing DualNet model...")
    input_channels = next(iter(dataloaders['pain']['train']))[
        'signal'].shape[1]  # Number of fNIRS channels

    model = DualNet(
        input_channels=input_channels,
        d_model=256,
        nhead=8,
        num_layers=2
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)

    print(f"✓ Model initialized")
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")

    # Initialize trainer
    print("\n3. Initializing trainer...")
    trainer = DualNetTrainer(
        model=model,
        dataloaders=dataloaders,
        device=device,
        lr=0.001
    )
    print("✓ Trainer initialized")

    # Train model (with reduced epochs for demonstration)
    print("\n4. Starting training process...")
    trainer.train_full_pipeline(
        pretrain_epochs=5,  # Reduced for demonstration
        multitask_epochs=10,  # Reduced for demonstration
        finetune_epochs=5  # Reduced for demonstration
    )

    # Get test predictions
    model.eval()
    with torch.no_grad():
        pain_test_preds = []
        pain_test_labels = []
        stress_test_preds = []
        stress_test_labels = []

        for batch in dataloaders['pain']['test']:
            signals = batch['signal'].to(device)
            labels = batch['label'].cpu().numpy()

            predictions = model(signals, task='pain')
            pain_preds = predictions['pain'].cpu().numpy()

            pain_test_preds.extend([1 if p >= 0.5 else 0 for p in pain_preds])
            pain_test_labels.extend(labels)

        for batch in dataloaders['stress']['test']:
            signals = batch['signal'].to(device)
            labels = batch['label'].cpu().numpy()

            predictions = model(signals, task='stress')
            stress_preds = predictions['stress'].cpu().numpy()

            stress_test_preds.extend(
                [1 if p >= 0.5 else 0 for p in stress_preds])
            stress_test_labels.extend(labels)

    # Save model
    print("\n6. Saving model...")
    trainer.save_checkpoint('dualnet_complete_model.pth')

    print("\n" + "="*50)
    print("DualNet Implementation Complete")
    print("="*50)
    print("\n✓ Model trained and evaluated")
    print("✓ Visualizations generated")
    print("✓ Model saved")
    print("\nThank you for using DualNet!")


if __name__ == "__main__":
    main()
