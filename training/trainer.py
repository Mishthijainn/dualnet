# ================================
# 9. MAIN TRAINING CLASS
# ================================
from sklearn.base import accuracy_score
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
import torch
import torch.optim as optim
from training.loss import ContrastiveLoss, DualNetLoss
from training.utils import create_mixed_batch, data_augmentation


class DualNetTrainer:
    """
    Main trainer class for DualNet
    """

    def __init__(self, model, dataloaders, device='cpu', lr=0.001):
        """
        Initialize the trainer

        Args:
            model: DualNet model
            dataloaders: Dict containing data loaders for pain and stress
            device: Device to use for training
            lr: Learning rate
        """
        self.model = model.to(device)
        self.dataloaders = dataloaders
        self.device = device

        # Optimizers
        self.optimizer = optim.Adam(
            model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )

        # Loss functions
        self.contrastive_loss = ContrastiveLoss()
        self.dualnet_loss = DualNetLoss()

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'pain_metrics': {
                'val': {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []},
                'test': {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []}
            },
            'stress_metrics': {
                'val': {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []},
                'test': {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []}
            }
        }

    def train_phase1_pretrain(self, epochs=50):
        """
        Phase 1: Pre-training with contrastive learning
        """
        print("\n" + "="*50)
        print("PHASE 1: Self-Supervised Pre-training")
        print("="*50)

        # Freeze classification branches
        for param in self.model.pain_classifier.parameters():
            param.requires_grad = False
        for param in self.model.stress_classifier.parameters():
            param.requires_grad = False

        self.model.train()

        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0

            # Create mixed batches
            mixed_batches = create_mixed_batch(
                self.dataloaders['pain']['train'],
                self.dataloaders['stress']['train']
            )

            for batch_idx, batch in enumerate(mixed_batches):
                if batch_idx >= min(len(self.dataloaders['pain']['train']),
                                    len(self.dataloaders['stress']['train'])):
                    break

                signals = batch['signal'].to(self.device)

                # Create augmented views
                aug1 = data_augmentation(signals)
                aug2 = data_augmentation(signals)

                # Extract features
                features1 = self.model.shared_extractor(aug1)
                features2 = self.model.shared_extractor(aug2)

                # Global average pooling to get embeddings
                emb1 = torch.mean(features1, dim=2)  # (batch, channels)
                emb2 = torch.mean(features2, dim=2)  # (batch, channels)

                # Compute contrastive loss
                loss = self.contrastive_loss(emb1, emb2)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                # Print progress
                if batch_idx % 10 == 0:
                    print(
                        f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{min(len(self.dataloaders['pain']['train']), len(self.dataloaders['stress']['train']))}, Loss: {loss.item():.4f}")

            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            print(
                f"Epoch {epoch+1}/{epochs}, Contrastive Loss: {avg_loss:.4f}")

        # Unfreeze classification branches
        for param in self.model.pain_classifier.parameters():
            param.requires_grad = True
        for param in self.model.stress_classifier.parameters():
            param.requires_grad = True

        print("✓ Phase 1 completed: Shared features pre-trained")

    def train_phase2_multitask(self, epochs=100):
        """
        Phase 2: Multi-task learning
        """
        print("\n" + "="*50)
        print("PHASE 2: Multi-task Learning")
        print("="*50)

        self.model.train()
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 20

        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0
            loss_components_sum = {
                'pain_loss': 0.0, 'stress_loss': 0.0, 'reg_loss': 0.0, 'total_loss': 0.0}

            # Create mixed batches
            mixed_batches = create_mixed_batch(
                self.dataloaders['pain']['train'],
                self.dataloaders['stress']['train']
            )

            # Training loop
            self.model.train()
            for batch_idx, batch in enumerate(mixed_batches):
                if batch_idx >= min(len(self.dataloaders['pain']['train']),
                                    len(self.dataloaders['stress']['train'])):
                    break

                signals = batch['signal'].to(self.device)
                labels = batch['label'].to(self.device)
                tasks = batch['task']

                # Forward pass
                predictions = self.model(signals)

                # Compute loss
                loss, loss_components = self.dualnet_loss(
                    predictions, labels, tasks)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                # Accumulate loss components
                for k, v in loss_components.items():
                    loss_components_sum[k] += v

                # Print progress
                if batch_idx % 10 == 0:
                    print(
                        f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{min(len(self.dataloaders['pain']['train']), len(self.dataloaders['stress']['train']))}, Loss: {loss.item():.4f}")

            # Calculate average loss
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            self.history['train_loss'].append(avg_loss)

            # Average loss components
            avg_components = {k: v / num_batches for k,
                              v in loss_components_sum.items()}

            # Validation
            val_loss, pain_metrics, stress_metrics = self.validate()
            self.history['val_loss'].append(val_loss)

            for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
                if pain_metrics:
                    self.history['pain_metrics']['val'][metric].append(
                        pain_metrics[metric])
                if stress_metrics:
                    self.history['stress_metrics']['val'][metric].append(
                        stress_metrics[metric])

            # Print epoch summary
            print(
                f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
            if pain_metrics:
                print(
                    f"Pain - Acc: {pain_metrics['accuracy']:.4f}, F1: {pain_metrics['f1']:.4f}, AUC: {pain_metrics['auc']:.4f}")
            if stress_metrics:
                print(
                    f"Stress - Acc: {stress_metrics['accuracy']:.4f}, F1: {stress_metrics['f1']:.4f}, AUC: {stress_metrics['auc']:.4f}")

            # Learning rate scheduler
            self.scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_dualnet_model.pth')
                print("✓ Model saved (best validation loss)")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping after {epoch+1} epochs")
                    break

        # Load best model
        self.model.load_state_dict(torch.load('best_dualnet_model.pth'))
        print("✓ Phase 2 completed: Multi-task training finished")

    def train_phase3_finetune(self, epochs=50):
        """
        Phase 3: Fine-tuning with reduced learning rate
        """
        print("\n" + "="*50)
        print("PHASE 3: Fine-tuning")
        print("="*50)

        # Reduce learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= 0.1

        # Fine-tune with reduced learning rate
        self.train_phase2_multitask(epochs)
        print("✓ Phase 3 completed: Fine-tuning finished")

    def validate(self):
        """
        Validate the model on validation sets

        Returns:
            Average validation loss, pain metrics, stress metrics
        """
        self.model.eval()
        total_val_loss = 0.0
        num_val_batches = 0

        pain_val_preds = []
        pain_val_labels = []
        stress_val_preds = []
        stress_val_labels = []

        with torch.no_grad():
            # Validate on pain dataset
            for batch in self.dataloaders['pain']['val']:
                signals = batch['signal'].to(self.device)
                labels = batch['label'].to(self.device)
                tasks = batch['task']

                predictions = self.model(signals, task='pain')
                loss, _ = self.dualnet_loss(predictions, labels, tasks)

                total_val_loss += loss.item()
                num_val_batches += 1

                pain_val_preds.extend(predictions['pain'].cpu().numpy())
                pain_val_labels.extend(labels.cpu().numpy())

            # Validate on stress dataset
            for batch in self.dataloaders['stress']['val']:
                signals = batch['signal'].to(self.device)
                labels = batch['label'].to(self.device)
                tasks = batch['task']

                predictions = self.model(signals, task='stress')
                loss, _ = self.dualnet_loss(predictions, labels, tasks)

                total_val_loss += loss.item()
                num_val_batches += 1

                stress_val_preds.extend(predictions['stress'].cpu().numpy())
                stress_val_labels.extend(labels.cpu().numpy())

        # Calculate average validation loss
        avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0

        # Calculate metrics
        pain_metrics = None
        if pain_val_preds:
            pain_val_preds_binary = [
                1 if p >= 0.5 else 0 for p in pain_val_preds]
            pain_metrics = {
                'accuracy': accuracy_score(pain_val_labels, pain_val_preds_binary),
                'precision': precision_score(pain_val_labels, pain_val_preds_binary, zero_division=0),
                'recall': recall_score(pain_val_labels, pain_val_preds_binary, zero_division=0),
                'f1': f1_score(pain_val_labels, pain_val_preds_binary, zero_division=0),
                'auc': roc_auc_score(pain_val_labels, pain_val_preds) if len(set(pain_val_labels)) > 1 else 0.5
            }

        stress_metrics = None
        if stress_val_preds:
            stress_val_preds_binary = [
                1 if p >= 0.5 else 0 for p in stress_val_preds]
            stress_metrics = {
                'accuracy': accuracy_score(stress_val_labels, stress_val_preds_binary),
                'precision': precision_score(stress_val_labels, stress_val_preds_binary, zero_division=0),
                'recall': recall_score(stress_val_labels, stress_val_preds_binary, zero_division=0),
                'f1': f1_score(stress_val_labels, stress_val_preds_binary, zero_division=0),
                'auc': roc_auc_score(stress_val_labels, stress_val_preds) if len(set(stress_val_labels)) > 1 else 0.5
            }

        self.model.train()
        return avg_val_loss, pain_metrics, stress_metrics

    def test(self):
        """
        Test the model on test sets

        Returns:
            Pain test metrics, stress test metrics
        """
        self.model.eval()

        pain_test_preds = []
        pain_test_labels = []
        stress_test_preds = []
        stress_test_labels = []

        with torch.no_grad():
            # Test on pain dataset
            for batch in self.dataloaders['pain']['test']:
                signals = batch['signal'].to(self.device)
                labels = batch['label'].to(self.device)

                predictions = self.model(signals, task='pain')

                pain_test_preds.extend(predictions['pain'].cpu().numpy())
                pain_test_labels.extend(labels.cpu().numpy())

            # Test on stress dataset
            for batch in self.dataloaders['stress']['test']:
                signals = batch['signal'].to(self.device)
                labels = batch['label'].to(self.device)

                predictions = self.model(signals, task='stress')

                stress_test_preds.extend(predictions['stress'].cpu().numpy())
                stress_test_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        pain_metrics = None
        if pain_test_preds:
            pain_test_preds_binary = [
                1 if p >= 0.5 else 0 for p in pain_test_preds]
            pain_metrics = {
                'accuracy': accuracy_score(pain_test_labels, pain_test_preds_binary),
                'precision': precision_score(pain_test_labels, pain_test_preds_binary, zero_division=0),
                'recall': recall_score(pain_test_labels, pain_test_preds_binary, zero_division=0),
                'f1': f1_score(pain_test_labels, pain_test_preds_binary, zero_division=0),
                'auc': roc_auc_score(pain_test_labels, pain_test_preds) if len(set(pain_test_labels)) > 1 else 0.5
            }

            # Confusion matrix
            pain_cm = confusion_matrix(
                pain_test_labels, pain_test_preds_binary)
            print("\nPain Detection Confusion Matrix:")
            print(pain_cm)

        stress_metrics = None
        if stress_test_preds:
            stress_test_preds_binary = [
                1 if p >= 0.5 else 0 for p in stress_test_preds]
            stress_metrics = {
                'accuracy': accuracy_score(stress_test_labels, stress_test_preds_binary),
                'precision': precision_score(stress_test_labels, stress_test_preds_binary, zero_division=0),
                'recall': recall_score(stress_test_labels, stress_test_preds_binary, zero_division=0),
                'f1': f1_score(stress_test_labels, stress_test_preds_binary, zero_division=0),
                'auc': roc_auc_score(stress_test_labels, stress_test_preds) if len(set(stress_test_labels)) > 1 else 0.5
            }

            # Confusion matrix
            stress_cm = confusion_matrix(
                stress_test_labels, stress_test_preds_binary)
            print("\nStress Detection Confusion Matrix:")
            print(stress_cm)

        # Save metrics to history
        if pain_metrics:
            for metric, value in pain_metrics.items():
                self.history['pain_metrics']['test'][metric] = value

        if stress_metrics:
            for metric, value in stress_metrics.items():
                self.history['stress_metrics']['test'][metric] = value

        return pain_metrics, stress_metrics

    def train_full_pipeline(self, pretrain_epochs=50, multitask_epochs=100, finetune_epochs=50):
        """
        Run the complete training pipeline
        """
        print("\n" + "="*50)
        print("DualNet Training Pipeline")
        print("="*50)

        # Phase 1: Pre-training
        self.train_phase1_pretrain(epochs=pretrain_epochs)

        # Phase 2: Multi-task learning
        self.train_phase2_multitask(epochs=multitask_epochs)

        # Phase 3: Fine-tuning
        self.train_phase3_finetune(epochs=finetune_epochs)

        # Final evaluation
        print("\n" + "="*50)
        print("Final Evaluation")
        print("="*50)

        pain_metrics, stress_metrics = self.test()

        print("\nPain Detection Metrics:")
        if pain_metrics:
            for metric, value in pain_metrics.items():
                print(f"{metric}: {value:.4f}")

        print("\nStress Detection Metrics:")
        if stress_metrics:
            for metric, value in stress_metrics.items():
                print(f"{metric}: {value:.4f}")

        print("\n✓ Training and evaluation completed")

    def save_checkpoint(self, filepath):
        """
        Save model checkpoint
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, filepath)
        print(f"Model checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath):
        """
        Load model checkpoint
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        print(f"Model checkpoint loaded from {filepath}")
