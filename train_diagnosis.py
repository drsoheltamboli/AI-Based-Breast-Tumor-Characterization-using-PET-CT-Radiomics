"""
Training Script for Breast Cancer Diagnosis Model
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import json
import logging
from datetime import datetime

from src.data_preparation import DataPreparation
from src.models.diagnosis_models import BreastCancerDiagnosisModel, ResNet3D, AttentionCNN3D, PETCTDataset, ModelTrainer
from src.utils.visualization import plot_training_history
from src.utils.metrics import calculate_diagnosis_metrics, print_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_prepared_dataset(dataset_path: str):
    """Load prepared dataset from disk"""
    dataset_path = Path(dataset_path)
    
    volumes = np.load(dataset_path / 'volumes.npy', allow_pickle=True)
    labels = np.load(dataset_path / 'labels.npy')
    patient_ids = np.load(dataset_path / 'patient_ids.npy', allow_pickle=True)
    
    with open(dataset_path / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    return {
        'volumes': volumes.tolist(),
        'labels': labels.tolist(),
        'patient_ids': patient_ids.tolist(),
        'metadata': metadata
    }


def train_model(args):
    """Main training function"""
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load or prepare dataset
    if args.dataset_path and Path(args.dataset_path).exists():
        logger.info(f"Loading dataset from {args.dataset_path}")
        full_dataset = load_prepared_dataset(args.dataset_path)
    else:
        logger.info("Preparing dataset from scratch...")
        prep = DataPreparation()
        
        # Load labels
        if args.labels_path and Path(args.labels_path).exists():
            labels_df = prep.load_labels_from_csv(args.labels_path)
        else:
            logger.warning("No labels file found. Creating synthetic labels for demonstration.")
            patients = prep.loader.get_patient_list()[:args.max_patients]
            labels_df = prep.create_synthetic_labels(patients, output_path='data/labels.csv')
        
        # Prepare dataset (skip saving to avoid memory issues with large volumes)
        full_dataset = prep.prepare_diagnosis_dataset(
            labels_df,
            max_patients=args.max_patients,
            save_path=None  # Skip saving to avoid memory issues
        )
    
    if len(full_dataset['volumes']) == 0:
        logger.error("No data available for training!")
        return
    
    # Split dataset
    prep = DataPreparation()
    train_set, val_set, test_set = prep.split_dataset(
        full_dataset,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state
    )
    
    # Create datasets
    train_dataset = PETCTDataset(train_set['volumes'], train_set['labels'], augment=args.augment)
    val_dataset = PETCTDataset(val_set['volumes'], val_set['labels'], augment=False)
    test_dataset = PETCTDataset(test_set['volumes'], test_set['labels'], augment=False)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Create model
    in_channels = 2 if len(full_dataset['volumes'][0].shape) == 4 else 1
    
    if args.model_type == 'cnn':
        model = BreastCancerDiagnosisModel(in_channels=in_channels, num_classes=2, dropout=args.dropout)
    elif args.model_type == 'resnet':
        model = ResNet3D(in_channels=in_channels, num_classes=2)
    elif args.model_type == 'attention':
        model = AttentionCNN3D(in_channels=in_channels, num_classes=2)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    logger.info(f"Created {args.model_type} model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create trainer
    trainer = ModelTrainer(model, device=device)
    trainer.setup_optimizer(lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc = trainer.train_epoch(train_loader)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Validate
        val_loss, val_acc = trainer.validate(val_loader)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        trainer.scheduler.step(val_loss)
        
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            model_path = Path(args.output_dir) / f'best_model_{args.model_type}.pth'
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'val_acc': val_acc,
                'model_type': args.model_type,
                'in_channels': in_channels
            }, model_path)
            logger.info(f"Saved best model to {model_path}")
        else:
            patience_counter += 1
        
        # Early stopping
        if args.early_stopping and patience_counter >= args.patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # Evaluate on test set
    logger.info("\nEvaluating on test set...")
    test_loss, test_acc = trainer.validate(test_loader)
    logger.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    # Calculate detailed metrics
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for volumes, labels in test_loader:
            volumes = volumes.to(device)
            outputs = model(volumes)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    metrics = calculate_diagnosis_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs)
    )
    
    print_metrics(metrics, "Test Set Metrics")
    
    # Save results
    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save training history
    plot_training_history(history, save_path=results_dir / 'training_history.png')
    
    # Save metrics
    with open(results_dir / 'test_metrics.json', 'w') as f:
        json.dump({k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                  for k, v in metrics.items()}, f, indent=2)
    
    logger.info(f"\nTraining completed!")
    logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")
    logger.info(f"Test accuracy: {test_acc:.2f}%")
    logger.info(f"Results saved to {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train Breast Cancer Diagnosis Model')
    
    # Data arguments
    parser.add_argument('--labels_path', type=str, help='Path to labels CSV file')
    parser.add_argument('--dataset_path', type=str, help='Path to pre-prepared dataset')
    parser.add_argument('--save_dataset_path', type=str, default='data/prepared/diagnosis',
                       help='Path to save prepared dataset')
    parser.add_argument('--max_patients', type=int, default=20, help='Maximum patients to use')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, choices=['cnn', 'resnet', 'attention'],
                       default='cnn', help='Model architecture')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--augment', action='store_true', help='Use data augmentation')
    
    # Data split arguments
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size')
    parser.add_argument('--val_size', type=float, default=0.1, help='Validation set size')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed')
    
    # Other arguments
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--output_dir', type=str, default='results/diagnosis', help='Output directory')
    parser.add_argument('--early_stopping', action='store_true', help='Use early stopping')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    
    args = parser.parse_args()
    
    train_model(args)


if __name__ == "__main__":
    main()

