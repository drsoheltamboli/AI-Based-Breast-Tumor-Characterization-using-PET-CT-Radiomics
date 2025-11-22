"""
Cross-Validation and Model Validation Script
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np
import json
import logging
from sklearn.model_selection import KFold, StratifiedKFold
from typing import Dict, List

from src.data_preparation import DataPreparation
from src.models.diagnosis_models import BreastCancerDiagnosisModel, ResNet3D, AttentionCNN3D, PETCTDataset, ModelTrainer
from src.models.treatment_models import TreatmentRecommender
from src.utils.metrics import calculate_diagnosis_metrics, calculate_treatment_metrics, print_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def cross_validate_diagnosis(args):
    """Perform cross-validation for diagnosis model"""
    logger.info("Starting cross-validation for diagnosis model...")
    
    # Prepare data
    prep = DataPreparation()
    
    if args.labels_path and Path(args.labels_path).exists():
        labels_df = prep.load_labels_from_csv(args.labels_path)
    else:
        logger.warning("No labels file found. Creating synthetic labels for demonstration.")
        patients = prep.loader.get_patient_list()[:args.max_patients]
        labels_df = prep.create_synthetic_labels(patients)
    
    dataset = prep.prepare_diagnosis_dataset(labels_df, max_patients=args.max_patients)
    
    if len(dataset['volumes']) == 0:
        logger.error("No data available!")
        return
    
    # Setup cross-validation
    kfold = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.random_state)
    labels_array = np.array(dataset['labels'])
    
    cv_results = {
        'fold_accuracies': [],
        'fold_losses': [],
        'fold_metrics': []
    }
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset['volumes'], labels_array)):
        logger.info(f"\n{'='*50}")
        logger.info(f"Fold {fold + 1}/{args.n_folds}")
        logger.info(f"{'='*50}")
        
        # Split data
        train_volumes = [dataset['volumes'][i] for i in train_idx]
        train_labels = [dataset['labels'][i] for i in train_idx]
        val_volumes = [dataset['volumes'][i] for i in val_idx]
        val_labels = [dataset['labels'][i] for i in val_idx]
        
        # Create datasets
        train_dataset = PETCTDataset(train_volumes, train_labels, augment=args.augment)
        val_dataset = PETCTDataset(val_volumes, val_labels, augment=False)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        )
        
        # Create model
        in_channels = 2 if len(dataset['volumes'][0].shape) == 4 else 1
        
        if args.model_type == 'cnn':
            model = BreastCancerDiagnosisModel(in_channels=in_channels, num_classes=2)
        elif args.model_type == 'resnet':
            model = ResNet3D(in_channels=in_channels, num_classes=2)
        elif args.model_type == 'attention':
            model = AttentionCNN3D(in_channels=in_channels, num_classes=2)
        else:
            raise ValueError(f"Unknown model type: {args.model_type}")
        
        # Train
        trainer = ModelTrainer(model, device=device)
        trainer.setup_optimizer(lr=args.learning_rate, weight_decay=args.weight_decay)
        
        # Train for fewer epochs per fold
        epochs_per_fold = max(10, args.epochs // args.n_folds)
        for epoch in range(epochs_per_fold):
            train_loss, train_acc = trainer.train_epoch(train_loader)
            val_loss, val_acc = trainer.validate(val_loader)
            trainer.scheduler.step(val_loss)
            
            if (epoch + 1) % 5 == 0:
                logger.info(f"  Epoch {epoch+1}/{epochs_per_fold}: Val Acc = {val_acc:.2f}%")
        
        # Final evaluation
        val_loss, val_acc = trainer.validate(val_loader)
        
        # Get predictions for metrics
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for volumes, labels in val_loader:
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
        
        cv_results['fold_accuracies'].append(float(val_acc))
        cv_results['fold_losses'].append(float(val_loss))
        cv_results['fold_metrics'].append(metrics)
        
        logger.info(f"Fold {fold + 1} Results:")
        logger.info(f"  Accuracy: {val_acc:.2f}%")
        logger.info(f"  Loss: {val_loss:.4f}")
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("Cross-Validation Summary")
    logger.info(f"{'='*50}")
    logger.info(f"Mean Accuracy: {np.mean(cv_results['fold_accuracies']):.2f}%")
    logger.info(f"Std Accuracy: {np.std(cv_results['fold_accuracies']):.2f}%")
    logger.info(f"Mean Loss: {np.mean(cv_results['fold_losses']):.4f}")
    
    # Save results
    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        'n_folds': args.n_folds,
        'mean_accuracy': float(np.mean(cv_results['fold_accuracies'])),
        'std_accuracy': float(np.std(cv_results['fold_accuracies'])),
        'fold_accuracies': cv_results['fold_accuracies'],
        'mean_loss': float(np.mean(cv_results['fold_losses'])),
        'std_loss': float(np.std(cv_results['fold_losses']))
    }
    
    with open(results_dir / 'cv_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nResults saved to {args.output_dir}")


def cross_validate_treatment(args):
    """Perform cross-validation for treatment model"""
    logger.info("Starting cross-validation for treatment model...")
    
    # Prepare data
    prep = DataPreparation()
    
    if args.labels_path and Path(args.labels_path).exists():
        labels_df = prep.load_labels_from_csv(args.labels_path)
    else:
        logger.warning("No labels file found. Creating synthetic labels for demonstration.")
        patients = prep.loader.get_patient_list()[:args.max_patients]
        labels_df = prep.create_synthetic_labels(patients)
    
    dataset = prep.prepare_treatment_dataset(labels_df, max_patients=args.max_patients)
    
    if len(dataset['features']) == 0:
        logger.error("No data available!")
        return
    
    # Setup cross-validation
    kfold = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.random_state)
    
    cv_results = {
        'fold_accuracies': [],
        'fold_metrics': []
    }
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset['features'], dataset['treatment_labels'])):
        logger.info(f"\n{'='*50}")
        logger.info(f"Fold {fold + 1}/{args.n_folds}")
        logger.info(f"{'='*50}")
        
        # Split data
        X_train = dataset['features'][train_idx]
        y_train = dataset['treatment_labels'][train_idx]
        X_val = dataset['features'][val_idx]
        y_val = dataset['treatment_labels'][val_idx]
        
        # Train model
        recommender = TreatmentRecommender(model_type=args.model_type)
        recommender.treatment_classes = dataset['treatment_classes']
        
        recommender.train(X_train, y_train, treatment_labels=dataset['treatment_classes'], test_size=0.0)
        
        # Evaluate
        val_pred = recommender.model.predict(recommender.scaler.transform(X_val))
        val_acc = np.mean(val_pred == y_val)
        
        metrics = calculate_treatment_metrics(y_val, val_pred, dataset['treatment_classes'])
        
        cv_results['fold_accuracies'].append(float(val_acc))
        cv_results['fold_metrics'].append(metrics)
        
        logger.info(f"Fold {fold + 1} Accuracy: {val_acc:.4f}")
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("Cross-Validation Summary")
    logger.info(f"{'='*50}")
    logger.info(f"Mean Accuracy: {np.mean(cv_results['fold_accuracies']):.2f}%")
    logger.info(f"Std Accuracy: {np.std(cv_results['fold_accuracies']):.2f}%")
    
    # Save results
    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        'n_folds': args.n_folds,
        'mean_accuracy': float(np.mean(cv_results['fold_accuracies'])),
        'std_accuracy': float(np.std(cv_results['fold_accuracies'])),
        'fold_accuracies': cv_results['fold_accuracies']
    }
    
    with open(results_dir / 'cv_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nResults saved to {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Cross-Validation for Models')
    parser.add_argument('--model', type=str, choices=['diagnosis', 'treatment'], 
                       required=True, help='Model type to validate')
    parser.add_argument('--labels_path', type=str, help='Path to labels CSV file')
    parser.add_argument('--max_patients', type=int, default=20, help='Maximum patients to use')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='results/cv', help='Output directory')
    
    # Diagnosis-specific
    parser.add_argument('--model_type', type=str, choices=['cnn', 'resnet', 'attention'],
                       default='cnn', help='Model architecture (for diagnosis)')
    parser.add_argument('--epochs', type=int, default=50, help='Total epochs (distributed across folds)')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--augment', action='store_true', help='Use data augmentation')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    
    args = parser.parse_args()
    
    if args.model == 'diagnosis':
        cross_validate_diagnosis(args)
    else:
        cross_validate_treatment(args)


if __name__ == "__main__":
    main()

