"""
Model Evaluation Script
Evaluates trained models on test data
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np
import json
import logging

from src.data_preparation import DataPreparation
from src.models.diagnosis_models import BreastCancerDiagnosisModel, ResNet3D, AttentionCNN3D, PETCTDataset
from src.models.treatment_models import TreatmentRecommender
from src.utils.metrics import calculate_diagnosis_metrics, calculate_treatment_metrics, print_metrics
from src.utils.visualization import plot_confusion_matrix

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate_diagnosis_model(args):
    """Evaluate diagnosis model"""
    logger.info("Evaluating diagnosis model...")
    
    # Load model
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.model_path, map_location=device)
    
    in_channels = checkpoint.get('in_channels', 2)
    model_type = checkpoint.get('model_type', 'cnn')
    
    if model_type == 'cnn':
        model = BreastCancerDiagnosisModel(in_channels=in_channels, num_classes=2)
    elif model_type == 'resnet':
        model = ResNet3D(in_channels=in_channels, num_classes=2)
    elif model_type == 'attention':
        model = AttentionCNN3D(in_channels=in_channels, num_classes=2)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded {model_type} model from {args.model_path}")
    
    # Load test data
    if args.test_dataset_path and Path(args.test_dataset_path).exists():
        from train_diagnosis import load_prepared_dataset
        test_dataset = load_prepared_dataset(args.test_dataset_path)
    else:
        logger.info("Preparing test dataset...")
        prep = DataPreparation()
        
        if args.labels_path and Path(args.labels_path).exists():
            labels_df = prep.load_labels_from_csv(args.labels_path)
        else:
            logger.warning("No labels file found. Creating synthetic labels.")
            patients = prep.loader.get_patient_list()[:args.max_patients]
            labels_df = prep.create_synthetic_labels(patients)
        
        full_dataset = prep.prepare_diagnosis_dataset(labels_df, max_patients=args.max_patients)
        _, _, test_dataset = prep.split_dataset(full_dataset, test_size=0.2, val_size=0.1)
    
    # Create data loader
    test_ds = PETCTDataset(test_dataset['volumes'], test_dataset['labels'], augment=False)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    
    # Evaluate
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
    
    # Calculate metrics
    metrics = calculate_diagnosis_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs)
    )
    
    print_metrics(metrics, "Diagnosis Model Evaluation")
    
    # Plot confusion matrix
    plot_confusion_matrix(
        np.array(all_labels),
        np.array(all_preds),
        ['Benign', 'Malignant'],
        save_path=Path(args.output_dir) / 'confusion_matrix.png'
    )
    
    # Save results
    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / 'evaluation_metrics.json', 'w') as f:
        json.dump({k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                  for k, v in metrics.items()}, f, indent=2)
    
    logger.info(f"Results saved to {args.output_dir}")


def evaluate_treatment_model(args):
    """Evaluate treatment model"""
    logger.info("Evaluating treatment model...")
    
    # Load model
    recommender = TreatmentRecommender()
    recommender.load_model(args.model_path)
    logger.info(f"Loaded model from {args.model_path}")
    
    # Load test data
    if args.test_dataset_path and Path(args.test_dataset_path).exists():
        from train_treatment import load_prepared_dataset
        test_dataset = load_prepared_dataset(args.test_dataset_path)
    else:
        logger.info("Preparing test dataset...")
        prep = DataPreparation()
        
        if args.labels_path and Path(args.labels_path).exists():
            labels_df = prep.load_labels_from_csv(args.labels_path)
        else:
            logger.warning("No labels file found. Creating synthetic labels.")
            patients = prep.loader.get_patient_list()[:args.max_patients]
            labels_df = prep.create_synthetic_labels(patients)
        
        full_dataset = prep.prepare_treatment_dataset(labels_df, max_patients=args.max_patients)
        _, _, test_dataset = prep.split_dataset(full_dataset, test_size=0.2, val_size=0.1)
    
    # Evaluate
    test_features_scaled = recommender.scaler.transform(test_dataset['features'])
    test_pred = recommender.model.predict(test_features_scaled)
    test_proba = recommender.model.predict_proba(test_features_scaled) if hasattr(recommender.model, 'predict_proba') else None
    
    # Calculate metrics
    metrics = calculate_treatment_metrics(
        test_dataset['treatment_labels'],
        test_pred,
        test_dataset['treatment_classes']
    )
    
    print_metrics(metrics, "Treatment Model Evaluation")
    
    # Plot confusion matrix
    plot_confusion_matrix(
        test_dataset['treatment_labels'],
        test_pred,
        test_dataset['treatment_classes'],
        save_path=Path(args.output_dir) / 'confusion_matrix.png'
    )
    
    # Save results
    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / 'evaluation_metrics.json', 'w') as f:
        json.dump({k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                  for k, v in metrics.items()}, f, indent=2)
    
    logger.info(f"Results saved to {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Trained Models')
    parser.add_argument('--model', type=str, choices=['diagnosis', 'treatment'],
                       required=True, help='Model type to evaluate')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--test_dataset_path', type=str, help='Path to test dataset')
    parser.add_argument('--labels_path', type=str, help='Path to labels CSV file')
    parser.add_argument('--max_patients', type=int, default=20, help='Maximum patients to use')
    parser.add_argument('--output_dir', type=str, default='results/evaluation', help='Output directory')
    
    # Diagnosis-specific
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    
    args = parser.parse_args()
    
    if args.model == 'diagnosis':
        evaluate_diagnosis_model(args)
    else:
        evaluate_treatment_model(args)


if __name__ == "__main__":
    main()

