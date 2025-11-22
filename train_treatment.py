"""
Training Script for Treatment Recommendation Model
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import json
import logging
from datetime import datetime

from src.data_preparation import DataPreparation
from src.models.treatment_models import TreatmentRecommender
from src.utils.metrics import calculate_treatment_metrics, print_metrics
from src.utils.visualization import plot_feature_importance

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_prepared_dataset(dataset_path: str):
    """Load prepared treatment dataset from disk"""
    dataset_path = Path(dataset_path)
    
    features = np.load(dataset_path / 'features.npy')
    treatment_labels = np.load(dataset_path / 'treatment_labels.npy')
    patient_ids = np.load(dataset_path / 'patient_ids.npy', allow_pickle=True)
    
    with open(dataset_path / 'treatment_classes.json', 'r') as f:
        treatment_classes = json.load(f)
    
    return {
        'features': features,
        'treatment_labels': treatment_labels,
        'patient_ids': patient_ids.tolist(),
        'treatment_classes': treatment_classes
    }


def train_model(args):
    """Main training function"""
    
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
        
        # Prepare dataset
        full_dataset = prep.prepare_treatment_dataset(
            labels_df,
            max_patients=args.max_patients,
            save_path=args.save_dataset_path
        )
    
    if len(full_dataset['features']) == 0:
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
    
    # Create recommender
    recommender = TreatmentRecommender(model_type=args.model_type)
    recommender.treatment_classes = full_dataset['treatment_classes']
    
    # Train model
    logger.info(f"Training {args.model_type} model...")
    results = recommender.train(
        train_set['features'],
        train_set['treatment_labels'],
        treatment_labels=full_dataset['treatment_classes'],
        test_size=0.0  # Already split
    )
    
    # Evaluate on validation set
    logger.info("\nEvaluating on validation set...")
    val_features_scaled = recommender.scaler.transform(val_set['features'])
    val_pred = recommender.model.predict(val_features_scaled)
    val_acc = np.mean(val_pred == val_set['treatment_labels'])
    logger.info(f"Validation Accuracy: {val_acc:.4f}")
    
    # Evaluate on test set
    logger.info("\nEvaluating on test set...")
    test_features_scaled = recommender.scaler.transform(test_set['features'])
    test_pred = recommender.model.predict(test_features_scaled)
    test_proba = recommender.model.predict_proba(test_features_scaled) if hasattr(recommender.model, 'predict_proba') else None
    
    test_acc = np.mean(test_pred == test_set['treatment_labels'])
    logger.info(f"Test Accuracy: {test_acc:.4f}")
    
    # Calculate detailed metrics
    metrics = calculate_treatment_metrics(
        test_set['treatment_labels'],
        test_pred,
        full_dataset['treatment_classes']
    )
    
    print_metrics(metrics, "Test Set Metrics")
    
    # Feature importance
    if hasattr(recommender.model, 'feature_importances_'):
        importances = recommender.model.feature_importances_
        feature_names = recommender.feature_names if recommender.feature_names else [f'feature_{i}' for i in range(len(importances))]
        importance_dict = {feature_names[i]: float(imp) for i, imp in enumerate(importances)}
        
        # Plot feature importance
        plot_feature_importance(importance_dict, top_n=15, 
                               save_path=Path(args.output_dir) / 'feature_importance.png')
        
        # Save feature importance
        with open(Path(args.output_dir) / 'feature_importance.json', 'w') as f:
            json.dump(importance_dict, f, indent=2)
    
    # Save model
    model_path = Path(args.output_dir) / f'treatment_model_{args.model_type}.pkl'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    recommender.save_model(str(model_path))
    logger.info(f"Saved model to {model_path}")
    
    # Save results
    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    with open(results_dir / 'test_metrics.json', 'w') as f:
        json.dump({k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                  for k, v in metrics.items()}, f, indent=2)
    
    # Save training summary
    summary = {
        'model_type': args.model_type,
        'train_accuracy': float(results['train_accuracy']),
        'test_accuracy': float(test_acc),
        'cv_accuracy': float(results.get('cv_accuracy', 0)),
        'num_features': int(full_dataset['features'].shape[1]),
        'num_samples': int(len(full_dataset['features'])),
        'treatment_classes': full_dataset['treatment_classes']
    }
    
    with open(results_dir / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nTraining completed!")
    logger.info(f"Test accuracy: {test_acc:.4f}")
    logger.info(f"Results saved to {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train Treatment Recommendation Model')
    
    # Data arguments
    parser.add_argument('--labels_path', type=str, help='Path to labels CSV file')
    parser.add_argument('--dataset_path', type=str, help='Path to pre-prepared dataset')
    parser.add_argument('--save_dataset_path', type=str, default='data/prepared/treatment',
                       help='Path to save prepared dataset')
    parser.add_argument('--max_patients', type=int, default=20, help='Maximum patients to use')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, 
                       choices=['random_forest', 'gradient_boosting', 'xgboost', 'logistic'],
                       default='xgboost', help='Model type')
    
    # Data split arguments
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size')
    parser.add_argument('--val_size', type=float, default=0.1, help='Validation set size')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='results/treatment', 
                       help='Output directory')
    
    args = parser.parse_args()
    
    train_model(args)


if __name__ == "__main__":
    main()

