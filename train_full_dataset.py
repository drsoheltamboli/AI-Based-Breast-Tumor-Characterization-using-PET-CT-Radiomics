"""
Train on Full Dataset
Trains models on the complete dataset (all available patients)
"""

import argparse
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_full_dataset(args):
    """Train on full dataset"""
    
    logger.info("="*60)
    logger.info("Training on Full Dataset")
    logger.info("="*60)
    
    # Check for labels
    if not args.labels_path or not Path(args.labels_path).exists():
        logger.error(f"Labels file not found: {args.labels_path}")
        logger.error("Please provide a valid labels CSV file using --labels_path")
        logger.error("\nTo create a label template, run:")
        logger.error("  python prepare_labels.py --action create_template --output data/labels_template.csv")
        return
    
    # Validate labels
    logger.info("\n[Step 1] Validating labels...")
    from prepare_labels import validate_labels
    validation = validate_labels(args.labels_path)
    
    if not validation['valid']:
        logger.error("Label validation failed. Please fix errors before training.")
        return
    
    logger.info(f"✓ Labels validated: {validation['stats']['labeled_patients']} labeled patients")
    
    # Prepare full dataset
    logger.info("\n[Step 2] Preparing full dataset...")
    from src.data_preparation import DataPreparation
    
    prep = DataPreparation()
    labels_df = prep.load_labels_from_csv(args.labels_path)
    
    logger.info(f"Total patients in labels: {len(labels_df)}")
    
    # Prepare datasets (no max_patients limit)
    logger.info("Preparing diagnosis dataset (this may take a while)...")
    diagnosis_dataset = prep.prepare_diagnosis_dataset(
        labels_df,
        max_patients=None,  # Use all patients
        save_path='data/prepared/diagnosis_full'
    )
    
    logger.info(f"Prepared {len(diagnosis_dataset['volumes'])} diagnosis samples")
    
    logger.info("Preparing treatment dataset (this may take a while)...")
    treatment_dataset = prep.prepare_treatment_dataset(
        labels_df,
        max_patients=None,  # Use all patients
        save_path='data/prepared/treatment_full'
    )
    
    logger.info(f"Prepared {len(treatment_dataset['features'])} treatment samples")
    
    if args.skip_training:
        logger.info("Dataset preparation complete. Skipping training (--skip_training).")
        return
    
    # Train diagnosis model
    if args.train_diagnosis:
        logger.info("\n[Step 3] Training diagnosis model on full dataset...")
        import subprocess
        
        train_cmd = [
            'python', 'train_diagnosis.py',
            '--dataset_path', 'data/prepared/diagnosis_full',
            '--model_type', args.diagnosis_model,
            '--epochs', str(args.epochs),
            '--batch_size', str(args.batch_size),
            '--learning_rate', str(args.learning_rate),
            '--output_dir', 'results/diagnosis_full'
        ]
        
        if args.augment:
            train_cmd.append('--augment')
        if args.early_stopping:
            train_cmd.extend(['--early_stopping', '--patience', str(args.patience)])
        
        logger.info(f"Running: {' '.join(train_cmd)}")
        result = subprocess.run(train_cmd)
        
        if result.returncode == 0:
            logger.info("✓ Diagnosis model training completed")
        else:
            logger.error("✗ Diagnosis model training failed")
    
    # Train treatment model
    if args.train_treatment:
        logger.info("\n[Step 4] Training treatment model on full dataset...")
        import subprocess
        
        train_cmd = [
            'python', 'train_treatment.py',
            '--dataset_path', 'data/prepared/treatment_full',
            '--model_type', args.treatment_model,
            '--output_dir', 'results/treatment_full'
        ]
        
        logger.info(f"Running: {' '.join(train_cmd)}")
        result = subprocess.run(train_cmd)
        
        if result.returncode == 0:
            logger.info("✓ Treatment model training completed")
        else:
            logger.error("✗ Treatment model training failed")
    
    logger.info("\n" + "="*60)
    logger.info("Full Dataset Training Complete!")
    logger.info("="*60)
    logger.info("\nResults saved in:")
    if args.train_diagnosis:
        logger.info("  - results/diagnosis_full/")
    if args.train_treatment:
        logger.info("  - results/treatment_full/")


def main():
    parser = argparse.ArgumentParser(description='Train on Full Dataset')
    
    # Required
    parser.add_argument('--labels_path', type=str, required=True,
                       help='Path to labels CSV file (REQUIRED)')
    
    # Training options
    parser.add_argument('--train_diagnosis', action='store_true',
                       help='Train diagnosis model')
    parser.add_argument('--train_treatment', action='store_true',
                       help='Train treatment model')
    parser.add_argument('--skip_training', action='store_true',
                       help='Only prepare dataset, skip training')
    
    # Model arguments
    parser.add_argument('--diagnosis_model', type=str,
                       choices=['cnn', 'resnet', 'attention'],
                       default='cnn', help='Diagnosis model type')
    parser.add_argument('--treatment_model', type=str,
                       choices=['random_forest', 'gradient_boosting', 'xgboost', 'logistic'],
                       default='xgboost', help='Treatment model type')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--augment', action='store_true',
                       help='Use data augmentation')
    parser.add_argument('--early_stopping', action='store_true',
                       help='Use early stopping')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    
    args = parser.parse_args()
    
    # Default: train both if neither specified
    if not args.train_diagnosis and not args.train_treatment and not args.skip_training:
        args.train_diagnosis = True
        args.train_treatment = True
    
    train_full_dataset(args)


if __name__ == "__main__":
    main()

