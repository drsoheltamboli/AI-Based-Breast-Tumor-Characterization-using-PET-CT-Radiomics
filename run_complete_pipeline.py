"""
Complete Pipeline Script
Runs the entire workflow from data preparation to model evaluation
"""

import argparse
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_complete_pipeline(args):
    """Run the complete pipeline"""
    
    logger.info("="*60)
    logger.info("Complete Precision Medicine AI Pipeline")
    logger.info("="*60)
    
    # Step 1: Prepare Data
    logger.info("\n[Step 1/5] Preparing data...")
    from src.data_preparation import DataPreparation
    
    prep = DataPreparation()
    
    if args.labels_path and Path(args.labels_path).exists():
        logger.info(f"Loading labels from {args.labels_path}")
        labels_df = prep.load_labels_from_csv(args.labels_path)
    else:
        logger.warning("No labels file found. Creating synthetic labels for demonstration.")
        patients = prep.loader.get_patient_list()[:args.max_patients]
        labels_df = prep.create_synthetic_labels(patients, output_path='data/labels.csv')
        logger.info(f"Created synthetic labels for {len(labels_df)} patients")
    
    # Prepare datasets
    logger.info("Preparing diagnosis dataset...")
    diagnosis_dataset = prep.prepare_diagnosis_dataset(
        labels_df,
        max_patients=args.max_patients,
        save_path='data/prepared/diagnosis'
    )
    
    logger.info("Preparing treatment dataset...")
    treatment_dataset = prep.prepare_treatment_dataset(
        labels_df,
        max_patients=args.max_patients,
        save_path='data/prepared/treatment'
    )
    
    if args.skip_training:
        logger.info("Skipping training (--skip_training flag set)")
        return
    
    # Step 2: Train Diagnosis Model
    logger.info("\n[Step 2/5] Training diagnosis model...")
    import subprocess
    
    train_diag_cmd = [
        'python', 'train_diagnosis.py',
        '--dataset_path', 'data/prepared/diagnosis',
        '--model_type', args.diagnosis_model,
        '--epochs', str(args.epochs),
        '--batch_size', str(args.batch_size),
        '--output_dir', 'results/diagnosis'
    ]
    
    if args.augment:
        train_diag_cmd.append('--augment')
    if args.early_stopping:
        train_diag_cmd.extend(['--early_stopping', '--patience', str(args.patience)])
    
    logger.info(f"Running: {' '.join(train_diag_cmd)}")
    result = subprocess.run(train_diag_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Diagnosis training failed: {result.stderr}")
    else:
        logger.info("Diagnosis model training completed")
    
    # Step 3: Train Treatment Model
    logger.info("\n[Step 3/5] Training treatment model...")
    
    train_treat_cmd = [
        'python', 'train_treatment.py',
        '--dataset_path', 'data/prepared/treatment',
        '--model_type', args.treatment_model,
        '--output_dir', 'results/treatment'
    ]
    
    logger.info(f"Running: {' '.join(train_treat_cmd)}")
    result = subprocess.run(train_treat_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Treatment training failed: {result.stderr}")
    else:
        logger.info("Treatment model training completed")
    
    # Step 4: Cross-Validation (optional)
    if args.run_cv:
        logger.info("\n[Step 4/5] Running cross-validation...")
        
        # CV for diagnosis
        cv_diag_cmd = [
            'python', 'validate_models.py',
            '--model', 'diagnosis',
            '--labels_path', 'data/labels.csv',
            '--model_type', args.diagnosis_model,
            '--n_folds', str(args.n_folds),
            '--max_patients', str(args.max_patients),
            '--output_dir', 'results/cv_diagnosis'
        ]
        
        logger.info(f"Running CV for diagnosis: {' '.join(cv_diag_cmd)}")
        subprocess.run(cv_diag_cmd)
        
        # CV for treatment
        cv_treat_cmd = [
            'python', 'validate_models.py',
            '--model', 'treatment',
            '--labels_path', 'data/labels.csv',
            '--model_type', args.treatment_model,
            '--n_folds', str(args.n_folds),
            '--max_patients', str(args.max_patients),
            '--output_dir', 'results/cv_treatment'
        ]
        
        logger.info(f"Running CV for treatment: {' '.join(cv_treat_cmd)}")
        subprocess.run(cv_treat_cmd)
    
    # Step 5: Evaluate Models
    logger.info("\n[Step 5/5] Evaluating models...")
    
    # Find model paths
    diag_model_path = Path('results/diagnosis') / f'best_model_{args.diagnosis_model}.pth'
    treat_model_path = Path('results/treatment') / f'treatment_model_{args.treatment_model}.pkl'
    
    if diag_model_path.exists():
        eval_diag_cmd = [
            'python', 'evaluate_models.py',
            '--model', 'diagnosis',
            '--model_path', str(diag_model_path),
            '--labels_path', 'data/labels.csv',
            '--max_patients', str(args.max_patients),
            '--output_dir', 'results/evaluation/diagnosis'
        ]
        
        logger.info(f"Evaluating diagnosis model: {' '.join(eval_diag_cmd)}")
        subprocess.run(eval_diag_cmd)
    else:
        logger.warning(f"Diagnosis model not found at {diag_model_path}")
    
    if treat_model_path.exists():
        eval_treat_cmd = [
            'python', 'evaluate_models.py',
            '--model', 'treatment',
            '--model_path', str(treat_model_path),
            '--labels_path', 'data/labels.csv',
            '--max_patients', str(args.max_patients),
            '--output_dir', 'results/evaluation/treatment'
        ]
        
        logger.info(f"Evaluating treatment model: {' '.join(eval_treat_cmd)}")
        subprocess.run(eval_treat_cmd)
    else:
        logger.warning(f"Treatment model not found at {treat_model_path}")
    
    logger.info("\n" + "="*60)
    logger.info("Pipeline completed!")
    logger.info("="*60)
    logger.info("\nResults saved in:")
    logger.info("  - results/diagnosis/")
    logger.info("  - results/treatment/")
    logger.info("  - results/evaluation/")
    if args.run_cv:
        logger.info("  - results/cv_diagnosis/")
        logger.info("  - results/cv_treatment/")


def main():
    parser = argparse.ArgumentParser(description='Run Complete Pipeline')
    
    # Data arguments
    parser.add_argument('--labels_path', type=str, help='Path to labels CSV file')
    parser.add_argument('--max_patients', type=int, default=20, help='Maximum patients to use')
    
    # Model arguments
    parser.add_argument('--diagnosis_model', type=str, choices=['cnn', 'resnet', 'attention'],
                       default='cnn', help='Diagnosis model type')
    parser.add_argument('--treatment_model', type=str,
                       choices=['random_forest', 'gradient_boosting', 'xgboost', 'logistic'],
                       default='xgboost', help='Treatment model type')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--augment', action='store_true', help='Use data augmentation')
    parser.add_argument('--early_stopping', action='store_true', help='Use early stopping')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    
    # Other arguments
    parser.add_argument('--run_cv', action='store_true', help='Run cross-validation')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--skip_training', action='store_true', 
                       help='Skip training (only prepare data)')
    
    args = parser.parse_args()
    
    run_complete_pipeline(args)


if __name__ == "__main__":
    main()

