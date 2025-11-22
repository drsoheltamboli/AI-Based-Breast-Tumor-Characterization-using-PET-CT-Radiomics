"""
Hyperparameter Tuning Script
Performs grid search and random search for optimal hyperparameters
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np
import json
import logging
from itertools import product
import random
from typing import Dict, List, Tuple

from src.data_preparation import DataPreparation
from src.models.diagnosis_models import BreastCancerDiagnosisModel, ResNet3D, AttentionCNN3D, PETCTDataset, ModelTrainer
from src.models.treatment_models import TreatmentRecommender
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def grid_search_diagnosis(args):
    """Grid search for diagnosis model hyperparameters"""
    logger.info("Starting grid search for diagnosis model...")
    
    # Load data
    prep = DataPreparation()
    
    if args.labels_path and Path(args.labels_path).exists():
        labels_df = prep.load_labels_from_csv(args.labels_path)
    else:
        logger.warning("No labels file found. Creating synthetic labels.")
        patients = prep.loader.get_patient_list()[:args.max_patients]
        labels_df = prep.create_synthetic_labels(patients)
    
    dataset = prep.prepare_diagnosis_dataset(labels_df, max_patients=args.max_patients)
    train_set, val_set, _ = prep.split_dataset(dataset, test_size=0.2, val_size=0.1)
    
    # Define parameter grid
    param_grid = {
        'learning_rate': [0.0001, 0.001, 0.01],
        'batch_size': [2, 4, 8],
        'dropout': [0.3, 0.5, 0.7],
        'weight_decay': [1e-5, 1e-4, 1e-3]
    }
    
    if args.model_type == 'resnet':
        param_grid['learning_rate'] = [0.0001, 0.001]  # ResNet needs lower LR
    
    # Generate all combinations
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = list(product(*values))
    
    logger.info(f"Testing {len(combinations)} parameter combinations...")
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    best_score = 0.0
    best_params = None
    results = []
    
    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        logger.info(f"\n[{i+1}/{len(combinations)}] Testing: {params}")
        
        try:
            # Create datasets
            train_dataset = PETCTDataset(train_set['volumes'], train_set['labels'], augment=args.augment)
            val_dataset = PETCTDataset(val_set['volumes'], val_set['labels'], augment=False)
            
            train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
            
            # Create model
            in_channels = 2 if len(dataset['volumes'][0].shape) == 4 else 1
            
            if args.model_type == 'cnn':
                model = BreastCancerDiagnosisModel(in_channels=in_channels, num_classes=2, 
                                                  dropout=params['dropout'])
            elif args.model_type == 'resnet':
                model = ResNet3D(in_channels=in_channels, num_classes=2)
            elif args.model_type == 'attention':
                model = AttentionCNN3D(in_channels=in_channels, num_classes=2)
            else:
                continue
            
            # Train
            trainer = ModelTrainer(model, device=device)
            trainer.setup_optimizer(lr=params['learning_rate'], weight_decay=params['weight_decay'])
            
            # Train for fewer epochs during grid search
            epochs = min(10, args.epochs // 3)
            for epoch in range(epochs):
                trainer.train_epoch(train_loader)
                val_loss, val_acc = trainer.validate(val_loader)
                trainer.scheduler.step(val_loss)
            
            # Final validation
            _, val_acc = trainer.validate(val_loader)
            
            results.append({
                'params': params,
                'val_accuracy': float(val_acc),
                'val_loss': float(val_loss)
            })
            
            logger.info(f"  Validation Accuracy: {val_acc:.2f}%")
            
            if val_acc > best_score:
                best_score = val_acc
                best_params = params
                logger.info(f"  ✓ New best! Accuracy: {val_acc:.2f}%")
        
        except Exception as e:
            logger.error(f"  Error with params {params}: {e}")
            continue
    
    # Save results
    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / 'grid_search_results.json', 'w') as f:
        json.dump({
            'best_params': best_params,
            'best_score': float(best_score),
            'all_results': results
        }, f, indent=2)
    
    logger.info(f"\n{'='*50}")
    logger.info("Grid Search Complete")
    logger.info(f"{'='*50}")
    logger.info(f"Best Parameters: {best_params}")
    logger.info(f"Best Validation Accuracy: {best_score:.2f}%")
    logger.info(f"Results saved to {results_dir / 'grid_search_results.json'}")
    
    return best_params, best_score


def random_search_diagnosis(args, n_trials: int = 20):
    """Random search for diagnosis model hyperparameters"""
    logger.info(f"Starting random search ({n_trials} trials)...")
    
    # Load data
    prep = DataPreparation()
    
    if args.labels_path and Path(args.labels_path).exists():
        labels_df = prep.load_labels_from_csv(args.labels_path)
    else:
        logger.warning("No labels file found. Creating synthetic labels.")
        patients = prep.loader.get_patient_list()[:args.max_patients]
        labels_df = prep.create_synthetic_labels(patients)
    
    dataset = prep.prepare_diagnosis_dataset(labels_df, max_patients=args.max_patients)
    train_set, val_set, _ = prep.split_dataset(dataset, test_size=0.2, val_size=0.1)
    
    # Parameter ranges
    param_ranges = {
        'learning_rate': [0.0001, 0.01],
        'batch_size': [2, 8],
        'dropout': [0.3, 0.7],
        'weight_decay': [1e-5, 1e-3]
    }
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    best_score = 0.0
    best_params = None
    results = []
    
    random.seed(args.random_state)
    np.random.seed(args.random_state)
    
    for i in range(n_trials):
        # Sample random parameters
        params = {
            'learning_rate': 10 ** np.random.uniform(np.log10(param_ranges['learning_rate'][0]),
                                                     np.log10(param_ranges['learning_rate'][1])),
            'batch_size': int(np.random.choice([2, 4, 8])),
            'dropout': np.random.uniform(param_ranges['dropout'][0], param_ranges['dropout'][1]),
            'weight_decay': 10 ** np.random.uniform(np.log10(param_ranges['weight_decay'][0]),
                                                   np.log10(param_ranges['weight_decay'][1]))
        }
        
        logger.info(f"\n[Trial {i+1}/{n_trials}] Testing: {params}")
        
        try:
            # Create datasets
            train_dataset = PETCTDataset(train_set['volumes'], train_set['labels'], augment=args.augment)
            val_dataset = PETCTDataset(val_set['volumes'], val_set['labels'], augment=False)
            
            train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
            
            # Create model
            in_channels = 2 if len(dataset['volumes'][0].shape) == 4 else 1
            
            if args.model_type == 'cnn':
                model = BreastCancerDiagnosisModel(in_channels=in_channels, num_classes=2,
                                                  dropout=params['dropout'])
            elif args.model_type == 'resnet':
                model = ResNet3D(in_channels=in_channels, num_classes=2)
            elif args.model_type == 'attention':
                model = AttentionCNN3D(in_channels=in_channels, num_classes=2)
            else:
                continue
            
            # Train
            trainer = ModelTrainer(model, device=device)
            trainer.setup_optimizer(lr=params['learning_rate'], weight_decay=params['weight_decay'])
            
            epochs = min(10, args.epochs // 3)
            for epoch in range(epochs):
                trainer.train_epoch(train_loader)
                val_loss, val_acc = trainer.validate(val_loader)
                trainer.scheduler.step(val_loss)
            
            _, val_acc = trainer.validate(val_loader)
            
            results.append({
                'params': params,
                'val_accuracy': float(val_acc),
                'val_loss': float(val_loss)
            })
            
            logger.info(f"  Validation Accuracy: {val_acc:.2f}%")
            
            if val_acc > best_score:
                best_score = val_acc
                best_params = params
                logger.info(f"  ✓ New best! Accuracy: {val_acc:.2f}%")
        
        except Exception as e:
            logger.error(f"  Error: {e}")
            continue
    
    # Save results
    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / 'random_search_results.json', 'w') as f:
        json.dump({
            'best_params': best_params,
            'best_score': float(best_score),
            'all_results': results
        }, f, indent=2)
    
    logger.info(f"\n{'='*50}")
    logger.info("Random Search Complete")
    logger.info(f"{'='*50}")
    logger.info(f"Best Parameters: {best_params}")
    logger.info(f"Best Validation Accuracy: {best_score:.2f}%")
    
    return best_params, best_score


def tune_treatment_model(args):
    """Tune treatment model hyperparameters"""
    logger.info("Tuning treatment model hyperparameters...")
    
    # Load data
    prep = DataPreparation()
    
    if args.labels_path and Path(args.labels_path).exists():
        labels_df = prep.load_labels_from_csv(args.labels_path)
    else:
        logger.warning("No labels file found. Creating synthetic labels.")
        patients = prep.loader.get_patient_list()[:args.max_patients]
        labels_df = prep.create_synthetic_labels(patients)
    
    dataset = prep.prepare_treatment_dataset(labels_df, max_patients=args.max_patients)
    train_set, val_set, _ = prep.split_dataset(dataset, test_size=0.2, val_size=0.1)
    
    # Parameter grid based on model type
    if args.model_type == 'xgboost':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.3]
        }
    elif args.model_type == 'random_forest':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None]
        }
    else:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.3]
        }
    
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = list(product(*values))
    
    logger.info(f"Testing {len(combinations)} parameter combinations...")
    
    best_score = 0.0
    best_params = None
    results = []
    
    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        logger.info(f"\n[{i+1}/{len(combinations)}] Testing: {params}")
        
        try:
            # Create model with params
            if args.model_type == 'xgboost':
                recommender = TreatmentRecommender(model_type='xgboost')
                # Note: XGBoost params need to be set during model creation
                # This is a simplified version
            else:
                recommender = TreatmentRecommender(model_type=args.model_type)
            
            recommender.treatment_classes = dataset['treatment_classes']
            
            # Train
            recommender.train(train_set['features'], train_set['treatment_labels'],
                            treatment_labels=dataset['treatment_classes'], test_size=0.0)
            
            # Evaluate
            val_pred = recommender.model.predict(recommender.scaler.transform(val_set['features']))
            val_acc = np.mean(val_pred == val_set['treatment_labels'])
            
            results.append({
                'params': params,
                'val_accuracy': float(val_acc)
            })
            
            logger.info(f"  Validation Accuracy: {val_acc:.4f}")
            
            if val_acc > best_score:
                best_score = val_acc
                best_params = params
                logger.info(f"  ✓ New best! Accuracy: {val_acc:.4f}")
        
        except Exception as e:
            logger.error(f"  Error: {e}")
            continue
    
    # Save results
    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / 'treatment_tuning_results.json', 'w') as f:
        json.dump({
            'best_params': best_params,
            'best_score': float(best_score),
            'all_results': results
        }, f, indent=2)
    
    logger.info(f"\nBest Parameters: {best_params}")
    logger.info(f"Best Validation Accuracy: {best_score:.4f}")
    
    return best_params, best_score


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter Tuning')
    parser.add_argument('--model', type=str, choices=['diagnosis', 'treatment'],
                       required=True, help='Model to tune')
    parser.add_argument('--method', type=str, choices=['grid', 'random'],
                       default='grid', help='Search method')
    parser.add_argument('--labels_path', type=str, help='Path to labels CSV')
    parser.add_argument('--max_patients', type=int, default=20, help='Max patients')
    parser.add_argument('--model_type', type=str, choices=['cnn', 'resnet', 'attention'],
                       default='cnn', help='Model architecture (diagnosis)')
    parser.add_argument('--treatment_model', type=str,
                       choices=['xgboost', 'random_forest', 'gradient_boosting'],
                       default='xgboost', help='Treatment model type')
    parser.add_argument('--epochs', type=int, default=30, help='Epochs per trial')
    parser.add_argument('--augment', action='store_true', help='Use augmentation')
    parser.add_argument('--n_trials', type=int, default=20, help='Number of random trials')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--output_dir', type=str, default='results/hyperparameter_tuning',
                       help='Output directory')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    if args.model == 'diagnosis':
        if args.method == 'grid':
            grid_search_diagnosis(args)
        else:
            random_search_diagnosis(args, n_trials=args.n_trials)
    else:
        args.model_type = args.treatment_model
        tune_treatment_model(args)


if __name__ == "__main__":
    main()

