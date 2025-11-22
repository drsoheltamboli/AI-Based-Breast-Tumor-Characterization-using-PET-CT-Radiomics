"""
Main script for Precision Medicine AI - Breast Cancer Diagnosis and Treatment
"""

import argparse
import sys
from pathlib import Path
import logging

from src.data_loader import DICOMLoader
from src.preprocessing import Preprocessor
from src.feature_extraction import FeatureExtractor
from src.models.diagnosis_models import BreastCancerDiagnosisModel, ModelTrainer, PETCTDataset
from src.models.treatment_models import TreatmentRecommender
from src.utils.visualization import visualize_scan, plot_training_history
from src.utils.metrics import calculate_diagnosis_metrics, print_metrics

import torch
from torch.utils.data import DataLoader
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Precision Medicine AI for Breast Cancer')
    parser.add_argument('--mode', type=str, choices=['explore', 'extract', 'train_diagnosis', 
                                                     'train_treatment', 'predict'],
                       default='explore', help='Operation mode')
    parser.add_argument('--patient_id', type=str, help='Patient ID to process')
    parser.add_argument('--max_patients', type=int, default=10, 
                       help='Maximum number of patients to process')
    parser.add_argument('--model_path', type=str, help='Path to saved model')
    parser.add_argument('--output_dir', type=str, default='results', 
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    if args.mode == 'explore':
        explore_data(args.max_patients)
    elif args.mode == 'extract':
        extract_features(args.max_patients)
    elif args.mode == 'train_diagnosis':
        train_diagnosis_model(args.max_patients)
    elif args.mode == 'train_treatment':
        train_treatment_model()
    elif args.mode == 'predict':
        predict_patient(args.patient_id, args.model_path)


def explore_data(max_patients=10):
    """Explore the dataset"""
    logger.info("Exploring dataset...")
    
    loader = DICOMLoader()
    patients = loader.get_patient_list()
    
    print(f"\nFound {len(patients)} patients in dataset")
    print(f"Exploring first {min(max_patients, len(patients))} patients...\n")
    
    for i, patient_id in enumerate(patients[:max_patients]):
        try:
            scans = loader.load_patient_scans(patient_id)
            print(f"{i+1}. {patient_id}:")
            print(f"   CT scans: {len(scans['ct_scans'])}")
            print(f"   PET scans: {len(scans['pet_scans'])}")
            
            if scans['ct_scans']:
                vol_shape = scans['ct_scans'][0]['data']['volume'].shape
                print(f"   CT volume shape: {vol_shape}")
        except Exception as e:
            logger.error(f"Error processing {patient_id}: {e}")


def extract_features(max_patients=10):
    """Extract features from scans"""
    logger.info("Extracting features...")
    
    loader = DICOMLoader()
    preprocessor = Preprocessor()
    extractor = FeatureExtractor()
    
    patients = loader.get_patient_list()[:max_patients]
    all_features = []
    
    for patient_id in patients:
        try:
            scans = loader.load_patient_scans(patient_id)
            processed = preprocessor.process_scans(scans)
            
            if processed['fused_scans']:
                volume = processed['fused_scans'][0]['fused_volume']
                features = extractor.extract_features_vector(volume, modality='FUSED')
                all_features.append({
                    'patient_id': patient_id,
                    'features': features
                })
                logger.info(f"Extracted features for {patient_id}: {features.shape}")
        except Exception as e:
            logger.error(f"Error extracting features for {patient_id}: {e}")
    
    # Save features
    output_path = Path('data/features')
    output_path.mkdir(parents=True, exist_ok=True)
    np.save(output_path / 'features.npy', all_features)
    logger.info(f"Saved features to {output_path / 'features.npy'}")


def train_diagnosis_model(max_patients=10):
    """Train diagnosis model"""
    logger.info("Training diagnosis model...")
    
    # Note: This is a simplified example. In practice, you would need:
    # 1. Ground truth labels for each patient
    # 2. Proper train/val/test split
    # 3. More data preprocessing
    
    logger.warning("Training requires labeled data. This is a template implementation.")
    logger.info("Please ensure you have ground truth labels before training.")
    
    # Example structure (would need actual labels)
    # loader = DICOMLoader()
    # preprocessor = Preprocessor()
    # 
    # # Load and preprocess data
    # volumes = []
    # labels = []  # Would need actual labels
    # 
    # # Create dataset
    # dataset = PETCTDataset(volumes, labels)
    # dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    # 
    # # Create model
    # model = BreastCancerDiagnosisModel(in_channels=2, num_classes=2)
    # trainer = ModelTrainer(model)
    # trainer.setup_optimizer()
    # 
    # # Train (would need validation set)
    # trainer.train(dataloader, val_dataloader, epochs=50)


def train_treatment_model():
    """Train treatment recommendation model"""
    logger.info("Training treatment model...")
    logger.warning("Training requires labeled treatment data. This is a template implementation.")
    
    # Example structure (would need actual treatment labels)
    # recommender = TreatmentRecommender(model_type='xgboost')
    # 
    # # Prepare features (would need actual data)
    # X = ...  # Feature matrix
    # y = ...  # Treatment labels
    # 
    # # Train
    # recommender.train(X, y)


def predict_patient(patient_id, model_path):
    """Predict diagnosis and treatment for a patient"""
    if not patient_id:
        logger.error("Please provide --patient_id")
        return
    
    logger.info(f"Predicting for patient {patient_id}...")
    
    # Load data
    loader = DICOMLoader()
    preprocessor = Preprocessor()
    extractor = FeatureExtractor()
    
    try:
        scans = loader.load_patient_scans(patient_id)
        processed = preprocessor.process_scans(scans)
        
        if processed['fused_scans']:
            volume = processed['fused_scans'][0]['fused_volume']
            
            # Extract features
            features = extractor.extract_all_features(volume[0], modality='CT')
            
            # Visualize
            visualize_scan(volume, modality='FUSED', 
                         save_path=f'results/{patient_id}_scan.png')
            
            logger.info(f"Processed patient {patient_id}")
            logger.info(f"Extracted {len(features)} feature groups")
            
    except Exception as e:
        logger.error(f"Error processing {patient_id}: {e}")


if __name__ == "__main__":
    main()

