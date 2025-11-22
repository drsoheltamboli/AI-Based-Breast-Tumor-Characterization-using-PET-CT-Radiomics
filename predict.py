"""
Prediction Script for Unknown PET-CT Images
Loads a trained model and predicts diagnosis for new patient scans
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np
import logging
from typing import Dict, Optional

from src.data_loader import DICOMLoader
from src.preprocessing import Preprocessor
from src.models.diagnosis_models import BreastCancerDiagnosisModel, ResNet3D, AttentionCNN3D, PETCTDataset
from src.utils.visualization import visualize_scan
from src.feature_extraction import FeatureExtractor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_trained_model(model_path: str, model_type: str = 'cnn', device: str = 'cpu'):
    """
    Load a trained diagnosis model
    
    Args:
        model_path: Path to saved model checkpoint
        model_type: Type of model ('cnn', 'resnet', 'attention')
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    in_channels = checkpoint.get('in_channels', 2)
    
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
    
    logger.info(f"Loaded {model_type} model from {model_path}")
    logger.info(f"Model was trained for {checkpoint.get('epoch', 'unknown')} epochs")
    logger.info(f"Validation accuracy: {checkpoint.get('val_acc', 'unknown'):.2f}%")
    
    return model


def predict_patient(model, patient_id: str, device: str = 'cpu', 
                   visualize: bool = True, output_dir: str = 'results/predictions'):
    """
    Predict diagnosis for a patient
    
    Args:
        model: Trained model
        patient_id: Patient ID or path to patient folder
        device: Device to run inference on
        visualize: Whether to create visualization
        output_dir: Directory to save results
        
    Returns:
        Dictionary with prediction results
    """
    logger.info(f"Processing patient: {patient_id}")
    
    # Load data
    loader = DICOMLoader()
    preprocessor = Preprocessor()
    extractor = FeatureExtractor()
    
    try:
        # Try loading as patient ID first
        if Path(patient_id).exists():
            # It's a path
            scans = loader.load_patient_scans_from_path(patient_id)
        else:
            # It's a patient ID
            scans = loader.load_patient_scans(patient_id)
    except Exception as e:
        logger.error(f"Error loading patient data: {e}")
        return None
    
    # Preprocess
    processed = preprocessor.process_scans(scans)
    
    if not processed['fused_scans']:
        logger.warning("No fused CT-PET scans available. Using CT only.")
        if processed['ct_scans']:
            volume = processed['ct_scans'][0]['volume']
            volume = np.expand_dims(volume, axis=0)  # Add channel dimension
        else:
            logger.error("No scans available for prediction")
            return None
    else:
        volume = processed['fused_scans'][0]['fused_volume']
    
    # Extract features for additional analysis
    if volume.ndim == 4:
        ct_volume = volume[0]
        pet_volume = volume[1]
    else:
        ct_volume = volume[0] if volume.shape[0] == 1 else volume
    
    features = extractor.extract_all_features(ct_volume, modality='CT')
    
    # Prepare volume for model
    if volume.ndim == 3:
        volume_tensor = torch.FloatTensor(volume).unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    else:
        volume_tensor = torch.FloatTensor(volume).unsqueeze(0)  # Add batch dim
    
    volume_tensor = volume_tensor.to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(volume_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
    
    # Get results
    prob_benign = probabilities[0][0].item()
    prob_malignant = probabilities[0][1].item()
    prediction = predicted[0].item()
    
    diagnosis = "Malignant" if prediction == 1 else "Benign"
    confidence = prob_malignant if prediction == 1 else prob_benign
    
    results = {
        'patient_id': patient_id,
        'diagnosis': diagnosis,
        'prediction': int(prediction),
        'confidence': float(confidence),
        'probabilities': {
            'benign': float(prob_benign),
            'malignant': float(prob_malignant)
        },
        'features': {
            'intensity_mean': features['intensity'].get('mean', 0),
            'intensity_max': features['intensity'].get('max', 0),
            'suv_max': features['intensity'].get('suv_max', 0),
            'texture_contrast': features['texture'].get('contrast', 0),
            'texture_homogeneity': features['texture'].get('homogeneity', 0),
        }
    }
    
    # Print results
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"Patient ID: {patient_id}")
    print(f"\nDiagnosis: {diagnosis}")
    print(f"Confidence: {confidence*100:.2f}%")
    print(f"\nProbabilities:")
    print(f"  Benign: {prob_benign*100:.2f}%")
    print(f"  Malignant: {prob_malignant*100:.2f}%")
    print(f"\nKey Features:")
    print(f"  Intensity Mean: {results['features']['intensity_mean']:.4f}")
    print(f"  Intensity Max: {results['features']['intensity_max']:.4f}")
    if results['features']['suv_max'] > 0:
        print(f"  SUV Max: {results['features']['suv_max']:.4f}")
    print(f"  Texture Contrast: {results['features']['texture_contrast']:.4f}")
    print(f"  Texture Homogeneity: {results['features']['texture_homogeneity']:.4f}")
    print("="*60)
    
    # Visualize if requested
    if visualize:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        vis_path = output_path / f'{patient_id.replace("/", "_")}_prediction.png'
        visualize_scan(volume, modality='FUSED' if volume.ndim == 4 else 'CT',
                      slice_idx=volume.shape[1]//2 if volume.ndim == 4 else volume.shape[0]//2,
                      save_path=str(vis_path))
        logger.info(f"Visualization saved to {vis_path}")
    
    return results


def predict_from_folder(model, folder_path: str, device: str = 'cpu', 
                       visualize: bool = True, output_dir: str = 'results/predictions'):
    """
    Predict for all patients in a folder
    
    Args:
        model: Trained model
        folder_path: Path to folder containing patient data
        device: Device to run inference on
        visualize: Whether to create visualizations
        output_dir: Directory to save results
        
    Returns:
        List of prediction results
    """
    folder_path = Path(folder_path)
    loader = DICOMLoader(base_path=str(folder_path))
    patients = loader.get_patient_list()
    
    all_results = []
    
    for patient_id in patients:
        try:
            result = predict_patient(model, patient_id, device, visualize, output_dir)
            if result:
                all_results.append(result)
        except Exception as e:
            logger.error(f"Error predicting for {patient_id}: {e}")
            continue
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Predict Diagnosis for PET-CT Scans')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--model_type', type=str, choices=['cnn', 'resnet', 'attention'],
                       default='cnn', help='Type of model')
    parser.add_argument('--patient_id', type=str,
                       help='Patient ID to predict (e.g., QIN-BREAST-01-0001)')
    parser.add_argument('--patient_folder', type=str,
                       help='Path to patient folder (alternative to patient_id)')
    parser.add_argument('--folder', type=str,
                       help='Path to folder containing multiple patients')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu or cuda)')
    parser.add_argument('--no_visualize', action='store_true',
                       help='Skip visualization')
    parser.add_argument('--output_dir', type=str, default='results/predictions',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = load_trained_model(args.model_path, args.model_type, args.device)
    
    # Make predictions
    if args.folder:
        # Predict for all patients in folder
        logger.info(f"Predicting for all patients in {args.folder}")
        results = predict_from_folder(model, args.folder, args.device, 
                                     not args.no_visualize, args.output_dir)
        logger.info(f"\nPredicted for {len(results)} patients")
        
        # Summary
        malignant_count = sum(1 for r in results if r['prediction'] == 1)
        benign_count = len(results) - malignant_count
        print(f"\nSummary:")
        print(f"  Total patients: {len(results)}")
        print(f"  Predicted Malignant: {malignant_count}")
        print(f"  Predicted Benign: {benign_count}")
    
    elif args.patient_id or args.patient_folder:
        # Predict for single patient
        patient_id = args.patient_id or args.patient_folder
        result = predict_patient(model, patient_id, args.device, 
                               not args.no_visualize, args.output_dir)
        
        if result:
            # Save results to file
            import json
            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            result_file = output_path / f'{patient_id.replace("/", "_")}_prediction.json'
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Results saved to {result_file}")
    else:
        logger.error("Please provide --patient_id, --patient_folder, or --folder")
        return


if __name__ == "__main__":
    main()

