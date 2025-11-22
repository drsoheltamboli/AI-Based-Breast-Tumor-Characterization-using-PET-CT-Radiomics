# Catalyzing Precision Medicine: AI-Powered Breast Cancer Diagnosis & Treatment

## Project Overview

This project leverages advanced AI and machine learning algorithms to analyze PET-CT scans for breast cancer diagnosis and provide personalized pharmaceutical treatment recommendations. The system combines deep learning models for image analysis with predictive algorithms for treatment strategy optimization.

## Features

- **PET-CT Scan Analysis**: Advanced image processing and feature extraction from DICOM files
- **Deep Learning Diagnosis**: 3D CNN and transformer-based models for cancer detection and classification
- **Treatment Recommendation**: ML algorithms for personalized pharmaceutical treatment strategies
- **Radiomics Features**: Extraction of quantitative imaging features for precision medicine
- **Data Pipeline**: Automated preprocessing and augmentation for medical imaging data

## Project Structure

```
QIN-BREAST/
├── data/
│   ├── raw/              # Original DICOM files
│   ├── processed/        # Preprocessed images
│   └── features/         # Extracted features
├── src/
│   ├── data_loader.py    # DICOM data loading utilities
│   ├── preprocessing.py  # Image preprocessing pipeline
│   ├── feature_extraction.py  # Radiomics and deep features
│   ├── models/
│   │   ├── diagnosis_models.py  # Cancer diagnosis models
│   │   └── treatment_models.py  # Treatment recommendation models
│   └── utils/
│       ├── visualization.py    # Visualization utilities
│       └── metrics.py          # Evaluation metrics
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_extraction.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_treatment_recommendation.ipynb
├── models/               # Saved model checkpoints
├── results/              # Training results and predictions
└── config/               # Configuration files
```

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preprocessing
```python
from src.data_loader import DICOMLoader
from src.preprocessing import Preprocessor

loader = DICOMLoader()
scans = loader.load_patient_scans("QIN-BREAST-01-0001")
preprocessor = Preprocessor()
processed = preprocessor.process_scans(scans)
```

### Model Training
```python
from src.models.diagnosis_models import BreastCancerDiagnosisModel

model = BreastCancerDiagnosisModel()
model.train(train_data, validation_data)
predictions = model.predict(test_data)
```

### Treatment Recommendation
```python
from src.models.treatment_models import TreatmentRecommender

recommender = TreatmentRecommender()
recommendations = recommender.recommend(patient_features, diagnosis_results)
```

## Dataset

This project uses the QIN-BREAST dataset from The Cancer Imaging Archive (TCIA), containing PET-CT scans of breast cancer patients. The dataset includes:
- CT Attenuation Correction (CTAC) scans
- PET Attenuation Corrected 3D Whole Body (PET AC 3DWB) scans
- Multiple time points for longitudinal analysis

## Model Architecture

### Diagnosis Models
- **3D CNN**: For volumetric PET-CT analysis
- **ResNet-3D**: Deep residual networks for feature extraction
- **Attention Mechanisms**: Focus on relevant regions
- **Multi-modal Fusion**: Combining CT and PET information

### Treatment Models
- **Random Forest**: For interpretable treatment recommendations
- **Gradient Boosting**: XGBoost for treatment outcome prediction
- **Neural Networks**: Deep learning for complex treatment patterns

## Training and Validation

### Quick Start

1. **Prepare data:**
```bash
# Generate synthetic labels for testing (or use your own labels CSV)
python -c "from src.data_preparation import DataPreparation; prep = DataPreparation(); patients = prep.loader.get_patient_list()[:20]; prep.create_synthetic_labels(patients, 'data/labels.csv')"
```

2. **Train models:**
```bash
# Train diagnosis model
python train_diagnosis.py --labels_path data/labels.csv --model_type cnn --epochs 30

# Train treatment model
python train_treatment.py --labels_path data/labels.csv --model_type xgboost
```

3. **Run complete pipeline:**
```bash
python run_complete_pipeline.py --max_patients 20 --run_cv
```

### Detailed Training Guide

See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for comprehensive instructions on:
- Data preparation and label format
- Model training with different architectures
- Cross-validation
- Model evaluation
- Hyperparameter tuning

## Results

Model performance metrics and treatment recommendation accuracy will be documented here after training.

## Citation

If you use this code or dataset, please cite:
- QIN-BREAST Dataset from TCIA
- This project repository

## Next Steps Tools

### Label Preparation
```bash
# Create label template
python prepare_labels.py --action create_template --output data/labels_template.csv

# Validate labels
python prepare_labels.py --action validate --input data/labels.csv
```

### Hyperparameter Tuning
```bash
# Grid search
python tune_hyperparameters.py --model diagnosis --method grid --labels_path data/labels.csv

# Random search
python tune_hyperparameters.py --model diagnosis --method random --n_trials 30 --labels_path data/labels.csv
```

### Full Dataset Training
```bash
python train_full_dataset.py --labels_path data/labels.csv --train_diagnosis --train_treatment --epochs 100
```

### Clinical Validation
```bash
# Generate clinical report
python clinical_validation.py --action generate_report --diagnosis_metrics results/diagnosis/test_metrics.json --treatment_metrics results/treatment/test_metrics.json

# Compare models
python compare_models.py --model_type all
```

See [NEXT_STEPS_COMPLETE.md](NEXT_STEPS_COMPLETE.md) for detailed instructions.

## License

This project follows the Creative Commons Attribution 3.0 Unported License as specified in the QIN-BREAST dataset license.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on the repository.

