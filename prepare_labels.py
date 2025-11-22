"""
Label Preparation Tool
Helps prepare medical labels in the correct format for training
"""

import argparse
import pandas as pd
from pathlib import Path
import json
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_label_template(output_path: str, patient_ids: Optional[List[str]] = None):
    """
    Create a template CSV file for labels
    
    Args:
        output_path: Path to save template CSV
        patient_ids: Optional list of patient IDs to include
    """
    from src.data_loader import DICOMLoader
    
    if patient_ids is None:
        loader = DICOMLoader()
        patient_ids = loader.get_patient_list()
    
    # Create template DataFrame
    template_data = {
        'patient_id': patient_ids,
        'diagnosis': [None] * len(patient_ids),  # 0=benign, 1=malignant
        'stage': [None] * len(patient_ids),      # 0-4
        'grade': [None] * len(patient_ids),      # 1-3
        'er_positive': [None] * len(patient_ids),  # 0 or 1
        'pr_positive': [None] * len(patient_ids),  # 0 or 1
        'her2_positive': [None] * len(patient_ids),  # 0 or 1
        'age': [None] * len(patient_ids),
        'treatment': [None] * len(patient_ids),  # hormonal, chemotherapy, targeted, combination
        'outcome': [None] * len(patient_ids)      # 0=poor, 1=good
    }
    
    df = pd.DataFrame(template_data)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Created label template with {len(patient_ids)} patients")
    logger.info(f"Template saved to: {output_path}")
    logger.info("\nPlease fill in the labels:")
    logger.info("  - diagnosis: 0 (benign) or 1 (malignant)")
    logger.info("  - stage: 0-4 (cancer stage)")
    logger.info("  - grade: 1-3 (tumor grade)")
    logger.info("  - er_positive, pr_positive, her2_positive: 0 or 1")
    logger.info("  - age: patient age")
    logger.info("  - treatment: hormonal, chemotherapy, targeted, or combination")
    logger.info("  - outcome: 0 (poor) or 1 (good)")
    
    return df


def validate_labels(labels_path: str) -> Dict:
    """
    Validate label file format and content
    
    Args:
        labels_path: Path to labels CSV file
        
    Returns:
        Dictionary with validation results
    """
    df = pd.read_csv(labels_path)
    
    required_columns = ['patient_id', 'diagnosis']
    missing = [col for col in required_columns if col not in df.columns]
    
    validation_results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    # Check required columns
    if missing:
        validation_results['valid'] = False
        validation_results['errors'].append(f"Missing required columns: {missing}")
        return validation_results
    
    # Check patient_id
    if df['patient_id'].isna().any():
        validation_results['errors'].append("patient_id contains missing values")
        validation_results['valid'] = False
    
    # Check diagnosis
    if 'diagnosis' in df.columns:
        if df['diagnosis'].isna().any():
            validation_results['warnings'].append(f"{df['diagnosis'].isna().sum()} missing diagnosis values")
        else:
            invalid = df[~df['diagnosis'].isin([0, 1])]
            if len(invalid) > 0:
                validation_results['errors'].append(f"Invalid diagnosis values (must be 0 or 1): {len(invalid)} rows")
                validation_results['valid'] = False
        
        validation_results['stats']['diagnosis'] = {
            'benign': int((df['diagnosis'] == 0).sum()),
            'malignant': int((df['diagnosis'] == 1).sum())
        }
    
    # Check stage
    if 'stage' in df.columns:
        invalid = df[~df['stage'].isin([0, 1, 2, 3, 4]) & df['stage'].notna()]
        if len(invalid) > 0:
            validation_results['warnings'].append(f"Invalid stage values (should be 0-4): {len(invalid)} rows")
    
    # Check grade
    if 'grade' in df.columns:
        invalid = df[~df['grade'].isin([1, 2, 3]) & df['grade'].notna()]
        if len(invalid) > 0:
            validation_results['warnings'].append(f"Invalid grade values (should be 1-3): {len(invalid)} rows")
    
    # Check binary columns
    binary_cols = ['er_positive', 'pr_positive', 'her2_positive', 'outcome']
    for col in binary_cols:
        if col in df.columns:
            invalid = df[~df[col].isin([0, 1]) & df[col].notna()]
            if len(invalid) > 0:
                validation_results['warnings'].append(f"Invalid {col} values (should be 0 or 1): {len(invalid)} rows")
    
    # Statistics
    validation_results['stats']['total_patients'] = len(df)
    validation_results['stats']['labeled_patients'] = int(df['diagnosis'].notna().sum()) if 'diagnosis' in df.columns else 0
    
    return validation_results


def merge_labels(existing_labels_path: str, new_labels_path: str, output_path: str):
    """
    Merge new labels with existing labels
    
    Args:
        existing_labels_path: Path to existing labels file
        new_labels_path: Path to new labels file
        output_path: Path to save merged labels
    """
    existing = pd.read_csv(existing_labels_path)
    new = pd.read_csv(new_labels_path)
    
    # Merge on patient_id
    merged = pd.merge(existing, new, on='patient_id', how='outer', suffixes=('_old', '_new'))
    
    # Prefer new values over old
    for col in new.columns:
        if col != 'patient_id':
            if f'{col}_new' in merged.columns:
                merged[col] = merged[f'{col}_new'].fillna(merged.get(f'{col}_old', None))
                merged = merged.drop(columns=[f'{col}_old', f'{col}_new'], errors='ignore')
    
    merged.to_csv(output_path, index=False)
    logger.info(f"Merged labels saved to: {output_path}")
    logger.info(f"Total patients: {len(merged)}")
    
    return merged


def export_for_clinical_review(labels_path: str, output_path: str):
    """
    Export labels in a format suitable for clinical review
    
    Args:
        labels_path: Path to labels CSV
        output_path: Path to save review file
    """
    df = pd.read_csv(labels_path)
    
    # Create review-friendly format
    review_data = []
    for _, row in df.iterrows():
        review_data.append({
            'Patient ID': row['patient_id'],
            'Diagnosis': 'Malignant' if row.get('diagnosis') == 1 else 'Benign' if row.get('diagnosis') == 0 else 'Unknown',
            'Stage': row.get('stage', 'N/A'),
            'Grade': row.get('grade', 'N/A'),
            'ER Status': 'Positive' if row.get('er_positive') == 1 else 'Negative' if row.get('er_positive') == 0 else 'Unknown',
            'PR Status': 'Positive' if row.get('pr_positive') == 1 else 'Negative' if row.get('pr_positive') == 0 else 'Unknown',
            'HER2 Status': 'Positive' if row.get('her2_positive') == 1 else 'Negative' if row.get('her2_positive') == 0 else 'Unknown',
            'Age': row.get('age', 'N/A'),
            'Treatment': row.get('treatment', 'N/A'),
            'Outcome': 'Good' if row.get('outcome') == 1 else 'Poor' if row.get('outcome') == 0 else 'Unknown'
        })
    
    review_df = pd.DataFrame(review_data)
    review_df.to_csv(output_path, index=False)
    logger.info(f"Clinical review file saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Label Preparation Tool')
    parser.add_argument('--action', type=str, 
                       choices=['create_template', 'validate', 'merge', 'export_review'],
                       required=True, help='Action to perform')
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--input', type=str, help='Input file path')
    parser.add_argument('--existing', type=str, help='Existing labels file (for merge)')
    parser.add_argument('--max_patients', type=int, help='Max patients for template')
    
    args = parser.parse_args()
    
    if args.action == 'create_template':
        from src.data_loader import DICOMLoader
        loader = DICOMLoader()
        patients = loader.get_patient_list()
        if args.max_patients:
            patients = patients[:args.max_patients]
        
        output = args.output or 'data/labels_template.csv'
        create_label_template(output, patients)
    
    elif args.action == 'validate':
        input_path = args.input or 'data/labels.csv'
        results = validate_labels(input_path)
        
        print("\n" + "="*50)
        print("Label Validation Results")
        print("="*50)
        print(f"Valid: {results['valid']}")
        print(f"\nStatistics:")
        for key, value in results['stats'].items():
            print(f"  {key}: {value}")
        
        if results['errors']:
            print(f"\nErrors ({len(results['errors'])}):")
            for error in results['errors']:
                print(f"  - {error}")
        
        if results['warnings']:
            print(f"\nWarnings ({len(results['warnings'])}):")
            for warning in results['warnings']:
                print(f"  - {warning}")
        
        if results['valid']:
            print("\n[OK] Labels are valid and ready for training!")
        else:
            print("\n[ERROR] Please fix errors before training")
    
    elif args.action == 'merge':
        if not args.existing or not args.input:
            print("Error: --existing and --input required for merge")
            return
        output = args.output or 'data/labels_merged.csv'
        merge_labels(args.existing, args.input, output)
    
    elif args.action == 'export_review':
        input_path = args.input or 'data/labels.csv'
        output = args.output or 'data/labels_for_review.csv'
        export_for_clinical_review(input_path, output)


if __name__ == "__main__":
    main()

