"""
Model Comparison Tool
Compares different model configurations and selects the best
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compare_diagnosis_models(results_dir: str = 'results'):
    """Compare different diagnosis model configurations"""
    results_dir = Path(results_dir)
    
    # Find all diagnosis model results
    model_results = []
    
    for model_dir in results_dir.glob('diagnosis*'):
        metrics_file = model_dir / 'test_metrics.json'
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            # Extract model info from directory name
            model_name = model_dir.name
            model_type = 'unknown'
            if 'cnn' in model_name.lower():
                model_type = 'CNN'
            elif 'resnet' in model_name.lower():
                model_type = 'ResNet-3D'
            elif 'attention' in model_name.lower():
                model_type = 'Attention-CNN'
            
            model_results.append({
                'model_name': model_name,
                'model_type': model_type,
                'accuracy': metrics.get('accuracy', 0),
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'f1_score': metrics.get('f1_score', 0),
                'roc_auc': metrics.get('roc_auc', 0),
                'sensitivity': metrics.get('sensitivity', 0),
                'specificity': metrics.get('specificity', 0)
            })
    
    if not model_results:
        logger.warning("No diagnosis model results found")
        return None
    
    df = pd.DataFrame(model_results)
    df = df.sort_values('accuracy', ascending=False)
    
    print("\n" + "="*80)
    print("Diagnosis Model Comparison")
    print("="*80)
    print(df.to_string(index=False))
    print("\nBest Model:", df.iloc[0]['model_name'])
    print("Best Accuracy:", f"{df.iloc[0]['accuracy']:.4f}")
    
    return df


def compare_treatment_models(results_dir: str = 'results'):
    """Compare different treatment model configurations"""
    results_dir = Path(results_dir)
    
    model_results = []
    
    for model_dir in results_dir.glob('treatment*'):
        metrics_file = model_dir / 'test_metrics.json'
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            model_name = model_dir.name
            model_type = 'unknown'
            if 'xgboost' in model_name.lower():
                model_type = 'XGBoost'
            elif 'random_forest' in model_name.lower() or 'rf' in model_name.lower():
                model_type = 'Random Forest'
            elif 'gradient' in model_name.lower() or 'gb' in model_name.lower():
                model_type = 'Gradient Boosting'
            elif 'logistic' in model_name.lower():
                model_type = 'Logistic Regression'
            
            model_results.append({
                'model_name': model_name,
                'model_type': model_type,
                'accuracy': metrics.get('accuracy', 0),
                'precision_macro': metrics.get('precision_macro', 0),
                'recall_macro': metrics.get('recall_macro', 0),
                'f1_macro': metrics.get('f1_macro', 0),
                'precision_weighted': metrics.get('precision_weighted', 0),
                'recall_weighted': metrics.get('recall_weighted', 0),
                'f1_weighted': metrics.get('f1_weighted', 0)
            })
    
    if not model_results:
        logger.warning("No treatment model results found")
        return None
    
    df = pd.DataFrame(model_results)
    df = df.sort_values('accuracy', ascending=False)
    
    print("\n" + "="*80)
    print("Treatment Model Comparison")
    print("="*80)
    print(df.to_string(index=False))
    print("\nBest Model:", df.iloc[0]['model_name'])
    print("Best Accuracy:", f"{df.iloc[0]['accuracy']:.4f}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Compare Models')
    parser.add_argument('--model_type', type=str, choices=['diagnosis', 'treatment', 'all'],
                       default='all', help='Type of models to compare')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Results directory')
    parser.add_argument('--output', type=str, help='Output file for comparison')
    
    args = parser.parse_args()
    
    results = {}
    
    if args.model_type in ['diagnosis', 'all']:
        diag_df = compare_diagnosis_models(args.results_dir)
        if diag_df is not None:
            results['diagnosis'] = diag_df.to_dict('records')
            if args.output:
                diag_df.to_csv(Path(args.output).parent / 'diagnosis_comparison.csv', index=False)
    
    if args.model_type in ['treatment', 'all']:
        treat_df = compare_treatment_models(args.results_dir)
        if treat_df is not None:
            results['treatment'] = treat_df.to_dict('records')
            if args.output:
                treat_df.to_csv(Path(args.output).parent / 'treatment_comparison.csv', index=False)
    
    if args.output and results:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Comparison saved to {args.output}")


if __name__ == "__main__":
    main()

