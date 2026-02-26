import argparse
import sys
import torch
import warnings
from pathlib import Path

# Suppress minor warnings for clean output
warnings.filterwarnings('ignore')

from src.config import Config
from src.data.collection import (
    check_tbx11k_availability, 
    process_tbx11k_images, 
    copy_images_enhanced, 
    collection_stats
)
from src.training.pipeline import TBPipeline
from src.evaluation.visualization import PatientVisualizationPipeline

def download_data(config: Config):
    """Data collection phase"""
    print("="*50)
    print("Starting Data Collection")
    print("="*50)
    config.create_dirs()
    
    # 1. TBX11K Tuberculosis Processing
    print("\n1. Processing TBX11K Dataset...")
    tbx_info = check_tbx11k_availability()
    if tbx_info['metadata_path']:
        print(f"Found TBX11K: {tbx_info['tb']} TB, {tbx_info['healthy']} Healthy")
        process_tbx11k_images(
            metadata_path=tbx_info['metadata_path'],
            target_class='tuberculosis',
            image_type_filter='tb',
            dest_dir=os.path.join(config.DATASET_PATH, 'tuberculosis'),
            limit=1000
        )
    else:
        print("TBX11K metadata not found in expected Kaggle paths. Ensure dataset is attached.")
        
    # Additional dataset collection would follow here...
    print("Data collection completed.")


def train_model(config: Config):
    """Execute the full 3-phase training pipeline."""
    print("="*50)
    print("Starting Model Training Pipeline")
    print("="*50)
    
    pipeline = TBPipeline(config)
    final_model = pipeline.execute()
    
    # Save the final model
    save_path = Path(config.OUTPUT_DIR) / "final_tb_model.pth"
    torch.save(final_model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


def evaluate_model(config: Config):
    """Run evaluation and generate visualizations on a test set."""
    print("="*50)
    print("Starting Evaluation & Visualization")
    print("="*50)
    
    print("Evaluation module loaded. (Implementation depends on loaded model weights).")
    

def main():
    parser = argparse.ArgumentParser(description="TB X-Ray Classification Pipeline")
    parser.add_argument("--data", action="store_true", help="Run data collection and preprocessing")
    parser.add_argument("--train", action="store_true", help="Run the full 3-phase training pipeline")
    parser.add_argument("--eval", action="store_true", help="Run evaluation and Grad-CAM visualizations")
    parser.add_argument("--all", action="store_true", help="Run all phases sequentially")
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    # Initialize Configuration
    config = Config()
    config.create_dirs()
    
    if args.data or args.all:
        download_data(config)
        
    if args.train or args.all:
        train_model(config)
        
    if args.eval or args.all:
        evaluate_model(config)

if __name__ == "__main__":
    main()
