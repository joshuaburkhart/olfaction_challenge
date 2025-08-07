#!/usr/bin/env python3
"""
Intensity DREAM Model - Advanced GPU-Accelerated Version
A machine learning pipeline for predicting molecular intensity from molecular descriptors.
Supports both GPU (CUDA) and CPU execution with automatic fallback.

Based on TASK1 challenge for intensity prediction.
"""

import os
import sys
import argparse
import warnings
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import StackingRegressor
from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb
import joblib

# GPU acceleration imports with fallbacks
try:
    import cupy as cp
    # Test if CuPy can actually access GPU
    cp.cuda.runtime.getDeviceCount()
    cp.array([1, 2, 3])  # Quick test
    GPU_AVAILABLE = True
    print("🚀 GPU acceleration enabled with CuPy!")
    
    # Try to import CuML, but don't fail if it's not available
    try:
        import cuml
        from cuml.preprocessing import StandardScaler as CumlStandardScaler
        from cuml.decomposition import KernelPCA as CumlKernelPCA
        CUML_AVAILABLE = True
        print("✅ CuML also available for enhanced GPU preprocessing!")
    except ImportError:
        from sklearn.preprocessing import StandardScaler as CumlStandardScaler
        from sklearn.decomposition import KernelPCA as CumlKernelPCA
        CUML_AVAILABLE = False
        print("⚙️  Using scikit-learn for preprocessing (CuML not available)")
        
except ImportError as e:
    import numpy as cp  # Use numpy as fallback
    from sklearn.preprocessing import StandardScaler as CumlStandardScaler
    from sklearn.decomposition import KernelPCA as CumlKernelPCA
    GPU_AVAILABLE = False
    CUML_AVAILABLE = False
    print("⚠️  GPU packages not found, falling back to CPU execution.")
    print(f"Install GPU packages with: pip install cupy-cuda11x")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def check_gpu_setup():
    """Check GPU availability and memory."""
    if not GPU_AVAILABLE:
        return False, "GPU packages not installed"
    
    try:
        # Check if GPU is accessible
        device_count = cp.cuda.runtime.getDeviceCount()
        if device_count == 0:
            return False, "No CUDA devices found"
        
        # Get GPU memory info
        meminfo = cp.cuda.Device().mem_info
        free_mem = meminfo[0] / 1024**3  # Convert to GB
        total_mem = meminfo[1] / 1024**3
        
        print(f"🎯 GPU detected: {device_count} device(s)")
        print(f"📊 GPU memory: {free_mem:.1f}GB free / {total_mem:.1f}GB total")
        
        if free_mem < 1.0:
            print("⚠️  Warning: Low GPU memory available (<1GB)")
        
        status_msg = f"GPU ready with {free_mem:.1f}GB available"
        if CUML_AVAILABLE:
            status_msg += " (CuML + CuPy)"
        else:
            status_msg += " (CuPy only)"
            
        return True, status_msg
    
    except Exception as e:
        return False, f"GPU check failed: {str(e)}"


def to_gpu_if_available(data):
    """Move data to GPU if available and beneficial."""
    if GPU_AVAILABLE and isinstance(data, cp.ndarray):
        return data
    elif GPU_AVAILABLE:
        try:
            return cp.asarray(data)
        except:
            return data
    return data


def to_cpu(data):
    """Move data back to CPU."""
    if GPU_AVAILABLE and isinstance(data, cp.ndarray):
        return cp.asnumpy(data)
    return data


def setup_data_directory():
    """Set up the data directory path."""
    # Default to a local data directory
    default_data_dir = os.path.join(os.path.dirname(__file__), 'data')
    
    parser = argparse.ArgumentParser(description='Run Intensity DREAM Model')
    parser.add_argument('--data-dir', 
                       default=default_data_dir,
                       help=f'Path to data directory (default: {default_data_dir})')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' does not exist.")
        print("Please specify the correct path using --data-dir argument.")
        sys.exit(1)
    
    return args.data_dir


def load_data(data_dir):
    """Load all required datasets for TASK1."""
    print("Loading TASK1 datasets...")
    
    try:
        mordred = pd.read_csv(os.path.join(data_dir, 'Mordred_Descriptors.csv'), 
                             index_col=0, encoding='latin1')
        stimulus_def = pd.read_csv(os.path.join(data_dir, 'TASK1_Stimulus_definition.csv'))
        train_data = pd.read_csv(os.path.join(data_dir, 'TASK1_training.csv'))
        leaderboard_form = pd.read_csv(os.path.join(data_dir, 'TASK1_leaderboard_set_Submission_form.csv'))
        test_form = pd.read_csv(os.path.join(data_dir, 'TASK1_test_set_Submission_form.csv'))
        
        # Load actual values for validation
        try:
            leaderboard_truth = pd.read_csv(os.path.join(data_dir, 'TASK1_Leaderboard_ActualValue.csv'))
            print("📊 Leaderboard ground truth loaded for validation")
        except FileNotFoundError:
            print("⚠️  TASK1_Leaderboard_ActualValue.csv not found - skipping validation")
            leaderboard_truth = None
        
        print("All TASK1 datasets loaded successfully!")
        return mordred, stimulus_def, train_data, leaderboard_form, test_form, leaderboard_truth
        
    except FileNotFoundError as e:
        print(f"Error: Could not find required data file: {e.filename}")
        print(f"Make sure all TASK1 data files are in: {data_dir}")
        sys.exit(1)


def preprocess_mordred(mordred):
    """Preprocess Mordred descriptors for TASK1."""
    print("Preprocessing Mordred descriptors...")
    
    # Convert to numeric and drop non-numeric columns
    mordred_numeric = mordred.select_dtypes(include=[np.number])
    print(f"Selected {mordred_numeric.shape[1]} numeric features from {mordred.shape[1]} total")
    
    # Drop columns with NaNs or constant values
    mordred_clean = mordred_numeric.dropna(axis=1)
    mordred_clean = mordred_clean.loc[:, mordred_clean.nunique() > 1]
    
    print(f"After cleaning: {mordred_clean.shape[1]} features")
    return mordred_clean


def extract_features_task1(data, stimulus_def, mordred_clean, desc="Extracting features"):
    """Extract features for TASK1 (direct stimulus to molecule mapping)."""
    print(f"{desc}...")
    
    # Create stimulus to molecule mapping
    stimulus_to_molecule = stimulus_def.set_index('stimulus')['molecule']
    
    # Map stimuli to molecules
    molecules = data['stimulus'].map(stimulus_to_molecule)
    
    # Extract features for these molecules
    X = mordred_clean.loc[molecules, :].values
    
    # Extract targets - TASK1 structure matches notebook approach
    if len(data.columns) > 3 and 'Intensity' in data.columns:  # TASK1 training data
        y = data.iloc[:, 3:].values  # Skip stimulus, Intensity, Pleasantness (like notebook)
        y = np.clip(y, 0, 5)  # Clip to [0,5] as in notebook
    elif 'subject' in data.columns:  # TASK2 training data format
        y = data.iloc[:, 3:].values  # Skip stimulus, subject, rep
    else:  # Submission forms - no targets
        y = None
    
    print(f"Extracted features for {X.shape[0]} samples with {X.shape[1]} features")
    if y is not None:
        print(f"Target shape: {y.shape}")
    
    return X, y


def prepare_advanced_dataset(X_train, y_train, X_val=None, y_val=None):
    """Prepare dataset with advanced preprocessing and KernelPCA."""
    print("Preparing advanced dataset with KernelPCA...")
    
    # Check GPU setup
    gpu_ready, gpu_status = check_gpu_setup()
    print(f"GPU Status: {gpu_status}")
    
    # Ensure we have targets
    if y_train is None:
        raise ValueError("y_train cannot be None")
    
    # Combine datasets if validation set provided
    if X_val is not None:
        X_all = np.vstack([X_train, X_val])
        if y_val is not None:
            y_all = np.vstack([y_train, y_val])
        else:
            y_all = y_train
    else:
        X_all = X_train
        y_all = y_train
    
    print(f"🔄 Advanced preprocessing pipeline on {'GPU' if gpu_ready else 'CPU'}...")
    
    # Step 1: Variance thresholding
    print("🔍 Applying variance thresholding...")
    variance_selector = VarianceThreshold(threshold=1e-5)
    X_all = variance_selector.fit_transform(X_all)
    print(f"Features after variance thresholding: {X_all.shape[1]}")
    
    # Step 2: Handle NaN values (fill with median)
    print("🔧 Handling missing values...")
    col_medians = np.nanmedian(X_all, axis=0)
    for i in range(X_all.shape[1]):
        X_all[np.isnan(X_all[:, i]), i] = col_medians[i]
    
    # Step 3: Scaling
    if gpu_ready and CUML_AVAILABLE:
        scaler = CumlStandardScaler()
        print("🚀 Using GPU-accelerated CuML StandardScaler")
        X_all = scaler.fit_transform(to_gpu_if_available(X_all))
        X_all = to_cpu(X_all)
    else:
        scaler = CumlStandardScaler()
        print("⚙️  Using CPU StandardScaler")
        X_all = scaler.fit_transform(X_all)
    
    # Step 4: KernelPCA dimensionality reduction
    print("🧮 Applying KernelPCA dimensionality reduction...")
    if gpu_ready and CUML_AVAILABLE:
        try:
            kpca = CumlKernelPCA(n_components=128, kernel='rbf', random_state=42)
            print("🚀 Using GPU-accelerated CuML KernelPCA")
            X_all = kpca.fit_transform(to_gpu_if_available(X_all))
            X_all = to_cpu(X_all)
        except:
            print("⚠️  CuML KernelPCA failed, falling back to CPU")
            kpca = KernelPCA(n_components=128, kernel='rbf', random_state=42, n_jobs=-1)
            X_all = kpca.fit_transform(X_all)
    else:
        kpca = KernelPCA(n_components=128, kernel='rbf', random_state=42, n_jobs=-1)
        print("⚙️  Using CPU KernelPCA")
        X_all = kpca.fit_transform(X_all)
    
    print(f"After KernelPCA: {X_all.shape[1]} components")
    
    # Step 5: Target clipping (ensure y_all is not None)
    if y_all is None:
        raise ValueError("Targets cannot be None after preprocessing")
    
    if gpu_ready:
        y_all_gpu = to_gpu_if_available(y_all)
        y_all = to_cpu(cp.clip(y_all_gpu, 0, 5))
    else:
        y_all = np.clip(y_all, 0, 5)
    
    print(f"Final dataset shape: X={X_all.shape}")
    print(f"Target shape: y={y_all.shape}")
    
    return X_all, y_all, variance_selector, scaler, kpca


def train_intensity_model(X_train, y_train, X_val, y_val):
    """Train advanced stacked model for intensity prediction."""
    print("Training advanced intensity prediction model...")
    
    # Check GPU availability for LightGBM
    gpu_ready, _ = check_gpu_setup()
    
    # Configure LightGBM parameters optimized for intensity prediction
    if gpu_ready:
        lgb_params = {
            'objective': 'regression',
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
            'n_estimators': 300,  # Balanced for intensity task
            'learning_rate': 0.05,
            'max_depth': 8,       # Reduced depth as in notebook
            'random_state': 42,
            'verbosity': -1,
            'num_leaves': 31,
            'min_data_in_leaf': 20,
        }
        print("🚀 Training with GPU-accelerated LightGBM")
    else:
        lgb_params = {
            'objective': 'regression',
            'n_estimators': 300,
            'learning_rate': 0.05,
            'max_depth': 8,
            'random_state': 42,
            'verbosity': -1,
            'n_jobs': -1
        }
        print("⚙️  Training with CPU LightGBM (all cores)")
    
    # Create stacked ensemble model
    print("🧠 Creating stacked ensemble for intensity prediction...")
    base_model = lgb.LGBMRegressor(**lgb_params)
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 7))
    
    stacking_regressor = StackingRegressor(
        estimators=[('lgbm', base_model)],
        final_estimator=ridge,
        passthrough=True,
        cv=3
    )
    
    # Wrap in MultiOutputRegressor for multi-target prediction
    model = MultiOutputRegressor(stacking_regressor)
    
    print("📈 Training stacked intensity model...")
    model.fit(X_train, y_train)
    
    # Make predictions
    val_preds = model.predict(X_val)
    
    # GPU-accelerated clipping
    if GPU_AVAILABLE:
        val_preds_gpu = to_gpu_if_available(val_preds)
        val_preds = to_cpu(cp.clip(val_preds_gpu, 0, 5))
    else:
        val_preds = np.clip(val_preds, 0, 5)
    
    return model, val_preds


def evaluate_intensity_model(y_val, val_preds, target_columns):
    """Evaluate intensity model performance."""
    print("\n" + "="*60)
    print("INTENSITY MODEL EVALUATION")
    print("="*60)
    
    # Convert to GPU arrays for faster computation
    y_val_gpu = to_gpu_if_available(y_val)
    val_preds_gpu = to_gpu_if_available(val_preds)
    
    # Overall RMSE
    rmse = cp.sqrt(cp.mean((y_val_gpu - val_preds_gpu) ** 2))
    rmse_cpu = to_cpu(rmse)
    print(f"Overall Validation RMSE: {rmse_cpu:.4f}")
    
    # Per-target RMSE if multiple targets
    if len(target_columns) > 1:
        per_target_rmse = cp.sqrt(cp.mean((y_val_gpu - val_preds_gpu) ** 2, axis=0))
        per_target_rmse_cpu = to_cpu(per_target_rmse)
        print("\nPer-Target RMSE:")
        for name, score in zip(target_columns, per_target_rmse_cpu):
            print(f"  {name}: {score:.4f}")


def prepare_predictions(submission_df, stimulus_def, mordred_clean, model,
                       variance_selector, scaler, kpca, target_columns):
    """Prepare predictions following notebook approach exactly."""
    # Map submission stimuli to molecules (following notebook)
    stim_col = 'stimulus'
    submission_molecules = submission_df[stim_col].map(stimulus_def.set_index('stimulus')['molecule'])
    
    # Get features for these molecules (following notebook)
    feats = mordred_clean.loc[submission_molecules, :]
    
    # Apply preprocessing pipeline
    feats = variance_selector.transform(feats)
    
    # Handle NaN values
    col_medians = np.nanmedian(feats, axis=0)
    for i in range(feats.shape[1]):
        feats[np.isnan(feats[:, i]), i] = col_medians[i]
    
    feats_scaled = scaler.transform(feats)
    feats_reduced = kpca.transform(feats_scaled)
    
    # Make predictions
    preds = model.predict(feats_reduced)
    
    # Clip to [0,5] as in notebook
    preds = np.clip(preds, 0, 5)
    
    # Create dataframe (following notebook)
    preds_df = pd.DataFrame(preds, columns=target_columns)
    preds_df.insert(0, stim_col, submission_df[stim_col].values)
    
    return preds_df


def make_intensity_predictions(submission_df, stimulus_def, mordred_clean, model, 
                             variance_selector, scaler, kpca, target_columns, 
                             data_dir, file_prefix):
    """Make intensity predictions for submission using notebook approach."""
    print(f"\nMaking intensity predictions for {file_prefix}...")
    
    # Use notebook approach
    preds_df = prepare_predictions(submission_df, stimulus_def, mordred_clean, model,
                                 variance_selector, scaler, kpca, target_columns)
    
    # Save predictions
    output_path = os.path.join(data_dir, f'{file_prefix}_Predictions.csv')
    preds_df.to_csv(output_path, index=False)
    
    print(f"✅ Saved {file_prefix} predictions to: {output_path}")
    return preds_df


def main():
    """Main execution function for intensity prediction."""
    print("Intensity DREAM Model - Advanced GPU-Accelerated Version")
    print("="*70)
    
    # Check and display GPU status
    gpu_ready, gpu_status = check_gpu_setup()
    print(f"🔧 System Status: {gpu_status}")
    if gpu_ready:
        print("✅ GPU acceleration will be used for training and preprocessing")
    else:
        print("⚡ CPU-only mode - consider installing GPU packages for speedup")
    print("="*70)
    
    # Setup data directory
    data_dir = setup_data_directory()
    print(f"Using data directory: {data_dir}")
    
    # Load TASK1 data
    mordred, stimulus_def, train_data, leaderboard_form, test_form, leaderboard_truth = load_data(data_dir)
    
    # Preprocess Mordred descriptors
    mordred_clean = preprocess_mordred(mordred)
    
    # Extract features for training
    X_train_full, y_train_full = extract_features_task1(train_data, stimulus_def, mordred_clean, 
                                                       "Extracting training features")
    
    # Get target column names - TASK1 olfactory descriptors start from column 3 (after stimulus, Intensity, Pleasantness)
    target_columns = train_data.columns[3:].tolist()
    print(f"📊 Predicting {len(target_columns)} olfactory descriptors (matching notebook approach)")
    
    # Prepare dataset with advanced preprocessing
    X_processed, y_processed, variance_selector, scaler, kpca = prepare_advanced_dataset(
        X_train_full, y_train_full)
    
    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_processed, y_processed, test_size=0.2, random_state=42)
    
    # Train intensity model
    print("\n🚀 Starting intensity model training...")
    model, val_preds = train_intensity_model(X_train, y_train, X_val, y_val)
    
    # Evaluate model
    evaluate_intensity_model(y_val, val_preds, target_columns)
    
    # Validate against leaderboard truth if available
    if leaderboard_truth is not None:
        print("\n🔍 Validating against leaderboard ground truth...")
        lb_preds = make_intensity_predictions(
            leaderboard_form, stimulus_def, mordred_clean, model,
            variance_selector, scaler, kpca, target_columns,
            data_dir, "TASK1_Leaderboard_Validation")
        
        # Compare with ground truth
        try:
            lb_truth_values = leaderboard_truth[target_columns].values
            lb_pred_values = lb_preds[target_columns].values
            
            # GPU-accelerated RMSE calculation
            if GPU_AVAILABLE:
                truth_gpu = to_gpu_if_available(lb_truth_values)
                pred_gpu = to_gpu_if_available(lb_pred_values)
                lb_rmse = cp.sqrt(cp.mean((truth_gpu - pred_gpu) ** 2))
                lb_rmse_cpu = to_cpu(lb_rmse)
            else:
                lb_rmse_cpu = np.sqrt(np.mean((lb_truth_values - lb_pred_values) ** 2))
            
            print(f"🎯 Leaderboard Validation RMSE: {lb_rmse_cpu:.4f}")
        except Exception as e:
            print(f"⚠️  Could not validate against ground truth: {e}")
    
    # Make final predictions using test set form (avoiding leaderboard stimulus mismatch)
    print("\n🎯 Making final predictions using test set stimuli...")
    final_preds = prepare_predictions(
        test_form, stimulus_def, mordred_clean, model,
        variance_selector, scaler, kpca, target_columns)
    
    # Save as final submission
    final_output_path = os.path.join(data_dir, 'TASK1_final_predictions.csv')
    final_preds.to_csv(final_output_path, index=False)
    print(f"✅ Final predictions saved to: {final_output_path}")
    
    # Also make regular test predictions for completeness
    test_preds = make_intensity_predictions(
        test_form, stimulus_def, mordred_clean, model,
        variance_selector, scaler, kpca, target_columns,
        data_dir, "TASK1_Test")
    
    # Save complete model
    model_path = os.path.join(data_dir, 'intensity_model_advanced.joblib')
    try:
        joblib.dump({
            'model': model,
            'variance_selector': variance_selector,
            'scaler': scaler,
            'kpca': kpca,
            'target_columns': target_columns,
            'stimulus_def': stimulus_def
        }, model_path)
        print(f"📦 Complete model saved to: {model_path}")
    except Exception as e:
        print(f"⚠️  Could not save model: {e}")
    
    print("\n" + "="*70)
    print("🎉 INTENSITY MODEL TRAINING COMPLETE!")
    print("\n🔬 Advanced Features Used:")
    print("   • Variance thresholding for feature selection")
    print("   • KernelPCA with RBF kernel (128 components)")
    print("   • Stacked ensemble (LightGBM + Ridge)")
    print("   • GPU-accelerated preprocessing and training")
    print("   • Robust NaN handling")
    print("   • Model persistence with joblib")
    
    if gpu_ready:
        print("\n⚡ GPU acceleration was used for:")
        if CUML_AVAILABLE:
            print("   • Data preprocessing (CuML StandardScaler, KernelPCA)")
        else:
            print("   • Array operations and computations (CuPy)")
        print("   • LightGBM model training")
        print("   • Performance evaluation computations")
        if GPU_AVAILABLE:
            try:
                meminfo = cp.cuda.Device().mem_info
                free_mem = meminfo[0] / 1024**3
                print(f"   • GPU memory remaining: {free_mem:.1f}GB")
            except:
                print("   • GPU memory status: Available")
    
    print("\n📁 Output Files:")
    print(f"   • TASK1_final_predictions.csv (🎯 PRIMARY SUBMISSION FILE)")
    print(f"   • TASK1_Test_Predictions.csv")
    if leaderboard_truth is not None:
        print(f"   • TASK1_Leaderboard_Validation_Predictions.csv")
    print(f"   • intensity_model_advanced.joblib")
    
    print("="*70)


if __name__ == "__main__":
    main()