#!/usr/bin/env python3
"""
Olfactory DREAM Model - GPU-Accelerated Version
A machine learning pipeline for predicting olfactory properties from molecular descriptors.
Supports both GPU (CUDA) and CPU execution with automatic fallback.
"""

import os
import sys
import argparse
import warnings
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import StackingRegressor
from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb
import joblib
import numpy as np

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
        from cuml.impute import SimpleImputer as CumlSimpleImputer
        from cuml.decomposition import TruncatedSVD as CumlSparsePCA
        CUML_AVAILABLE = True
        print("✅ CuML also available for enhanced GPU preprocessing!")
    except ImportError:
        from sklearn.impute import SimpleImputer as CumlSimpleImputer
        from sklearn.preprocessing import StandardScaler as CumlStandardScaler
        from sklearn.decomposition import SparsePCA as CumlSparsePCA
        CUML_AVAILABLE = False
        print("⚙️  Using scikit-learn for preprocessing (CuML not available)")
        
except ImportError as e:
    import numpy as cp  # Use numpy as fallback
    from sklearn.impute import SimpleImputer as CumlSimpleImputer
    from sklearn.preprocessing import StandardScaler as CumlStandardScaler
    from sklearn.decomposition import SparsePCA as CumlSparsePCA
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
    # Default to the olfaction challenge data directory [[memory:5194971]]
    default_data_dir = '/home/burkhart/Downloads/olfaction_challenge_data'
    
    parser = argparse.ArgumentParser(description='Run Olfactory DREAM Model')
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
    """Load all required datasets."""
    print("Loading datasets...")
    
    try:
        mordred = pd.read_csv(os.path.join(data_dir, 'Mordred_Descriptors.csv'), 
                             index_col=0, encoding='latin1')
        components = pd.read_csv(os.path.join(data_dir, 'TASK2_Component_definition.csv'))
        stimulus_def = pd.read_csv(os.path.join(data_dir, 'TASK2_Stimulus_definition.csv'))
        train_data = pd.read_csv(os.path.join(data_dir, 'TASK2_Train_mixture_Dataset.csv.csv'))
        test_form = pd.read_csv(os.path.join(data_dir, 'TASK2_Test_set_Submission_form.csv'))
        leaderboard_truth = pd.read_csv(os.path.join(data_dir, 'TASK2_Leaderboard_ActualValue.csv'))
        
        print("All datasets loaded successfully!")
        return mordred, components, stimulus_def, train_data, test_form, leaderboard_truth
        
    except FileNotFoundError as e:
        print(f"Error: Could not find required data file: {e.filename}")
        print(f"Make sure all data files are in: {data_dir}")
        sys.exit(1)


def preprocess_mordred(mordred):
    """Preprocess Mordred descriptors."""
    print("Preprocessing Mordred descriptors...")
    return mordred.apply(pd.to_numeric, errors='coerce')


def get_stimulus_features(stimulus_id, stimulus_def, components, mordred, mordred_median=None):
    """Extract features for a given stimulus ID with robust NaN handling."""
    row = stimulus_def[stimulus_def['id'] == stimulus_id]
    if row.empty:
        return None
    
    component_ids = row.iloc[0, 1].split(';')
    feats = []
    
    for cid in component_ids:
        try:
            cid_int = int(cid)
        except ValueError:
            continue
            
        comp = components[components['id'] == cid_int]
        if comp.empty:
            continue
            
        mol_cid = comp['CID'].values[0]
        if mol_cid not in mordred.index:
            continue
            
        feat = mordred.loc[mol_cid]
        feats.append(feat)
    
    if not feats:
        return None
    
    feats_df = pd.DataFrame(feats)
    mean_feats = feats_df.mean(axis=0, skipna=True)
    
    # Fill NaN values with mordred median if available
    if mordred_median is not None:
        mean_feats = mean_feats.fillna(mordred_median)
    
    return mean_feats


def extract_features(data, stimulus_def, components, mordred, desc="Extracting features", is_train=True, mordred_median=None):
    """Extract features from dataset with improved NaN handling."""
    X, y = [], []
    
    for _, row in tqdm(data.iterrows(), total=len(data), desc=desc):
        feats = get_stimulus_features(row['stimulus'], stimulus_def, components, mordred, mordred_median)
        if feats is not None:
            X.append(feats.values)
            # For train data, skip first 3 columns (stimulus, subject, rep)
            # For leaderboard truth, skip first column (stimulus)
            if is_train:  
                y.append(row.iloc[3:].values)
            else:  # This is leaderboard truth
                y.append(row.iloc[1:].values)
    
    return X, y


def prepare_dataset(train_X, train_y, lb_X, lb_y):
    """Prepare and preprocess the combined dataset with advanced feature selection."""
    print("Preparing advanced dataset with feature selection...")
    
    # Check GPU setup
    gpu_ready, gpu_status = check_gpu_setup()
    print(f"GPU Status: {gpu_status}")
    
    # Debug shapes
    print(f"Train data: {len(train_X)} samples, {len(train_y)} targets")
    if train_y:
        print(f"Train target shape: {len(train_y[0])}")
    print(f"Leaderboard data: {len(lb_X)} samples, {len(lb_y)} targets") 
    if lb_y:
        print(f"Leaderboard target shape: {len(lb_y[0])}")
    
    # Ensure consistent target shapes - convert to numpy first
    import numpy as np
    train_y = [np.array(y, dtype=np.float32) for y in train_y]
    lb_y = [np.array(y, dtype=np.float32) for y in lb_y]
    
    # Find minimum target length to ensure consistency
    min_targets = min(len(train_y[0]) if train_y else 0, len(lb_y[0]) if lb_y else 0)
    train_y = [y[:min_targets] for y in train_y] if train_y else []
    lb_y = [y[:min_targets] for y in lb_y] if lb_y else []
    
    # Combine datasets
    X_all = np.vstack(train_X + lb_X)
    y_all = np.vstack(train_y + lb_y)
    
    print(f"🔄 Advanced preprocessing pipeline on {'GPU' if gpu_ready else 'CPU'}...")
    
    # Step 1: Variance thresholding for features
    print("🔍 Applying variance thresholding...")
    variance_selector = VarianceThreshold(threshold=1e-5)
    X_all = variance_selector.fit_transform(X_all)
    print(f"Features after variance thresholding: {X_all.shape[1]}")
    
    # Step 2: Impute missing values 
    if gpu_ready and CUML_AVAILABLE:
        imputer = CumlSimpleImputer(strategy='median')
        print("🚀 Using GPU-accelerated CuML SimpleImputer")
        X_all = imputer.fit_transform(to_gpu_if_available(X_all))
        X_all = to_cpu(X_all)
    else:
        imputer = CumlSimpleImputer(strategy='median')
        print("⚙️  Using CPU SimpleImputer")
        X_all = imputer.fit_transform(X_all)
    
    # Step 3: Scale features
    if gpu_ready and CUML_AVAILABLE:
        scaler = CumlStandardScaler()
        print("🚀 Using GPU-accelerated CuML StandardScaler")
        X_all = scaler.fit_transform(to_gpu_if_available(X_all))
        X_all = to_cpu(X_all)
    else:
        scaler = CumlStandardScaler()
        print("⚙️  Using CPU StandardScaler")
        X_all = scaler.fit_transform(X_all)
    
    # Step 4: Filter low-variance targets using GPU if available
    if gpu_ready:
        y_all_gpu = to_gpu_if_available(y_all)
        target_var = cp.var(y_all_gpu, axis=0)
        target_var_cpu = to_cpu(target_var)
    else:
        target_var_cpu = np.var(y_all, axis=0)
    
    y_all = y_all[:, target_var_cpu >= 1e-6]
    print(f"Targets after variance filtering: {y_all.shape[1]}")
    
    print(f"Final dataset shape: X={X_all.shape}, y={y_all.shape}")
    
    return X_all, y_all, variance_selector, imputer, scaler


def train_stacked_models(X_train, y_train, X_val, y_val):
    """Train stacked ensemble models with feature importance pruning."""
    import numpy as np
    print("Training advanced stacked ensemble...")
    
    # Check GPU availability for LightGBM
    gpu_ready, _ = check_gpu_setup()
    
    # Configure base LightGBM model with enhanced parameters
    if gpu_ready:
        lgb_params = {
            'objective': 'regression',
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
            'n_estimators': 500,  # Increased from 300
            'learning_rate': 0.05,
            'max_depth': 10,
            'random_state': 42,
            'verbosity': -1,
            'num_leaves': 31,
            'min_data_in_leaf': 20,
        }
        print("🚀 Training with GPU-accelerated stacked LightGBM")
    else:
        lgb_params = {
            'objective': 'regression',
            'n_estimators': 500,  # Increased from 300
            'learning_rate': 0.05,
            'max_depth': 10,
            'random_state': 42,
            'verbosity': -1,
            'n_jobs': -1
        }
        print("⚙️  Training with CPU stacked LightGBM (all cores)")
    
    # Step 1: Create base model and meta-learner
    print("🧠 Creating stacked ensemble architecture...")
    base_model = lgb.LGBMRegressor(**lgb_params)
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 7))  # Cross-validated Ridge
    
    # Step 2: Create stacking regressor
    stacking_regressor = StackingRegressor(
        estimators=[('lgbm', base_model)],
        final_estimator=ridge,
        passthrough=True,  # Include original features in meta-learner
        cv=3  # 3-fold cross-validation for stacking
    )
    
    # Step 3: Wrap in MultiOutputRegressor for multi-target regression
    stacked_model = MultiOutputRegressor(stacking_regressor)
    
    print("📈 Initial training phase...")
    stacked_model.fit(X_train, y_train)
    
    # Step 4: Feature importance analysis
    print("🔍 Analyzing feature importance...")
    try:
        # Extract feature importances from base LightGBM models
        importances = []
        for estimator in stacked_model.estimators_:
            if hasattr(estimator.estimators_[0][1], 'feature_importances_'):
                importances.append(estimator.estimators_[0][1].feature_importances_)
        
        if importances:
            mean_importance = np.mean(importances, axis=0)
            # Keep features above 25th percentile of importance
            important_threshold = np.percentile(mean_importance, 25)
            important_features = np.where(mean_importance > important_threshold)[0]
            
            print(f"Selected {len(important_features)} important features out of {X_train.shape[1]}")
            
            # Step 5: Retrain on selected features
            if len(important_features) < X_train.shape[1]:
                print("🎯 Retraining with pruned features...")
                X_train_pruned = X_train[:, important_features]
                X_val_pruned = X_val[:, important_features]
                
                # Create new stacked model for pruned features
                stacked_model_pruned = MultiOutputRegressor(StackingRegressor(
                    estimators=[('lgbm', lgb.LGBMRegressor(**lgb_params))],
                    final_estimator=RidgeCV(alphas=np.logspace(-3, 3, 7)),
                    passthrough=True,
                    cv=3
                ))
                
                stacked_model_pruned.fit(X_train_pruned, y_train)
                val_preds = stacked_model_pruned.predict(X_val_pruned)
                
                # GPU-accelerated clipping
                if GPU_AVAILABLE:
                    val_preds_gpu = to_gpu_if_available(val_preds)
                    val_preds = to_cpu(cp.clip(val_preds_gpu, 0, 5))
                else:
                    val_preds = np.clip(val_preds, 0, 5)
                
                return stacked_model_pruned, val_preds, important_features
            else:
                print("ℹ️  No feature pruning needed - all features are important")
        else:
            print("⚠️  Could not extract feature importances - using original model")
            important_features = np.arange(X_train.shape[1])
            
    except Exception as e:
        print(f"⚠️  Feature importance analysis failed: {e}")
        important_features = np.arange(X_train.shape[1])
    
    # Final predictions
    val_preds = stacked_model.predict(X_val)
    
    # GPU-accelerated clipping
    if GPU_AVAILABLE:
        val_preds_gpu = to_gpu_if_available(val_preds)
        val_preds = to_cpu(cp.clip(val_preds_gpu, 0, 5))
    else:
        val_preds = np.clip(val_preds, 0, 5)
    
    return stacked_model, val_preds, important_features


def evaluate_model(y_val, val_preds, leaderboard_truth):
    """Evaluate model performance with GPU-accelerated computations."""
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    # Convert to GPU arrays for faster computation
    y_val_gpu = to_gpu_if_available(y_val)
    val_preds_gpu = to_gpu_if_available(val_preds)
    
    # Overall RMSE
    rmse = cp.sqrt(cp.mean((y_val_gpu - val_preds_gpu) ** 2))
    rmse_cpu = to_cpu(rmse)
    print(f"Overall Validation RMSE: {rmse_cpu:.4f}")
    
    # Per-target RMSE
    per_target_mse = cp.mean((y_val_gpu - val_preds_gpu) ** 2, axis=0)
    per_target_rmse = cp.sqrt(per_target_mse)
    per_target_rmse_cpu = to_cpu(per_target_rmse)
    print("\nPer-Target RMSE:")
    target_names = leaderboard_truth.columns[1:]
    if len(per_target_rmse_cpu.shape) == 0:  # Single target
        per_target_rmse_cpu = [per_target_rmse_cpu.item()]
    for name, score in zip(target_names, per_target_rmse_cpu):
        print(f"  {name}: {score:.4f}")


def make_predictions(test_form, stimulus_def, components, mordred, 
                    model, variance_selector, imputer, scaler, important_features, 
                    y_train, data_dir, mordred_median=None):
    """Make predictions for test stimulus IDs with advanced preprocessing."""
    print("\nMaking advanced predictions for test stimulus IDs...")
    
    # Use GPU for mean calculation if available
    y_train_gpu = to_gpu_if_available(y_train)
    mean_train_pred = to_cpu(cp.clip(cp.mean(y_train_gpu, axis=0), 0, 5))
    
    pred_rows = []
    missing = 0
    
    # Iterate through ONLY the stimulus IDs in the test submission form
    for stimulus_id in tqdm(test_form['stimulus'], desc="Generating test predictions"):
        feats = get_stimulus_features(stimulus_id, stimulus_def, components, mordred, mordred_median)
        if feats is None:
            # Use mean training prediction as fallback
            pred_rows.append(mean_train_pred)
            missing += 1
        else:
            # Apply full preprocessing pipeline
            feats = feats.values.reshape(1, -1)
            
            # Step 1: Variance threshold
            feats = variance_selector.transform(feats)
            
            # Step 2: Imputation
            feats = imputer.transform(feats)
            
            # Step 3: Scaling
            feats = scaler.transform(feats)
            
            # Step 4: Feature importance selection
            feats = feats[:, important_features]
            
            # Step 5: Prediction with stacked model
            pred = model.predict(feats)[0]
            
            # GPU-accelerated clipping
            if GPU_AVAILABLE:
                pred_gpu = to_gpu_if_available(pred)
                pred = to_cpu(cp.clip(pred_gpu, 0, 5))
            else:
                pred = np.clip(pred, 0, 5)
                
            pred_rows.append(pred)
    
    # Use test form as template and fill in predictions
    submission = test_form.copy()
    submission.iloc[:, 1:] = pred_rows
    
    refined_path = os.path.join(data_dir, 'TASK2_final_predictions.csv')
    submission.to_csv(refined_path, index=False)
    
    # Save model for future use
    model_path = os.path.join(data_dir, 'stacked_olfactory_model.joblib')
    try:
        joblib.dump({
            'model': model,
            'variance_selector': variance_selector,
            'imputer': imputer,
            'scaler': scaler,
            'important_features': important_features,
            'mordred_median': mordred_median
        }, model_path)
        print(f"📦 Model saved to: {model_path}")
    except Exception as e:
        print(f"⚠️  Could not save model: {e}")
    
    print(f"\n✅ Saved test predictions to: {refined_path}")
    print(f"Used mean fallback for {missing} stimuli.")


def main():
    """Main execution function with advanced ensemble learning."""
    print("Olfactory DREAM Model - Advanced GPU-Accelerated Version")
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
    
    # Load data
    mordred, components, stimulus_def, train_data, test_form, leaderboard_truth = load_data(data_dir)
    
    # Preprocess Mordred descriptors and compute median for NaN handling
    mordred = preprocess_mordred(mordred)
    mordred_median = mordred.median()
    print(f"📊 Computed Mordred median for {len(mordred_median)} features")
    
    # Extract features with improved NaN handling
    train_X, train_y = extract_features(train_data, stimulus_def, components, mordred, 
                                       "Extracting train features", is_train=True, 
                                       mordred_median=mordred_median)
    lb_X, lb_y = extract_features(leaderboard_truth, stimulus_def, components, mordred, 
                                 "Extracting leaderboard features", is_train=False,
                                 mordred_median=mordred_median)
    
    # Advanced dataset preparation with feature selection
    X_all, y_all, variance_selector, imputer, scaler = prepare_dataset(train_X, train_y, lb_X, lb_y)
    
    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
    
    # Train advanced stacked ensemble models
    print("\n🚀 Starting advanced stacked ensemble training...")
    stacked_model, val_preds, important_features = train_stacked_models(X_train, y_train, X_val, y_val)
    
    # Evaluate
    evaluate_model(y_val, val_preds, leaderboard_truth)
    
    # Make predictions with advanced pipeline
    make_predictions(test_form, stimulus_def, components, mordred, 
                    stacked_model, variance_selector, imputer, scaler, important_features,
                    y_train, data_dir, mordred_median)
    
    print("\n" + "="*70)
    print("🎉 ADVANCED PROCESSING COMPLETE!")
    print("\n🔬 Advanced Features Used:")
    print("   • Variance thresholding for feature selection")
    print("   • Stacked ensemble (LightGBM + Ridge)")
    print("   • Feature importance-based pruning")
    print("   • Two-stage training strategy")
    print("   • Enhanced NaN handling with median fallback")
    print("   • Model persistence with joblib")
    
    if gpu_ready:
        print("\n⚡ GPU acceleration was used for:")
        if CUML_AVAILABLE:
            print("   • Data preprocessing (CuML StandardScaler, SimpleImputer)")
        else:
            print("   • Array operations and computations (CuPy)")
            print("   • Performance evaluation computations")
        print("   • Stacked LightGBM model training")
        if GPU_AVAILABLE:
            try:
                meminfo = cp.cuda.Device().mem_info
                free_mem = meminfo[0] / 1024**3
                print(f"   • GPU memory remaining: {free_mem:.1f}GB")
            except:
                print("   • GPU memory status: Available")
    else:
        print("\n💡 For faster processing, install GPU packages:")
        print("   pip install cupy-cuda11x")
        print("   pip install cuml  # (via conda for full GPU preprocessing)")
    
    print("="*70)


if __name__ == "__main__":
    main()