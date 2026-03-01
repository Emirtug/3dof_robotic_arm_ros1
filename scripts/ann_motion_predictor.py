#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
"""
ANN Motion Predictor Module
===========================
TÜBİTAK 2209-B Projesi için Yapay Sinir Ağı Modülü

Bu modül ArUco marker'dan gelen pozisyon verilerini:
1. Smoothing (titreşim azaltma)
2. Prediction (gecikme telafisi)
için işler.

Kullanım:
    from ann_motion_predictor import ANNMotionPredictor
    
    predictor = ANNMotionPredictor()
    predictor.load_model("model.h5")  # veya predictor.train(data)
    
    smoothed_pos = predictor.smooth(current_pos)
    predicted_pos = predictor.predict_next(position_history)

Author: Emirtuğ Kacar
Date: 2024
"""

import numpy as np
import os
import json
from collections import deque
from datetime import datetime

# TensorFlow/Keras import with error handling
try:
    import tensorflow as tf
    import keras
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout, Input
    from keras.optimizers import Adam
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    from keras.saving import load_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[WARN] TensorFlow/Keras not installed. Install with: pip install tensorflow")
    print("[WARN] ANN features will be disabled, falling back to simple filtering")


class SimpleMovingAverage:
    """Fallback smoothing when TensorFlow is not available"""
    
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.history = {
            'x': deque(maxlen=window_size),
            'y': deque(maxlen=window_size),
            'z': deque(maxlen=window_size)
        }
    
    def smooth(self, position):
        """Apply simple moving average"""
        x, y, z = position
        self.history['x'].append(x)
        self.history['y'].append(y)
        self.history['z'].append(z)
        
        return (
            np.mean(self.history['x']),
            np.mean(self.history['y']),
            np.mean(self.history['z'])
        )
    
    def predict_next(self, history):
        """Simple linear extrapolation"""
        if len(history) < 2:
            return history[-1] if history else (0, 0, 0)
        
        # Linear extrapolation from last 2 points
        p1 = np.array(history[-2])
        p2 = np.array(history[-1])
        velocity = p2 - p1
        predicted = p2 + velocity
        return tuple(predicted)


class ANNMotionPredictor:
    """
    LSTM-based motion predictor for robot arm control.
    
    Features:
    - Position smoothing (noise reduction)
    - Next position prediction (latency compensation)
    - Online learning capability
    - Model save/load
    """
    
    def __init__(self, sequence_length=10, prediction_horizon=1):
        """
        Initialize the ANN Motion Predictor.
        
        Args:
            sequence_length: Number of past positions to consider (default: 10)
            prediction_horizon: How many steps ahead to predict (default: 1)
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.input_dim = 3  # x, y, z
        
        # Position history for smoothing and prediction
        self.position_history = deque(maxlen=sequence_length + prediction_horizon)
        
        # Models
        self.smoothing_model = None
        self.prediction_model = None
        
        # Training data buffer
        self.training_buffer = []
        self.buffer_size = 1000
        
        # Statistics for normalization
        self.mean = np.zeros(3)
        self.std = np.ones(3)
        self.is_normalized = False
        
        # Fallback for when TF is not available
        self.fallback = SimpleMovingAverage()
        
        # Model directory
        self.model_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', 'models'
        )
        os.makedirs(self.model_dir, exist_ok=True)
        
        if TF_AVAILABLE:
            # Suppress TF warnings
            tf.get_logger().setLevel('ERROR')
            print("[ANN] TensorFlow loaded successfully")
        else:
            print("[ANN] Using fallback simple filtering")
    
    def _build_smoothing_model(self):
        """Build LSTM model for position smoothing (denoising)"""
        if not TF_AVAILABLE:
            return None
        
        model = Sequential([
            Input(shape=(self.sequence_length, self.input_dim)),
            LSTM(32, return_sequences=True),
            Dropout(0.2),
            LSTM(16),
            Dropout(0.2),
            Dense(self.input_dim)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _build_prediction_model(self):
        """Build LSTM model for next position prediction"""
        if not TF_AVAILABLE:
            return None
        
        model = Sequential([
            Input(shape=(self.sequence_length, self.input_dim)),
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            LSTM(32, return_sequences=True),
            Dropout(0.3),
            LSTM(16),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(self.input_dim * self.prediction_horizon)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _normalize(self, data):
        """Normalize data using stored statistics"""
        return (data - self.mean) / (self.std + 1e-8)
    
    def _denormalize(self, data):
        """Denormalize data"""
        return data * self.std + self.mean
    
    def update_normalization(self, positions):
        """Update normalization statistics from position data"""
        positions = np.array(positions)
        self.mean = np.mean(positions, axis=0)
        self.std = np.std(positions, axis=0)
        self.std = np.where(self.std < 1e-8, 1.0, self.std)  # Prevent division by zero
        self.is_normalized = True
    
    def add_position(self, position):
        """
        Add a new position to history.
        
        Args:
            position: Tuple or list of (x, y, z)
        """
        self.position_history.append(np.array(position))
        
        # Add to training buffer
        if len(self.training_buffer) < self.buffer_size:
            self.training_buffer.append(np.array(position))
    
    def smooth(self, position):
        """
        Apply smoothing to current position.
        
        Args:
            position: Current (x, y, z) position
            
        Returns:
            Smoothed (x, y, z) position
        """
        self.add_position(position)
        
        if not TF_AVAILABLE or self.smoothing_model is None:
            return self.fallback.smooth(position)
        
        if len(self.position_history) < self.sequence_length:
            return position
        
        # Prepare input sequence
        sequence = np.array(list(self.position_history)[-self.sequence_length:])
        
        if self.is_normalized:
            sequence = self._normalize(sequence)
        
        sequence = sequence.reshape(1, self.sequence_length, self.input_dim)
        
        # Predict smoothed position
        smoothed = self.smoothing_model.predict(sequence, verbose=0)[0]
        
        if self.is_normalized:
            smoothed = self._denormalize(smoothed)
        
        return tuple(smoothed)
    
    def predict_next(self, position=None):
        """
        Predict next position(s) based on history.
        
        Args:
            position: Optional current position to add first
            
        Returns:
            Predicted (x, y, z) position
        """
        if position is not None:
            self.add_position(position)
        
        if not TF_AVAILABLE or self.prediction_model is None:
            history = list(self.position_history)
            return self.fallback.predict_next(history)
        
        if len(self.position_history) < self.sequence_length:
            return tuple(self.position_history[-1]) if self.position_history else (0, 0, 0)
        
        # Prepare input sequence
        sequence = np.array(list(self.position_history)[-self.sequence_length:])
        
        if self.is_normalized:
            sequence = self._normalize(sequence)
        
        sequence = sequence.reshape(1, self.sequence_length, self.input_dim)
        
        # Predict next position
        predicted = self.prediction_model.predict(sequence, verbose=0)[0]
        
        # Reshape if multiple predictions
        if self.prediction_horizon > 1:
            predicted = predicted.reshape(self.prediction_horizon, self.input_dim)
            predicted = predicted[0]  # Return first prediction
        
        if self.is_normalized:
            predicted = self._denormalize(predicted)
        
        return tuple(predicted)
    
    def smooth_and_predict(self, position):
        """
        Apply both smoothing and prediction.
        
        Args:
            position: Current (x, y, z) position
            
        Returns:
            Tuple of (smoothed_position, predicted_next_position)
        """
        smoothed = self.smooth(position)
        predicted = self.predict_next()
        return smoothed, predicted
    
    def prepare_training_data(self, positions):
        """
        Prepare sequences for training from raw position data.
        
        Args:
            positions: List of (x, y, z) positions
            
        Returns:
            X, y arrays for training
        """
        positions = np.array(positions)
        
        if len(positions) < self.sequence_length + self.prediction_horizon:
            raise ValueError(f"Need at least {self.sequence_length + self.prediction_horizon} positions")
        
        X, y = [], []
        
        for i in range(len(positions) - self.sequence_length - self.prediction_horizon + 1):
            X.append(positions[i:i + self.sequence_length])
            y.append(positions[i + self.sequence_length:i + self.sequence_length + self.prediction_horizon].flatten())
        
        return np.array(X), np.array(y)
    
    def train(self, positions, epochs=50, validation_split=0.2, verbose=1):
        """
        Train the prediction model on position data.
        
        Args:
            positions: List of (x, y, z) positions
            epochs: Number of training epochs
            validation_split: Fraction of data for validation
            verbose: Training verbosity
            
        Returns:
            Training history
        """
        if not TF_AVAILABLE:
            print("[ANN] TensorFlow not available, cannot train")
            return None
        
        print(f"[ANN] Training on {len(positions)} positions...")
        
        # Update normalization
        self.update_normalization(positions)
        
        # Prepare data
        X, y = self.prepare_training_data(positions)
        
        # Normalize
        X_norm = np.array([self._normalize(seq) for seq in X])
        y_norm = self._normalize(y.reshape(-1, self.input_dim)).flatten()
        y_norm = y_norm.reshape(-1, self.input_dim * self.prediction_horizon)
        
        # Build model if needed
        if self.prediction_model is None:
            self.prediction_model = self._build_prediction_model()
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ModelCheckpoint(
                os.path.join(self.model_dir, 'best_prediction_model.h5'),
                save_best_only=True
            )
        ]
        
        # Train
        history = self.prediction_model.fit(
            X_norm, y_norm,
            epochs=epochs,
            batch_size=32,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )
        
        print(f"[ANN] Training complete. Final loss: {history.history['loss'][-1]:.6f}")
        
        return history
    
    def train_smoothing(self, noisy_positions, clean_positions=None, epochs=30, verbose=1):
        """
        Train the smoothing model.
        
        Args:
            noisy_positions: Input noisy positions
            clean_positions: Target clean positions (if None, uses moving average)
            epochs: Training epochs
            verbose: Verbosity
        """
        if not TF_AVAILABLE:
            print("[ANN] TensorFlow not available, cannot train")
            return None
        
        print(f"[ANN] Training smoothing model on {len(noisy_positions)} positions...")
        
        # If no clean data, create pseudo-labels using moving average
        if clean_positions is None:
            clean_positions = []
            window = 5
            for i in range(len(noisy_positions)):
                start = max(0, i - window // 2)
                end = min(len(noisy_positions), i + window // 2 + 1)
                clean_positions.append(np.mean(noisy_positions[start:end], axis=0))
            clean_positions = np.array(clean_positions)
        
        # Update normalization
        self.update_normalization(noisy_positions)
        
        # Prepare sequences
        X, y = [], []
        for i in range(len(noisy_positions) - self.sequence_length):
            X.append(noisy_positions[i:i + self.sequence_length])
            y.append(clean_positions[i + self.sequence_length - 1])
        
        X = np.array(X)
        y = np.array(y)
        
        # Normalize
        X_norm = np.array([self._normalize(seq) for seq in X])
        y_norm = self._normalize(y)
        
        # Build model
        if self.smoothing_model is None:
            self.smoothing_model = self._build_smoothing_model()
        
        # Train
        history = self.smoothing_model.fit(
            X_norm, y_norm,
            epochs=epochs,
            batch_size=32,
            validation_split=0.2,
            verbose=verbose
        )
        
        print(f"[ANN] Smoothing training complete. Final loss: {history.history['loss'][-1]:.6f}")
        
        return history
    
    def save_model(self, name="motion_predictor"):
        """Save models and normalization parameters"""
        if not TF_AVAILABLE:
            print("[ANN] TensorFlow not available, cannot save")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = os.path.join(self.model_dir, f"{name}_{timestamp}")
        
        # Save prediction model
        if self.prediction_model is not None:
            self.prediction_model.save(f"{base_path}_prediction.h5")
            print(f"[ANN] Saved prediction model: {base_path}_prediction.h5")
        
        # Save smoothing model
        if self.smoothing_model is not None:
            self.smoothing_model.save(f"{base_path}_smoothing.h5")
            print(f"[ANN] Saved smoothing model: {base_path}_smoothing.h5")
        
        # Save normalization parameters
        params = {
            'mean': self.mean.tolist(),
            'std': self.std.tolist(),
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon,
            'is_normalized': self.is_normalized
        }
        
        with open(f"{base_path}_params.json", 'w') as f:
            json.dump(params, f, indent=2)
        
        print(f"[ANN] Saved parameters: {base_path}_params.json")
        
        return base_path
    
    def load_model(self, base_path):
        """Load models and normalization parameters"""
        if not TF_AVAILABLE:
            print("[ANN] TensorFlow not available, cannot load")
            return False
        
        try:
            # Load prediction model
            pred_path = f"{base_path}_prediction.h5"
            if os.path.exists(pred_path):
                self.prediction_model = load_model(pred_path)
                print(f"[ANN] Loaded prediction model: {pred_path}")
            
            # Load smoothing model
            smooth_path = f"{base_path}_smoothing.h5"
            if os.path.exists(smooth_path):
                self.smoothing_model = load_model(smooth_path)
                print(f"[ANN] Loaded smoothing model: {smooth_path}")
            
            # Load parameters
            params_path = f"{base_path}_params.json"
            if os.path.exists(params_path):
                with open(params_path, 'r') as f:
                    params = json.load(f)
                
                self.mean = np.array(params['mean'])
                self.std = np.array(params['std'])
                self.sequence_length = params['sequence_length']
                self.prediction_horizon = params['prediction_horizon']
                self.is_normalized = params['is_normalized']
                print(f"[ANN] Loaded parameters: {params_path}")
            
            return True
            
        except Exception as e:
            print(f"[ANN] Error loading model: {e}")
            return False
    
    def get_comparison_stats(self, test_positions):
        """
        Compare ANN prediction vs simple method.
        
        Args:
            test_positions: List of positions to test
            
        Returns:
            Dictionary with comparison statistics
        """
        if len(test_positions) < self.sequence_length + 2:
            return None
        
        ann_errors = []
        simple_errors = []
        
        # Reset
        self.position_history.clear()
        simple_filter = SimpleMovingAverage()
        
        for i in range(len(test_positions) - 1):
            pos = test_positions[i]
            actual_next = test_positions[i + 1]
            
            # Add to history
            self.add_position(pos)
            simple_filter.smooth(pos)
            
            if i >= self.sequence_length:
                # ANN prediction
                ann_pred = self.predict_next()
                ann_error = np.linalg.norm(np.array(ann_pred) - np.array(actual_next))
                ann_errors.append(ann_error)
                
                # Simple prediction
                history = list(self.position_history)
                simple_pred = simple_filter.predict_next(history)
                simple_error = np.linalg.norm(np.array(simple_pred) - np.array(actual_next))
                simple_errors.append(simple_error)
        
        stats = {
            'ann_mean_error': np.mean(ann_errors) if ann_errors else float('inf'),
            'ann_std_error': np.std(ann_errors) if ann_errors else 0,
            'simple_mean_error': np.mean(simple_errors) if simple_errors else float('inf'),
            'simple_std_error': np.std(simple_errors) if simple_errors else 0,
            'improvement_percent': 0
        }
        
        if stats['simple_mean_error'] > 0:
            stats['improvement_percent'] = (
                (stats['simple_mean_error'] - stats['ann_mean_error']) / 
                stats['simple_mean_error'] * 100
            )
        
        return stats


# =============================================================================
# Test / Demo
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("ANN Motion Predictor - Test")
    print("=" * 60)
    
    # Create predictor
    predictor = ANNMotionPredictor(sequence_length=10, prediction_horizon=1)
    
    # Generate synthetic training data (sine wave motion)
    print("\n[TEST] Generating synthetic motion data...")
    t = np.linspace(0, 10 * np.pi, 500)
    
    # Simulated 3D trajectory
    x = 0.3 + 0.1 * np.sin(t)
    y = 0.0 + 0.1 * np.cos(t)
    z = 0.25 + 0.05 * np.sin(2 * t)
    
    # Add noise
    noise_level = 0.01
    x_noisy = x + np.random.normal(0, noise_level, len(t))
    y_noisy = y + np.random.normal(0, noise_level, len(t))
    z_noisy = z + np.random.normal(0, noise_level, len(t))
    
    positions = list(zip(x_noisy, y_noisy, z_noisy))
    clean_positions = list(zip(x, y, z))
    
    print(f"[TEST] Generated {len(positions)} positions")
    
    # Split data
    train_size = int(len(positions) * 0.8)
    train_data = positions[:train_size]
    test_data = positions[train_size:]
    
    if TF_AVAILABLE:
        # Train prediction model
        print("\n[TEST] Training prediction model...")
        predictor.train(train_data, epochs=30, verbose=0)
        
        # Train smoothing model
        print("\n[TEST] Training smoothing model...")
        predictor.train_smoothing(train_data, epochs=20, verbose=0)
        
        # Save model
        print("\n[TEST] Saving model...")
        predictor.save_model("test_model")
        
        # Get comparison stats
        print("\n[TEST] Comparing ANN vs Simple prediction...")
        stats = predictor.get_comparison_stats(test_data)
        
        if stats:
            print(f"\n  ANN Mean Error:    {stats['ann_mean_error']:.6f}")
            print(f"  Simple Mean Error: {stats['simple_mean_error']:.6f}")
            print(f"  Improvement:       {stats['improvement_percent']:.2f}%")
    else:
        print("\n[TEST] TensorFlow not available, testing fallback...")
        
        # Test fallback
        for i, pos in enumerate(test_data[:20]):
            smoothed = predictor.smooth(pos)
            predicted = predictor.predict_next()
            
            if i % 5 == 0:
                print(f"  Position {i}: {pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}")
                print(f"  Smoothed:    {smoothed[0]:.3f}, {smoothed[1]:.3f}, {smoothed[2]:.3f}")
                print(f"  Predicted:   {predicted[0]:.3f}, {predicted[1]:.3f}, {predicted[2]:.3f}")
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)
