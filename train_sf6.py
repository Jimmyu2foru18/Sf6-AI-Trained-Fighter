#!/usr/bin/env python3
"""
Street Fighter 6 Training Script
Adapted from the existing SF2 training setup
"""

import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

# Import our SF6 environment
from sf6_env.street_fighter_6_base import StreetFighter6Base

class SF6TrainingCallback(BaseCallback):
    """Custom callback for SF6 training"""
    
    def __init__(self, check_freq, save_path, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
    
    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
    
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, f'sf6_model_{self.n_calls}')
            self.model.save(model_path)
            print(f"Model saved at step {self.n_calls}")
        return True

def create_sf6_environment():
    """Create and wrap the SF6 environment"""
    # Create base environment
    env = StreetFighter6Base(use_cv=True)
    
    # Wrap with Monitor for logging
    env = Monitor(env, './sf6_logs/')
    
    # Vectorize environment
    env = DummyVecEnv([lambda: env])
    
    # Frame stacking (same as your SF2 setup)
    env = VecFrameStack(env, 4, channels_order='last')
    
    return env

def train_sf6_model():
    """Train a PPO model on SF6"""
    print("Setting up SF6 training environment...")
    
    # Create environment
    env = create_sf6_environment()
    
    # Model parameters (adapted from your SF2 setup)
    model_params = {
        'n_steps': 2560,  # Adjusted for SF6
        'gamma': 0.99,
        'learning_rate': 2.5e-4,
        'clip_range': 0.2,
        'gae_lambda': 0.95
    }
    
    # Create PPO model
    model = PPO(
        'CnnPolicy', 
        env, 
        tensorboard_log='./sf6_logs/', 
        verbose=1, 
        **model_params
    )
    
    # Setup callback
    callback = SF6TrainingCallback(
        check_freq=10000, 
        save_path='./sf6_models/'
    )
    
    print("Starting SF6 model training...")
    print("Make sure Street Fighter 6 is running and in training mode!")
    
    # Wait for user confirmation
    input("Press Enter when SF6 is ready for training...")
    
    # Train the model
    model.learn(
        total_timesteps=100000,  # Start with smaller number for testing
        callback=callback
    )
    
    # Save final model
    model.save('./sf6_models/sf6_final_model')
    print("Training completed!")
    
    # Close environment
    env.close()

def test_sf6_model(model_path='./sf6_models/sf6_final_model.zip'):
    """Test a trained SF6 model"""
    print(f"Testing SF6 model: {model_path}")
    
    # Create environment
    env = create_sf6_environment()
    
    # Load model
    model = PPO.load(model_path)
    
    print("Make sure Street Fighter 6 is running!")
    input("Press Enter when ready to test...")
    
    # Test the model
    obs = env.reset()
    total_reward = 0
    
    for step in range(1000):  # Test for 1000 steps
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        if done:
            print(f"Episode finished at step {step} with total reward: {total_reward}")
            obs = env.reset()
            total_reward = 0
        
        time.sleep(0.016)  # ~60 FPS
    
    env.close()
    print("Testing completed!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        # Test mode
        model_path = sys.argv[2] if len(sys.argv) > 2 else './sf6_models/sf6_final_model.zip'
        test_sf6_model(model_path)
    else:
        # Training mode
        train_sf6_model()
