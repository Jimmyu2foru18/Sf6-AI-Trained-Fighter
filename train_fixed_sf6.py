
#!/usr/bin/env python3
"""
Train with the Fixed SF6 Environment
"""

import sys
import os
import argparse
from pathlib import Path
sys.path.append('./sf6_env')
sys.path.append('./sf6_configs')

from street_fighter_6_fixed import StreetFighter6Fixed
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import time

class SF6FixedCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
    
    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_path = self.save_path / f"model_{self.n_calls}_steps.zip"
            self.model.save(str(model_path))
            if self.verbose > 0:
                print(f"Model saved to {model_path}")
        return True

def create_env(character='ryu'):
    """Create and wrap the environment"""
    def _init():
        env = StreetFighter6Fixed(character=character, debug_mode=False)
        env = Monitor(env)
        return env
    return _init

def train_fixed_sf6(character='ryu', timesteps=10000, save_freq=2000):
    """Train the fixed SF6 environment"""
    print(f"Training Fixed SF6 Environment with {character}")
    print(f"Timesteps: {timesteps}")
    print("="*50)
    
    # Create environment
    env = DummyVecEnv([create_env(character)])
    env = VecFrameStack(env, n_stack=4)
    
    # Create model
    model = PPO(
        'CnnPolicy',
        env,
        verbose=1,
        n_steps=512,  # Smaller steps for faster feedback
        batch_size=64,
        learning_rate=2.5e-4,
        gamma=0.99,
        clip_range=0.2,
        tensorboard_log="./sf6_logs/fixed_training/"
    )
    
    # Create callback
    callback = SF6FixedCallback(
        save_freq=save_freq,
        save_path="./sf6_models/fixed_training/"
    )
    
    # Train
    print("Starting training...")
    start_time = time.time()
    
    model.learn(
        total_timesteps=timesteps,
        callback=callback,
        progress_bar=True
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Save final model
    final_path = f"./sf6_models/fixed_training/final_model_{character}.zip"
    model.save(final_path)
    print(f"Final model saved to: {final_path}")
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Fixed SF6 Environment')
    parser.add_argument('--character', default='ryu', help='Character to train with')
    parser.add_argument('--timesteps', type=int, default=10000, help='Training timesteps')
    parser.add_argument('--save-freq', type=int, default=2000, help='Model save frequency')
    
    args = parser.parse_args()
    
    model = train_fixed_sf6(
        character=args.character,
        timesteps=args.timesteps,
        save_freq=args.save_freq
    )
    
    print("
[OK] Training completed successfully!")
