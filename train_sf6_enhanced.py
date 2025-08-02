#!/usr/bin/env python3
"""
Enhanced Street Fighter 6 Training Script
Integrates community project concepts with your existing SF2 approach
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

# Core RL libraries
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed

# Hyperparameter optimization
import optuna
from optuna.integration import TensorBoardCallback

# Utilities
import numpy as np
import logging
from datetime import datetime

# Import our enhanced SF6 environment
sys.path.append('./sf6_env')
sys.path.append('./sf6_configs')
from street_fighter_6_enhanced import StreetFighter6Enhanced
from sf6_config import SF6Config

class SF6TrainingCallback(BaseCallback):
    """Enhanced callback for SF6 training with detailed logging"""
    
    def __init__(self, check_freq: int, save_path: str, eval_env=None, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.eval_env = eval_env
        self.best_mean_reward = -np.inf
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
    
    def _on_step(self) -> bool:
        # Save model periodically
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, f'sf6_model_{self.n_calls}')
            self.model.save(model_path)
            
            if self.verbose > 0:
                print(f"Model saved at step {self.n_calls}")
            
            # Log additional metrics
            if hasattr(self.training_env, 'get_attr'):
                try:
                    env_stats = self.training_env.get_attr('get_performance_stats')[0]
                    for key, value in env_stats.items():
                        self.logger.record(f"sf6/{key}", value)
                except:
                    pass  # Ignore if stats not available
        
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout"""
        # Log episode statistics
        if len(self.episode_rewards) > 0:
            mean_reward = np.mean(self.episode_rewards[-100:])  # Last 100 episodes
            self.logger.record("sf6/mean_episode_reward", mean_reward)
            
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                # Save best model
                best_model_path = os.path.join(self.save_path, 'best_model')
                self.model.save(best_model_path)
                if self.verbose > 0:
                    print(f"New best model saved with mean reward: {mean_reward:.2f}")

class SF6HyperparameterOptimizer:
    """Hyperparameter optimization for SF6 using Optuna"""
    
    def __init__(self, config: SF6Config, n_trials: int = 50, n_timesteps: int = 50000):
        self.config = config
        self.n_trials = n_trials
        self.n_timesteps = n_timesteps
        self.study_name = f"sf6_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    def optimize_hyperparameters(self, trial) -> float:
        """Objective function for hyperparameter optimization"""
        # Suggest hyperparameters
        n_steps = trial.suggest_categorical('n_steps', [512, 1024, 2048, 2560, 4096])
        gamma = trial.suggest_float('gamma', 0.9, 0.9999, log=True)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        clip_range = trial.suggest_float('clip_range', 0.1, 0.4)
        gae_lambda = trial.suggest_float('gae_lambda', 0.8, 0.99)
        
        # Create environment
        env = self.create_training_environment()
        
        # Create model with suggested hyperparameters
        model = PPO(
            'CnnPolicy',
            env,
            n_steps=n_steps,
            gamma=gamma,
            learning_rate=learning_rate,
            clip_range=clip_range,
            gae_lambda=gae_lambda,
            verbose=0,
            tensorboard_log=f"./sf6_logs/optuna_{trial.number}/"
        )
        
        # Train the model
        try:
            model.learn(total_timesteps=self.n_timesteps)
            
            # Evaluate the model
            mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5, deterministic=True)
            
        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            mean_reward = -1000  # Penalty for failed trials
        finally:
            env.close()
        
        return mean_reward
    
    def create_training_environment(self):
        """Create training environment for optimization"""
        env = StreetFighter6Enhanced()
        env = Monitor(env, './sf6_logs/optuna_monitor/')
        env = DummyVecEnv([lambda: env])
        env = VecFrameStack(env, self.config.training.frame_stack, channels_order='last')
        return env
    
    def run_optimization(self) -> Dict[str, Any]:
        """Run the hyperparameter optimization"""
        print(f"Starting hyperparameter optimization with {self.n_trials} trials...")
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            study_name=self.study_name,
            storage=f'sqlite:///sf6_optuna_{self.study_name}.db',
            load_if_exists=True
        )
        
        # Add TensorBoard callback
        tensorboard_callback = TensorBoardCallback(
            dirname=f"./sf6_logs/optuna_tensorboard/",
            metric_name="value"
        )
        
        # Run optimization
        study.optimize(
            self.optimize_hyperparameters,
            n_trials=self.n_trials,
            callbacks=[tensorboard_callback]
        )
        
        # Save results
        best_params = study.best_params
        results_path = f"./sf6_configs/best_hyperparams_{self.study_name}.json"
        with open(results_path, 'w') as f:
            json.dump({
                'best_params': best_params,
                'best_value': study.best_value,
                'n_trials': len(study.trials),
                'study_name': self.study_name
            }, f, indent=2)
        
        print(f"Optimization completed!")
        print(f"Best parameters: {best_params}")
        print(f"Best value: {study.best_value:.2f}")
        print(f"Results saved to: {results_path}")
        
        return best_params

class SF6Trainer:
    """Main trainer class for SF6 RL"""
    
    def __init__(self, config_path: Optional[str] = None, character: str = 'ryu'):
        # Load configuration
        if config_path and Path(config_path).exists():
            self.config = SF6Config.load_config(config_path)
        else:
            self.config = SF6Config()
        
        self.character = character
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Setup logging
        self.setup_logging()
        
        # Create directories
        self.create_directories()
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path(self.config.training.tensorboard_log)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f'training_{self.timestamp}.log'),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def create_directories(self):
        """Create necessary directories"""
        directories = [
            self.config.training.tensorboard_log,
            self.config.training.model_save_path,
            './sf6_logs/monitor/',
            './sf6_logs/eval/',
            './sf6_configs/'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def create_training_environment(self):
        """Create the training environment with all wrappers"""
        self.logger.info(f"Creating training environment for character: {self.character}")
        
        # Create base environment
        env = StreetFighter6Enhanced(character=self.character)
        
        # Wrap with Monitor for logging
        env = Monitor(env, f'./sf6_logs/monitor/training_{self.timestamp}/')
        
        # Vectorize environment
        env = DummyVecEnv([lambda: env])
        
        # Frame stacking
        env = VecFrameStack(env, self.config.training.frame_stack, channels_order='last')
        
        return env
    
    def create_evaluation_environment(self):
        """Create evaluation environment"""
        env = StreetFighter6Enhanced(character=self.character)
        env = Monitor(env, f'./sf6_logs/eval/eval_{self.timestamp}/')
        return env
    
    def train_model(self, hyperparams: Optional[Dict] = None, resume_from: Optional[str] = None):
        """Train the SF6 model"""
        self.logger.info("Starting SF6 model training...")
        
        # Use provided hyperparams or default config
        if hyperparams:
            model_params = hyperparams
        else:
            model_params = {
                'n_steps': self.config.training.n_steps,
                'gamma': self.config.training.gamma,
                'learning_rate': self.config.training.learning_rate,
                'clip_range': self.config.training.clip_range,
                'gae_lambda': self.config.training.gae_lambda
            }
        
        self.logger.info(f"Using hyperparameters: {model_params}")
        
        # Create environments
        train_env = self.create_training_environment()
        eval_env = self.create_evaluation_environment()
        
        # Create model
        if resume_from and Path(resume_from).exists():
            self.logger.info(f"Resuming training from: {resume_from}")
            model = PPO.load(resume_from, env=train_env)
            # Update hyperparameters if provided
            for key, value in model_params.items():
                if hasattr(model, key):
                    setattr(model, key, value)
        else:
            model = PPO(
                'CnnPolicy',
                train_env,
                verbose=1,
                tensorboard_log=self.config.training.tensorboard_log,
                **model_params
            )
        
        # Setup callbacks
        training_callback = SF6TrainingCallback(
            check_freq=self.config.training.save_freq,
            save_path=self.config.training.model_save_path,
            eval_env=eval_env
        )
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"{self.config.training.model_save_path}/best_model_{self.timestamp}",
            log_path=f"./sf6_logs/eval/",
            eval_freq=self.config.training.eval_freq,
            deterministic=self.config.training.eval_deterministic,
            n_eval_episodes=self.config.training.eval_episodes
        )
        
        callbacks = [training_callback, eval_callback]
        
        # Pre-training setup
        self.logger.info("Make sure Street Fighter 6 is running and in training mode!")
        self.logger.info("Recommended setup:")
        self.logger.info("1. Start SF6 in windowed mode")
        self.logger.info("2. Go to Training Mode")
        self.logger.info("3. Set up your preferred training scenario")
        self.logger.info("4. Position the window for optimal screen capture")
        
        input("Press Enter when SF6 is ready for training...")
        
        # Start training
        start_time = time.time()
        try:
            model.learn(
                total_timesteps=self.config.training.total_timesteps,
                callback=callbacks,
                tb_log_name=f"sf6_{self.character}_{self.timestamp}"
            )
            
            training_time = time.time() - start_time
            self.logger.info(f"Training completed in {training_time:.2f} seconds")
            
            # Save final model
            final_model_path = f"{self.config.training.model_save_path}/sf6_final_{self.character}_{self.timestamp}"
            model.save(final_model_path)
            self.logger.info(f"Final model saved to: {final_model_path}")
            
            # Save training configuration
            config_save_path = f"./sf6_configs/training_config_{self.timestamp}.json"
            training_config = {
                'character': self.character,
                'hyperparameters': model_params,
                'training_time': training_time,
                'total_timesteps': self.config.training.total_timesteps,
                'timestamp': self.timestamp
            }
            
            with open(config_save_path, 'w') as f:
                json.dump(training_config, f, indent=2)
            
            return final_model_path
            
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
            # Save current model
            interrupt_model_path = f"{self.config.training.model_save_path}/sf6_interrupted_{self.timestamp}"
            model.save(interrupt_model_path)
            self.logger.info(f"Model saved to: {interrupt_model_path}")
            return interrupt_model_path
            
        finally:
            train_env.close()
            eval_env.close()
    
    def test_model(self, model_path: str, n_episodes: int = 5):
        """Test a trained model"""
        self.logger.info(f"Testing model: {model_path}")
        
        if not Path(model_path).exists():
            self.logger.error(f"Model file not found: {model_path}")
            return
        
        # Create test environment
        env = self.create_training_environment()
        
        # Load model
        model = PPO.load(model_path)
        
        self.logger.info("Make sure Street Fighter 6 is running!")
        input("Press Enter when ready to test...")
        
        total_rewards = []
        episode_lengths = []
        
        for episode in range(n_episodes):
            obs = env.reset()
            episode_reward = 0
            episode_length = 0
            
            self.logger.info(f"Starting test episode {episode + 1}/{n_episodes}")
            
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_reward += reward[0]
                episode_length += 1
                
                if done[0]:
                    break
            
            total_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            self.logger.info(f"Episode {episode + 1} completed:")
            self.logger.info(f"  Reward: {episode_reward:.2f}")
            self.logger.info(f"  Length: {episode_length} steps")
            
            if hasattr(info[0], 'get') and 'player_health' in info[0]:
                self.logger.info(f"  Final Health: P={info[0]['player_health']:.1f}, E={info[0]['enemy_health']:.1f}")
        
        # Summary statistics
        mean_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        mean_length = np.mean(episode_lengths)
        
        self.logger.info(f"\nTest Results Summary:")
        self.logger.info(f"  Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
        self.logger.info(f"  Mean Episode Length: {mean_length:.1f} steps")
        self.logger.info(f"  Best Episode Reward: {max(total_rewards):.2f}")
        self.logger.info(f"  Worst Episode Reward: {min(total_rewards):.2f}")
        
        env.close()
        return {
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'mean_length': mean_length,
            'all_rewards': total_rewards
        }

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Street Fighter 6 RL Training')
    parser.add_argument('--mode', choices=['train', 'test', 'optimize'], default='train',
                       help='Mode: train, test, or optimize hyperparameters')
    parser.add_argument('--character', default='ryu', help='Character to train/test with')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--model', help='Path to model file (for testing or resuming training)')
    parser.add_argument('--episodes', type=int, default=5, help='Number of test episodes')
    parser.add_argument('--trials', type=int, default=50, help='Number of optimization trials')
    parser.add_argument('--timesteps', type=int, help='Override total timesteps for training')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = SF6Trainer(config_path=args.config, character=args.character)
    
    if args.timesteps:
        trainer.config.training.total_timesteps = args.timesteps
    
    if args.mode == 'train':
        # Training mode
        model_path = trainer.train_model(resume_from=args.model)
        print(f"\nTraining completed! Model saved to: {model_path}")
        
    elif args.mode == 'test':
        # Testing mode
        if not args.model:
            print("Error: --model argument required for testing")
            return
        
        results = trainer.test_model(args.model, args.episodes)
        print(f"\nTesting completed! Mean reward: {results['mean_reward']:.2f}")
        
    elif args.mode == 'optimize':
        # Hyperparameter optimization mode
        optimizer = SF6HyperparameterOptimizer(
            trainer.config, 
            n_trials=args.trials,
            n_timesteps=args.timesteps or 50000
        )
        
        best_params = optimizer.run_optimization()
        
        # Train with best parameters
        print("\nTraining with optimized hyperparameters...")
        model_path = trainer.train_model(hyperparams=best_params)
        print(f"Optimized model saved to: {model_path}")

if __name__ == "__main__":
    main()