#!/usr/bin/env python3
"""
Street Fighter 6 RL Project Setup Script
This script clones community projects and sets up the initial environment structure.
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(command, cwd=None):
    """Run a shell command and return the result"""
    try:
        result = subprocess.run(command, shell=True, cwd=cwd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error running command: {command}")
            print(f"Error output: {result.stderr}")
            return False
        print(f"Successfully ran: {command}")
        return True
    except Exception as e:
        print(f"Exception running command {command}: {e}")
        return False

def clone_community_projects():
    """Clone relevant community projects for SF6 RL development"""
    projects = {
        'AIVO-StreetFighter': 'https://github.com/corbosiny/AIVO-StreetFigherReinforcementLearning.git',
        'street-fighter-ai': 'https://github.com/linyiLYi/street-fighter-ai.git',
        'Street-Fighter-AI-DQN': 'https://github.com/ayanmali/Street-Fighter-AI.git',
        'Retro-Street-Fighter-RL': 'https://github.com/Tqualizer/Retro-Street-Fighter-reinforcement-learning.git'
    }
    
    # Create community projects directory
    community_dir = Path('community_projects')
    community_dir.mkdir(exist_ok=True)
    
    print("Cloning community projects...")
    for project_name, repo_url in projects.items():
        project_path = community_dir / project_name
        if project_path.exists():
            print(f"Project {project_name} already exists, skipping...")
            continue
            
        print(f"Cloning {project_name}...")
        if run_command(f"git clone {repo_url} {project_path}"):
            print(f"Successfully cloned {project_name}")
        else:
            print(f"Failed to clone {project_name}")

def install_dependencies():
    """Install required dependencies for SF6 RL development"""
    dependencies = [
        'opencv-python',
        'pytesseract',
        'mss',
        'pillow',
        'pyautogui',
        'pynput',
        'websockets',
        'stable-baselines3[extra]',
        'optuna',
        'tensorboard'
    ]
    
    print("Installing dependencies...")
    for dep in dependencies:
        print(f"Installing {dep}...")
        if run_command(f"pip install {dep}"):
            print(f"Successfully installed {dep}")
        else:
            print(f"Failed to install {dep}")

def create_sf6_environment_structure():
    """Create the basic SF6 environment structure"""
    directories = [
        'sf6_env',
        'sf6_env/interfaces',
        'sf6_env/controllers',
        'sf6_env/utils',
        'sf6_models',
        'sf6_logs',
        'sf6_configs'
    ]
    
    print("Creating SF6 project structure...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

def create_base_sf6_environment():
    """Create the base SF6 environment file"""
    sf6_env_content = '''import cv2
import numpy as np
import pyautogui
import mss
import time
from gym import Env
from gym.spaces import MultiBinary, Box
from typing import Dict, Any, Tuple, Optional

class StreetFighter6Base(Env):
    """Base Street Fighter 6 Environment for Reinforcement Learning"""
    
    def __init__(self, use_cv=True, use_mod=False):
        super().__init__()
        
        # Environment spaces (same as your SF2 setup)
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.action_space = MultiBinary(12)
        
        # Configuration
        self.use_cv = use_cv
        self.use_mod = use_mod
        
        # Screen capture setup
        if self.use_cv:
            self.sct = mss.mss()
            self.monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}
        
        # Game state tracking
        self.previous_frame = None
        self.player_health = 100
        self.enemy_health = 100
        self.score = 0
        
        # Input mapping for SF6
        self.key_mapping = {
            0: 'j',      # Light Punch
            1: 'k',      # Medium Punch  
            2: 'l',      # Heavy Punch
            3: 'u',      # Light Kick
            4: 'i',      # Medium Kick
            5: 'o',      # Heavy Kick
            6: 'w',      # Up
            7: 's',      # Down
            8: 'a',      # Left
            9: 'd',      # Right
            10: 'space', # Special
            11: 'shift'  # Block
        }
    
    def capture_screen(self) -> np.ndarray:
        """Capture the game screen"""
        if not self.use_cv:
            raise RuntimeError("Computer vision not enabled")
        
        screenshot = self.sct.grab(self.monitor)
        frame = np.array(screenshot)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame similar to your SF2 implementation"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Resize to 84x84 (same as your SF2 setup)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_CUBIC)
        
        # Add channel dimension
        processed = np.reshape(resized, (84, 84, 1))
        
        return processed
    
    def extract_health_bars(self, frame: np.ndarray) -> Dict[str, int]:
        """Extract health information using computer vision"""
        # TODO: Implement health bar detection
        # This is a placeholder - you'll need to implement actual CV detection
        # based on SF6's UI layout
        
        # For now, return dummy values
        return {
            'player_health': self.player_health,
            'enemy_health': self.enemy_health
        }
    
    def send_action(self, action: np.ndarray) -> None:
        """Send action to the game via keyboard input"""
        # Release all keys first
        for key in self.key_mapping.values():
            pyautogui.keyUp(key)
        
        # Press keys based on action array
        for i, pressed in enumerate(action):
            if pressed and i < len(self.key_mapping):
                pyautogui.keyDown(self.key_mapping[i])
        
        # Small delay to ensure input registration
        time.sleep(0.016)  # ~60 FPS timing
    
    def calculate_reward(self, prev_state: Dict, current_state: Dict) -> float:
        """Calculate reward based on game state changes"""
        # Health-based rewards (similar to your SF2 implementation)
        health_diff = (prev_state['enemy_health'] - current_state['enemy_health']) * 2
        damage_taken = (current_state['player_health'] - prev_state['player_health'])
        
        # Basic reward calculation
        reward = health_diff + damage_taken
        
        return reward
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step in the environment"""
        # Store previous state
        prev_state = {
            'player_health': self.player_health,
            'enemy_health': self.enemy_health
        }
        
        # Send action to game
        self.send_action(action)
        
        # Capture new frame
        if self.use_cv:
            raw_frame = self.capture_screen()
            processed_frame = self.preprocess_frame(raw_frame)
            
            # Extract game state
            health_info = self.extract_health_bars(raw_frame)
            self.player_health = health_info['player_health']
            self.enemy_health = health_info['enemy_health']
        else:
            # Placeholder for mod-based data extraction
            processed_frame = np.zeros((84, 84, 1), dtype=np.uint8)
        
        # Calculate frame delta (similar to your SF2 implementation)
        if self.previous_frame is not None:
            frame_delta = processed_frame - self.previous_frame
        else:
            frame_delta = processed_frame
        
        self.previous_frame = processed_frame
        
        # Calculate reward
        current_state = {
            'player_health': self.player_health,
            'enemy_health': self.enemy_health
        }
        reward = self.calculate_reward(prev_state, current_state)
        
        # Check if episode is done
        done = self.player_health <= 0 or self.enemy_health <= 0
        
        # Info dictionary
        info = {
            'player_health': self.player_health,
            'enemy_health': self.enemy_health,
            'score': self.score
        }
        
        return frame_delta, reward, done, info
    
    def reset(self) -> np.ndarray:
        """Reset the environment"""
        # Reset game state
        self.player_health = 100
        self.enemy_health = 100
        self.score = 0
        self.previous_frame = None
        
        # TODO: Implement actual game reset logic
        # This might involve sending specific key combinations to restart a match
        
        # Capture initial frame
        if self.use_cv:
            raw_frame = self.capture_screen()
            processed_frame = self.preprocess_frame(raw_frame)
        else:
            processed_frame = np.zeros((84, 84, 1), dtype=np.uint8)
        
        self.previous_frame = processed_frame
        return processed_frame
    
    def render(self, mode='human'):
        """Render the environment (SF6 handles its own rendering)"""
        # SF6 renders itself, so this is mostly a pass-through
        pass
    
    def close(self):
        """Close the environment"""
        # Release all keys
        for key in self.key_mapping.values():
            pyautogui.keyUp(key)
        
        if hasattr(self, 'sct'):
            self.sct.close()

# Example usage and testing
if __name__ == "__main__":
    # Create environment
    env = StreetFighter6Base(use_cv=True)
    
    # Test basic functionality
    print("Testing SF6 Environment...")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Test reset
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    # Test a few random actions
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"Step {i}: Reward={reward}, Done={done}, Info={info}")
        
        if done:
            obs = env.reset()
    
    env.close()
    print("SF6 Environment test completed!")
'''
    
    with open('sf6_env/street_fighter_6_base.py', 'w') as f:
        f.write(sf6_env_content)
    print("Created base SF6 environment file")

def create_training_script():
    """Create a training script adapted from your existing SF2 setup"""
    training_script = '''#!/usr/bin/env python3
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
'''
    
    with open('train_sf6.py', 'w') as f:
        f.write(training_script)
    print("Created SF6 training script")

def create_readme():
    """Create a README for the SF6 project"""
    readme_content = '''# Street Fighter 6 Reinforcement Learning Project

This project adapts your existing Street Fighter II reinforcement learning setup to work with Street Fighter 6.

## Setup

1. Run the setup script:
   ```bash
   python setup_sf6_project.py
   ```

2. Install additional dependencies if needed:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

1. Start Street Fighter 6 and go to Training Mode
2. Set up your preferred training scenario
3. Run the training script:
   ```bash
   python train_sf6.py
   ```

### Testing

1. Start Street Fighter 6
2. Run the test script:
   ```bash
   python train_sf6.py test [model_path]
   ```

## Project Structure

- `sf6_env/` - SF6 environment implementation
- `sf6_models/` - Trained models
- `sf6_logs/` - Training logs and tensorboard data
- `sf6_configs/` - Configuration files
- `community_projects/` - Cloned community projects for reference

## Community Projects Integrated

- **AIVO-StreetFighter**: Training platform architecture
- **street-fighter-ai**: Advanced RL implementations
- **Street-Fighter-AI-DQN**: DQN approach for comparison
- **Retro-Street-Fighter-RL**: Additional RL techniques

## Key Differences from SF2 Setup

1. **Input Method**: Uses keyboard input instead of gym-retro
2. **State Extraction**: Computer vision for health bars and game state
3. **Environment**: Custom environment class for SF6
4. **Reward Function**: Adapted for SF6 mechanics

## Next Steps

1. Implement accurate health bar detection
2. Add support for Fluffy mod integration
3. Expand to multiple characters
4. Implement tournament mode

## Notes

- Make sure SF6 is in windowed mode for screen capture
- Adjust screen capture coordinates in the environment if needed
- The current implementation is a starting point - you'll need to refine the computer vision components
'''
    
    with open('SF6_README.md', 'w') as f:
        f.write(readme_content)
    print("Created SF6 project README")

def main():
    """Main setup function"""
    print("Setting up Street Fighter 6 Reinforcement Learning Project...")
    print("=" * 60)
    
    # Step 1: Clone community projects
    clone_community_projects()
    print()
    
    # Step 2: Install dependencies
    install_dependencies()
    print()
    
    # Step 3: Create project structure
    create_sf6_environment_structure()
    print()
    
    # Step 4: Create base environment
    create_base_sf6_environment()
    print()
    
    # Step 5: Create training script
    create_training_script()
    print()
    
    # Step 6: Create README
    create_readme()
    print()
    
    print("=" * 60)
    print("Setup completed successfully!")
    print()
    print("Next steps:")
    print("1. Review the implementation plan: sf6_implementation_plan.md")
    print("2. Check the SF6 environment: sf6_env/street_fighter_6_base.py")
    print("3. Start Street Fighter 6 in training mode")
    print("4. Run: python train_sf6.py")
    print()
    print("For more details, see: SF6_README.md")

if __name__ == "__main__":
    main()