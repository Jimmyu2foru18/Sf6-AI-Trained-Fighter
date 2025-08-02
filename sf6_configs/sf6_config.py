#!/usr/bin/env python3
"""
Street Fighter 6 Configuration File
Centralized configuration for the SF6 RL environment
"""

import os
from dataclasses import dataclass
from typing import Dict, Tuple, List

@dataclass
class ScreenConfig:
    """Screen capture configuration"""
    # Monitor settings (adjust based on your setup)
    monitor_top: int = 0
    monitor_left: int = 0
    monitor_width: int = 1920
    monitor_height: int = 1080
    
    # Game window detection (SF6 specific)
    game_window_title: str = "STREET FIGHTER 6"
    
    # Health bar regions (you'll need to calibrate these)
    player_health_region: Tuple[int, int, int, int] = (100, 50, 400, 80)  # (x, y, width, height)
    enemy_health_region: Tuple[int, int, int, int] = (1520, 50, 400, 80)
    
    # Timer region
    timer_region: Tuple[int, int, int, int] = (960, 30, 100, 50)
    
    # Super meter regions
    player_super_region: Tuple[int, int, int, int] = (100, 1000, 300, 30)
    enemy_super_region: Tuple[int, int, int, int] = (1520, 1000, 300, 30)

@dataclass
class InputConfig:
    """Input control configuration"""
    # Key mappings for SF6 (adjust based on your control scheme)
    key_mapping: Dict[int, str] = None
    
    def __post_init__(self):
        if self.key_mapping is None:
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
                10: 'space', # Special/EX
                11: 'shift'  # Block/Parry
            }
    
    # Input timing
    input_delay: float = 0.016  # ~60 FPS timing
    key_hold_duration: float = 0.033  # How long to hold keys
    
    # Safety settings
    enable_input_safety: bool = True
    max_actions_per_second: int = 60

@dataclass
class EnvironmentConfig:
    """Environment-specific configuration"""
    # Observation space
    obs_width: int = 84
    obs_height: int = 84
    obs_channels: int = 1
    
    # Action space
    action_space_size: int = 12
    
    # Frame processing
    use_frame_delta: bool = True
    frame_stack_size: int = 4
    
    # Game state
    max_health: int = 100
    round_time_limit: int = 99  # seconds
    
    # Episode settings
    max_episode_steps: int = 3600  # ~60 seconds at 60 FPS
    reset_on_round_end: bool = True

@dataclass
class RewardConfig:
    """Reward function configuration"""
    # Health-based rewards
    damage_dealt_multiplier: float = 2.0
    damage_taken_multiplier: float = -1.0
    
    # Special move rewards
    special_move_bonus: float = 0.1
    super_move_bonus: float = 0.5
    
    # Positional rewards
    forward_movement_bonus: float = 0.01
    backward_movement_penalty: float = -0.005
    
    # Round outcome rewards
    round_win_bonus: float = 10.0
    round_loss_penalty: float = -10.0
    perfect_round_bonus: float = 5.0
    
    # Combo rewards
    combo_hit_bonus: float = 0.2
    combo_length_multiplier: float = 0.1
    
    # Defensive rewards
    successful_block_bonus: float = 0.1
    successful_parry_bonus: float = 0.3
    
    # Time-based penalties
    time_penalty_per_step: float = -0.001

@dataclass
class TrainingConfig:
    """Training-specific configuration"""
    # PPO hyperparameters (optimized from your SF2 setup)
    n_steps: int = 2560
    gamma: float = 0.99
    learning_rate: float = 2.5e-4
    clip_range: float = 0.2
    gae_lambda: float = 0.95
    
    # Training settings
    total_timesteps: int = 1000000
    save_freq: int = 10000
    eval_freq: int = 25000
    
    # Logging
    tensorboard_log: str = "./sf6_logs/"
    model_save_path: str = "./sf6_models/"
    
    # Environment settings
    n_envs: int = 1  # Start with single environment
    frame_stack: int = 4
    
    # Evaluation
    eval_episodes: int = 10
    eval_deterministic: bool = True

@dataclass
class CVConfig:
    """Computer Vision configuration"""
    # Health bar detection
    health_bar_color_range: Dict[str, Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = None
    
    def __post_init__(self):
        if self.health_bar_color_range is None:
            # HSV color ranges for health bars (you'll need to calibrate)
            self.health_bar_color_range = {
                'green': ((40, 50, 50), (80, 255, 255)),    # Healthy
                'yellow': ((20, 50, 50), (40, 255, 255)),   # Medium damage
                'red': ((0, 50, 50), (20, 255, 255))        # Critical
            }
    
    # OCR settings
    tesseract_config: str = '--psm 8 -c tessedit_char_whitelist=0123456789'
    
    # Image processing
    gaussian_blur_kernel: Tuple[int, int] = (5, 5)
    canny_threshold1: int = 50
    canny_threshold2: int = 150
    
    # Template matching
    template_match_threshold: float = 0.8
    
    # Debug settings
    save_debug_images: bool = False
    debug_image_path: str = "./sf6_logs/debug_images/"

class SF6Config:
    """Main configuration class that combines all configs"""
    
    def __init__(self):
        self.screen = ScreenConfig()
        self.input = InputConfig()
        self.environment = EnvironmentConfig()
        self.reward = RewardConfig()
        self.training = TrainingConfig()
        self.cv = CVConfig()
    
    def save_config(self, filepath: str):
        """Save configuration to file"""
        import json
        from dataclasses import asdict
        
        config_dict = {
            'screen': asdict(self.screen),
            'input': asdict(self.input),
            'environment': asdict(self.environment),
            'reward': asdict(self.reward),
            'training': asdict(self.training),
            'cv': asdict(self.cv)
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load_config(cls, filepath: str):
        """Load configuration from file"""
        import json
        
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        config = cls()
        
        # Update configurations
        for key, value in config_dict.get('screen', {}).items():
            setattr(config.screen, key, value)
        
        for key, value in config_dict.get('input', {}).items():
            setattr(config.input, key, value)
        
        for key, value in config_dict.get('environment', {}).items():
            setattr(config.environment, key, value)
        
        for key, value in config_dict.get('reward', {}).items():
            setattr(config.reward, key, value)
        
        for key, value in config_dict.get('training', {}).items():
            setattr(config.training, key, value)
        
        for key, value in config_dict.get('cv', {}).items():
            setattr(config.cv, key, value)
        
        return config
    
    def get_character_specific_config(self, character: str):
        """Get character-specific configurations"""
        # Character-specific key mappings and strategies
        character_configs = {
            'ryu': {
                'preferred_range': 'mid',
                'special_moves': ['hadoken', 'shoryuken', 'tatsumaki'],
                'combo_priority': ['cr.mk', 'hadoken']
            },
            'chun-li': {
                'preferred_range': 'close',
                'special_moves': ['kikoken', 'spinning_bird_kick'],
                'combo_priority': ['st.mp', 'cr.mk']
            },
            'ken': {
                'preferred_range': 'close',
                'special_moves': ['hadoken', 'shoryuken', 'tatsumaki'],
                'combo_priority': ['cr.mk', 'shoryuken']
            }
            # Add more characters as needed
        }
        
        return character_configs.get(character.lower(), {})

# Create default configuration instance
default_config = SF6Config()

# Example usage
if __name__ == "__main__":
    # Create and save default config
    config = SF6Config()
    config.save_config('./sf6_configs/default_config.json')
    print("Default configuration saved!")
    
    # Example of loading and modifying config
    loaded_config = SF6Config.load_config('./sf6_configs/default_config.json')
    loaded_config.training.learning_rate = 1e-4  # Modify learning rate
    loaded_config.save_config('./sf6_configs/modified_config.json')
    print("Modified configuration saved!")