
import cv2
import numpy as np
import pyautogui
import mss
import time
import os
import win32gui
import win32con
from gymnasium import Env
from gymnasium.spaces import MultiBinary, Box
from typing import Dict, Any, Tuple, Optional, List
import logging
from pathlib import Path

# Import config
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'sf6_configs'))
from sf6_config import SF6Config

class StreetFighter6Fixed(Env):
    """Fixed Street Fighter 6 Environment with proper game integration"""
    
    def __init__(self, config_path: Optional[str] = None, character: str = 'ryu', debug_mode: bool = True):
        super().__init__()
        
        # Load configuration
        if config_path and Path(config_path).exists():
            self.config = SF6Config.load_config(config_path)
        else:
            self.config = SF6Config()
        
        self.character = character
        self.debug_mode = debug_mode
        
        # Environment spaces
        self.observation_space = Box(
            low=0, high=255, 
            shape=(self.config.environment.obs_height, 
                   self.config.environment.obs_width, 
                   self.config.environment.obs_channels), 
            dtype=np.uint8
        )
        self.action_space = MultiBinary(self.config.environment.action_space_size)
        
        # Screen capture setup
        self.sct = mss.mss()
        self.monitor = {
            "top": self.config.screen.monitor_top,
            "left": self.config.screen.monitor_left,
            "width": self.config.screen.monitor_width,
            "height": self.config.screen.monitor_height
        }
        
        # Game state tracking (using realistic defaults)
        self.reset_game_state()
        
        # Logging setup
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.step_count = 0
        self.episode_count = 0
        self.total_reward = 0
        
        # Input safety
        self.last_action_time = 0
        self.max_actions_per_second = 30  # Reduced for safety
        
        # Game connection status
        self.game_connected = self.check_game_connection()
        
        if self.debug_mode:
            self.logger.info(f"SF6 Environment initialized - Game connected: {self.game_connected}")
    
    def check_game_connection(self) -> bool:
        """Check if SF6 is running and accessible"""
        try:
            def enum_windows_callback(hwnd, windows):
                if win32gui.IsWindowVisible(hwnd):
                    window_title = win32gui.GetWindowText(hwnd)
                    if self.config.screen.game_window_title.lower() in window_title.lower():
                        windows.append(hwnd)
                return True
            
            windows = []
            win32gui.EnumWindows(enum_windows_callback, windows)
            return len(windows) > 0
        except Exception as e:
            self.logger.warning(f"Failed to check game connection: {e}")
            return False
    
    def reset_game_state(self):
        """Reset all game state variables to realistic defaults"""
        # Use realistic health values (0-100 scale)
        self.player_health = 100.0
        self.enemy_health = 100.0
        self.player_super = 0.0
        self.enemy_super = 0.0
        self.combo_count = 0
        self.round_time = self.config.environment.round_time_limit
        self.previous_frame = None
        self.step_count = 0
        self.total_reward = 0
    
    def capture_screen(self) -> np.ndarray:
        """Capture the game screen"""
        try:
            screenshot = self.sct.grab(self.monitor)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            return frame
        except Exception as e:
            self.logger.warning(f"Screen capture failed: {e}")
            # Return a black frame as fallback
            return np.zeros((self.config.screen.monitor_height, self.config.screen.monitor_width, 3), dtype=np.uint8)
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for the neural network"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Resize to target dimensions
        resized = cv2.resize(
            gray, 
            (self.config.environment.obs_width, self.config.environment.obs_height), 
            interpolation=cv2.INTER_CUBIC
        )
        
        # Add channel dimension
        processed = np.reshape(resized, 
                             (self.config.environment.obs_height, 
                              self.config.environment.obs_width, 
                              self.config.environment.obs_channels))
        
        return processed
    
    def extract_game_state(self, frame: np.ndarray) -> Dict[str, Any]:
        """Extract game state - simplified version for now"""
        if not self.game_connected:
            # Return mock values when game is not connected
            return {
                'player_health': self.player_health,
                'enemy_health': self.enemy_health,
                'player_super': self.player_super,
                'enemy_super': self.enemy_super,
                'combo_count': self.combo_count
            }
        
        # TODO: Implement actual computer vision detection
        # For now, simulate gradual health changes to test the learning
        
        # Simulate some health changes based on actions
        if hasattr(self, '_last_action') and self._last_action is not None:
            # Simulate damage based on action patterns
            if np.sum(self._last_action) > 6:  # Many buttons pressed = combo attempt
                damage_to_enemy = np.random.uniform(1, 5)
                self.enemy_health = max(0, self.enemy_health - damage_to_enemy)
                
                # Small chance of taking damage
                if np.random.random() < 0.2:
                    damage_to_player = np.random.uniform(0.5, 2)
                    self.player_health = max(0, self.player_health - damage_to_player)
            
            # Build super meter slowly
            self.player_super = min(100, self.player_super + np.random.uniform(0.1, 0.5))
        
        return {
            'player_health': self.player_health,
            'enemy_health': self.enemy_health,
            'player_super': self.player_super,
            'enemy_super': self.enemy_super,
            'combo_count': self.combo_count
        }
    
    def send_action_to_game(self, action: np.ndarray) -> bool:
        """Send action to the game with safety checks"""
        if not self.game_connected:
            if self.debug_mode:
                self.logger.debug("Game not connected - simulating action")
            return True
        
        current_time = time.time()
        
        # Rate limiting
        time_since_last = current_time - self.last_action_time
        min_interval = 1.0 / self.max_actions_per_second
        if time_since_last < min_interval:
            time.sleep(min_interval - time_since_last)
        
        # Send keys to game
        try:
            # Release all keys first
            for key in self.config.input.key_mapping.values():
                pyautogui.keyUp(key)
            
            # Press keys based on action
            pressed_keys = []
            for i, pressed in enumerate(action):
                if pressed and i < len(self.config.input.key_mapping):
                    key = self.config.input.key_mapping[i]
                    pyautogui.keyDown(key)
                    pressed_keys.append(key)
            
            # Hold briefly
            time.sleep(self.config.input.key_hold_duration)
            
            # Release keys
            for key in pressed_keys:
                pyautogui.keyUp(key)
            
            self.last_action_time = current_time
            return True
            
        except Exception as e:
            self.logger.warning(f"Failed to send action: {e}")
            return False
    
    def calculate_reward(self, prev_state: Dict, current_state: Dict) -> float:
        """Calculate reward based on state changes"""
        reward = 0.0
        
        # Health-based rewards
        damage_dealt = prev_state['enemy_health'] - current_state['enemy_health']
        damage_taken = prev_state['player_health'] - current_state['player_health']
        
        reward += damage_dealt * self.config.reward.damage_dealt_multiplier
        reward += damage_taken * self.config.reward.damage_taken_multiplier
        
        # Super meter rewards
        super_gained = current_state['player_super'] - prev_state['player_super']
        reward += super_gained * 0.01
        
        # Survival bonus (small positive reward for staying alive)
        reward += 0.01
        
        # Round outcome rewards
        if current_state['enemy_health'] <= 0:
            reward += self.config.reward.round_win_bonus
        
        if current_state['player_health'] <= 0:
            reward += self.config.reward.round_loss_penalty
        
        # Time penalty (very small)
        reward += self.config.reward.time_penalty_per_step
        
        return reward
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        # Store previous state
        prev_state = {
            'player_health': self.player_health,
            'enemy_health': self.enemy_health,
            'player_super': self.player_super,
            'enemy_super': self.enemy_super,
            'combo_count': self.combo_count
        }
        
        # Store action for game state simulation
        self._last_action = action
        
        # Send action to game
        action_sent = self.send_action_to_game(action)
        
        # Small delay to let the game process
        time.sleep(self.config.input.input_delay)
        
        # Capture new frame
        raw_frame = self.capture_screen()
        processed_frame = self.preprocess_frame(raw_frame)
        
        # Extract game state
        game_state = self.extract_game_state(raw_frame)
        self.player_health = game_state['player_health']
        self.enemy_health = game_state['enemy_health']
        self.player_super = game_state['player_super']
        self.enemy_super = game_state['enemy_super']
        self.combo_count = game_state['combo_count']
        
        # Calculate frame delta if using frame delta
        if self.config.environment.use_frame_delta and self.previous_frame is not None:
            observation = processed_frame - self.previous_frame
        else:
            observation = processed_frame
        
        self.previous_frame = processed_frame
        
        # Calculate reward
        reward = self.calculate_reward(prev_state, game_state)
        self.total_reward += reward
        
        # Check if episode is done
        terminated = (
            self.player_health <= 0 or 
            self.enemy_health <= 0
        )
        
        truncated = self.step_count >= self.config.environment.max_episode_steps
        
        # Update step count
        self.step_count += 1
        
        # Info dictionary
        info = {
            'player_health': self.player_health,
            'enemy_health': self.enemy_health,
            'player_super': self.player_super,
            'enemy_super': self.enemy_super,
            'combo_count': self.combo_count,
            'step_count': self.step_count,
            'total_reward': self.total_reward,
            'episode_count': self.episode_count,
            'game_connected': self.game_connected,
            'action_sent': action_sent
        }
        
        return observation, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)
        
        self.logger.info(f"Resetting environment (Episode {self.episode_count})")
        
        # Check game connection
        self.game_connected = self.check_game_connection()
        
        # Reset game state
        self.reset_game_state()
        
        # If game is connected, try to reset it
        if self.game_connected:
            # TODO: Implement actual game reset logic
            # For now, just wait a moment
            time.sleep(0.5)
        
        # Capture initial frame
        raw_frame = self.capture_screen()
        processed_frame = self.preprocess_frame(raw_frame)
        
        # Extract initial game state
        game_state = self.extract_game_state(raw_frame)
        self.player_health = game_state['player_health']
        self.enemy_health = game_state['enemy_health']
        self.player_super = game_state['player_super']
        self.enemy_super = game_state['enemy_super']
        self.combo_count = game_state['combo_count']
        
        self.previous_frame = processed_frame
        self.episode_count += 1
        
        info = {
            'game_connected': self.game_connected,
            'player_health': self.player_health,
            'enemy_health': self.enemy_health
        }
        
        return processed_frame, info
    
    def render(self, mode='human'):
        """Render the environment"""
        if mode == 'human':
            # The game itself provides the rendering
            pass
    
    def close(self):
        """Clean up resources"""
        # Release all keys
        try:
            for key in self.config.input.key_mapping.values():
                pyautogui.keyUp(key)
        except:
            pass
        
        if hasattr(self, 'sct'):
            self.sct.close()
        
        self.logger.info("SF6 Environment closed")
