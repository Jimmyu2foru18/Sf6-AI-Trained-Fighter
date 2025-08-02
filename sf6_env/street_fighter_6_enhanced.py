#!/usr/bin/env python3
"""
Enhanced Street Fighter 6 Environment
Integrates community project concepts and advanced features
"""

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

# Import our configuration
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'sf6_configs'))
from sf6_config import SF6Config

class HealthBarDetector:
    """Computer vision module for detecting health bars and game state"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def detect_health_percentage(self, frame: np.ndarray, region: Tuple[int, int, int, int]) -> float:
        """Detect health percentage from a health bar region"""
        x, y, w, h = region
        
        # Extract health bar region
        health_region = frame[y:y+h, x:x+w]
        
        if health_region.size == 0:
            return 100.0  # Default to full health if region is invalid
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(health_region, cv2.COLOR_RGB2HSV)
        
        # Create masks for different health states
        total_pixels = 0
        health_pixels = 0
        
        for color_name, (lower, upper) in self.config.cv.health_bar_color_range.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            color_pixels = cv2.countNonZero(mask)
            
            if color_name in ['green', 'yellow', 'red']:  # Health colors
                health_pixels += color_pixels
            
            total_pixels = max(total_pixels, health_region.shape[0] * health_region.shape[1])
        
        # Calculate health percentage
        if total_pixels > 0:
            health_percentage = (health_pixels / total_pixels) * 100
            return min(100.0, max(0.0, health_percentage))
        
        return 100.0
    
    def detect_super_meter(self, frame: np.ndarray, region: Tuple[int, int, int, int]) -> float:
        """Detect super meter percentage"""
        x, y, w, h = region
        super_region = frame[y:y+h, x:x+w]
        
        if super_region.size == 0:
            return 0.0
        
        # Convert to HSV and look for blue/cyan colors (typical super meter)
        hsv = cv2.cvtColor(super_region, cv2.COLOR_RGB2HSV)
        
        # Blue/cyan range for super meter
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        super_pixels = cv2.countNonZero(mask)
        total_pixels = super_region.shape[0] * super_region.shape[1]
        
        if total_pixels > 0:
            return (super_pixels / total_pixels) * 100
        
        return 0.0
    
    def detect_combo_counter(self, frame: np.ndarray) -> int:
        """Detect combo counter using OCR"""
        # This would require more sophisticated OCR implementation
        # For now, return 0 as placeholder
        return 0

class InputController:
    """Advanced input controller with safety features"""
    
    def __init__(self, config):
        self.config = config
        self.last_action_time = 0
        self.action_count = 0
        self.action_history = []
        self.logger = logging.getLogger(__name__)
        
        # Safety settings
        self.max_actions_per_second = config.input.max_actions_per_second
        self.input_delay = config.input.input_delay
    
    def send_action(self, action: np.ndarray) -> bool:
        """Send action with safety checks"""
        current_time = time.time()
        
        # Rate limiting
        if self.config.input.enable_input_safety:
            time_since_last = current_time - self.last_action_time
            if time_since_last < (1.0 / self.max_actions_per_second):
                time.sleep((1.0 / self.max_actions_per_second) - time_since_last)
        
        # Release all keys first
        self._release_all_keys()
        
        # Press keys based on action array
        pressed_keys = []
        for i, pressed in enumerate(action):
            if pressed and i < len(self.config.input.key_mapping):
                key = self.config.input.key_mapping[i]
                try:
                    pyautogui.keyDown(key)
                    pressed_keys.append(key)
                except Exception as e:
                    self.logger.warning(f"Failed to press key {key}: {e}")
        
        # Hold keys for specified duration
        time.sleep(self.config.input.key_hold_duration)
        
        # Release pressed keys
        for key in pressed_keys:
            try:
                pyautogui.keyUp(key)
            except Exception as e:
                self.logger.warning(f"Failed to release key {key}: {e}")
        
        # Update tracking
        self.last_action_time = current_time
        self.action_count += 1
        self.action_history.append((current_time, action.copy()))
        
        # Keep only recent history
        if len(self.action_history) > 100:
            self.action_history = self.action_history[-100:]
        
        return True
    
    def _release_all_keys(self):
        """Release all mapped keys"""
        for key in self.config.input.key_mapping.values():
            try:
                pyautogui.keyUp(key)
            except:
                pass  # Ignore errors when releasing keys
    
    def emergency_stop(self):
        """Emergency stop - release all keys"""
        self._release_all_keys()
        self.logger.info("Emergency stop executed - all keys released")

class WindowManager:
    """Manages SF6 window detection and focus"""
    
    def __init__(self, config):
        self.config = config
        self.game_window = None
        self.logger = logging.getLogger(__name__)
    
    def find_sf6_window(self) -> Optional[int]:
        """Find SF6 window handle"""
        def enum_windows_callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                window_title = win32gui.GetWindowText(hwnd)
                if self.config.screen.game_window_title.lower() in window_title.lower():
                    windows.append(hwnd)
            return True
        
        windows = []
        win32gui.EnumWindows(enum_windows_callback, windows)
        
        if windows:
            self.game_window = windows[0]
            return self.game_window
        
        return None
    
    def focus_sf6_window(self) -> bool:
        """Bring SF6 window to focus"""
        if not self.game_window:
            self.game_window = self.find_sf6_window()
        
        if self.game_window:
            try:
                win32gui.SetForegroundWindow(self.game_window)
                return True
            except Exception as e:
                self.logger.warning(f"Failed to focus SF6 window: {e}")
        
        return False
    
    def get_window_rect(self) -> Optional[Tuple[int, int, int, int]]:
        """Get SF6 window rectangle"""
        if not self.game_window:
            self.game_window = self.find_sf6_window()
        
        if self.game_window:
            try:
                rect = win32gui.GetWindowRect(self.game_window)
                return rect
            except Exception as e:
                self.logger.warning(f"Failed to get window rect: {e}")
        
        return None

class StreetFighter6Enhanced(Env):
    """Enhanced Street Fighter 6 Environment with community project integration"""
    
    def __init__(self, config_path: Optional[str] = None, character: str = 'ryu'):
        super().__init__()
        
        # Load configuration
        if config_path and Path(config_path).exists():
            self.config = SF6Config.load_config(config_path)
        else:
            self.config = SF6Config()
        
        # Character-specific settings
        self.character = character
        self.character_config = self.config.get_character_specific_config(character)
        
        # Environment spaces
        self.observation_space = Box(
            low=0, high=255, 
            shape=(self.config.environment.obs_height, 
                   self.config.environment.obs_width, 
                   self.config.environment.obs_channels), 
            dtype=np.uint8
        )
        self.action_space = MultiBinary(self.config.environment.action_space_size)
        
        # Initialize components
        self.health_detector = HealthBarDetector(self.config)
        self.input_controller = InputController(self.config)
        self.window_manager = WindowManager(self.config)
        
        # Screen capture setup
        self.sct = mss.mss()
        self.monitor = {
            "top": self.config.screen.monitor_top,
            "left": self.config.screen.monitor_left,
            "width": self.config.screen.monitor_width,
            "height": self.config.screen.monitor_height
        }
        
        # Game state tracking
        self.reset_game_state()
        
        # Frame history for delta calculation
        self.frame_history = []
        
        # Logging setup
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.step_count = 0
        self.episode_count = 0
        self.total_reward = 0
        
        # Debug mode
        self.debug_mode = self.config.cv.save_debug_images
        if self.debug_mode:
            Path(self.config.cv.debug_image_path).mkdir(parents=True, exist_ok=True)
    
    def reset_game_state(self):
        """Reset all game state variables"""
        self.player_health = self.config.environment.max_health
        self.enemy_health = self.config.environment.max_health
        self.player_super = 0
        self.enemy_super = 0
        self.combo_count = 0
        self.round_time = self.config.environment.round_time_limit
        self.previous_frame = None
        self.step_count = 0
        self.total_reward = 0
    
    def capture_screen(self) -> np.ndarray:
        """Capture the game screen with window management"""
        # Try to focus SF6 window first
        self.window_manager.focus_sf6_window()
        
        # Capture screen
        screenshot = self.sct.grab(self.monitor)
        frame = np.array(screenshot)
        
        # Convert BGRA to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        
        return frame
    
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
        """Extract game state information from the frame"""
        # Detect health bars
        player_health = self.health_detector.detect_health_percentage(
            frame, self.config.screen.player_health_region
        )
        enemy_health = self.health_detector.detect_health_percentage(
            frame, self.config.screen.enemy_health_region
        )
        
        # Detect super meters
        player_super = self.health_detector.detect_super_meter(
            frame, self.config.screen.player_super_region
        )
        enemy_super = self.health_detector.detect_super_meter(
            frame, self.config.screen.enemy_super_region
        )
        
        # Detect combo counter
        combo_count = self.health_detector.detect_combo_counter(frame)
        
        return {
            'player_health': player_health,
            'enemy_health': enemy_health,
            'player_super': player_super,
            'enemy_super': enemy_super,
            'combo_count': combo_count
        }
    
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
        reward += super_gained * 0.01  # Small bonus for building super
        
        # Combo rewards
        if current_state['combo_count'] > prev_state['combo_count']:
            reward += self.config.reward.combo_hit_bonus
            reward += current_state['combo_count'] * self.config.reward.combo_length_multiplier
        
        # Round outcome rewards
        if current_state['enemy_health'] <= 0:
            reward += self.config.reward.round_win_bonus
            if current_state['player_health'] >= self.config.environment.max_health:
                reward += self.config.reward.perfect_round_bonus
        
        if current_state['player_health'] <= 0:
            reward += self.config.reward.round_loss_penalty
        
        # Time penalty
        reward += self.config.reward.time_penalty_per_step
        
        return reward
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step in the environment"""
        # Store previous state
        prev_state = {
            'player_health': self.player_health,
            'enemy_health': self.enemy_health,
            'player_super': self.player_super,
            'enemy_super': self.enemy_super,
            'combo_count': self.combo_count
        }
        
        # Send action to game
        self.input_controller.send_action(action)
        
        # Small delay to let the game process the input
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
        done = (
            self.player_health <= 0 or 
            self.enemy_health <= 0 or 
            self.step_count >= self.config.environment.max_episode_steps
        )
        
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
            'episode_count': self.episode_count
        }
        
        # Save debug image if enabled
        if self.debug_mode and self.step_count % 60 == 0:  # Save every 60 steps
            debug_path = Path(self.config.cv.debug_image_path) / f"step_{self.step_count}.png"
            cv2.imwrite(str(debug_path), cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR))
        
        # For gymnasium compatibility, we need to return terminated and truncated separately
        terminated = done  # Episode ended due to game logic (health depleted, etc.)
        truncated = False  # Episode ended due to time limit or other external factors
        
        return observation, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)
        
        self.logger.info(f"Resetting environment (Episode {self.episode_count})")
        
        # Reset game state
        self.reset_game_state()
        
        # TODO: Implement actual game reset logic
        # This might involve:
        # 1. Pressing ESC to pause
        # 2. Navigating to restart option
        # 3. Confirming restart
        # For now, we'll just wait a moment
        time.sleep(1.0)
        
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
        
        return processed_frame, {}
    
    def render(self, mode='human'):
        """Render the environment (SF6 handles its own rendering)"""
        # SF6 renders itself, so this is mostly a pass-through
        # Could add overlay information here if needed
        pass
    
    def close(self):
        """Close the environment and cleanup"""
        self.logger.info("Closing SF6 environment")
        
        # Emergency stop to release all keys
        self.input_controller.emergency_stop()
        
        # Close screen capture
        if hasattr(self, 'sct'):
            self.sct.close()
        
        self.logger.info("SF6 environment closed successfully")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'total_episodes': self.episode_count,
            'total_steps': self.step_count,
            'average_reward_per_episode': self.total_reward / max(1, self.episode_count),
            'actions_per_second': self.input_controller.action_count / max(1, time.time() - getattr(self, 'start_time', time.time()))
        }

# Example usage and testing
if __name__ == "__main__":
    # Create environment with default config
    env = StreetFighter6Enhanced(character='ryu')
    
    print("Testing Enhanced SF6 Environment...")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Character: {env.character}")
    
    # Test basic functionality
    try:
        obs = env.reset()
        print(f"Initial observation shape: {obs.shape}")
        
        # Test a few random actions
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            print(f"Step {i}: Reward={reward:.3f}, Done={done}")
            print(f"  Health: P={info['player_health']:.1f}, E={info['enemy_health']:.1f}")
            
            if done:
                obs = env.reset()
                break
        
        # Print performance stats
        stats = env.get_performance_stats()
        print(f"Performance stats: {stats}")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        env.close()
        print("Enhanced SF6 Environment test completed!")