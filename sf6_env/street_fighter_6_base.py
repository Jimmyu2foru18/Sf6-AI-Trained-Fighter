import cv2
import numpy as np
import pyautogui
import mss
import time
from gymnasium import Env
from gymnasium.spaces import MultiBinary, Box
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
