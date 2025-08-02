#!/usr/bin/env python3
"""
Street Fighter 6 Diagnostic Tool
Tests actual game connection, input functionality, and computer vision components
"""

import cv2
import numpy as np
import pyautogui
import mss
import time
import win32gui
import win32con
from pathlib import Path
import sys
import os

# Add sf6_configs to path
sys.path.append('./sf6_configs')
from sf6_config import SF6Config

class SF6Diagnostic:
    def __init__(self):
        self.config = SF6Config()
        self.sct = mss.mss()
        
    def test_game_window_detection(self):
        """Test if SF6 window can be found and focused"""
        print("\n=== Testing Game Window Detection ===")
        
        def enum_windows_callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                window_title = win32gui.GetWindowText(hwnd)
                if window_title:
                    windows.append((hwnd, window_title))
            return True
        
        windows = []
        win32gui.EnumWindows(enum_windows_callback, windows)
        
        print(f"Found {len(windows)} visible windows:")
        sf6_windows = []
        for hwnd, title in windows:
            if 'street fighter' in title.lower() or 'sf6' in title.lower():
                sf6_windows.append((hwnd, title))
                print(f"  ✓ SF6 Window Found: {title} (Handle: {hwnd})")
            elif len(title) > 3:  # Only show non-empty titles
                print(f"    {title}")
        
        if not sf6_windows:
            print("  ❌ No Street Fighter 6 windows found!")
            print("  Please make sure SF6 is running and try again.")
            return False
        
        # Try to focus the first SF6 window
        hwnd, title = sf6_windows[0]
        try:
            win32gui.SetForegroundWindow(hwnd)
            print(f"  ✓ Successfully focused: {title}")
            return True
        except Exception as e:
            print(f"  ❌ Failed to focus window: {e}")
            return False
    
    def test_screen_capture(self):
        """Test screen capture functionality"""
        print("\n=== Testing Screen Capture ===")
        
        monitor = {
            "top": self.config.screen.monitor_top,
            "left": self.config.screen.monitor_left,
            "width": self.config.screen.monitor_width,
            "height": self.config.screen.monitor_height
        }
        
        try:
            screenshot = self.sct.grab(monitor)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            
            print(f"  ✓ Screen capture successful")
            print(f"  ✓ Frame shape: {frame.shape}")
            print(f"  ✓ Frame dtype: {frame.dtype}")
            
            # Save a test screenshot
            test_path = Path("./sf6_logs/diagnostic_screenshot.png")
            test_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(test_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            print(f"  ✓ Test screenshot saved to: {test_path}")
            
            return frame
        except Exception as e:
            print(f"  ❌ Screen capture failed: {e}")
            return None
    
    def test_health_detection(self, frame):
        """Test health bar detection"""
        print("\n=== Testing Health Detection ===")
        
        if frame is None:
            print("  ❌ No frame available for health detection")
            return
        
        # Test player health region
        player_region = self.config.screen.player_health_region
        enemy_region = self.config.screen.enemy_health_region
        
        print(f"  Player health region: {player_region}")
        print(f"  Enemy health region: {enemy_region}")
        
        # Extract regions
        try:
            player_crop = frame[player_region[1]:player_region[1]+player_region[3], 
                              player_region[0]:player_region[0]+player_region[2]]
            enemy_crop = frame[enemy_region[1]:enemy_region[1]+enemy_region[3], 
                             enemy_region[0]:enemy_region[0]+enemy_region[2]]
            
            print(f"  ✓ Player health crop shape: {player_crop.shape}")
            print(f"  ✓ Enemy health crop shape: {enemy_crop.shape}")
            
            # Save cropped regions for inspection
            cv2.imwrite("./sf6_logs/player_health_region.png", cv2.cvtColor(player_crop, cv2.COLOR_RGB2BGR))
            cv2.imwrite("./sf6_logs/enemy_health_region.png", cv2.cvtColor(enemy_crop, cv2.COLOR_RGB2BGR))
            print("  ✓ Health region crops saved for inspection")
            
        except Exception as e:
            print(f"  ❌ Health detection failed: {e}")
    
    def test_input_system(self):
        """Test input system with user confirmation"""
        print("\n=== Testing Input System ===")
        
        print("  This will test keyboard input to the game.")
        print("  Make sure SF6 is focused and in a safe state (training mode).")
        response = input("  Continue with input test? (y/n): ")
        
        if response.lower() != 'y':
            print("  Input test skipped.")
            return
        
        print("  Testing key mappings:")
        for i, key in self.config.input.key_mapping.items():
            print(f"    Action {i}: {key}")
        
        print("\n  Testing individual keys (3 second delay between each):")
        test_keys = ['w', 'a', 's', 'd']  # Basic movement
        
        for key in test_keys:
            print(f"    Testing key: {key}")
            try:
                pyautogui.keyDown(key)
                time.sleep(0.1)
                pyautogui.keyUp(key)
                print(f"    ✓ Key {key} sent successfully")
            except Exception as e:
                print(f"    ❌ Failed to send key {key}: {e}")
            
            time.sleep(1)  # Wait between key presses
        
        print("  ✓ Input test completed")
    
    def test_reward_calculation(self):
        """Test reward calculation with mock data"""
        print("\n=== Testing Reward Calculation ===")
        
        # Mock previous and current states
        prev_state = {
            'player_health': 100,
            'enemy_health': 100,
            'player_super': 0,
            'enemy_super': 0,
            'combo_count': 0
        }
        
        current_state = {
            'player_health': 95,  # Took 5 damage
            'enemy_health': 85,   # Dealt 15 damage
            'player_super': 10,   # Gained some super
            'enemy_super': 5,
            'combo_count': 3      # Hit a 3-hit combo
        }
        
        # Calculate reward
        reward = 0.0
        
        # Health-based rewards
        damage_dealt = prev_state['enemy_health'] - current_state['enemy_health']
        damage_taken = prev_state['player_health'] - current_state['player_health']
        
        reward += damage_dealt * self.config.reward.damage_dealt_multiplier
        reward += damage_taken * self.config.reward.damage_taken_multiplier
        
        # Super meter rewards
        super_gained = current_state['player_super'] - prev_state['player_super']
        reward += super_gained * 0.01
        
        # Combo rewards
        if current_state['combo_count'] > prev_state['combo_count']:
            reward += self.config.reward.combo_hit_bonus
            reward += current_state['combo_count'] * self.config.reward.combo_length_multiplier
        
        print(f"  Damage dealt: {damage_dealt} (reward: {damage_dealt * self.config.reward.damage_dealt_multiplier})")
        print(f"  Damage taken: {damage_taken} (reward: {damage_taken * self.config.reward.damage_taken_multiplier})")
        print(f"  Super gained: {super_gained} (reward: {super_gained * 0.01})")
        print(f"  Combo hits: {current_state['combo_count']} (reward: {self.config.reward.combo_hit_bonus + current_state['combo_count'] * self.config.reward.combo_length_multiplier})")
        print(f"  ✓ Total reward: {reward}")
    
    def run_full_diagnostic(self):
        """Run complete diagnostic suite"""
        print("Street Fighter 6 RL Environment Diagnostic")
        print("="*50)
        
        # Test 1: Window detection
        window_ok = self.test_game_window_detection()
        
        # Test 2: Screen capture
        frame = self.test_screen_capture()
        
        # Test 3: Health detection
        if frame is not None:
            self.test_health_detection(frame)
        
        # Test 4: Input system
        self.test_input_system()
        
        # Test 5: Reward calculation
        self.test_reward_calculation()
        
        print("\n=== Diagnostic Summary ===")
        if window_ok and frame is not None:
            print("  ✓ Basic functionality appears to be working")
            print("  ✓ Check the saved images in ./sf6_logs/ to verify health detection regions")
            print("  ✓ If health bars are not visible in the crops, adjust the regions in sf6_config.py")
        else:
            print("  ❌ Critical issues detected:")
            if not window_ok:
                print("    - Street Fighter 6 window not found or not focusable")
            if frame is None:
                print("    - Screen capture failed")
        
        print("\n=== Recommendations ===")
        print("  1. Make sure Street Fighter 6 is running in windowed mode")
        print("  2. Go to Training Mode for consistent environment")
        print("  3. Position the window so health bars are clearly visible")
        print("  4. Use the CV calibration tool to fine-tune detection regions")
        print("  5. Check that the game window title matches the config")

if __name__ == "__main__":
    diagnostic = SF6Diagnostic()
    diagnostic.run_full_diagnostic()