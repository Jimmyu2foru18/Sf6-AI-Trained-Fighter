#!/usr/bin/env python3
"""
Analyze SF6 Environment Issues
Identifies why episodes are ending quickly and what the environment is actually detecting
"""

import sys
import os
import time
import numpy as np

# Add paths
sys.path.append('./sf6_env')
sys.path.append('./sf6_configs')

from street_fighter_6_enhanced import StreetFighter6Enhanced
from sf6_config import SF6Config

def analyze_environment_behavior():
    """Analyze what the environment is actually doing"""
    print("Street Fighter 6 Environment Analysis")
    print("="*50)
    
    # Create environment
    print("\n1. Creating SF6 Environment...")
    env = StreetFighter6Enhanced(character='ryu')
    
    print(f"   ✓ Observation space: {env.observation_space}")
    print(f"   ✓ Action space: {env.action_space}")
    print(f"   ✓ Max episode steps: {env.config.environment.max_episode_steps}")
    
    # Test reset
    print("\n2. Testing Environment Reset...")
    start_time = time.time()
    obs, info = env.reset()
    reset_time = time.time() - start_time
    
    print(f"   ✓ Reset completed in {reset_time:.2f} seconds")
    print(f"   ✓ Observation shape: {obs.shape}")
    print(f"   ✓ Initial game state:")
    print(f"     - Player health: {env.player_health}")
    print(f"     - Enemy health: {env.enemy_health}")
    print(f"     - Player super: {env.player_super}")
    print(f"     - Episode count: {env.episode_count}")
    
    # Test a few steps
    print("\n3. Testing Environment Steps...")
    step_count = 0
    total_reward = 0
    
    for i in range(10):
        # Random action
        action = env.action_space.sample()
        
        step_start = time.time()
        obs, reward, terminated, truncated, info = env.step(action)
        step_time = time.time() - step_start
        
        step_count += 1
        total_reward += reward
        
        print(f"   Step {i+1}:")
        print(f"     - Action: {action}")
        print(f"     - Reward: {reward:.4f}")
        print(f"     - Terminated: {terminated}")
        print(f"     - Truncated: {truncated}")
        print(f"     - Step time: {step_time:.3f}s")
        print(f"     - Player health: {info['player_health']}")
        print(f"     - Enemy health: {info['enemy_health']}")
        print(f"     - Step count: {info['step_count']}")
        
        if terminated or truncated:
            print(f"     ❌ Episode ended after {step_count} steps!")
            print(f"     Total reward: {total_reward:.4f}")
            break
        
        time.sleep(0.1)  # Small delay
    
    # Analyze why episodes end quickly
    print("\n4. Analyzing Episode Termination Logic...")
    print(f"   Current termination conditions:")
    print(f"   - Player health <= 0: {env.player_health <= 0}")
    print(f"   - Enemy health <= 0: {env.enemy_health <= 0}")
    print(f"   - Max steps reached: {env.step_count >= env.config.environment.max_episode_steps}")
    
    # Test health detection
    print("\n5. Testing Health Detection...")
    raw_frame = env.capture_screen()
    game_state = env.extract_game_state(raw_frame)
    
    print(f"   Raw health detection results:")
    print(f"   - Player health: {game_state['player_health']}")
    print(f"   - Enemy health: {game_state['enemy_health']}")
    print(f"   - Player super: {game_state['player_super']}")
    print(f"   - Enemy super: {game_state['enemy_super']}")
    print(f"   - Combo count: {game_state['combo_count']}")
    
    # Check if health detection is working
    if game_state['player_health'] == 0 and game_state['enemy_health'] == 0:
        print("   ❌ Health detection is not working - both healths are 0!")
        print("   This explains why episodes end immediately.")
    
    env.close()
    
    return game_state

def analyze_health_detection_regions():
    """Analyze the health detection regions"""
    print("\n6. Analyzing Health Detection Regions...")
    
    config = SF6Config()
    
    print(f"   Player health region: {config.screen.player_health_region}")
    print(f"   Enemy health region: {config.screen.enemy_health_region}")
    print(f"   Monitor settings: {config.screen.monitor_width}x{config.screen.monitor_height}")
    print(f"   Game window title: '{config.screen.game_window_title}'")
    
    # Check if saved health region images exist
    import os
    player_img = "./sf6_logs/player_health_region.png"
    enemy_img = "./sf6_logs/enemy_health_region.png"
    
    if os.path.exists(player_img) and os.path.exists(enemy_img):
        print(f"   ✓ Health region images saved for inspection:")
        print(f"     - Player: {player_img}")
        print(f"     - Enemy: {enemy_img}")
        print(f"   📝 Open these images to verify they show actual health bars")
    else:
        print(f"   ❌ Health region images not found")

def provide_solutions():
    """Provide solutions based on analysis"""
    print("\n" + "="*50)
    print("ANALYSIS SUMMARY & SOLUTIONS")
    print("="*50)
    
    print("\n🔍 LIKELY ISSUES:")
    print("1. Street Fighter 6 is not running or not in the correct mode")
    print("2. Health detection regions are incorrectly configured")
    print("3. Game window is not being captured properly")
    print("4. Health bars are not visible in the captured regions")
    
    print("\n💡 SOLUTIONS:")
    print("1. VERIFY GAME SETUP:")
    print("   - Start Street Fighter 6")
    print("   - Go to Training Mode")
    print("   - Set to Windowed mode (not fullscreen)")
    print("   - Position window so health bars are clearly visible")
    
    print("\n2. CALIBRATE HEALTH DETECTION:")
    print("   - Run: python cv_calibration_tool.py")
    print("   - Manually select health bar regions")
    print("   - Test color detection")
    print("   - Save calibrated settings")
    
    print("\n3. MODIFY ENVIRONMENT FOR TESTING:")
    print("   - Temporarily disable health-based episode termination")
    print("   - Use time-based episodes only")
    print("   - Add more debug logging")
    
    print("\n4. ALTERNATIVE APPROACH:")
    print("   - Use the original SF2 environment as a baseline")
    print("   - Implement SF6 gradually with working components")
    print("   - Focus on input control first, then add CV detection")

if __name__ == "__main__":
    try:
        game_state = analyze_environment_behavior()
        analyze_health_detection_regions()
        provide_solutions()
        
        print("\n" + "="*50)
        print("NEXT STEPS:")
        print("1. Check the health region images in ./sf6_logs/")
        print("2. If they don't show health bars, recalibrate regions")
        print("3. Ensure SF6 is running in Training Mode")
        print("4. Consider using the CV calibration tool")
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        print("\nThis suggests a fundamental issue with the environment setup.")
        print("Please ensure all dependencies are installed and SF6 is running.")