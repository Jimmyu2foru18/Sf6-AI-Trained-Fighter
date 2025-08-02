
#!/usr/bin/env python3
"""
Test the Fixed SF6 Environment
"""

import sys
import os
import time
sys.path.append('./sf6_env')
sys.path.append('./sf6_configs')

from street_fighter_6_fixed import StreetFighter6Fixed

def test_fixed_environment():
    """Test the fixed SF6 environment"""
    print("Testing Fixed SF6 Environment")
    print("="*40)
    
    # Create environment
    env = StreetFighter6Fixed(character='ryu', debug_mode=True)
    
    print(f"Game connected: {env.game_connected}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Test reset
    obs, info = env.reset()
    print(f"
Initial state:")
    print(f"  Player health: {info['player_health']}")
    print(f"  Enemy health: {info['enemy_health']}")
    print(f"  Observation shape: {obs.shape}")
    
    # Test steps
    print(f"
Testing 20 steps...")
    total_reward = 0
    
    for i in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if i % 5 == 0:  # Print every 5 steps
            print(f"  Step {i+1}: R={reward:.3f}, P_HP={info['player_health']:.1f}, E_HP={info['enemy_health']:.1f}")
        
        if terminated or truncated:
            print(f"  Episode ended at step {i+1}")
            break
        
        time.sleep(0.1)  # Small delay
    
    print(f"
Total reward: {total_reward:.3f}")
    print(f"Final player health: {info['player_health']:.1f}")
    print(f"Final enemy health: {info['enemy_health']:.1f}")
    
    env.close()
    print("
[OK] Test completed successfully!")

if __name__ == "__main__":
    test_fixed_environment()
