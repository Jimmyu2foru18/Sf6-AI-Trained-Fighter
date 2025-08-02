# Street Fighter 6 Reinforcement Learning Implementation Plan

## Overview
This document outlines the implementation strategy for adapting your existing Street Fighter II reinforcement learning project to work with Street Fighter 6, combining computer vision approaches with community projects.

## Phase 1: Clone and Setup Community Projects

### 1.1 SmartCV-SF6 (Computer Vision Approach)
```bash
git clone https://github.com/community/SmartCV-SF6
```
- **Purpose**: Pixel detection and OCR for real-time data extraction
- **Integration**: Use for health bar detection, timer reading, and UI element recognition
- **Advantages**: No game modification required

### 1.2 Fluffy Mod Integration
```bash
git clone https://github.com/community/fluffy-mod-sf6
```
- **Purpose**: Access to in-game script variables and frame data
- **Integration**: Real-time access to:
  - Player health and meter values
  - Frame data and timing
  - Input history
  - Character states
- **Advantages**: More accurate data than computer vision

### 1.3 AIVO-StreetFighter (Reference Architecture)
```bash
git clone https://github.com/corbosiny/AIVO-StreetFigherReinforcementLearning
```
- **Purpose**: Reference implementation for tournament-style training
- **Integration**: Adapt training platform architecture
- **Key Components**: Agent interface, training environment, logging system

## Phase 2: Environment Architecture

### 2.1 Hybrid Environment Class
```python
class StreetFighter6Hybrid(Env):
    def __init__(self, use_mod=True, use_cv=True):
        super().__init__()
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.action_space = MultiBinary(12)
        
        # Initialize data sources
        self.mod_interface = FluffyModInterface() if use_mod else None
        self.cv_interface = SmartCVInterface() if use_cv else None
        self.input_controller = SF6InputController()
        
    def get_game_state(self):
        """Get game state from available sources"""
        if self.mod_interface and self.mod_interface.is_connected():
            return self.mod_interface.get_state()
        elif self.cv_interface:
            return self.cv_interface.extract_state()
        else:
            raise RuntimeError("No data source available")
```

### 2.2 Data Source Interfaces

#### Fluffy Mod Interface
```python
class FluffyModInterface:
    def __init__(self):
        self.connection = None
        self.connect_to_mod()
    
    def get_state(self):
        """Extract real-time game data from mod"""
        return {
            'player_health': self.get_player_health(),
            'enemy_health': self.get_enemy_health(),
            'player_meter': self.get_player_meter(),
            'frame_data': self.get_frame_data(),
            'inputs': self.get_input_history()
        }
```

#### Computer Vision Interface
```python
class SmartCVInterface:
    def __init__(self):
        self.screen_capture = mss.mss()
        self.health_templates = self.load_health_templates()
        self.ocr_engine = pytesseract
    
    def extract_state(self):
        """Extract game state using computer vision"""
        screenshot = self.capture_screen()
        return {
            'player_health': self.detect_health_bar(screenshot, 'player'),
            'enemy_health': self.detect_health_bar(screenshot, 'enemy'),
            'timer': self.read_timer(screenshot),
            'round_info': self.detect_round_info(screenshot)
        }
```

## Phase 3: Input Control System

### 3.1 SF6 Input Controller
```python
class SF6InputController:
    def __init__(self):
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
    
    def send_action(self, action_array):
        """Convert action array to keyboard inputs"""
        for i, pressed in enumerate(action_array):
            if pressed:
                pyautogui.keyDown(self.key_mapping[i])
            else:
                pyautogui.keyUp(self.key_mapping[i])
```

## Phase 4: Training Integration

### 4.1 Adapt Existing Training Loop
```python
# Modify your existing training setup
class SF6TrainingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.performance_tracker = SF6PerformanceTracker()
    
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            # Save model and log performance
            model_path = os.path.join(self.save_path, f'sf6_model_{self.n_calls}')
            self.model.save(model_path)
            
            # Log SF6-specific metrics
            self.performance_tracker.log_metrics(self.locals, self.globals)
        return True
```

### 4.2 Reward Function Adaptation
```python
def calculate_sf6_reward(self, prev_state, current_state, action):
    """SF6-specific reward calculation"""
    # Health-based rewards (similar to your existing system)
    health_reward = (prev_state['enemy_health'] - current_state['enemy_health']) * 2
    damage_penalty = (current_state['player_health'] - prev_state['player_health'])
    
    # SF6-specific rewards
    meter_reward = (current_state['player_meter'] - prev_state['player_meter']) * 0.1
    combo_reward = self.detect_combo_bonus(prev_state, current_state)
    
    # Frame advantage rewards (if using mod data)
    frame_reward = self.calculate_frame_advantage_reward(current_state)
    
    return health_reward + damage_penalty + meter_reward + combo_reward + frame_reward
```

## Phase 5: Implementation Steps

### Step 1: Repository Setup
1. Clone community projects into subdirectories
2. Install dependencies for each project
3. Test individual components

### Step 2: Environment Development
1. Create base SF6 environment class
2. Implement computer vision interface
3. Integrate Fluffy mod interface (if available)
4. Test data extraction accuracy

### Step 3: Input System
1. Implement keyboard/controller input mapping
2. Test input responsiveness and timing
3. Calibrate input delays for optimal performance

### Step 4: Training Adaptation
1. Modify existing PPO training loop
2. Adapt reward functions for SF6
3. Update hyperparameters for new environment
4. Implement SF6-specific logging and metrics

### Step 5: Testing and Optimization
1. Test environment stability
2. Validate data accuracy (mod vs CV)
3. Optimize performance and reduce latency
4. Fine-tune reward functions

## Phase 6: Advanced Features

### 6.1 Multi-Character Support
```python
class SF6MultiCharacterEnv(StreetFighter6Hybrid):
    def __init__(self, character='ryu'):
        super().__init__()
        self.character = character
        self.character_specific_rewards = self.load_character_rewards(character)
```

### 6.2 Training Mode Integration
- Utilize SF6's training mode features
- Implement scenario-based training
- Use recording function for consistent opponent behavior

### 6.3 Tournament Mode
- Adapt AIVO's tournament architecture
- Implement agent vs agent training
- Create ranking and evaluation systems

## Dependencies

```bash
# Computer Vision
pip install opencv-python
pip install pytesseract
pip install mss
pip install pillow

# Input Control
pip install pyautogui
pip install pynput

# Existing RL Dependencies
pip install stable-baselines3
pip install gym
pip install numpy
pip install torch

# Mod Integration (if available)
pip install websockets  # For mod communication
pip install json
```

## Expected Outcomes

1. **Robust Data Collection**: Hybrid approach ensures data availability
2. **Accurate State Representation**: Mod data provides precise game state
3. **Reliable Input Control**: Keyboard/controller integration for consistent actions
4. **Scalable Architecture**: Support for multiple characters and game modes
5. **Community Integration**: Leverage existing SF6 AI development efforts

## Next Steps

1. Begin with computer vision approach for immediate testing
2. Integrate Fluffy mod when available
3. Adapt your existing SF2 models as starting points
4. Gradually expand to multi-character and tournament modes

This implementation plan provides a comprehensive roadmap for transitioning your Street Fighter II reinforcement learning project to Street Fighter 6 while leveraging community resources and maintaining the core architecture you've already developed.