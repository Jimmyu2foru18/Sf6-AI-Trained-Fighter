# Street Fighter 6 Reinforcement Learning Project

This project adapts your existing Street Fighter II reinforcement learning setup to work with Street Fighter 6, integrating community projects and advanced computer vision techniques.

## 🚀 Quick Start

### 1. Initial Setup
```bash
# Run the automated setup script
python setup_sf6_project.py

# Install additional dependencies
pip install -r requirements.txt
```

### 2. Computer Vision Calibration
```bash
# Launch the CV calibration tool
python sf6_env/cv_calibration_tool.py
```

### 3. Training
```bash
# Basic training
python train_sf6_enhanced.py --mode train --character ryu

# Hyperparameter optimization
python train_sf6_enhanced.py --mode optimize --trials 50

# Resume training from checkpoint
python train_sf6_enhanced.py --mode train --model ./sf6_models/checkpoint.zip
```

### 4. Testing
```bash
# Test a trained model
python train_sf6_enhanced.py --mode test --model ./sf6_models/best_model.zip --episodes 10
```

## 📁 Project Structure

```
StreetFighterRL-main/
├── sf6_env/                          # SF6 Environment Implementation
│   ├── street_fighter_6_base.py      # Basic SF6 environment
│   ├── street_fighter_6_enhanced.py  # Enhanced environment with CV
│   ├── cv_calibration_tool.py        # GUI tool for CV calibration
│   ├── interfaces/                   # Interface modules
│   ├── controllers/                  # Input controllers
│   └── utils/                        # Utility functions
├── sf6_configs/                      # Configuration Files
│   ├── sf6_config.py                # Main configuration system
│   └── *.json                       # Saved configurations
├── sf6_models/                       # Trained Models
├── sf6_logs/                         # Training Logs & TensorBoard
├── community_projects/               # Integrated Community Projects
│   ├── AIVO-StreetFighter/          # Training platform architecture
│   ├── street-fighter-ai/           # Advanced RL implementations
│   ├── Street-Fighter-AI-DQN/       # DQN approach
│   └── Retro-Street-Fighter-RL/     # Additional RL techniques
├── train_sf6_enhanced.py            # Enhanced training script
├── setup_sf6_project.py             # Automated setup
├── requirements.txt                  # Dependencies
└── sf6_implementation_plan.md       # Implementation strategy
```

## 🎮 Street Fighter 6 Setup

### Recommended Game Settings
1. **Display Mode**: Windowed or Borderless Windowed
2. **Resolution**: 1920x1080 (adjust monitor settings in config if different)
3. **Training Mode**: Use Training Mode for consistent environment
4. **Character**: Start with Ryu (default) or specify with `--character`

### Training Mode Setup
1. Go to **Training Mode**
2. Select your character and opponent
3. Set **Dummy Settings**:
   - Guard: Standing/Crouching (varies)
   - Counter Attack: Off (initially)
   - Recovery: Normal
4. Position characters in center stage

## 🔧 Configuration System

### Using the Configuration System
```python
from sf6_configs.sf6_config import SF6Config

# Load default configuration
config = SF6Config()

# Modify settings
config.training.learning_rate = 1e-4
config.reward.damage_dealt_multiplier = 3.0

# Save configuration
config.save_config('./sf6_configs/my_config.json')

# Load saved configuration
config = SF6Config.load_config('./sf6_configs/my_config.json')
```

### Key Configuration Categories
- **ScreenConfig**: Monitor settings, health bar regions
- **InputConfig**: Key mappings, input timing
- **EnvironmentConfig**: Observation space, episode settings
- **RewardConfig**: Reward function parameters
- **TrainingConfig**: PPO hyperparameters, training settings
- **CVConfig**: Computer vision parameters

## 🖥️ Computer Vision Calibration

### Using the CV Calibration Tool
1. **Launch Tool**: `python sf6_env/cv_calibration_tool.py`
2. **Start SF6**: Position game window appropriately
3. **Capture Frame**: Use "Capture Screenshot" or "Start Live Capture"
4. **Select Regions**: Click region buttons and drag to select areas
   - Player Health Bar
   - Enemy Health Bar
   - Player Super Meter
   - Enemy Super Meter
   - Timer (optional)
5. **Test Detection**: Use "Test Health Detection" and "Test Super Detection"
6. **Save Configuration**: Export settings for use in training

### Manual Region Configuration
If you prefer manual setup, edit the regions in `sf6_configs/sf6_config.py`:
```python
# Health bar regions (x, y, width, height)
player_health_region: Tuple[int, int, int, int] = (100, 50, 400, 80)
enemy_health_region: Tuple[int, int, int, int] = (1520, 50, 400, 80)
```

## 🤖 Training Modes

### 1. Basic Training
```bash
python train_sf6_enhanced.py --mode train
```

### 2. Hyperparameter Optimization
```bash
# Run optimization study
python train_sf6_enhanced.py --mode optimize --trials 100 --timesteps 50000

# Train with optimized parameters
python train_sf6_enhanced.py --mode train --config ./sf6_configs/best_hyperparams_*.json
```

### 3. Character-Specific Training
```bash
# Train different characters
python train_sf6_enhanced.py --mode train --character ryu
python train_sf6_enhanced.py --mode train --character chun-li
python train_sf6_enhanced.py --mode train --character ken
```

### 4. Resume Training
```bash
python train_sf6_enhanced.py --mode train --model ./sf6_models/checkpoint_100000.zip
```

## 📊 Monitoring Training

### TensorBoard
```bash
# Launch TensorBoard
tensorboard --logdir ./sf6_logs/

# View at http://localhost:6006
```

### Key Metrics to Monitor
- **Episode Reward**: Overall performance
- **Episode Length**: How long episodes last
- **Health Differential**: Damage dealt vs taken
- **Learning Rate**: Training progress
- **Policy Loss**: Model optimization

## 🎯 Community Project Integration

### AIVO-StreetFighter
- **Architecture**: Multi-agent training framework
- **Features**: Tournament mode, character diversity
- **Integration**: Training loop structure, evaluation metrics

### street-fighter-ai
- **Techniques**: Advanced reward shaping, curriculum learning
- **Features**: Frame analysis, combo detection
- **Integration**: Reward function design, state representation

### Street-Fighter-AI-DQN
- **Algorithm**: Deep Q-Network implementation
- **Features**: Experience replay, target networks
- **Integration**: Alternative to PPO for comparison

### Retro-Street-Fighter-RL
- **Techniques**: Various RL algorithms
- **Features**: Environment wrappers, preprocessing
- **Integration**: Environment design patterns

## 🔍 Troubleshooting

### Common Issues

#### 1. Health Bar Detection Not Working
- **Solution**: Use CV calibration tool to recalibrate regions
- **Check**: Game resolution matches monitor settings
- **Verify**: Health bar colors in HSV ranges

#### 2. Input Not Registering
- **Solution**: Check key mappings in configuration
- **Verify**: SF6 window has focus
- **Test**: Manual key presses work in game

#### 3. Training Crashes
- **Solution**: Reduce batch size or learning rate
- **Check**: Available memory and GPU usage
- **Verify**: Environment reset functionality

#### 4. Poor Performance
- **Solution**: Adjust reward function parameters
- **Try**: Different hyperparameters via optimization
- **Check**: Training environment consistency

### Debug Mode
Enable debug mode in configuration:
```python
config.cv.save_debug_images = True
config.cv.debug_image_path = "./sf6_logs/debug_images/"
```

## 🚀 Advanced Features

### Multi-Character Training
```python
# Train on multiple characters
characters = ['ryu', 'chun-li', 'ken', 'luke']
for character in characters:
    # Train individual models
    # Or use character-agnostic training
```

### Tournament Mode
```python
# Implement tournament between trained models
# Compare different training approaches
# Evaluate against human players
```

### Mod Integration (Future)
```python
# Integration with Fluffy mod for direct game state access
# Real-time frame data extraction
# Advanced input timing
```

## 📈 Performance Optimization

### Training Speed
- Use GPU acceleration for PPO training
- Optimize screen capture frequency
- Reduce observation resolution if needed
- Use vectorized environments for parallel training

### Memory Usage
- Limit frame history size
- Use efficient image preprocessing
- Clear debug images periodically
- Monitor memory usage during training

## 🤝 Contributing

### Adding New Characters
1. Add character config in `sf6_config.py`
2. Define character-specific key mappings
3. Implement character-specific reward functions
4. Test with CV calibration tool

### Improving Computer Vision
1. Enhance health bar detection algorithms
2. Add combo counter detection
3. Implement move recognition
4. Add training mode state detection

### Algorithm Improvements
1. Implement alternative RL algorithms (SAC, TD3)
2. Add curriculum learning
3. Implement self-play training
4. Add imitation learning from human play

## 📚 References

- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [OpenAI Gym Documentation](https://gym.openai.com/)
- [Street Fighter 6 Official Site](https://www.streetfighter.com/6)
- [Computer Vision with OpenCV](https://opencv.org/)

## 📄 License

This project builds upon your existing SF2 RL work and integrates various community projects. Please respect the licenses of the integrated community projects.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the configuration documentation
3. Test with the CV calibration tool
4. Check community project documentation

---

**Note**: This is an advanced implementation that requires Street Fighter 6, proper setup, and calibration. Start with the basic environment and gradually move to the enhanced features as you become familiar with the system.
