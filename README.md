# SF6 Street Fighter 6 - Attempt at AI Trained Fighter

> 🚧 *Disclaimer: This project is an experimental and currently non-functional attempt to create an AI-trained fighter for Street Fighter 6 (SF6),
> inspired by a successful similar project built around Street Fighter 2 (SF2).*

## 📌 Project Summary

This repository documents an attempt to build an AI-trained Street Fighter 6 (SF6) character using machine learning techniques and gameplay emulation, modeled after a successful 
SF2 AI fighter implementation by another developer. While the implementation here did not achieve a fully working AI, the effort, structure,
and findings may serve as a valuable learning reference for others exploring AI and gaming.

---

## 🎯 Project Goals

- Recreate the structure used to train an AI to play Street Fighter 2, adapting it for Street Fighter 6.
- Automate gameplay capture, processing, and neural network training.
- Use computer vision and reinforcement learning to allow the AI to "learn" how to play.
- Create an AI that can make decisions and fight effectively in SF6.

---

## 🧱 Project Structure
```bash
Sf6-AI-Trained-Fighter/
      ├── community_projects/            # Placeholder for external contributions or forks
      ├── opt/                           # Optional configurations or utility scripts (currently undefined)
      ├── sf6_configs/                   # Game and training configuration files (e.g., keybinds, training settings)
      ├── sf6_env/                       # Environment setup and game interaction scripts
      ├── sf6_logs/                      # Logs generated during training, testing, or diagnostics
      ├── train/                         # Training routines, model versions, and AI logic scripts

      ├── README.md                      # Main project overview and instructions
      ├── SF6_README.md                  # Supplemental README (could contain legacy or extended notes)

          # Notebooks
            ├── StreetFighter-NoDelta.ipynb    # Notebook experiment (possibly without delta state comparisons)
            ├── StreetFighter-Test.ipynb       # Notebook for testing model interactions
            ├── StreetFighter-Tutorial.ipynb   # Tutorial or walkthrough of the AI training process

                  # Scripts & Utilities
                    ├── analyze_sf6_issue.py               # Debugging tool for analyzing issues in training or input parsing
                    ├── calabrate.json                     # JSON file likely used for screen or input calibration
                    ├── def calculate_reward(self, c.py    # Misnamed/malformed script (recommend renaming)
                    ├── fix_sf6_environment.py             # Script to patch/fix environment-related bugs
                    ├── import cv2.py                      # Misnamed script — likely meant to test OpenCV (recommend renaming)
                    ├── requirements.txt                   # Python dependencies required for the project
                    ├── setup_sf6_project.py               # Project setup and initialization helper
                    ├── sf6_diagnostic.py                  # Diagnostic script to test system/game compatibility or errors
                    ├── sf6_implementation_plan.md         # Written plan describing intended structure and methodology

      # Training Scripts
        ├── test_fixed_sf6.py              # Test script for improved/modified training pipeline
        ├── train_fixed_sf6.py             # Updated training script with bug fixes or improvements
        ├── train_sf6.py                   # Original/standard training script
        ├── train_sf6_enhanced.py          # Enhanced training script with experimental changes
```

---

## 🧠 AI Training Approach

The AI design was heavily inspired by:
- The **SF2 AI project**, which successfully used reinforcement learning with screen capture and gamepad simulation.
- A convolutional neural network (CNN) planned to interpret screen inputs and select controller actions.

**Core Concepts Attempted:**
- Frame-by-frame screen capture of SF6 using OBS or Windows API.
- Mapping visual input to in-game states using OpenCV or similar tools.
- Using reinforcement learning or supervised learning from recorded data.
- Outputting gamepad inputs to simulate human play.

**Tools & Libraries Considered:**
- Python
- PyTorch/TensorFlow
---

## 🚫 Known Limitations & Issues

This project is **not currently functional**. Key issues and challenges encountered:

- **SF6 is more complex and modern than SF2**, making screen parsing and input mapping harder.
- **Timing and latency** issues when simulating inputs.
- **No reinforcement learning rewards** were successfully implemented due to lack of internal game state access.
- **Difficulty capturing reliable training data**, especially without internal API hooks.
- **SF6 is not emulator-friendly**, limiting ability to intercept RAM/state like SF2 with MAME or SNES emulators.

Despite these blockers, the structure is in place and could be continued or forked by more experienced AI or computer vision developers.

---

## 💡 Lessons Learned

- Game complexity drastically impacts the feasibility of AI training via computer vision.
- Simpler or older games (like SF2) are more AI-training friendly due to emulation and accessible state info.
- Modern games often need more advanced hooks or custom SDKs to access meaningful state data.
- AI training without a reward system or reliable state metrics leads to poor/no learning.

---

## 📁 How to Use (for Exploration)

While this repo is non-functional, you can still explore:
1. The data capture scripts to understand the collection method.
2. The neural network architecture drafts in `models/`.
3. The overall pipeline concept for AI training in a gaming context.

---

## 🙏 Credits & Inspiration

- **Original SF2 AI Project** (inspiration for this repo): [Insert link if available]
- 'https://github.com/nicknochnack/StreetFighterRL'

---

