<div align="center">

# 🤖 Autonomous Robotic Arm Control
### using Reinforcement Learning (TD3)

![Python](https://img.shields.io/badge/Python-3.10.8-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![Robosuite](https://img.shields.io/badge/Sim-Robosuite-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

<br>

![Demo](Assets/demo.gif)

**Training a Franka Emika Panda robot to open a door using the Twin Delayed DDPG (TD3) algorithm.**

</div>

---

## 📖 Overview

This project demonstrates the power of **Deep Reinforcement Learning** in continuous control robotics. Using the **Robosuite** simulation framework (powered by MuJoCo), we trained an agent to solve the complex `Door` task from scratch.

The agent operates in a continuous state and action space, learning to coordinate 7 degrees of freedom to approach, grasp, rotate, and push the door handle.

## 🚀 Key Features

-   **🧠 Advanced RL:** Custom implementation of **TD3** (Twin Delayed Deep Deterministic Policy Gradient), improving over DDPG by reducing overestimation bias.
-   **🦾 High-Fidelity Sim:** Built on **Robosuite**, offering realistic physics and collisions.
-   **📈 Proven Convergence:** Solved the environment with a stable high score of **~275**.
-   **🛡️ Robustness:** Includes mechanisms for **Checkpointing** and **Best Model Preservation**.

---

## 📊 Performance

The agent was trained for **8,500+ episodes**. It demonstrates a clear "S-curve" learning trajectory, mastering the task after an initial exploration phase.

<div align="center">
<img src="Assets/training_graph.png" width="800" alt="Training Graph">
</div>

---

## 🛠️ Installation

1.  **Clone the Repo**:
```bash
git clone https://github.com/ranjitsatpute/Intelligent-Robotic-Arm-Control-via-Reinforcement-Learning.git
cd Autonomous-Robotic-Arm-Control-using-Reinforcement-Learning
```

2.  **Setup Environment** (Recommended):
*Requires Python 3.10.8*
```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # source venv/bin/activate # Linux/Mac
   ```

3.  **Install Dependencies**:
```bash
   pip install -r requirements.txt
   ```

---

## 🕹️ Usage

### 👁️ Watch the AI (Visualization)
To see the trained agent in action:
```bash
python test.py
```
*Loads the best weights and renders the simulation.*

### 🏋️ Train from Scratch
To start a new training session:
```bash
python main.py
```
*Logs are saved to `logs/` (viewable with TensorBoard).*

---

## 🧠 Model Architecture (TD3)

The solution uses an **Actor-Critic** architecture with **Twin Delayed** stabilization:

```mermaid
graph TD
    %% 1. INTERACTION LOOP (The "What Happened" Phase)
    subgraph Environment_Interaction [Interaction Phase]
        direction TB
        Env[Robosuite Environment]
        State[State Vector]
        Action[Action + Noise]
        Reward[Reward Signal]
    end

    %% 2. MEMORY (The "Remember" Phase)
    subgraph Memory_System [Memory Phase]
        Buffer[(Replay Buffer<br>Capacity 1M)]
    end

    %% 3. LEARNING (The "Improve" Phase)
    subgraph Learning_System [Training Phase - TD3]
        direction TB
        
        subgraph Critic_Setup [Value Estimation]
            Batch(Sample Mini-Batch)
            Critic1[Critic 1 Network]
            Critic2[Critic 2 Network]
            TargetC[Target Critics]
            LossC(Minimize MSE Loss)
        end
        
        subgraph Actor_Setup [Policy Optimization]
            Actor[Actor Network]
            TargetA[Target Actor]
            LossA(Maximize Q-Value)
        end
    end

    Env --> State
    State --> Actor
    Actor --> Action
    Action --> Env
    Env --> Reward

    State --> Buffer
    Action --> Buffer
    Reward --> Buffer

    Buffer --> Batch
    
    Batch --> Critic1
    Batch --> Critic2
    Batch --> TargetC
    Batch --> TargetA

    TargetA --> TargetC
    TargetC --> LossC
    LossC --> Critic1
    LossC --> Critic2

    Batch --> Actor
    Actor --> Critic1
    Critic1 --> LossA
    LossA --> Actor

    Actor -.-> TargetA
    Critic1 -.-> TargetC
    Critic2 -.-> TargetC
```
## 🔗 References
*   [TD3 Paper (Fujimoto et al.)](https://arxiv.org/abs/1802.09477)
*   [Robosuite Documentation](https://robosuite.ai/)

<div align="center">
<sub>Built by Ranjit Satpute</sub>
</div>
