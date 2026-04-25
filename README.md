<div align="center">

# 🤖 Autonomous Robotic Arm Control
### using Reinforcement Learning (TD3)

![Python](https://img.shields.io/badge/Python-3.10.8-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![Robosuite](https://img.shields.io/badge/Sim-Robosuite-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

<br>

![Demo](assets/demo.gif)

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
<img src="assets/training_graph.png" width="800" alt="Training Graph">
</div>

---

## 🛠️ Installation

1.  **Clone the Repo**:
```bash
git clone https://github.com/Omkarkkale/Autonomous-Robotic-Arm-Control-using-Reinforcement-Learning.git
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
       
       %% Critic Section
       subgraph Critic_Setup [Value Estimation]
           Batch(Sample Mini-Batch)
           Critic1[Critic 1 Network]
           Critic2[Critic 2 Network]
           TargetC[Target Critics]
           LossC(Minimize MSE Loss)
       end
       
       %% Actor Section
       subgraph Actor_Setup [Policy Optimization]
           Actor[Actor Network]
           TargetA[Target Actor]
           LossA(Maximize Q-Value)
       end
   end

   %% --- CONNECTIONS & FLOW ---

   %% 1. Interaction Flow
   Env -->|Generates| State
   State -->|Input to| Actor
   Actor -->|Add Exploration Noise| Action
   Action -->|Execute| Env
   Env -->|Feedback| Reward

   %% 2. Storage Flow
   State & Action & Reward -->|Store Transition| Buffer

   %% 3. Training Flow
    Buffer -->|Random Batch (256)| Batch
    Buffer -->|Random Batch 256| Batch
   
   %% Critic Update (Twin Delayed)
   Batch --> Critic1 & Critic2
   Batch --> TargetC & TargetA
   TargetA -->|Next Action| TargetC
   TargetC -->|Target Q-Value| LossC
   LossC -->|Backpropagate| Critic1 & Critic2

   %% Actor Update (Delayed)
   Batch --> Actor
   Actor -->|Proposed Action| Critic1
   Critic1 -->|Policy Gradient| LossA
   LossA -->|Backpropagate| Actor

   %% Stability (Polyak Averaging)
   Actor -.->|Soft Update| TargetA
   Critic1 & Critic2 -.->|Soft Update| TargetC
```

*   **Actor:** Maps states to continuous actions (Joint Velocities).
*   **Critic (x2):** Estimates the Q-value of state-action pairs (Twin Critics to reduce bias).
*   **Target Networks:** Normalized using Polyak averaging for stability.

---

## 🔗 References
*   [TD3 Paper (Fujimoto et al.)](https://arxiv.org/abs/1802.09477)
*   [Robosuite Documentation](https://robosuite.ai/)

<div align="center">
<sub>Built by Omkar Kale</sub>
</div>
