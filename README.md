# EFEDQN for STSP-TVRAR

This repository provides the official PyTorch implementation of the **EFEDQN** algorithm, designed to solve the Asymmetric Time-Varying Steiner Traveling Salesman Problem with Road Access Restrictions (STSP-TVRAR). 

## 1. Key Features
- **Explicit Edge Embedding & Step-Wise Re-Encoding:** The `Net.py` implements a bespoke Graph Attention framework that refreshes spatio-temporal features step-by-step, naturally avoiding the pitfalls of static problem reduction.
- **Decoupled Architecture:** Separates graph representation learning from sequential pathfinding logic using a highly optimized Multi-Head Attention decoder.
- **Advanced DRL Agent:** Features a Value-based Double Deep Q-Network (DDQN) combined with Prioritized Experience Replay (PER) and Importance Sampling weights (`Agent.py` & `Replaybuffer.py`).
- **Inference-Time Instance Augmentation:** The testing phase natively incorporates an augmentation strategy (random permutation of edge indices) to generate diverse routing plans and improve zero-shot generalization on large-scale networks.

## 2. Repository Structure
All core scripts are organized in the root directory for straightforward execution:
- `train.py`: The main entry script. It handles synthetic instance generation, model training (up to 5000 episodes), and periodic zero-shot evaluation on varying network scales (25 to 100 nodes).
- `Net.py`: Contains the Neural Network architectures (`TransformerConv`, `CrossAttention`, `MultiHeadAttention`, and the complete `Net_DQN`).
- `Agent.py`: Implements the Deep Reinforcement Learning interactions, Q-value estimations, and network synchronization.
- `Env.py`: The custom RL environment modeling the complex traffic networks, time-windows, and step-wise transitions.
- `Graph.py`: The topology generator for asymmetric, weakly-connected directed graphs with random transit attributes.
- `Replaybuffer.py`: The memory buffer supporting prioritized sampling based on TD-errors.

## 3. Dependencies
The framework is built with Python 3.8+ and PyTorch. Please install the required dependencies:
```bash
pip install torch numpy networkx pandas matplotlib openpyxl
