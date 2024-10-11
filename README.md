# Expandable Subspace Ensemble for Pre-Trained Model-Based Class-Incremental Learning 🌟

## Overview 📚

This repository implements an **EASE (Efficient Adapter for Subspace Expansion)** paper using PyTorch, designed for incremental learning on the CIFAR-10 dataset. The model consists of a simple CNN backbone and task-specific adapters that allow for efficient learning of new classes while preserving knowledge of previously learned tasks.

## Features ⚙️

- **Incremental Learning**: Trains the model on multiple tasks without forgetting previously learned classes. 🔄
- **Prototype Management**: Extracts and manages class prototypes for few-shot learning. 🏷️
- **Modular Architecture**: Easy to extend with different backbones and adapters. 🛠️

## Requirements 📦

- Python 3.6+
- PyTorch 1.7+
- torchvision
- NumPy

## Installation 🖥️

1. Create an environment
2. Clone the repository
3. Install the required packages
4. Run the main.py for a high level implementation, app.py file for a more robust implementation
   
