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
   
# Class Methods Documentation

## SimpleCNN

### `__init__(self)`
- **Purpose**: Initializes the SimpleCNN model.
- **Inputs**: None.
- **Outputs**: Configures the model architecture, including convolutional and pooling layers, as well as the fully connected layer.

### `forward(self, x)`
- **Purpose**: Defines the forward pass of the model.
- **Inputs**: 
  - `x`: Input tensor of shape `(N, 3, 32, 32)` (N = batch size).
- **Outputs**: 
  - Feature tensor of shape `(N, 128)` after processing through the network.

---

## Adapter

### `__init__(self, input_dim, reduction_dim)`
- **Purpose**: Initializes the Adapter with specified input and reduced dimensions.
- **Inputs**: 
  - `input_dim`: Dimension of the input feature vector.
  - `reduction_dim`: Dimension for the reduced representation.
- **Outputs**: Configures the down-projection, activation, and up-projection layers.

### `forward(self, x)`
- **Purpose**: Defines the forward pass with a residual connection.
- **Inputs**: 
  - `x`: Input tensor (feature vector).
- **Outputs**: 
  - Adapted feature vector with the same dimension as input.

---

## EASE

### `__init__(self, backbone, reduction_dim=64, num_classes=10)`
- **Purpose**: Initializes the EASE model with a backbone and task specifications.
- **Inputs**: 
  - `backbone`: Instance of the `SimpleCNN`.
  - `reduction_dim`: Dimensionality for the adapters (default: 64).
  - `num_classes`: Number of output classes (default: 10).
- **Outputs**: Configures the model with adapters and classifiers for multiple tasks.

### `add_task(self)`
- **Purpose**: Adds a new adapter and classifier for a new task.
- **Inputs**: None.
- **Outputs**: Updates the model to include a new task-specific adapter and classifier.

### `forward(self, x, task_idx)`
- **Purpose**: Defines the forward pass through the backbone, adapter, and classifier for the specified task.
- **Inputs**: 
  - `x`: Input tensor for prediction.
  - `task_idx`: Index of the current task.
- **Outputs**: 
  - Class logits for the input tensor corresponding to the task.

---

## PrototypeManager

### `__init__(self)`
- **Purpose**: Initializes the PrototypeManager.
- **Inputs**: None.
- **Outputs**: Configures an empty dictionary to hold prototypes.

### `extract_prototypes(self, model, task_idx, dataset, device)`
- **Purpose**: Extracts class prototypes for a specific task.
- **Inputs**: 
  - `model`: Instance of the EASE model.
  - `task_idx`: Index of the current task.
  - `dataset`: Dataset used for prototype extraction.
  - `device`: Device (CPU/GPU) for computations.
- **Outputs**: 
  - `prototypes`: Dictionary of normalized prototypes for each class in the task.

### `complement_old_prototypes(self, old_prototypes, new_prototypes)`
- **Purpose**: Synthesizes old prototypes in the new semantic space using similarity.
- **Inputs**: 
  - `old_prototypes`: Dictionary of prototypes from the previous task.
  - `new_prototypes`: Dictionary of prototypes from the current task.
- **Outputs**: 
  - `complemented`: Dictionary of complemented old prototypes in the new subspace.

# Functions Documentation

## `weighted_inference(model, images, task_idx, prototypes)`
- **Purpose**: Performs inference across tasks using class prototypes for weighted logits.
- **Inputs**: 
  - `model`: Instance of the EASE model.
  - `images`: Input batch of images.
  - `task_idx`: Current task index.
  - `prototypes`: Dictionary of class prototypes for the current task.
- **Outputs**: 
  - Aggregated logits tensor after applying weights based on prototype similarity.

---

## `train_task(model, prototype_manager, dataloader, task_idx, device, epochs=10, lr=1e-3)`
- **Purpose**: Trains the model on a specific task and updates prototypes.
- **Inputs**: 
  - `model`: Instance of the EASE model.
  - `prototype_manager`: Instance of the PrototypeManager.
  - `dataloader`: DataLoader for the training data.
  - `task_idx`: Index of the current task.
  - `device`: Device for computations.
  - `epochs`: Number of training epochs (default: 10).
  - `lr`: Learning rate (default: 1e-3).
- **Outputs**: 
  - Updates the model and prototype manager with new task adapters and prototypes.

---

## `evaluate_model(model, prototype_manager, test_loader, device, task_idx)`
- **Purpose**: Evaluates the model on the test dataset for the current task.
- **Inputs**: 
  - `model`: Instance of the EASE model.
  - `prototype_manager`: Instance of the PrototypeManager.
  - `test_loader`: DataLoader for the test data.
  - `device`: Device for computations.
  - `task_idx`: Index of the current task.
- **Outputs**: 
  - Prints the accuracy of the model on the test dataset for the current task.

---

## `prepare_dataloaders(task_idx, batch_size=64)`
- **Purpose**: Prepares training and testing data loaders for a specific task.
- **Inputs**: 
  - `task_idx`: Index of the current task.
  - `batch_size`: Batch size for DataLoader (default: 64).
- **Outputs**: 
  - `train_loader`: DataLoader for the training data of the current task.
  - `test_loader`: DataLoader for the test data of the current task.
