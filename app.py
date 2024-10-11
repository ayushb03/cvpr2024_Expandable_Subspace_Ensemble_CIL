import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.model_selection import train_test_split

# Define Simple CNN Backbone
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # Assuming CIFAR-10 images are 32x32
        self.output_dim = 128

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)  # Flatten
        x = self.fc1(x)
        return x


# Adapter for task-specific subspace
class Adapter(nn.Module):
    def __init__(self, input_dim, reduction_dim):
        super(Adapter, self).__init__()
        self.down_projection = nn.Linear(input_dim, reduction_dim)
        self.activation = nn.ReLU()
        self.up_projection = nn.Linear(reduction_dim, input_dim)

    def forward(self, x):
        down = self.down_projection(x)
        activated = self.activation(down)
        up = self.up_projection(activated)
        return up + x  # Residual connection


# Main EASE model
class EASE(nn.Module):
    def __init__(self, backbone, adapter_dim=16):
        super(EASE, self).__init__()
        self.backbone = backbone
        self.adapters = nn.ModuleList()
        self.adapter_dim = adapter_dim
        self.prototypes = {}

    def add_adapter(self):
        input_dim = self.backbone.output_dim
        adapter = Adapter(input_dim, self.adapter_dim)
        self.adapters.append(adapter)

    def forward(self, x, task_idx):
        features = self.backbone(x)
        task_features = self.adapters[task_idx](features)
        return task_features


# Prototype Manager for Few-Shot Learning
class PrototypeManager:
    def __init__(self):
        self.prototypes = {}

    def extract_prototypes(self, model, task_idx, dataset, device):
        """Extract class prototypes for the current task using the model's adapter."""
        prototypes = {}
        original_targets = dataset.dataset.targets  # Access the original targets from the dataset

        for label in set(original_targets):
            class_indices = [i for i, target in enumerate(original_targets) if target == label]
            class_data = [dataset.dataset.data[i] for i in class_indices]  # Use indices to get data

            # Convert images to the correct shape and datatype
            class_features = []
            for x in class_data:
                # Ensure x is a tensor with shape [3, 32, 32]
                x_tensor = torch.tensor(x).permute(2, 0, 1).unsqueeze(0).float().to(device)  # [1, 3, 32, 32]
                class_features.append(model.forward(x_tensor, task_idx).detach())

            prototypes[label] = torch.mean(torch.stack(class_features), dim=0)

        return prototypes


    def complement_old_prototypes(self, old_prototypes, new_prototypes):
        """Complement old prototypes in the new subspace."""
        complemented_prototypes = {}
        similarity_matrix = torch.zeros(len(old_prototypes), len(new_prototypes)).to(next(iter(old_prototypes.values())).device)

        old_classes = list(old_prototypes.keys())
        new_classes = list(new_prototypes.keys())

        # Step 1: Compute cosine similarity between old and new prototypes
        for i, old_class in enumerate(old_classes):
            for j, new_class in enumerate(new_classes):
                if old_class in old_prototypes and new_class in new_prototypes:
                    # Check for correct dimensions
                    old_proto = old_prototypes[old_class]
                    new_proto = new_prototypes[new_class]

                    if old_proto.dim() == 1 and new_proto.dim() == 1:  # Ensure both are 1D tensors
                        similarity_matrix[i, j] = torch.cosine_similarity(old_proto.unsqueeze(0), new_proto.unsqueeze(0), dim=1)
                    else:
                        print(f"Skipping similarity computation for old class {old_class} and new class {new_class} due to dimension mismatch.")
                else:
                    print(f"Prototype missing for old class {old_class} or new class {new_class}.")

        # Step 2: Normalize the similarity matrix using softmax
        similarity_weights = torch.softmax(similarity_matrix, dim=1)

        # Step 3: Reconstruct old prototypes using new class prototypes
        for i, old_class in enumerate(old_classes):
            reconstructed_prototype = torch.zeros_like(new_prototypes[new_classes[0]])
            for j, new_class in enumerate(new_classes):
                reconstructed_prototype += similarity_weights[i, j] * new_prototypes[new_class]
            complemented_prototypes[old_class] = reconstructed_prototype

        return complemented_prototypes




# Training the model
def train_task(model, prototype_manager, dataloader, task_idx, device, epochs=10, lr=1e-3):
    model.add_adapter()  # Add a new adapter for the task
    optimizer = optim.Adam(model.adapters[task_idx].parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move inputs and labels to the same device as the model
            outputs = model(inputs, task_idx)  # Ensure output is on the correct device
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{epochs}] Loss: {avg_loss:.4f}")

    # Extract new prototypes for the current task
    new_prototypes = prototype_manager.extract_prototypes(model, task_idx, dataloader.dataset, device)
    if task_idx > 0:
        old_prototypes = prototype_manager.prototypes[task_idx - 1]
        complemented_prototypes = prototype_manager.complement_old_prototypes(old_prototypes, new_prototypes)
        prototype_manager.prototypes[task_idx - 1] = complemented_prototypes

    prototype_manager.prototypes[task_idx] = new_prototypes


# Prepare CIFAR-10 dataset and split it into 5 tasks
def prepare_dataloaders(task_idx, batch_size=64):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    cifar10_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    cifar10_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Split dataset into 5 tasks
    num_classes_per_task = 2
    classes_for_task = list(range(task_idx * num_classes_per_task, (task_idx + 1) * num_classes_per_task))

    task_train_idx = [i for i, label in enumerate(cifar10_train.targets) if label in classes_for_task]
    task_test_idx = [i for i, label in enumerate(cifar10_test.targets) if label in classes_for_task]

    task_train_dataset = Subset(cifar10_train, task_train_idx)
    task_test_dataset = Subset(cifar10_test, task_test_idx)

    train_loader = DataLoader(task_train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(task_test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def main(num_tasks=5, epochs=2, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize backbone and model
    backbone = SimpleCNN().to(device)
    model = EASE(backbone, adapter_dim=64).to(device)

    prototype_manager = PrototypeManager()

    # Loop over the tasks
    for task_index in range(num_tasks):
        print(f"Training on Task {task_index + 1}")

        # Prepare the data for the current task
        train_loader, test_loader = prepare_dataloaders(task_index, batch_size=batch_size)

        # Train the model on the current task
        train_task(model, prototype_manager, train_loader, task_index, device, epochs=epochs)

        print(f"Finished training on task {task_index + 1}")

    # Evaluate the model on the test dataset after all tasks have been trained
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for task_index in range(num_tasks):
            print(f"Evaluating on Task {task_index + 1}")
            _, test_loader = prepare_dataloaders(task_index, batch_size=batch_size)

            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)

                # Use the task-specific adapter for this task during evaluation
                adapted_features = model(images, task_index)
                _, predicted = torch.max(adapted_features, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

    print(f"Overall accuracy on the test set after {num_tasks} tasks: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    main(num_tasks=5)