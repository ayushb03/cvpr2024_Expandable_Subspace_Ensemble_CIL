import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np


# Define transforms for the dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 Dataset
cifar10_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
cifar10_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Create DataLoader for train and test
train_loader = DataLoader(cifar10_train, batch_size=32, shuffle=True)
test_loader = DataLoader(cifar10_test, batch_size=32, shuffle=False)


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # CIFAR-10 images are 32x32
        self.output_dim = 128

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.fc1(x)
        return x

class Adapter(nn.Module):
    def __init__(self, input_dim, reduction_dim):
        super(Adapter, self).__init__()
        self.down_projection = nn.Linear(input_dim, reduction_dim)
        self.activation = nn.ReLU()
        self.up_projection = nn.Linear(reduction_dim, input_dim)

    def forward(self, x):
        down = self.down_projection(x)  # shape: [batch_size, reduction_dim]
        activated = self.activation(down)  # shape: [batch_size, reduction_dim]
        up = self.up_projection(activated)  # shape: [batch_size, input_dim]
        return up + x  # Now x is of shape [batch_size, input_dim]



class EASE(nn.Module):
    def __init__(self, backbone, num_tasks, reduction_dim):
        super(EASE, self).__init__()
        self.backbone = backbone
        self.adapters = nn.ModuleList([Adapter(backbone.output_dim, reduction_dim) for _ in range(num_tasks)])
        self.num_tasks = num_tasks
        self.prototypes = []

    def forward(self, x, task_index):
        features = self.backbone(x)
        adapted_features = self.adapters[task_index](features)
        return adapted_features, features  # Return both adapted and raw features

    def extract_prototypes(self, features, labels, task_index):
        if task_index >= len(self.prototypes):
            self.prototypes.append({})  # Create a new dictionary for the new task
        for label in labels.unique():
            if label.item() not in self.prototypes[task_index]:
                self.prototypes[task_index][label.item()] = []
        self.prototypes[task_index][label.item()].append(features)

    def get_class_prototypes(self, task_index):
        return {label: torch.mean(torch.stack(features), dim=0) for label, features in self.prototypes[task_index].items()}

    def semantic_mapping(self, old_protos, new_protos):
        new_mapped_protos = {}
        for old_label, old_proto in old_protos.items():
            # Compute similarities to new prototypes
            similarities = {new_label: F.cosine_similarity(old_proto.unsqueeze(0), proto.unsqueeze(0)).item()
                            for new_label, proto in new_protos.items()}
            # Get the new prototype that is most similar
            most_similar_label = max(similarities, key=similarities.get)
            new_mapped_protos[old_label] = new_protos[most_similar_label]
        return new_mapped_protos


def train(model, data_loader, optimizer, criterion, task_index):
    model.train()
    for images, labels in data_loader:
        # Move images and labels to the same device as the model
        images, labels = images.to(next(model.parameters()).device), labels.to(next(model.parameters()).device)
        
        optimizer.zero_grad()
        adapted_features, raw_features = model(images, task_index)
        loss = criterion(adapted_features, labels)
        loss.backward()
        optimizer.step()
        model.extract_prototypes(raw_features, labels, task_index)

def infer(model, images, task_index):
    model.eval()
    # Move images to the same device as the model
    images = images.to(next(model.parameters()).device)
    
    with torch.no_grad():
        adapted_features, _ = model(images, task_index)
        return adapted_features  # Modify this to return class predictions


def main(num_tasks=5):
    # Initialize backbone
    backbone = SimpleCNN()
    
    # Initialize model
    model = EASE(backbone, num_tasks, reduction_dim=64)
    model.to('cuda')  # Ensure model is on GPU

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Split CIFAR-10 into tasks
    num_classes_per_task = 10 // num_tasks  # 2 classes per task
    task_indices = np.arange(10)  # All CIFAR-10 classes

    for task_index in range(num_tasks):
        # Get the classes for the current task
        classes_for_task = task_indices[task_index * num_classes_per_task: (task_index + 1) * num_classes_per_task]
        
        # Create a subset of the dataset for the current task
        task_dataset = Subset(cifar10_train, [i for i, (_, label) in enumerate(cifar10_train) if label in classes_for_task])
        train_loader = DataLoader(task_dataset, batch_size=32, shuffle=True)

        # Train the model on the current task
        train(model, train_loader, optimizer, criterion, task_index)
        print(f'Trained on task {task_index + 1} with classes {classes_for_task}')

    # Evaluate the model
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to('cuda'), labels.to('cuda')  # Move both to CUDA
            adapted_features, _ = model(images, 0)  # Unpack the tuple
            _, predicted = torch.max(adapted_features, 1)  # Use adapted_features directly
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy on test set: {100 * correct / total:.2f}%')



if __name__ == "__main__":
    main(num_tasks=5)
