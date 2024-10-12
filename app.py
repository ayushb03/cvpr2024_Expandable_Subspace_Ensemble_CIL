import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)  # Output: 16 x 32 x 32
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # Output: 16 x 16 x 16
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # Output: 32 x 16 x 16
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # Output: 32 x 8 x 8
        self.fc1 = nn.Linear(32 * 8 * 8, 128)  # Fully connected layer

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Pass through conv1 and pooling
        x = self.pool2(F.relu(self.conv2(x)))  # Pass through conv2 and pooling
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)  # Fully connected layer
        return x


class Adapter(nn.Module):
    def __init__(self, input_dim, reduction_dim):
        super(Adapter, self).__init__()
        self.down_project = nn.Linear(input_dim, reduction_dim)
        self.activation = nn.ReLU()
        self.up_project = nn.Linear(reduction_dim, input_dim)

    def forward(self, x):
        # Add a residual connection to preserve original features.
        return x + self.up_project(self.activation(self.down_project(x)))


class EASE(nn.Module):
    def __init__(self, backbone, reduction_dim=64, num_classes=10):
        super(EASE, self).__init__()
        self.backbone = backbone  # Backbone model
        self.adapters = nn.ModuleList()  # Store task-specific adapters
        self.classifiers = nn.ModuleList()  # Store task-specific classifiers
        self.reduction_dim = reduction_dim
        self.num_classes = num_classes

    def add_task(self):
        """Add a new adapter and classifier for a new task."""
        input_dim = self.backbone.fc1.out_features  # Use fc1 output directly
        adapter = Adapter(input_dim, self.reduction_dim)  # Task-specific adapter
        classifier = nn.Linear(input_dim, self.num_classes)  # Task-specific classifier
        
        self.adapters.append(adapter)
        self.classifiers.append(classifier)

    def forward(self, x, task_idx):
        """Forward pass through the backbone, adapter, and classifier."""
        features = self.backbone(x)  # Get features from the backbone
        adapted_features = self.adapters[task_idx](features)  # Task-specific adapter
        logits = self.classifiers[task_idx](adapted_features)  # Task-specific classifier

        return logits


class PrototypeManager:
    def __init__(self):
        self.prototypes = {}

    def extract_prototypes(self, model, task_idx, dataset, device):
        """Extract class prototypes for a specific task."""
        model.eval()
        prototypes = {}
        with torch.no_grad():
            for label in set(dataset.dataset.targets):
                indices = [i for i, target in enumerate(dataset.dataset.targets) if target == label]
                data_points = torch.stack([dataset.dataset[i][0] for i in indices]).to(device)

                # Compute the mean feature vector per class
                features = model.backbone(data_points)
                prototype = torch.mean(features, dim=0)
                prototypes[label] = F.normalize(prototype, p=2, dim=0)  # Normalize

        return prototypes

    def complement_old_prototypes(self, old_prototypes, new_prototypes):
        """Synthesize old prototypes in the new subspace using semantic similarity."""
        complemented = {}
        similarity_matrix = torch.zeros(len(old_prototypes), len(new_prototypes)).to(
            next(iter(old_prototypes.values())).device
        )

        # Compute pairwise cosine similarity between old and new prototypes
        for i, old_class in enumerate(old_prototypes):
            for j, new_class in enumerate(new_prototypes):
                similarity_matrix[i, j] = torch.cosine_similarity(
                    old_prototypes[old_class], new_prototypes[new_class], dim=0
                )

        # Apply softmax to normalize similarities
        weights = torch.softmax(similarity_matrix, dim=1)

        # Reconstruct old prototypes in the new subspace
        for i, old_class in enumerate(old_prototypes):
            reconstructed = sum(
                weights[i, j] * new_prototypes[new_class]
                for j, new_class in enumerate(new_prototypes)
            )
            complemented[old_class] = F.normalize(reconstructed, p=2, dim=0)

        return complemented


def weighted_inference(model, images, task_idx, prototypes):
    """Perform weighted inference using multiple subspaces."""
    device = images.device
    logits_sum = torch.zeros(images.size(0), model.num_classes).to(device)

    # Extract normalized features from the backbone
    features = F.normalize(model.backbone(images), p=2, dim=1)

    # Iterate through all adapters used so far
    for i in range(task_idx + 1):
        adapted_features = F.normalize(model.adapters[i](features), p=2, dim=1)
        logits = model.classifiers[i](adapted_features)

        for j, prototype in prototypes.items():
            similarity = torch.cosine_similarity(adapted_features, prototype.unsqueeze(0), dim=1)
            logits[:, j] *= similarity  # Reweight logits using prototype similarity

        # Assign higher weight to the current task's adapter
        weight = 1.0 if i == task_idx else 0.1  # Current task prioritized
        logits_sum += weight * logits

    return logits_sum


def train_task(model, prototype_manager, dataloader, task_idx, device, epochs=10, lr=1e-3):
    model.add_task()  # Add a new adapter and classifier for the task
    optimizer = optim.Adam(model.adapters[task_idx].parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs, task_idx)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}] Loss: {total_loss / len(dataloader):.4f}")

    # Extract new prototypes and complement old ones
    new_prototypes = prototype_manager.extract_prototypes(model, task_idx, dataloader.dataset, device)
    if task_idx > 0:
        old_prototypes = prototype_manager.prototypes[task_idx - 1]
        complemented = prototype_manager.complement_old_prototypes(old_prototypes, new_prototypes)
        prototype_manager.prototypes[task_idx - 1] = complemented

    prototype_manager.prototypes[task_idx] = new_prototypes


def evaluate_model(model, prototype_manager, test_loader, device, task_idx):
    """Evaluate the model on the current task."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            prototypes = prototype_manager.prototypes[task_idx]  # Current task prototypes

            # Perform weighted inference across all subspaces
            logits_sum = weighted_inference(model, inputs, task_idx, prototypes)
            _, predicted = torch.max(logits_sum, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Task {task_idx}: Test Accuracy = {accuracy:.2f}%")

def prepare_dataloaders(task_idx, batch_size=64):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Select data for the specific task based on the task index
    classes_per_task = 2  # Number of classes per task
    task_classes = list(range(task_idx * classes_per_task, (task_idx + 1) * classes_per_task))
    train_indices = [i for i, label in enumerate(train_data.targets) if label in task_classes]
    test_indices = [i for i, label in enumerate(test_data.targets) if label in task_classes]

    train_subset = Subset(train_data, train_indices)
    test_subset = Subset(test_data, test_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def main(num_tasks=5, epochs=2, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EASE(SimpleCNN(), reduction_dim=64).to(device)
    prototype_manager = PrototypeManager()

    for task_idx in range(num_tasks):
        train_loader, test_loader = prepare_dataloaders(task_idx, batch_size)
        train_task(model, prototype_manager, train_loader, task_idx, device, epochs)

        # Evaluate the model after training on the current task
        evaluate_model(model, prototype_manager, test_loader, device, task_idx)

    print("Training and evaluation complete.")

if __name__ == "__main__":
    main()