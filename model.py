import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class CustomImageDataset(Dataset):
    """Custom dataset for loading emotion recognition images"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        self._load_dataset()
        self._create_label_mapping()

    def _load_dataset(self):
        """Load image paths and labels from directory structure"""
        for label_name in os.listdir(self.root_dir):
            label_folder = os.path.join(self.root_dir, label_name)
            if os.path.isdir(label_folder):
                for img_name in os.listdir(label_folder):
                    img_path = os.path.join(label_folder, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(label_name)

    def _create_label_mapping(self):
        """Create mapping from label names to integers"""
        self.label_map = {label: idx for idx, label in enumerate(sorted(set(self.labels)))}
        self.labels = [self.label_map[label] for label in self.labels]
        print(self.labels)
        
    def get_label_mapping(self):
        """Return the label mapping dictionary"""
        return {v: k for k, v in self.label_map.items()}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        
        return img, label

# class CNN_GRU_Model(nn.Module):
#     """Combined CNN and GRU model for emotion recognition"""
#     def __init__(self, num_classes, hidden_size=256):
#         super(CNN_GRU_Model, self).__init__()
        
#         # CNN layers with proper dimensionality for 47x47 input
#         self.cnn = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
            
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
            
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
            
#             nn.Conv2d(128, 256, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
#         self.feature_size = 256 * 2 * 2
#         self.fc1 = nn.Linear(self.feature_size, 512)
#         self.fc_dropout = nn.Dropout(0.3)
        
#         self.gru = nn.GRU(
#             input_size=512,
#             hidden_size=hidden_size,
#             num_layers=2,
#             batch_first=True,
#             dropout=0.25
#         )
        
#         self.fc2 = nn.Linear(hidden_size, num_classes)

#     def forward(self, x):
#         x = self.cnn(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         x = torch.relu(x)
#         x = self.fc_dropout(x)
#         x = x.unsqueeze(1)
#         x, _ = self.gru(x)
#         x = self.fc2(x[:, -1, :])
#         return x

class CNN_Model(nn.Module):
    """CNN model for emotion recognition"""
    def __init__(self, num_classes):
        super(CNN_Model, self).__init__()
        
        # CNN layers with proper dimensionality for 47x47 input
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Fully connected layers for classification
        self.feature_size = 256 * 2 * 2
        self.fc1 = nn.Linear(self.feature_size, 512)
        self.fc_dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc_dropout(x)
        x = self.fc2(x)
        return x
class EmotionRecognitionTrainer:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.best_val_accuracy = 0.0
        self.best_val_loss = float('inf')
        self.model_dir = "models"
        self.model_save_path = os.path.join(self.model_dir, "best_model.pth")
        os.makedirs(self.model_dir, exist_ok=True)
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        return total_loss / len(train_loader), 100. * correct / total

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(val_loader)
        
        if accuracy > self.best_val_accuracy:
            self.best_val_accuracy = accuracy
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            
        return avg_loss, accuracy, all_predictions, all_labels

    def plot_confusion_matrix(self, true_labels, predictions, label_mapping):
        """Plot confusion matrix using seaborn"""
        # Convert numeric labels to emotion names
        label_names = [label_mapping[i] for i in range(len(label_mapping))]
        
        # Compute confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        # Create figure and axes
        plt.figure(figsize=(10, 8))
        
        # Plot confusion matrix
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d',
            cmap='Blues',
            xticklabels=label_names,
            yticklabels=label_names
        )
        
        plt.title('Emotion Recognition Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('confusion_matrix.png')
        plt.close()
       
    def train(self, train_loader, val_loader, num_epochs):
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc, predictions, true_labels = self.validate(val_loader)
            
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.2f}%')
            print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%')
            torch.save(self.model, self.model_save_path)
            # if val_acc > self.best_val_accuracy:
            #     self.best_val_accuracy = val_acc
            #     torch.save(self.model, self.model_save_path)
            #     print(f"New best model saved with validation accuracy: {val_acc:.2f}%")
                
            # Plot confusion matrix at the end of training
            if epoch == num_epochs - 1:
                label_mapping = train_loader.dataset.dataset.get_label_mapping() \
                    if isinstance(train_loader.dataset, torch.utils.data.Subset) \
                    else train_loader.dataset.get_label_mapping()
                self.plot_confusion_matrix(true_labels, predictions, label_mapping)
                print("Confusion matrix has been saved as 'confusion_matrix.png'")

def get_transforms():
    """Return the image transformations needed for the model"""
    return transforms.Compose([
        transforms.Resize((47, 47)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

def main():
    NUM_CLASSES = 4
    BATCH_SIZE = 32 
    NUM_EPOCHS = 12
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 0.0001
    ROOT_DIR = 'train'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    VAL_SPLIT = 0.4

    transform = get_transforms()
    full_dataset = CustomImageDataset(root_dir=ROOT_DIR, transform=transform)
    
    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    model = CNN_Model(num_classes=NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    trainer = EmotionRecognitionTrainer(model, criterion, optimizer, DEVICE)
    trainer.train(train_loader, val_loader, NUM_EPOCHS)
    
    print("Training complete!")
    print(f"Best validation accuracy: {trainer.best_val_accuracy:.2f}%")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")

if __name__ == "__main__":
    main()