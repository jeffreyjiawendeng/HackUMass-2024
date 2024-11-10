import torch
from torchvision import transforms
from PIL import Image
import os
from model import CNN_Model
import torch.nn as nn

class CNN_Model(nn.Module):
    def __init__(self, num_classes, hidden_size=256):
        super(CNN_Model, self).__init__()
        # Define the model architecture as in training
        self.num_classes = num_classes
    
    def forward(self, x):
        # Forward pass logic
        return x

def load_saved_model(model_path):
    try:
        model = torch.load(model_path, weights_only=False)
        model.eval()  # Set the model to evaluation mode
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def create_transform(image_size=(224, 224)):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def process_image(image_path, transform):
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img)
    return img_tensor.unsqueeze(0)  # Add batch dimension

def get_true_label(img_path, emotion_labels):
    # Get the parent folder name from the image path
    parent_folder = os.path.basename(os.path.dirname(img_path)).lower()
    # Try to find matching emotion label
    for idx, label in enumerate(emotion_labels):
        if label.lower() == parent_folder:
            return idx
    return None

def predict_emotions(model_path='best_emotion_model.pth', test_dir='test', image_size=(224, 224)):
    # Define emotion labels (4 classes as per your model)
    emotion_labels = ['Angry', 'Happy', 'Neutral', 'Sad']
    
    # Load the model
    model = load_saved_model(model_path)
    if model is None:
        print("Model loading failed.")
        return
    
    # Create transformation pipeline
    transform = create_transform(image_size)
    
    # Recursively find all image files in the subfolders of test_dir
    valid_extensions = ('.jpg', '.jpeg', '.png')
    image_files = []
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.lower().endswith(valid_extensions):
                image_files.append(os.path.join(root, file))
    
    if not image_files:
        print(f"No valid images found in {test_dir}")
        return

    # Initialize counters for accuracy calculation
    total_predictions = 0
    correct_predictions = 0
    class_predictions = {label: {'correct': 0, 'total': 0} for label in emotion_labels}
    
    for img_path in image_files:
        print(f"\nProcessing: {img_path}")
        
        try:
            # Get true label from folder name
            true_label_idx = get_true_label(img_path, emotion_labels)
            
            # Process image
            img_tensor = process_image(img_path, transform)
            
            # Make prediction
            with torch.no_grad():
                output = model(img_tensor)
                
                # Ensure output is properly shaped and get probabilities
                if len(output.shape) > 2:
                    output = output.view(output.size(0), -1)
                    output = output[:, :4]
                
                probabilities = torch.softmax(output, dim=1)[0].cpu().numpy()
                probabilities = probabilities[:4]
            
            # Create list of (probability, index) tuples for first 4 classes
            predictions = [(prob, idx) for idx, prob in enumerate(probabilities)]
            predictions.sort(reverse=True)  # Sort by probability in descending order
            
            # Print predictions with percentages
            print("\nPrediction Results:")
            print("-" * 30)
            
            # Print each emotion with its probability percentage
            for prob, idx in predictions:
                emotion = emotion_labels[idx]
                print(f"{emotion}: {prob*100:.2f}%")
            
            # Print the most confident prediction
            top_prob, top_idx = predictions[0]
            print("\nPredicted Emotion:", emotion_labels[top_idx])
            print(f"Confidence: {top_prob*100:.2f}%")
            
            # Update accuracy statistics if we have the true label
            if true_label_idx is not None:
                total_predictions += 1
                true_label = emotion_labels[true_label_idx]
                
                # Update class-specific statistics
                class_predictions[true_label]['total'] += 1
                
                if top_idx == true_label_idx:
                    correct_predictions += 1
                    class_predictions[true_label]['correct'] += 1
                
                print(f"True Emotion: {true_label}")
                print("Prediction: Correct" if top_idx == true_label_idx else "Prediction: Incorrect")
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            import traceback
            print(traceback.format_exc())
            continue
        
        print("-" * 50)
    
    # Print overall accuracy statistics
    print("\nOverall Accuracy Statistics:")
    print("=" * 50)
    if total_predictions > 0:
        overall_accuracy = (correct_predictions / total_predictions) * 100
        print(f"\nTotal Accuracy: {overall_accuracy:.2f}% ({correct_predictions}/{total_predictions} correct)")
        
        print("\nPer-Class Accuracy:")
        for emotion in emotion_labels:
            stats = class_predictions[emotion]
            if stats['total'] > 0:
                class_accuracy = (stats['correct'] / stats['total']) * 100
                print(f"{emotion}: {class_accuracy:.2f}% ({stats['correct']}/{stats['total']} correct)")
            else:
                print(f"{emotion}: No samples")
    else:
        print("No predictions were made")

if __name__ == "__main__":
    # Customize parameters as needed
    predict_emotions(
        model_path='models/best_model.pth',
        test_dir='test',
        image_size=(47, 47)
    )