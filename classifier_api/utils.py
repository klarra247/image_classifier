import torch
import torchvision.models as models  # Changed from torchvision.models import resnet101
import torchvision.transforms as transforms
from PIL import Image
import io
import os

# Define constants
CLASS_NAMES = ['2d_character', '3d_character', 'box_mini', 'layer', 'lettering', 'photo']
IMAGE_SIZE = (224, 224)

# Define the model path
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'best_model_by_acc.pth')

# Image transformation
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_model():
    """Load the trained PyTorch model"""
    try:
        # Check if GPU is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Check if model file exists
        if not os.path.exists(MODEL_PATH):
            print(f"Error: Model file not found at {MODEL_PATH}")
            return None, device
            
        # Initialize model architecture
        model = models.resnet101(pretrained=False)
        num_features = model.fc.in_features
        model.fc = torch.nn.Sequential(
            torch.nn.Linear(num_features, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, len(CLASS_NAMES))
        )
        
        # Load trained weights
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        
        # Move model to GPU
        model = model.to(device)
        model.eval()
        print(f"Model successfully loaded from {MODEL_PATH}")
        return model, device
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None, torch.device("cpu")

# Initialize the model and device when the module is loaded
try:
    MODEL, DEVICE = load_model()
except Exception as e:
    print(f"Error initializing model: {e}")
    MODEL, DEVICE = None, torch.device("cpu")

def predict_image(image_bytes):
    """
    Make a prediction from image bytes
    
    Args:
        image_bytes: Bytes of the image file
    
    Returns:
        dict: Dictionary containing prediction results
    """
    if MODEL is None:
        return {'error': 'Model not loaded. Check server logs for details.'}
    
    try:
        # Open and preprocess the image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        print(f"Image opened successfully. Size: {image.size}")
        
        tensor = transform(image).unsqueeze(0)  # Add batch dimension
        
        # Move tensor to the same device as the model
        tensor = tensor.to(DEVICE)
        print(f"Image transformed to tensor with shape: {tensor.shape}")
        
        # Get prediction
        with torch.no_grad():
            outputs = MODEL(tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidences, predicted = torch.max(probabilities, 1)
        
        # Get class name and confidence
        predicted_class = CLASS_NAMES[predicted.item()]
        confidence = confidences.item()
        print(f"Prediction successful: {predicted_class} with confidence {confidence:.4f}")
        
        # Get top 3 predictions
        top3_values, top3_indices = torch.topk(probabilities, 3, dim=1)
        top3_predictions = []
        
        for i in range(3):
            class_idx = top3_indices[0][i].item()
            class_conf = float(top3_values[0][i].item())
            top3_predictions.append({
                'class': CLASS_NAMES[class_idx],
                'confidence': class_conf
            })
            print(f"Top {i+1}: {CLASS_NAMES[class_idx]} ({class_conf:.4f})")
        
        return {
            'predicted_class': predicted_class,
            'confidence': float(confidence),  # Convert to float for JSON serialization
            'top3': top3_predictions
        }
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Error in predict_image: {e}")
        print(error_traceback)
        return {'error': str(e)}