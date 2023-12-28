import torch
import torchvision.transforms as transforms
from PIL import Image
from efficientnet_pytorch import EfficientNet

# Define the transformation for data preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert to RGB (EfficientNet expects 3 channels)
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard ImageNet normalization
])

# Load your trained model
model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=3)  # Adjust the number of classes
model.load_state_dict(torch.load('source_best_model.pth'))  # Load the best model checkpoint
model.eval()


class_names = {
    0: 'NP',
    1: 'NZ',
    2: 'QZ',
    # Add more class names if you have additional classes
}


def predict_single_image(image_path, model, transform):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    outputs = model(image)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)  # Calculate softmax to get probabilities
    return probabilities.detach().numpy()[0]  # Convert to NumPy array and return


def classify_images_in_folder(folder_path, model, transform):
    import os
    results = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            probabilities = predict_single_image(image_path, model, transform)
            predicted_class = int(torch.argmax(torch.tensor(probabilities)))  # Get the index of the maximum probability
            class_name = class_names.get(predicted_class, 'Unknown')  # Get class name
            results.append((filename, class_name, probabilities))
    return results


folder_path = 'test4/test'
results = classify_images_in_folder(folder_path, model, transform)

for filename, predicted_class, probabilities in results:
    print(f'Image: {filename}, Predicted Class: {predicted_class}, Probabilities: {probabilities}')

    # Display probabilities as percentages
    for i, prob in enumerate(probabilities):
        class_label = class_names[i]
        print(f'   {class_label}: {prob * 100:.2f}%')
