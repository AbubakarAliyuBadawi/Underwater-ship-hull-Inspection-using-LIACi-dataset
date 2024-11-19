import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from src.build_model import build_model
from src.args import ArgumentParser
import os

def create_color_label_map():
    colors = [
        (0, 0, 0),         # Void - Black
        (0, 0, 255),       # Ship hull - Blue
        (0, 128, 0),       # Marine growth - Green
        (0, 255, 255),     # Anode - Cyan
        (64, 224, 208),    # Overboard valve - Turquoise
        (128, 0, 128),     # Propeller - Purple
        (255, 0, 0),       # Paint peel - Red
        (255, 165, 0),     # Bilge keel - Orange
        (255, 192, 203),   # Defect - Pink
        (255, 255, 0),     # Corrosion - Yellow
        (255, 255, 255),   # Sea chest grating - White
    ]
    return np.array(colors, dtype=np.uint8)

def predict_image(model, image_path, device, target_size=(512, 1024)):
    # Read and preprocess image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (target_size[1], target_size[0]))
    
    # Normalize and convert to tensor
    image = image.astype(np.float32) / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    image = image.to(device)

    # Get prediction
    model.eval()
    with torch.no_grad():
        prediction, _ = model(image)
        prediction = torch.softmax(prediction, dim=1)
        prediction = torch.argmax(prediction, dim=1).squeeze()

    return prediction.cpu().numpy()

def visualize_prediction(image_path, ground_truth_path, prediction, output_path=None):
    # Read original image
    original = cv2.imread(image_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # Read ground truth mask
    ground_truth = cv2.imread(ground_truth_path)
    ground_truth = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2RGB)
    
    # Create colored prediction
    colors = create_color_label_map()
    colored_pred = colors[prediction]
    
    # Resize all images to match prediction size
    original = cv2.resize(original, (prediction.shape[1], prediction.shape[0]))
    ground_truth = cv2.resize(ground_truth, (prediction.shape[1], prediction.shape[0]), 
                            interpolation=cv2.INTER_NEAREST)
    
    # Create visualization
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(original)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(ground_truth)
    plt.title('Ground Truth')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(colored_pred)
    plt.title('Prediction')
    plt.axis('off')
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()

def main():
    # Parse arguments
    parser = ArgumentParser()
    parser.set_common_args()
    args = parser.parse_args([])  # Empty list for default args
    
    # Update args for your model
    args.num_classes = 11
    args.dataset = "cityscapes"
    args.height = 512
    args.width = 1024
    
    # Build model and load weights
    model, device = build_model(args, n_classes=args.num_classes)
    checkpoint_path = "/cluster/home/abubakb/ContMAV_3/results_closeworld/cityscapes/LIACi-02-Open_world/05_11_2024-08_59_54-575462/epoch_279.pth"
    
    # Load the state dict
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict)
    model.eval()

    # Directories
    test_dir = "/cluster/home/abubakb/final_dataset_str/leftImg8bit/test"
    masks_dir = "/cluster/home/abubakb/LIACi_dataset_pretty/masks"  # Ground truth masks
    output_dir = "/cluster/home/abubakb/ContMAV_3/LIACi-predictions-Open_world"
    os.makedirs(output_dir, exist_ok=True)

    # Process each image
    for image_name in os.listdir(test_dir):
        if image_name.endswith(('.png', '.jpg', '.jpeg')):
            print(f"Processing {image_name}")
            
            # Get paths
            image_path = os.path.join(test_dir, image_name)
            
            # Get corresponding mask name (assuming same name)
            mask_name = os.path.splitext(image_name)[0] + '.png'
            ground_truth_path = os.path.join(masks_dir, mask_name)
            
            output_path = os.path.join(output_dir, f"pred_{image_name}")
            
            # Check if ground truth exists
            if not os.path.exists(ground_truth_path):
                print(f"Warning: No ground truth found for {image_name}")
                continue
            
            # Get prediction
            prediction = predict_image(model, image_path, device, target_size=(args.height, args.width))
            
            # Visualize and save
            visualize_prediction(image_path, ground_truth_path, prediction, output_path)
            print(f"Saved prediction to {output_path}")

if __name__ == "__main__":
    main()
