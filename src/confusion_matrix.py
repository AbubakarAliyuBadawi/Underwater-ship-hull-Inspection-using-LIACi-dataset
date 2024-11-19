import os
import numpy as np
from tqdm import tqdm
import torch
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from src.build_model import build_model
from src.prepare_data import prepare_data
from src.args import ArgumentParser

def plot_confusion_matrix():
    parser = ArgumentParser()
    parser.set_common_args()
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    _, val_loader, _ = prepare_data(args)
    
    model, _ = build_model(args, n_classes=args.num_classes)
    model.load_state_dict(torch.load(args.load_weights))
    model.eval()
    model.to(device)
    
    all_preds = []
    all_targets = []
    
    print("Collecting predictions...")
    with torch.no_grad():
        for sample in tqdm(val_loader):
            images = sample["image"].to(device)
            targets = sample["label"]
            
            pred_ss, _ = model(images)
            preds = torch.argmax(pred_ss, dim=1)
            
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(targets.numpy().flatten())
    
    # Create confusion matrix
    print("Creating confusion matrix...")
    conf_matrix = confusion_matrix(
        [t for t in all_targets if t != 0],
        [p for p, t in zip(all_preds, all_targets) if t != 0],
        labels=range(1, args.num_classes)
    )
    
    # Normalize confusion matrix to get decimals (divide by 100 to convert from percentage)
    conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    
    # Class names
    class_names = [
        "Ship hull",
        "Marine growth",
        "Anode",
        "Overboard valve",
        "Propeller",
        "Paint peel",
        "Bilge keel",
        "Defect",
        "Corrosion",
        "Sea chest grating"
    ]
    
    # Plot normalized matrix
    plt.figure(figsize=(20, 20))
    sns.heatmap(conf_matrix_norm, 
                annot=True, 
                fmt='.2f',  # Show two decimal places
                xticklabels=class_names,
                yticklabels=class_names,
                cmap='YlOrRd')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Normalized)')
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save normalized matrix
    os.makedirs(args.results_dir, exist_ok=True)
    save_path = os.path.join(args.results_dir, "confusion_matrix_normalized.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    # Also save raw counts
    plt.figure(figsize=(20, 20))
    sns.heatmap(conf_matrix, 
                annot=True, 
                fmt='d',  # Use integer format for raw counts
                xticklabels=class_names,
                yticklabels=class_names,
                cmap='YlOrRd')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Raw Counts)')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    save_path_raw = os.path.join(args.results_dir, "confusion_matrix_raw.png")
    plt.savefig(save_path_raw, bbox_inches='tight', dpi=300)
    plt.close()
    
    # Print class-wise accuracies
    print("\nClass-wise accuracies:")
    for i, name in enumerate(class_names):
        accuracy = conf_matrix_norm[i, i]
        print(f"{name}: {accuracy:.2f}")

if __name__ == "__main__":
    plot_confusion_matrix()
    
    
# (openworld) (base) [abubakb@idun-login2 ~]$ python confusion_matrix.py     --dataset_dir /cluster/home/abubakb/final_dataset     --load_weights /cluster/home/abubakb/ContMAV_3/results_closeworld/cityscapes/LIACi-01/05_11_2024-08_03_14-120322/epoch_219.pth  --results_dir /cluster/home/abubakb/ContMAV_3
