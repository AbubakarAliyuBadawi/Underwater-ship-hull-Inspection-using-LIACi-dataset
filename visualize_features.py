import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
import os
from src.prepare_data import prepare_data
from src.build_model import build_model
from src.args import ArgumentParser

class CityscapesBase:
    CLASS_NAMES_FULL = [
        'Void',            # 0
        'Ship hull',       # 1
        'Marine growth',   # 2
        'Anode',          # 3
        'Overboard valve', # 4
        'Propeller',      # 5
        'Paint peel',      # 6
        'Bilge keel',     # 7
        'Defect',         # 8
        'Corrosion',      # 9
        'Sea chest grating'# 10
    ]

def collect_balanced_features(model, data_loader, device, samples_per_class=5000):
    """Collect balanced number of samples from each class."""
    features_obj = {i: [] for i in range(11)}
    features_cont = {i: [] for i in range(11)}
    
    print(f"Collecting {samples_per_class} samples per class...")
    total_samples = {i: 0 for i in range(11)}
    
    # Add debug counters
    unknown_pixel_count = 0
    total_pixel_count = 0
    
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx % 10 == 0:
                print(f"Processing batch {batch_idx}...")
                
            images = batch["image"].to(device)
            batch_labels = batch["label"].to(device)
            
            # Debug: Print unique labels in batch
            unique_labels = torch.unique(batch_labels)
            print(f"\nUnique labels in batch: {unique_labels}")
            
            # Get features
            try:
                pred_scales, ow_res = model(images)
            except Exception as e:
                print(f"Error during model forward pass: {str(e)}")
                continue
            
            # For each class in the batch
            for class_idx in range(11):
                if class_idx == 0:  # Unknown/void class
                    class_mask = (batch_labels == 0)
                    unknown_count = class_mask.sum().item()
                    unknown_pixel_count += unknown_count
                    print(f"\nFound {unknown_count} unknown pixels in current batch")
                else:
                    class_mask = batch_labels == class_idx
                
                total_pixel_count += class_mask.sum().item()
                
                if not class_mask.any():
                    continue
                
                n_pixels = min(samples_per_class - len(features_obj[class_idx]), 
                             class_mask.sum().item())
                if n_pixels <= 0:
                    continue
                
                indices = torch.where(class_mask.view(-1))[0]
                if len(indices) > n_pixels:
                    indices = indices[torch.randperm(len(indices))[:n_pixels]]
                
                pred_flat = pred_scales.view(-1, pred_scales.shape[1])
                ow_flat = ow_res.view(-1, ow_res.shape[1])
                
                features_obj[class_idx].extend(pred_flat[indices].cpu().numpy())
                features_cont[class_idx].extend(ow_flat[indices].cpu().numpy())
                total_samples[class_idx] = len(features_obj[class_idx])
    
    print("\n=== Unknown Class Statistics ===")
    print(f"Total unknown pixels found: {unknown_pixel_count}")
    print(f"Percentage unknown: {(unknown_pixel_count/total_pixel_count)*100:.2f}%")
    print(f"Unknown samples collected: {len(features_obj[0])}")
    
    # Validate collected features
    features_obj_final = []
    features_cont_final = []
    labels_final = []
    
    print("\nFeatures collected per class:")
    for class_idx in range(11):
        if len(features_obj[class_idx]) > 0:
            features_obj_final.append(np.array(features_obj[class_idx][:samples_per_class]))
            features_cont_final.append(np.array(features_cont[class_idx][:samples_per_class]))
            labels_final.append(np.full(samples_per_class, class_idx))
            print(f"Class {CityscapesBase.CLASS_NAMES_FULL[class_idx]}: {len(features_obj[class_idx])} samples")

    # Debug: Print final label distribution
    final_labels = np.hstack(labels_final)
    unique, counts = np.unique(final_labels, return_counts=True)
    print("\nFinal label distribution:")
    for u, c in zip(unique, counts):
        print(f"Class {CityscapesBase.CLASS_NAMES_FULL[u]}: {c} samples")
    
    return (np.vstack(features_obj_final), np.vstack(features_cont_final), 
            np.hstack(labels_final))

def visualize_paper_style_features(features_obj, features_cont, labels, save_dir):
    """Create visualizations matching the paper's style for marine dataset."""
    # Debug: Print input statistics
    print("\n=== Visualization Input Statistics ===")
    print(f"Total samples: {len(labels)}")
    unique, counts = np.unique(labels, return_counts=True)
    print("\nLabel distribution:")
    for u, c in zip(unique, counts):
        print(f"Class {CityscapesBase.CLASS_NAMES_FULL[u]}: {c} samples")
    
    print("\nFeature shapes:")
    print(f"Object features shape: {features_obj.shape}")
    print(f"Contrastive features shape: {features_cont.shape}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Define colors for the 10 known classes (excluding void/unknown)
    colors = plt.cm.tab20(np.linspace(0, 1, 10))
    
    # 1. Objectosphere Visualization
    print("\nGenerating Objectosphere visualization...")
    plt.figure(figsize=(8, 8))
    
    # Normalize features
    features_obj_norm = features_obj / (np.linalg.norm(features_obj, axis=1, keepdims=True) + 1e-8)
    print(f"Normalized object features shape: {features_obj_norm.shape}")
    print(f"Object features norm range: {np.linalg.norm(features_obj_norm, axis=1).min():.3f} - {np.linalg.norm(features_obj_norm, axis=1).max():.3f}")
    
    # Apply TSNE
    features_2d = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(features_obj_norm)
    print(f"TSNE features shape: {features_2d.shape}")
    
    # Scale to preserve unit circle structure
    max_norm = np.max(np.linalg.norm(features_2d, axis=1))
    features_2d = features_2d / max_norm
    print(f"Scaled features norm range: {np.linalg.norm(features_2d, axis=1).min():.3f} - {np.linalg.norm(features_2d, axis=1).max():.3f}")
    
    # Plot known vs unknown
    known_mask = labels != 0  # 0 is void/unknown
    unknown_count = (~known_mask).sum()
    known_count = known_mask.sum()
    print(f"\nPlotting {known_count} known and {unknown_count} unknown samples")
    
    plt.scatter(features_2d[known_mask, 0], features_2d[known_mask, 1],
               c='black', alpha=0.5, s=20, label=f'Known ({known_count} samples)')
    if unknown_count > 0:
        plt.scatter(features_2d[~known_mask, 0], features_2d[~known_mask, 1],
                   c='red', alpha=0.5, s=20, label=f'Unknown ({unknown_count} samples)')
    
    circle = plt.Circle((0, 0), 1.0, fill=False, color='red', linestyle='--')
    plt.gca().add_artist(circle)
    plt.axis('equal')
    plt.title("A. Objectosphere")
    plt.legend()
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.savefig(os.path.join(save_dir, "A_objectosphere.png"), bbox_inches='tight', dpi=300)
    plt.close()
    
    # 2. Contrastive Visualization
    print("\nGenerating Contrastive visualization...")
    plt.figure(figsize=(8, 8))
    
    # Normalize contrastive features
    features_cont_norm = features_cont / (np.linalg.norm(features_cont, axis=1, keepdims=True) + 1e-8)
    print(f"Normalized contrastive features shape: {features_cont_norm.shape}")
    print(f"Contrastive features norm range: {np.linalg.norm(features_cont_norm, axis=1).min():.3f} - {np.linalg.norm(features_cont_norm, axis=1).max():.3f}")
    
    features_2d = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(features_cont_norm)
    
    # Scale features
    max_norm = np.max(np.linalg.norm(features_2d, axis=1))
    features_2d = features_2d / max_norm
    
    # Plot known classes evenly on circle
    angles = np.linspace(0, 2*np.pi, 10, endpoint=False)  # 10 known classes
    print("\nPlotting known classes on circle:")
    for i, angle in enumerate(range(1, 11)):  # Skip class 0 (unknown)
        mask = labels == angle
        samples_count = mask.sum()
        print(f"Class {CityscapesBase.CLASS_NAMES_FULL[angle]}: {samples_count} samples")
        if mask.any():
            center = np.array([np.cos(angles[i]), np.sin(angles[i])])
            class_features = features_2d[mask]
            centered = class_features - np.mean(class_features, axis=0)
            scaled = 0.2 * centered + center
            plt.scatter(scaled[:, 0], scaled[:, 1],
                       c=[colors[i]], alpha=0.5, s=20,
                       label=f'{CityscapesBase.CLASS_NAMES_FULL[angle]} ({samples_count})')
    
    circle = plt.Circle((0, 0), 1.0, fill=False, color='red', linestyle='--')
    plt.gca().add_artist(circle)
    plt.axis('equal')
    plt.title("B. Contrastive")
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(os.path.join(save_dir, "B_contrastive.png"), bbox_inches='tight', dpi=300)
    plt.close()
    
    # 3. Combined Visualization
    print("\nGenerating Combined visualization...")
    plt.figure(figsize=(8, 8))
    
    # Plot known classes on circle
    print("\nPlotting known classes and unknown samples:")
    for i, angle in enumerate(range(1, 11)):
        mask = labels == angle
        samples_count = mask.sum()
        if mask.any():
            print(f"Class {CityscapesBase.CLASS_NAMES_FULL[angle]}: {samples_count} samples")
            center = np.array([np.cos(angles[i]), np.sin(angles[i])])
            class_features = features_2d[mask]
            centered = class_features - np.mean(class_features, axis=0)
            scaled = 0.15 * centered + center
            plt.scatter(scaled[:, 0], scaled[:, 1],
                       c=[colors[i]], alpha=0.5, s=20,
                       label=f'{CityscapesBase.CLASS_NAMES_FULL[angle]} ({samples_count})')
    
    # Add unknown class at center
    unknown_mask = labels == 0
    unknown_count = unknown_mask.sum()
    print(f"Unknown class: {unknown_count} samples")
    if unknown_mask.any():
        unknown_features = features_2d[unknown_mask]
        scaled_unknown = 0.1 * unknown_features
        plt.scatter(scaled_unknown[:, 0], scaled_unknown[:, 1],
                   c='red', alpha=0.5, s=20, label=f'Unknown ({unknown_count})')
    
    circle = plt.Circle((0, 0), 1.0, fill=False, color='red', linestyle='--')
    plt.gca().add_artist(circle)
    plt.axis('equal')
    plt.title("C. Objectosphere + Contrastive")
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(os.path.join(save_dir, "C_combined.png"), bbox_inches='tight', dpi=300)
    plt.close()

    print("\nVisualization complete. Check the output directory for the plots.")

def main():
    parser = ArgumentParser(description="Feature Visualization")
    parser.set_common_args()
    args = parser.parse_args()

    print("\nInitializing feature visualization...")
    print(f"Loading weights from: {args.load_weights}")
    print(f"Dataset directory: {args.dataset_dir}")
    
    # Set up model and data
    try:
        data_loaders = prepare_data(args)
        _, valid_loader, _ = data_loaders
        print("Data loaders prepared successfully")
    except Exception as e:
        print(f"Error preparing data: {str(e)}")
        return

    # Build model
    try:
        model, device = build_model(args, n_classes=11)
        print(f"Model built successfully. Using device: {device}")
    except Exception as e:
        print(f"Error building model: {str(e)}")
        return

    # Load weights
    try:
        checkpoint = torch.load(args.load_weights)
        model.load_state_dict(checkpoint)
        print("Model weights loaded successfully")
    except Exception as e:
        print(f"Error loading weights: {str(e)}")
        return

    # Collect features
    try:
        features_obj, features_cont, labels = collect_balanced_features(
            model, valid_loader, device, samples_per_class=5000
        )
        print("Features collected successfully")
    except Exception as e:
        print(f"Error collecting features: {str(e)}")
        return

    # Create visualizations
    try:
        viz_dir = "paper_style_viz"
        visualize_paper_style_features(features_obj, features_cont, labels, viz_dir)
        print(f"\nAll visualizations completed successfully. Check the '{viz_dir}' directory.")
    except Exception as e:
        print(f"Error creating visualizations: {str(e)}")

if __name__ == "__main__":
    main()
