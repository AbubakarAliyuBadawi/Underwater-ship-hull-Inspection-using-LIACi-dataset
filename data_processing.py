import cv2
import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def create_color_to_label_mapping():
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
    
    color_to_label = np.zeros((256, 256, 256), dtype=np.uint8)
    for label_id, color in enumerate(colors):
        color_to_label[color[0], color[1], color[2]] = label_id
    return color_to_label

def process_dataset(rgb_dir, colored_mask_dir, output_dir, split_ratios=[0.9, 0.07, 0.03], target_size=(512, 512)):
    """
    Process dataset: split into train/val/test, convert masks, and resize all images to the same size.
    
    Args:
        rgb_dir: Directory containing RGB images
        colored_mask_dir: Directory containing colored masks
        output_dir: Output directory
        split_ratios: Train/val/test split ratios
        target_size: (height, width) tuple for resizing all images
    """
    if not os.path.exists(rgb_dir):
        raise ValueError(f"RGB directory does not exist: {rgb_dir}")
    if not os.path.exists(colored_mask_dir):
        raise ValueError(f"Mask directory does not exist: {colored_mask_dir}")
    
    # Create Cityscapes directory structure
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, 'leftImg8bit', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'gtFine', split), exist_ok=True)
    
    color_to_label = create_color_to_label_mapping()
    
    # Get all image files
    image_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    mask_files = sorted([f for f in os.listdir(colored_mask_dir) if f.endswith(('.png'))])
    
    assert len(image_files) == len(mask_files), "Number of images and masks must match"
    print(f"Found {len(image_files)} images and masks")
    print(f"Will resize all images to {target_size}")
    
    # Create splits
    train_ratio, val_ratio, test_ratio = split_ratios
    train_images, temp_images = train_test_split(
        list(zip(image_files, mask_files)), train_size=train_ratio, random_state=42
    )
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_images, test_images = train_test_split(
        temp_images, train_size=val_ratio_adjusted, random_state=42
    )
    
    split_mapping = {
        'train': train_images,
        'val': val_images,
        'test': test_images
    }
    
    successful = 0
    failed = 0
    
    # Process each split
    for split, file_pairs in split_mapping.items():
        print(f"\nProcessing {split} split ({len(file_pairs)} images)...")
        
        for img_file, mask_file in tqdm(file_pairs):
            try:
                # Read files
                rgb_path = os.path.join(rgb_dir, img_file)
                mask_path = os.path.join(colored_mask_dir, mask_file)
                
                # Read and resize RGB image
                rgb_img = cv2.imread(rgb_path)
                if rgb_img is None:
                    print(f"\nError: Could not read image: {rgb_path}")
                    failed += 1
                    continue
                
                # Resize RGB image
                rgb_img = cv2.resize(rgb_img, (target_size[1], target_size[0]), 
                                   interpolation=cv2.INTER_LINEAR)
                
                # Read and resize mask
                colored_mask = cv2.imread(mask_path)
                if colored_mask is None:
                    print(f"\nError: Could not read mask: {mask_path}")
                    failed += 1
                    continue
                
                # Resize mask (use NEAREST for masks to preserve label values)
                colored_mask = cv2.resize(colored_mask, (target_size[1], target_size[0]), 
                                        interpolation=cv2.INTER_NEAREST)
                
                # Convert BGR to RGB for mask conversion
                colored_mask = cv2.cvtColor(colored_mask, cv2.COLOR_BGR2RGB)
                
                # Convert colored mask to label mask
                label_mask = color_to_label[colored_mask[:,:,0], 
                                          colored_mask[:,:,1], 
                                          colored_mask[:,:,2]]
                
                # Save files
                rgb_save_name = os.path.splitext(img_file)[0] + '.png'
                cv2.imwrite(os.path.join(output_dir, 'leftImg8bit', split, rgb_save_name), rgb_img)
                
                label_save_name = f"{os.path.splitext(img_file)[0]}_gtFine_labelTrainIds.png"
                cv2.imwrite(os.path.join(output_dir, 'gtFine', split, label_save_name), label_mask)
                
                successful += 1
                
            except Exception as e:
                print(f"\nError processing {img_file} with mask {mask_file}: {str(e)}")
                failed += 1
                continue
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {successful} files")
    print(f"Failed to process: {failed} files")
    
    # Verify output
    for split in ['train', 'val', 'test']:
        n_images = len(os.listdir(os.path.join(output_dir, 'leftImg8bit', split)))
        n_labels = len(os.listdir(os.path.join(output_dir, 'gtFine', split)))
        print(f"\n{split} split:")
        print(f"Images: {n_images}")
        print(f"Labels: {n_labels}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process dataset into Cityscapes format with mask conversion')
    parser.add_argument('rgb_dir', help='Directory containing RGB images')
    parser.add_argument('mask_dir', help='Directory containing colored masks')
    parser.add_argument('output_dir', help='Directory to save processed dataset')
    parser.add_argument('--height', type=int, default=512, help='Target height for resizing')
    parser.add_argument('--width', type=int, default=1024, help='Target width for resizing')
    args = parser.parse_args()
    
    process_dataset(args.rgb_dir, args.mask_dir, args.output_dir, 
                   target_size=(args.height, args.width))
    

#    python labels.py /cluster/home/abubakb/LIACi_dataset_pretty/images /cluster/home/abubakb/LIACi_dataset_pretty/masks /cluster/home/abubakb/final_dataset
