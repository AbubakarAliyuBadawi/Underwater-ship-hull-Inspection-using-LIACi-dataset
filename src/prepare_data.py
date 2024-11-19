# prepare_data.py
# location (/cluster/home/abubakb/ContMAV/src/prepare_data.py)


from torch.utils.data import DataLoader
from src import preprocessing
from src.datasets import Cityscapes
import torch

def prepare_data(args, ckpt_dir=None, with_input_orig=False, split=None):
    """
    Modified prepare_data function for your custom dataset with 11 classes
    """
    train_preprocessor_kwargs = {}
    
    # Modified for your dataset
    if args.dataset == "cityscapes":
        Dataset = Cityscapes
        dataset_kwargs = {
            "n_classes": 11,  # Changed to 11 for your classes
            "classes": args.num_classes  # Use the value from args
        }
        valid_set = split if split in ["val", "valid", "test"] else "val"
    else:
        raise ValueError(f"Unknown dataset: `{args.dataset}`")

    # Data augmentation settings
    if args.aug_scale_min != 1 or args.aug_scale_max != 1.4:
        train_preprocessor_kwargs["train_random_rescale"] = (
            args.aug_scale_min,
            args.aug_scale_max,
        )

    # Initialize datasets
    train_data = Dataset(
        data_dir=args.dataset_dir,
        split="train",
        with_input_orig=with_input_orig,
        overfit=args.overfit,
        **dataset_kwargs  # Remove the duplicate classes parameter
    )
    print(f"Train dataset size: {len(train_data)} images")

    valid_data = Dataset(
        data_dir=args.dataset_dir,
        split=valid_set,
        with_input_orig=with_input_orig,
        overfit=args.overfit,
        **dataset_kwargs  # Remove the duplicate classes parameter
    )
    print(f"Validation dataset size: {len(valid_data)} images")

    test_data = Dataset(
        data_dir=args.dataset_dir,
        split="test",
        with_input_orig=with_input_orig,
        overfit=args.overfit,
        **dataset_kwargs  # Remove the duplicate classes parameter
    )
    print(f"Test dataset size: {len(test_data)} images")
    
    # Set up preprocessors
    train_preprocessor = preprocessing.get_preprocessor(
        height=args.height,
        width=args.width,
        phase="train",
        **train_preprocessor_kwargs,
    )
    train_data.preprocessor = train_preprocessor

    valid_preprocessor = preprocessing.get_preprocessor(
        height=args.height,
        width=args.width,
        phase="test",
    )

    if args.valid_full_res:
        valid_preprocessor_full_res = preprocessing.get_preprocessor(
            phase="test",
        )

    valid_data.preprocessor = valid_preprocessor
    test_data.preprocessor = valid_preprocessor

    # Handle case where dataset directory is not provided
    if args.dataset_dir is None:
        if args.valid_full_res:
            return valid_data, valid_preprocessor_full_res
        else:
            return valid_data, valid_preprocessor

    # Configure batch sizes
    if args.overfit:
        args.batch_size = 2
        args.batch_size_valid = 2

    # Create data loaders
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        num_workers=args.workers,
        drop_last=True,
        shuffle=not args.overfit,
    )
    print(f"Train loader batches: {len(train_loader)}")

    batch_size_valid = args.batch_size_valid or args.batch_size
    valid_loader = DataLoader(
        valid_data,
        batch_size=batch_size_valid,
        num_workers=args.workers,
        shuffle=False
    )
    print(f"Validation loader batches: {len(valid_loader)}") 

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size_valid,
        num_workers=args.workers,
        shuffle=False
    )

    return train_loader, valid_loader, test_loader

def count_classes(train_loader, valid_loader):
    """
    Modified to count 11 classes instead of 20
    """
    # Initialize counters with correct size (11 classes including void)
    n_classes = train_loader.dataset._n_classes + 1  # Add 1 for void class
    train_count = torch.zeros(n_classes)
    val_count = torch.zeros(n_classes)
    
    print("\nAnalyzing dataset class distribution...")
    
    # Count training set classes
    print("Processing training set...")
    for data in train_loader:
        label = data["label"]
        for i in range(n_classes):
            train_count[i] += torch.sum(label == i).item()
    
    # Count validation set classes
    print("Processing validation set...")
    for data in valid_loader:
        label = data["label"]
        for i in range(n_classes):
            val_count[i] += torch.sum(label == i).item()
    
    # Calculate percentages
    train_total = train_count.sum()
    val_total = val_count.sum()
    train_percentages = (train_count / train_total * 100).numpy()
    val_percentages = (val_count / val_total * 100).numpy()
    
    # Print results
    print("\nClass Distribution Summary:")
    print(f"{'Class':<20} {'Train %':<10} {'Val %':<10}")
    print("-" * 40)
    
    class_names = train_loader.dataset.class_names
    for i in range(n_classes):
        class_name = class_names[i] if i < len(class_names) else f"Class {i}"
        print(f"{class_name:<20} {train_percentages[i]:<10.2f} {val_percentages[i]:<10.2f}")
    
    return train_count, val_count
