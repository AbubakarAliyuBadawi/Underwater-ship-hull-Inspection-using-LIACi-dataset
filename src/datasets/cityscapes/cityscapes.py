# cityscapes.py
# location (/cluster/home/abubakb/ContMAV/src/datasets/cityscapes/cityscapes.py)

class CityscapesBase:
    # Define splits available in your dataset
    SPLITS = ["train", "valid", "test", "val"]

    # Define number of classes (10 classes + void)
    N_CLASSES = [11]

    # Class names and colors for your dataset
    CLASS_NAMES_FULL = [
        'Void',            # 0
        'Ship hull',       # 1
        'Marine growth',   # 2
        'Anode',          # 3
        'Overboard valve', # 4
        'Propeller',      # 5
        'Paint peel',     # 6
        'Bilge keel',     # 7
        'Defect',         # 8
        'Corrosion',      # 9
        'Sea chest grating'# 10
    ]

    CLASS_COLORS_FULL = [
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

    # Since we're not reducing classes, use the same for reduced
    CLASS_NAMES_REDUCED = CLASS_NAMES_FULL
    CLASS_COLORS_REDUCED = CLASS_COLORS_FULL

    # Mapping classes (we're not reducing, so straight mapping)
    CLASS_MAPPING_REDUCED = {
        0: 0,     # Void
        1: 1,     # Ship hull
        2: 2,     # Marine growth
        3: 3,     # Anode
        4: 4,     # Overboard valve
        5: 5,     # Propeller
        6: 6,     # Paint peel
        7: 7,     # Bilge keel
        8: 8,     # Defect
        9: 9,     # Corrosion
        10: 10    # Sea chest grating
    }

    # Directory structure (matching your processed dataset)
    RGB_DIR = "leftImg8bit"  # Directory containing your RGB images
    LABELS_FULL_DIR = "gtFine"  # Directory containing your ground truth labels
    LABELS_FULL_COLORED_DIR = "gtFine_color"  # Colored visualizations
    LABELS_REDUCED_DIR = "gtFine"  # Same as full since we don't reduce classes
    LABELS_REDUCED_COLORED_DIR = "gtFine_color"  # Same as full colored
