# Open-world Semantic Segmentation for Underwater Shiphull Inspection

This project implements an open-world semantic segmentation framework for underwater ship hull inspection, adapting the approach from recent advances in autonomous driving to address the unique challenges of underwater imagery analysis.

## Overview

The system utilizes a dual-decoder architecture to perform both traditional semantic segmentation and novel defect detection on underwater ship hull imagery. The implementation achieves a mean IoU of 50.9% across 11 classes while maintaining the ability to identify unknown anomalies.

### Key Features

- Dual-decoder architecture with ResNet34 backbone
- Open-world semantic segmentation capabilities
- Specialized loss functions for underwater imagery 
- Support for both known defect detection and novel anomaly identification
- Comprehensive feature space organization through contrastive learning

## Dataset

The project uses the LIACI dataset from SINTEF Ocean, which includes:
- 1893 annotated images
- 11 semantic classes 
- Diverse underwater conditions and defect types

### Classes
1. Void (background)
2. Ship hull
3. Marine growth
4. Anode
5. Overboard valve
6. Propeller
7. Paint peel
8. Bilge keel
9. Defect
10. Corrosion
11. Sea chest grating

## Project Structure

```
final_dataset/
├── leftImg8bit/
│   ├── train/
│   ├── val/
│   └── test/
└── gtFine/
    ├── train/
    ├── val/
    └── test/
```

## Implementation Details

### Loss Functions
- Semantic Segmentation Loss (weight: 0.9)
- Objectosphere Loss (weight: 0.5)
- MAV Loss (weight: 0.1)
- Contrastive Loss (weight: 0.5)

### Training Configuration
- Image size: 512×512 pixels
- Learning rate: 1e-4
- Optimizer: Adam
- Batch size: 8
- Gradient accumulation steps: 4
- Training epochs: 2000

## Performance

### Closed-World Performance
| Class Category | IoU (%) |
|----------------|---------|
| Ship hull | 78.5 |
| Marine growth | 81.2 |
| Sea chest grating | 84.7 |
| Anode | 39.8 |
| Overboard valve | 63.4 |
| Paint peel | 29.1 |
| Mean IoU | 50.03 |

### Open-World Performance
| Class Category | IoU (%) |
|----------------|---------|
| Void | 78.0 |
| Ship hull | 80.0 |
| Sea chest grating | 82.0 |
| Propeller | 70.0 |
| Mean IoU | 50.9 |

## Usage

1. Dataset Preparation:
```bash
python prepare_dataset.py --input_dir /path/to/raw/data --output_dir final_dataset
```

2. Training:
```bash
python train.py --data_dir final_dataset --batch_size 8 --epochs 2000
```

3. Evaluation:
```bash
python evaluate.py --model_path /path/to/model --test_dir final_dataset/test
```

## Future Improvements

1. Dataset Enhancement:
   - Expand dataset with additional rare defect examples
   - Incorporate domain-specific augmentation techniques

2. Model Improvements:
   - Implement attention mechanisms for varying illumination
   - Add temporal consistency checks
   - Develop confidence estimation metrics

3. Deployment Optimization:
   - Create lightweight model variants
   - Implement online learning capabilities
   - Optimize for real-time ROV deployment

## Citation

```bibtex
@article{underwater_inspection_2024,
  title={Open-world Semantic Segmentation for Underwater Shiphull Inspection},
  author={Badawi, Abubakar Aliyu and Ishfaq-Bhat},
  journal={Department of Marine Technology, NTNU},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- SINTEF Ocean for providing the LIACI dataset
- Norwegian University of Science and Technology (NTNU) for research support
- The authors of the original open-world semantic segmentation paper

## Contact

For questions or collaborations, please contact:
- Abubakar Aliyu Badawi - abubakb@stud.ntnu.com
- Ishfaq-Bhat - ishfaqb@stud.ntnu.com
