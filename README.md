# MedianNerveDL
A record code for "Automated segmentation of median nerve in dynamic sonography using deep learning: Evaluation of model performance".\
\
Current codes are only for inference.


# Requirements
## python environment
- PyTorch 1.1 or newer version. (for U-Net, FPN, MaskRCNN)
- torchvision 0.3.0 or newer version (for U-Net, FPN, MaskRCNN)
- tensorflow 1.9.0 or newer version (for Deeplabv3+)
  - tensorflow 2.X is not supported
- opencv
- pillow
- matplotlib
- pandas
- numpy
## Ground truth map
Ground truth should follow below polcy.
- Naming: <input_file_name>_mask.png
- Format: png
- A binary map with 1 for Median Nerve pixels, and 0 for background pixels.

# Model
Model_zoo will be further updated.
## U-Net
### Weights
- [u-net_resnet-101](https://drive.google.com/file/d/1GiNDp3t5vI4F884hbTv8O21pH6yDmwQY/view?usp=sharing)
- [u-net_resnext101-32x8d](https://drive.google.com/file/d/1bckxkq9bJrVnZrLw2u0r3rVUMb6icT3D/view?usp=sharing)
### Command
With ground truth
   
    python ./inference_deeplab_option.py --predict_dir <folder_contains_input_images> --model_type unet --backbone <resnet101/resnext101_32x8d> --output_dir <folder_for_output> --gt_dir <folder_contains_ground_truth_masks> --model_path <weights_file>

Without ground truth

    python ./inference_option_withoutgt.py --predict_dir <folder_contains_input_images> --model_type unet --backbone <resnet101/resnext101_32x8d> --output_dir <folder_for_output> --model_path <weights_file>

## FPN
### Weights
- [FPN_resnet-101](https://drive.google.com/file/d/12EWlBX-y-J2ks6vKgOAMP0sW41xXaCQz/view?usp=sharing)
- [FPN_resnext101-32x8d](https://drive.google.com/file/d/1x57kYYT7u0R_FUYk0MDSH6IqKrz96RCd/view?usp=sharing)
### Command
With ground truth

    python ./inference_deeplab_option.py --predict_dir <folder_contains_input_images> --model_type fpn --backbone <resnet101/resnext101_32x8d> --output_dir <folder_for_output> --gt_dir <folder_contains_ground_truth_masks> --model_path <weights_file>

Without ground truth

    python ./inference_option_withoutgt.py --predict_dir <folder_contains_input_images> --model_type fpn --backbone <resnet101/resnext101_32x8d> --output_dir <folder_for_output> --model_path <weights_file>
## Deeplabv3+
Deeplabv3+ was trained with [Deeplab project](https://github.com/tensorflow/models/tree/master/research/deeplab)
### Weights
- [Deeplabv3+_xception-65](https://drive.google.com/file/d/1mxwIXh-XXPrvIilaoiGUiToM4fpTsmTW/view?usp=sharing)
- [Deeplabv3+_resnet-101](https://drive.google.com/file/d/1sOp9Z5gx2oqj1qWbbZknc0eGbPYItf0_/view?usp=sharing)
### Command
With ground truth

    python ./inference_deeplab.py --predict_dir <folder_contains_input_images> --output_dir <folder_for_output> --gt_dir <folder_contains_ground_truth_masks> --model_path <frozen_graph_file>

Without ground truth

    python ./inference_deeplab_withoutgt.py --predict_dir <folder_contains_input_images> --output_dir <folder_for_output> --model_path <frozen_graph_file>

## MaskRCNN
For Mask R-CNN, we use [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)

