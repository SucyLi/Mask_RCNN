"""
Mask R-CNN
Train on the fMRI tissue dataset

Author: Xinhui Li
------------------------------------------------------------
Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 tissue.py train --dataset=/path/to/tissue/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 tissue.py train --dataset=/path/to/tissue/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 tissue.py train --dataset=/path/to/tissue/dataset --weights=imagenet
"""

import os
import sys
import json
import datetime
import numpy as np
import nibabel as nb
import skimage.draw

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
# TISSUE_DIR = os.path.join(ROOT_DIR, "datasets/tissue")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
print(os.getcwd())
import mrcnn
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################

class TissueConfig(Config):
    """Configuration for training on the fMRI tissue dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "tissue"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    # IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 6  # Background + tissue

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    IMAGE_MIN_DIM = 256
    
    IMAGE_MAX_DIM = 256

############################################################
#  Dataset
############################################################

class TissueDataset(utils.Dataset):

    def load_tissue(self, dataset_dir, subset):
        """Load a subset of the tissue dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or valid
        """
        # Add classes. We have only one class to add.
        self.add_class("tissue", 1, "wm")
        self.add_class("tissue", 2, "gm")
        self.add_class("tissue", 3, "csf")
        self.add_class("tissue", 4, "skull")
        self.add_class("tissue", 5, "skin")
        self.add_class("tissue", 6, "eye")

        # Train or validation dataset?
        assert subset in ["train", "valid"]
        dataset_dir = os.path.join(dataset_dir, subset)

        t1_dir = os.path.join(dataset_dir, "t1w")
        mask_dir = os.path.join(dataset_dir, "mask")
        t1_list = os.listdir(t1_dir)
        t1_list.sort()
        mask_list = os.listdir(mask_dir)
        mask_list.sort()

        # Add images
        for i, t1 in enumerate(t1_list):
            t1_image_path = os.path.join(t1_dir, t1)
            t1_image = nb.load(t1_image_path).get_fdata()

            mask_image_path = os.path.join(mask_dir, mask_list[i])
            mask_image = nb.load(mask_image_path).get_fdata()

            height, width = t1_image.shape[:2]

            for j in range(t1_image.shape[2]):
                self.add_image(
                    "tissue",
                    image_id=t1.split('.nii.gz')[0]+'_'+str(j),  # use file name + slice number as a unique image id
                    path=t1_image_path,
                    height=height,
                    width=width, 
                    slice=j,
                    tissue=np.unique(mask_image[:,:,j]),
                    mask=mask_image[:,:,j])

    def load_image(self, image_id):
        # Load image
        image = nb.load(self.image_info[image_id]['path']).get_fdata()
        slice = self.image_info[image_id]['slice']
        image = image[:,:,slice]
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        mask_raw = info["mask"]
        tissue_list = np.unique(mask_raw).tolist() 
        tissue_list.remove(0) # remove background
        mask = np.zeros([info["height"], info["width"], len(tissue_list)], dtype=np.bool)

        for i, t in enumerate(tissue_list):
            mask[:,:,i] = mask_raw == t
        
        return mask, np.array(tissue_list)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "tissue":
            return info["image_id"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = TissueDataset()
    dataset_train.load_tissue(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = TissueDataset()
    dataset_val.load_tissue(args.dataset, "valid")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')

############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect tissues given fMRI scans.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/tissue/dataset/",
                        help='Directory of the tissue dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    
    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = TissueConfig()
    else:
        class InferenceConfig(TissueConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    else:
        print("'{}' is not recognized. "
              "Use 'train'".format(args.command))
