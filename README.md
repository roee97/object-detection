# Object-Detection (yolo)

This is mostly a very basic and incomplete implementation of yolo (single shot detection).


## Current Features and Model

This version predicts bounding boxes and confidence for each one.

Currently there is only one anchor per prediction position, and classification is not yet implemented.

The model uses a pre-trained ResNet-50 from torchvision as a backbone (this can easily be changed).


## Usage (Training)

Currently, the main functionality is in [train.py](train.py) which uses
 [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning)
 to train the detection model on the COCO dataset.
You can use the supplied [Dockerfile](Dockerfile) to create a suitable environment.

### Data

The directory structure is expected to be the following:
```
path/to/coco/
  annotations/  # annotation json files
  train2017/    # train images
  val2017/      # val images
```

## Logging

The training process is logged to tensorboard. The default directory (created after running) is tb_logs.
 
