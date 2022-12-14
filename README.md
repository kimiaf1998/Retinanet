# RetinaNet - Object Detection
<img src="https://badgen.net/badge/Version/1.1/blue?icon=github"> <img src="https://badgen.net/badge/Status/Stable/green?icon=git">
<br/>


| ![scr1](https://github.com/kimiaf1998/Retinanet/blob/master/screenshots/5.png "Detection Result 1") | ![scr2](https://github.com/kimiaf1998/Retinanet/blob/master/screenshots/6.png "Detection Result 2") |
| ------------ | ------------ |
| ![scr4](https://github.com/kimiaf1998/Retinanet/blob/master/screenshots/1.png "Detection Result 3") | ![scr3](https://github.com/kimiaf1998/Retinanet/blob/master/screenshots/3.png "Detection Result 4") |

RetinaNet is a one-stage detection algorithm introduced in 2017 and 2018 by Facebook AI Researchers (FAIR) in a paper called [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002 "Focal Loss for Dense Object Detection").
In this repository, an attempt has been made to implement the algorithm on ResNet-50 and ResNet-101 backbones, primarily using the [Pytorch](https://pytorch.org/ "Pytorch") library with [CUDA](https://en.wikipedia.org/wiki/CUDA "CUDA") support under the COCO-2014 dataset.

<br>

## What is RETINANET?
According to the [definition](https://tinyurl.com/5bksrrrr "definition"):
> RetinaNet is a single, unified network composed of a backbone network and two task-specific subnetworks. The backbone is responsible for computing a conv feature map over an entire input image and is an off-the-self convolution network. The first subnet performs classification on the backbones output; the second subnet performs convolution bounding box regression.

<br>

 ![scr1](https://pbs.twimg.com/media/D_TF0tjUEAElZkp.jpg "RetinaNet Architecture")

<br>This network employs two key ideas to resolve the class imbalance problem:
- **Feature Pyramud Network**: By [definition](https://paperswithcode.com/method/fpn "definition"), a Feature Pyramid Network, or FPN, is a feature extractor that takes a single-scale image of an arbitrary size as input, and outputs proportionally sized feature maps at multiple levels, in a fully convolutional fashion. This process is independent of the backbone convolutional architectures. It therefore acts as a generic solution for building feature pyramids inside deep convolutional networks to be used in tasks like object detection.


- **Focal Loss Function**:  By [defenition](https://paperswithcode.com/method/focal-loss#:~:text=Focal%20loss%20applies%20a%20modulating,in%20the%20correct%20class%20increases. "defenition"), Focal loss applies a modulating term to the cross entropy loss in order to focus learning on hard misclassified examples. It is a dynamically scaled cross entropy loss, where the scaling factor decays to zero as confidence in the correct class increases. Intuitively, this scaling factor can automatically down-weight the contribution of easy examples during training and rapidly focus the model on hard examples
<br>

## Requriements
We have utilized the [PyCharm](https://www.jetbrains.com/pycharm/ "PyCharm") IDE to develop the project. Besides, the following libraries have been used throughout the development:
- Pytorch (Cuda-Support Version)
- TorchVision
- Numpy
- OpenCV
- Pycocotools
- Skimage
<br>

## Run
1. **Train the network**
The first step to executing the project is to train it. You must train the network and use the generated weight file to perform the object detection algorithm. Hence, open the project in your IDE and change the `DATASET_PATH` variable to point to the directory of your dataset (i.e., COCO). This variable is located in the first line of the [constants.py](https://github.com/kimiaf1998/Retinanet/blob/master/net/utility/constants.py "constants.py"). You can also change other parameters including epochs and loss within this file. Finally, start training the network by running the [train.py](https://github.com/kimiaf1998/Retinanet/blob/master/train.py "train.py") file to store .pt files on your local disk.<br>(You can also use this [pre-trained model](https://drive.google.com/file/d/1VWnq0Okc5mXDdUX5RxwxhdAfVbPtdUNG/view "pre-trained model") instead of training the network from scratch.)


2. **Visualize outputs**
Modify the `MODEL_PATH` variable to your desired weight file and run the [visualization.py](https://github.com/kimiaf1998/Retinanet/blob/master/visualization.py "visualization.py") file. Depending on your system configuration, it may take a while to show corresponding bounding boxes around the detected objects within a batch of images.
