# Semantic Segmentation
### Introduction
In this project a Fully Convolution Network is built based on the VGG 16 architecture used for image classification and it is trained on [Kitti Road dataset](http://www.cvlibs.net/download.php?file=data_road.zip). This is used to perform sematic segmentation to identify road surface from the test set.

### Architecture
The [VGG 16](https://github.com/srikanth-narayanan/CarND-Semantic-Segmentation/blob/master/Graph_VGG_16.png) architecture is originally used for image classification. This architecure consists of 5 convolution layer and 2 fully connected layer.
- The final fully connected layer is converted to 1x1 convolution by setting the depth equal to number of classes, which is road or not a road in this use case. This layer is upsampled to represent the original image size.
- In order to maintain / preserve the feature information skip connections are used. The skip connection 1 carries information from the 4th convolution layer of the VGG 16 architecture. This is converted to a 1x1 convolution layer and element wise addtion is performed with upsampled 1x1 convolution layer to form skip connection.
- The resultant is upsampled and skip connection is repeated with 3rd convolution layer to preserve more feature information.
- The result is upsampled to generate the final layer.
- All convolution layers, upsampled layers uses a l2-regularisation and kernel initialiser.

### Optimiser

The loss function is calculated using cross-entropy and Adam Optimiser is used to reduce the cost.

### Training and Hyperparameters

The following hyperparameters are used for training.

- EPOCH - 60
- BATCH SIZE - 5
- Learning Rate - 0.0009
- keep probability - 0.5

### Results

The network performed overall well in training and test. The average loss per batch for 50 epoch was reduced up to 0.03.

The following are some of the samples of the result, but there are scenrios where the results could be improved.

![Sample 1](https://github.com/srikanth-narayanan/CarND-Semantic-Segmentation/blob/master/runs/Sample_Outputs/um_000013.png)

![Sample 2](https://github.com/srikanth-narayanan/CarND-Semantic-Segmentation/blob/master/runs/Sample_Outputs/um_000014.png)

![Sample 3](https://github.com/srikanth-narayanan/CarND-Semantic-Segmentation/blob/master/runs/Sample_Outputs/um_000027.png)

![Sample 4](https://github.com/srikanth-narayanan/CarND-Semantic-Segmentation/blob/master/runs/Sample_Outputs/um_000066.png)

![Sample 5](https://github.com/srikanth-narayanan/CarND-Semantic-Segmentation/blob/master/runs/Sample_Outputs/um_000077.png)

![Sample 6](https://github.com/srikanth-narayanan/CarND-Semantic-Segmentation/blob/master/runs/Sample_Outputs/uu_000021.png)

### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

 ### Tips
- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip)
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [forum post](https://discussions.udacity.com/t/here-is-some-advice-and-clarifications-about-the-semantic-segmentation-project/403100/8?u=subodh.malgonde) for more information.  A summary of additional points, follow. 
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy. 
- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.
