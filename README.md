# [CVPR 2021] PCLs: Geometry-aware Neural Reconstruction of 3D Pose with Perspective Crop Layers



### [[Paper]](https://arxiv.org/abs/2011.13607)

**PCLs: Geometry-aware Neural Reconstruction of 3D Pose with Perspective Crop Layers**
<br/>
[Frank Yu](https://yu-frank.github.io/),
[Mathieu Salzmann](https://people.epfl.ch/mathieu.salzmann),
[Pascal Fua](https://people.epfl.ch/pascal.fua/bio?lang=en), and
[Helge Rhodin](https://www.cs.ubc.ca/~rhodin/)
<br/>
In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2021.

PyTorch implementation for removing perspective distortions from images or 2D poses to improve accuracy of 3D human pose estimation techniques. Shown below are examples of this perspective distortion and its correction using PCL. Images shown below are from the Human3.6M (left) and MPI-INF-3DHP (right) datasets. 

<img src="imgs/Overview.jpg" style="zoom:25%;" />

## Abstract

Local processing is an essential feature of CNNs and other neural network architectures - it is one of the reasons why they work so well on images where relevant information is, to a large extent, local. However, perspective effects stemming from the projection in a conventional camera vary for different global positions in the image. We introduce Perspective Crop Layers (PCLs) - a form of perspective crop of the region of interest based on the camera geometry - and show that accounting for the perspective consistently improves the accuracy of state-of-the-art 3D pose reconstruction methods. PCLs are modular neural network layers, which, when inserted into existing CNN and MLP architectures, deterministically remove the location-dependent perspective effects while leaving end-to-end training and the number of parameters of the underlying neural network unchanged. We demonstrate that PCL leads to improved 3D human pose reconstruction accuracy for CNN architectures that use cropping operations, such as spatial transformer networks (STN), and, somewhat surprisingly, MLPs used for 2D-to-3D keypoint lifting. Our conclusion is that it is important to utilize camera calibration information when available, for classical and deep-learning-based computer vision alike. PCL offers an easy way to improve the accuracy of existing 3D reconstruction networks by making them geometry-aware.

## Setup

### Prerequisites

- Linux or Windows
- NVIDIA GPU + CUDA CuDNN
- Python 3.6 (Tested on Python 3.6.2)
- PyTorch 1.6.0, 
- Python dependencies listed in [requirements.txt](https://github.com/yu-frank/PerspectiveCropLayers/blob/main/requirements.txt)

To get started, please run the following commands:

```bash
conda create -n pcl python=3.6.2
conda activate pcl
conda install --file requirements.txt
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=pcl
conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.2 -c pytorch
```


## Running the Demos and Using PCL

We have included 2 Jupyter notebook demos for you to try out PCLs on both a general setting (**RECOMMENDED**) [**pcl_demo.ipynb**](https://github.com/yu-frank/PerspectiveCropLayers/blob/main/pcl_demo.ipynb) (which does not require any pretrained models and goes through step-by-step how to use PCL) as well as one geared towards human pose estimation on extracted samples from Human3.6m as well as MPI-INF-3DHP ([**humanPose-demo.ipynb**](https://github.com/yu-frank/PerspectiveCropLayers/blob/main/humanPose-demo.ipynb)) (which requires a pretrained model and additional data)

### Using PCLs

#### Image Input

- To process an image using PCL using the following lines of code:

```python
P_virt2orig, R_virt2orig, K_virt = pcl.pcl_transforms({Crop Position [px; Nx2]}, {Crop Scale [px; Nx2]}, {Camera Intrinsic Matrix [px; Nx3x3]})
grid_sparse = pcl.perspective_grid(P_virt2orig, {Input IMG Dim.}, {Output IMG Dim}, transform_to_pytorch=True)
PCL_cropped_img = F.grid_sample({Original IMG}, grid_perspective)
```

**NOTE:** All input tensors to PCL MUST be in pixel coordinates (including the camera matrix). This means, for a 512x512 image, the range of the coordinates should be [0, 512)

- Once the you pass this to the network you must then convert the predicted pose (which is in the virtual camera coordinates) back to the world coordinates. Please use the following code 

**NOTE:** If the output of the network is normalized you must first deformalized the output before running this line

```python
# Using the same R_virt2orig from the above command
pose_3d = pcl.virtPose2CameraPose(virt_3d_pose, R_virt2orig, batch_size, num_joints)
```

#### 2D Pose Input

- To process an input 2D pose please follow the following steps:

**NOTE:** The input 2D pose is not root centered yet (ie. the hip joint 2D coordinate should **NOT** be (0,0); this should be done afterwards (possibly during the normalization step).

```python
virt_2d_pose, R_virt2orig, P_virt2orig = pcl.pcl_transforms_2d(pose_2d, {Crop Position [px; Nx2]}, {Crop Scale [px; Nx2]}, {Camera Intrinsic Matrix [px; Nx3x3]})
```

- Normalize virt_2d_pose w/ the mean and standard deviation of virt_2d_pose over the whole dataset. This normalized 2D pose is what should be passed to the network.
- Once the network has made its prediction use the following code to convert the predicted 3D pose in virtual camera coordinates back to world coordinates with the following code:

```python
# Using the same R_virt2orig from the above command
pose_3d = pcl.virtPose2CameraPose(virt_3d_pose, R_virt2orig, batch_size, num_joints)
```

#### Coordinate Systems Used:

**Image coordinates:** First coordinate is the horizontal axis, second coordinate is the y axis, and the origin is in the top left.

**3D coordinates (left-handed coordinate system)**: First coordinate is the horizontal axis (left to right), second coordinate is the vertical axis (up), and the third is in depth direction (positive values in front of camera).

## Training and Evaluation

### Preprocessing the Dataset:

Please follow the instructions from [Margipose](https://github.com/anibali/margipose) for downloading, pre-processing, and storing the data.

### Pretrained Models:

Included in the GitHub are 4 sets of pretrained models that are used in [**humanPose-demo.ipynb**](https://github.com/yu-frank/PerspectiveCropLayers/blob/main/humanPose-demo.ipynb)

### Train and Evaluation Code:

We have also included training and evaluation code.

## License

This work is licensed under MIT License. See [LICENSE](https://github.com/yu-frank/PerspectiveCropLayers/blob/main/LICENSE) for details.

If you find our code helpful, please consider citing the following paper:

```
@article{yu2020pcls,
  title={PCLs: Geometry-aware Neural Reconstruction of 3D Pose with Perspective Crop Layers},
  author={Yu, Frank and Salzmann, Mathieu and Fua, Pascal and Rhodin, Helge},
  journal={arXiv preprint arXiv:2011.13607},
  year={2020}
}
```

## Acknowledgements

- We would like to thank Dushyant Mehta and Jim Little for valuable discussion and feedback. Thank you to Shih-Yang Su for his feedback in the creation of this repository. This work was funded in part by Compute Canada and a Microsoft Swiss JRC project. Frank Yu was supported by NSERC-CGSM.
- We would also like to thank the authors of [Nibali et al. 2018](https://github.com/anibali/margipose), [Martinez et al. 2017](https://github.com/una-dinosauria/3d-pose-baseline), and [Pavllo et al. 2019](https://dariopavllo.github.io/VideoPose3D/) for making their implementations available.
- Our code builds upon [3d-pose-baseline](https://github.com/weigq/3d_pose_baseline_pytorch), [Margipose](https://github.com/anibali/margipose), and [VideoPose3D](https://github.com/facebookresearch/VideoPose3D)

