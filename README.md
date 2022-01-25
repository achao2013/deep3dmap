
# deep3dmap
This is an 3d reconstruction pipeline in deep learning.

# Introduction

<details open>
<summary>Major features</summary>

- **Modular Design**

  We decompose the reconstruction framework into different components and one can easily construct a customized reconstruction framework by combining different modules.

- **Support of multiple frameworks of 3d reconstruction from image or images**

  The toolbox directly supports popular and contemporary reconstruction frameworks, *e.g.* , etc.

- **differential renderers**

  The toolbox support different differential renderers include neural rerender, pyrender, pytorch3d, 

- **support varigrained reconstructon**
  
  We suppert face, body, indoor, outdoor and so on
  
- **multiple demo applications**
  
  We provide abundant toy demos 

</details>

# Supported methods:

- [x] Learning Multi-Path Architectures for Multi-view Consistent 3D Face Alignment
- [x] Do 2D GANs Know 3D Shape? Unsupervised 3D Shape Reconstruction from 2D Image GANs
- [x] NeuralRecon: Real-Time Coherent 3D Reconstruction from Monocular Video
- [x] Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network
- [ ] GNeRF: GAN-based Neural Radiance Field without Posed Camera
- [ ] GIRAFFE: Representing Scenes as Compositional Generative Neural Feature Fields



# Installation

Please refer to [get_started.md](docs/get_started.md) for installation.


# reference

<details open>
<summary>different from mmcv</summary>

- **can use both dataset and dataloader to prepare input data**
  
- **can define input key in the config which will be used in forward**
  
- **model inputs is formulated as an dictionary with keys**

</details>

