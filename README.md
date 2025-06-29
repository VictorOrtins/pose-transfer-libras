This is a modification of the "Official code for Multi-scale Attention Guided Pose Transfer" repository to perform pose transfer in sign language. This small work was for the project of the "Graphic and Multimedia Systems" discipline of the master's degree.


[![badge_torch](https://img.shields.io/badge/made_with-PyTorch_2.0-EE4C2C?style=flat-square&logo=PyTorch)](https://pytorch.org/)

<br>

### :zap: Getting Started
```bash
mkdir pose2pose
cd pose2pose
git clone https://github.com/prasunroy/pose-transfer.git
cd pose-transfer
pip install -r requirements.txt
```

<br>

### Data and Weights
* Download dataset files from [HowSign Dataset](https://how2sign.github.io/#download) and extract into `dataset` directory.
  *   Green Screen RGB clips* (frontal view) and B-F-H 2D Keypoints clips* (frontal view)
  *   Remember to use the generate_frames_and_csv code
* Download pretrained checkpoints from [Google Drive](https://drive.google.com/file/d/1-iwbykju_Bz8l0EloabnYSMqrECXeGIe/view?usp=sharing) into `wherever you want` directory (change the path in the code).

<br>

### External Links
<h4>
  <a href="https://arxiv.org/abs/2202.06777">Original paper arXiv</a>&nbsp;&nbsp;&bull;&nbsp;&nbsp;
  <a href="https://github.com/prasunroy/pose-transfer">Original repository</a>&nbsp;&nbsp;&bull;&nbsp;&nbsp;
  <a href="https://how2sign.github.io/">Dataset</a>&nbsp;&nbsp;&bull;&nbsp;&nbsp;
  <a href="https://drive.google.com/drive/folders/1SDSEfWyP5ZFR8nA-zQLhwjBsRm7ggfWj">Pretrained Models</a>&nbsp;&nbsp;&bull;&nbsp;&nbsp;
</h4>

<br>

### License
```
Copyright 2023 by the authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
