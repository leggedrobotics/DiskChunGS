# DiskChunGS License

Copyright (C) 2025 Robotic Systems Lab, ETH Zurich

## GNU General Public License v3.0

DiskChunGS is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.

## Upstream Projects

DiskChunGS builds upon work from the following projects:

1. **CaRtGS** (Copyright 2024 Dapeng Feng, Sun Yat-sen University)
   - License: GPL v3
   - Repository: https://github.com/DapengFeng/cartgs

2. **Photo-SLAM** (Copyright 2023-2024 Longwei Li, Hui Cheng)
   - License: GPL v3
   - Repository: https://github.com/HuajianUP/Photo-SLAM

3. **ORB-SLAM3** (Copyright 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gomez Rodriguez, Jose M.M. Montiel, Juan D. Tardos, University of Zaragoza)
   - License: GPL v3
   - Repository: https://github.com/UZ-SLAMLab/ORB_SLAM3

4. **depth-anything-tensorrt** (Copyright spacewalk01)
   - License: MIT
   - Repository: https://github.com/spacewalk01/depth-anything-tensorrt

5. **xfeat_cpp** (Copyright udaysankar01)
   - License: Apache-2.0
   - Repository: https://github.com/udaysankar01/xfeat_cpp

## External Components (third_party/gaussian_splatting/)

The `third_party/gaussian_splatting/` module contains code derived from INRIA
research projects. These components are distributed under their own licenses,
which are **not** GPL v3. Their use is restricted to non-commercial research
and evaluation.

1. **3D Gaussian Splatting** (Copyright 2023 Inria & Max Planck Institut for Informatik)
   - License: Inria Non-Commercial Research License
   - Repository: https://github.com/graphdeco-inria/gaussian-splatting
   - **Restriction: Non-commercial use only**
   - Full license text: [third_party/gaussian_splatting/LICENSE_3DGS.md](third_party/gaussian_splatting/LICENSE_3DGS.md)

2. **On-The-Fly-NVS** (Copyright 2025 Inria, GRAPHDECO research group)
   - License: Inria Non-Commercial Research License
   - Repository: https://github.com/graphdeco-inria/on-the-fly-nvs
   - **Restriction: Non-commercial use only**
   - Full license text: [third_party/gaussian_splatting/LICENSE_ONTHEFLY.md](third_party/gaussian_splatting/LICENSE_ONTHEFLY.md)

For commercial licensing of the INRIA components, contact:
stip-sophia.transfert@inria.fr

## Contact

- DiskChunGS: casimir.feldmann@gmail.com

## Citation

If you use this software in your research, please cite:

```bibtex
@ARTICLE{11417439,
  author={Feldmann, Casimir and Wilder-Smith, Maximum and Patil, Vaishakh and Oechsle, Michael and Niemeyer, Michael and Tateno, Keisuke and Hutter, Marco},
  journal={IEEE Robotics and Automation Letters},
  title={DiskChunGS: Large-Scale 3D Gaussian SLAM Through Chunk-Based Memory Management},
  year={2026},
  volume={11},
  number={4},
  pages={5009-5016},
  doi={10.1109/LRA.2026.3668704}}

@ARTICLE{10900401,
  author={Feng, Dapeng and Chen, Zhiqiang and Yin, Yizhen and Zhong, Shipeng and Qi, Yuhua and Chen, Hongbo},
  journal={IEEE Robotics and Automation Letters},
  title={CaRtGS: Computational Alignment for Real-Time Gaussian Splatting SLAM},
  year={2025},
  volume={10},
  number={5},
  pages={4340-4347},
  doi={10.1109/LRA.2025.3544928}}

@INPROCEEDINGS{10657868,
  author={Huang, Huajian and Li, Longwei and Cheng, Hui and Yeung, Sai-Kit},
  booktitle={2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  title={Photo-SLAM: Real-Time Simultaneous Localization and Photorealistic Mapping for Monocular, Stereo, and RGB-D Cameras},
  year={2024},
  pages={21584-21593},
  doi={10.1109/CVPR52733.2024.02039}}

@article{10.1145/3592433,
  author = {Kerbl, Bernhard and Kopanas, Georgios and Leimkuehler, Thomas and Drettakis, George},
  title = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
  year = {2023},
  volume = {42},
  number = {4},
  doi = {10.1145/3592433},
  journal = {ACM Trans. Graph.},
  articleno = {139}}

@article{10.1145/3730913,
  author = {Meuleman, Andreas and Shah, Ishaan and Lanvin, Alexandre and Kerbl, Bernhard and Drettakis, George},
  title = {On-the-fly Reconstruction for Large-Scale Novel View Synthesis from Unposed Images},
  year = {2025},
  volume = {44},
  number = {4},
  doi = {10.1145/3730913},
  journal = {ACM Trans. Graph.},
  articleno = {125}}

@ARTICLE{10330699,
  author={Xu, Gangwei and Wang, Yun and Cheng, Junda and Tang, Jinhui and Yang, Xin},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  title={Accurate and Efficient Stereo Matching via Attention Concatenation Volume},
  year={2024},
  volume={46},
  number={4},
  pages={2461-2474},
  doi={10.1109/TPAMI.2023.3335480}}

@ARTICLE{9440682,
  author={Campos, Carlos and Elvira, Richard and Rodriguez, Juan J. Gomez and M. Montiel, Jose M. and D. Tardos, Juan},
  journal={IEEE Transactions on Robotics},
  title={ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial, and Multimap SLAM},
  year={2021},
  volume={37},
  number={6},
  pages={1874-1890},
  doi={10.1109/TRO.2021.3075644}}
```
