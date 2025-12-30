# PRNet implementation


If you find this project useful, please cite:

```
@inproceedings{10.1145/3746027.3755581,
author = {Huang, Xiangui and Lai, Taotao and Liu, Yizhang and Lin, Shuyuan and Li, Zuoyong},
title = {Two-View Correspondence Pruning via Channel-Spatial Interaction and Bidirectional Consensus Interaction},
year = {2025},
booktitle = {Proceedings of the 33rd ACM International Conference on Multimedia},
pages = {8577–8585},
numpages = {9},
keywords = {camera pose estimation., consensus learning, correspondence pruning, feature matching, outlier removal},
location = {Dublin, Ireland},
series = {MM '25}
}
```

## Requirements

Please use Python 3.6, opencv-contrib-python (3.4.0.12) and Pytorch (>= 1.7.0). Other dependencies should be easily installed through pip or conda.


## Example scripts

### Run the demo

For a quick start, clone the repo and download the pretrained model.
```bash
git clone https://github.com/jiaowohxg/CSBCNet/

```bash
cd ./core 
python main.py
```

## Acknowledgement
This code is borrowed from [NCMNet](https://github.com/xinliu29/NCMNet)  If using the part of code related to data generation, testing and evaluation, please cite these papers.

```
@inproceedings{zhang2019learning,
  title={Learning two-view correspondences and geometry using order-aware network},
  author={Zhang, Jiahui and Sun, Dawei and Luo, Zixin and Yao, Anbang and Zhou, Lei and Shen, Tianwei and Chen, Yurong and Quan, Long and Liao, Hongen},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={5845--5854},
  year={2019}
}
@inproceedings{zhao2021progressive,
  title={Progressive correspondence pruning by consensus learning},
  author={Zhao, Chen and Ge, Yixiao and Zhu, Feng and Zhao, Rui and Li, Hongsheng and Salzmann, Mathieu},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={6464--6473},
  year={2021}
}
@inproceedings{liu2023ncmnet,
  author    = {Liu, Xin and Yang, Jufeng},
  title     = {Progressive Neighbor Consistency Mining for Correspondence Pruning},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2023},
  pages     = {9527-9537}
}
```
