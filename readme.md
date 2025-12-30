\# PRNet implementation





If you find this project useful, please cite:



```

@article{kang2025prnet,

&nbsp; title={PRNet: Parallel Reinforcement Network for two-view correspondence learning},

&nbsp; author={Kang, Zheng and Lai, Taotao and Li, Zuoyong and Wei, Lifang and Chen, Riqing},

&nbsp; journal={Knowledge-Based Systems},

&nbsp; volume={310},

&nbsp; pages={112978},

&nbsp; year={2025},

&nbsp; publisher={Elsevier}

}

```



\## Requirements



Please use Python 3.6, opencv-contrib-python (3.4.0.12) and Pytorch (>= 1.7.0). Other dependencies should be easily installed through pip or conda.





\## Example scripts



\### Run the demo



For a quick start, clone the repo and download the pretrained model.

```bash

git clone https://github.com/kangzheng1999/PRNet



```

Then download the pretrained models from \[here](https://drive.google.com/drive/folders/1JKuIWhMXe9ve3wRPb\_xZmX37ZH1bxAC3).



Then run the feature matching with demo ConvMatch.



```bash

cd ./demo \&\& python demo.py

```



\### Generate training and testing data



First download YFCC100M dataset.

```bash

bash download\_data.sh raw\_data raw\_data\_yfcc.tar.gz 0 8

tar -xvf raw\_data\_yfcc.tar.gz

```



Download SUN3D testing (1.1G) and training (31G) dataset if you need.

```bash

bash download\_data.sh raw\_sun3d\_test raw\_sun3d\_test.tar.gz 0 2

tar -xvf raw\_sun3d\_test.tar.gz

bash download\_data.sh raw\_sun3d\_train raw\_sun3d\_train.tar.gz 0 63

tar -xvf raw\_sun3d\_train.tar.gz

```



Then generate matches for YFCC100M and SUN3D (only testing) with SIFT.

```bash

cd ../dump\_match

python extract\_feature.py

python yfcc.py

python extract\_feature.py --input\_path=../raw\_data/sun3d\_test

python sun3d.py

```

Generate SUN3D training data if you need by following the same procedure and uncommenting corresponding lines in `sun3d.py`.







\### Test pretrained model



We provide the model trained on YFCC100M and SUN3D described in our AAAI paper. Run the test script to get similar results in our paper (the generated putative matches are different once regenerating the data).



```bash

cd ./test 

python test.py

```

You can change the default settings for test in `./test/config.py`.



\### Train model on YFCC100M or SUN3D



After generating dataset for YFCC100M, run the tranining script.

```bash

cd ./core 

python main.py

```



You can change the default settings for network structure and training process in `./core/config.py`.



\### Train with your own local feature or data 



The provided models are trained using SIFT. You had better retrain the model if you want to use PRNet with your own local feature, such as RootSIFT, SuperPoint and etc. 



You can follow the provided example scirpts in `./dump\_match` to generate dataset for your own local feature or data.



\## Acknowledgement

This code is borrowed from \[ConvMatch](https://github.com/krbangzi/ConvMatch)  If using the part of code related to data generation, testing and evaluation, please cite these papers.



```

@inproceedings{zhang2019learning,

&nbsp; title={Learning two-view correspondences and geometry using order-aware network},

&nbsp; author={Zhang, Jiahui and Sun, Dawei and Luo, Zixin and Yao, Anbang and Zhou, Lei and Shen, Tianwei and Chen, Yurong and Quan, Long and Liao, Hongen},

&nbsp; booktitle={Proceedings of the IEEE/CVF international conference on computer vision},

&nbsp; pages={5845--5854},

&nbsp; year={2019}

}

@inproceedings{zhao2021progressive,

&nbsp; title={Progressive correspondence pruning by consensus learning},

&nbsp; author={Zhao, Chen and Ge, Yixiao and Zhu, Feng and Zhao, Rui and Li, Hongsheng and Salzmann, Mathieu},

&nbsp; booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},

&nbsp; pages={6464--6473},

&nbsp; year={2021}

}

```

