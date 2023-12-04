## [MulT: An End-to End Mutitask Learning Transformer](https://ivrl.github.io/MulT/)
### CVPR 2022

------------------------------------------------------------------------------------------------------------------------
### Installation

Our project is developed based on [mmsegmentation](https://github.com/open-mmlab/mmsegmentation). Please follow the official docs for installation and dataset preparation.

#### A from-scratch setup script (Linux)

Here is a full script for setting up with conda.

```shell
conda update --all
conda install pytorch=1.7.0 torchvision torchaudio cudatoolkit=11.0 -c pytorch
pip install mmcv-full==1.2.7 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
pip install -e .  # or "python setup.py develop"
pip install -r requirements/optional.txt
cd ../..
cd project/Swin-Transformer-MulT
pip install -e .  # or "python setup.py develop"
pip install -r requirements/optional.txt
cd ..
sudo apt update
sudo apt install libgl1-mesa-glx  # Originally posted by @rsignell-usgs in https://github.com/conda-forge/pygridgen-feedstock/issues/10#issuecomment-365914605
```

### Training & Evaluation

####  Shared-attention encoding
```shell
cd SETR

# Training
# bash ./tools/dist_train.sh ${config}                          ${gpu_num}
  bash ./tools/dist_train.sh configs/METR/MMETR_40k_NYU_bs_8.py 2

# Evaluation
# bash ./tools/dist_test.sh ${config}                          ${pth_file}                         ${gpu_num} --eval mIoU
  bash ./tools/dist_test.sh configs/METR/MMETR_40k_NYU_bs_8.py /metr.pth 2          --eval mIoU
```


####  MulT
```shell
cd Swin-Transformer-MulT

# Training
# bash ./tools/dist_train.sh ${config}                                               ${gpu_num}
  bash ./tools/dist_train.sh configs/swind/swind_dea_large_patch4_window7_40k_NYU.py 2

# Evaluation
# bash ./tools/dist_test.sh ${config}                                               ${pth_file}                         ${gpu_num} --eval mIoU
  bash ./tools/dist_test.sh configs/swind/swind_dea_large_patch4_window7_40k_NYU.py <path>/dea4.pth 2          --eval mIoU
```
In the example above, [model decode_head type](Swin-Transformer-MulT/configs/swind/swind_dea_large_patch4_window7_40k_NYU.py#L19) is set to ``SwinTransformerDEA4``.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### Reference
```
  @InProceedings{Bhattacharjee_2022_CVPR,
    author    = {Bhattacharjee, Deblina and Zhang, Tong and S\"usstrunk, Sabine and Salzmann, Mathieu},
    title     = {MulT: An End-to-End Multitask Learning Transformer},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {12031-12041}
}
```
License: [Creative Commons Attribution Non-commercial No Derivatives](http://creativecommons.org/licenses/by-nc-nd/3.0/)
