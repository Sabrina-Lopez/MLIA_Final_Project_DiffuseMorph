<img src="./img/representative.png">

# DiffuseMorph

This repository is for the implementation of "DiffuseMorph: Unsupervised Deformable Image Registration Using Diffusion Model" for Machine Learning in Image Analysis of Fall 2025.

[[Original Paper](https://arxiv.org/abs/2112.05149)]

## Requirements
  * OS : Linux
  * Python == 3.11.6
  * The requirements in requirements.txt in a conda environment

## Data
In our experiments, we used the following datasets:
* brain MR: [Downloaded from MLIA class final project materials]
* MNIST: [Downloaded from MLIA class final project materials]
* Google Draw: [Downloaded from MLIA class final project materials]

## Training or Testing Preparation

Before running training or testing, check the data/\_\_init\_\_.py and ensure that the correct dataset class is being imported for the 'create_dataset_2D' function. If you want to train or test with the Brain MR dataset, uncomment 'from data.brainMR_dataset import brainMRDataset as D' and comment the other dataset class imports. Complete the same process for the Google Draw and MNIST datasets.

Before testing, ensure that the path for the 'resume_state' variable of the testing configuration file has the correct file path to your pre-trained model with the same format for the checkpoint reference. For the project, the testing configuration files for the Brain MR and Google Draw datasets should be already correctly configured.

## Training

To train our model for 2D image registration on Brain MR dataset, runthis command:

```train
python main_2D.py -p train -c config/diffuseMorph_train_2D_brainMR.json
```
The key hyperparameters that were tuned the following: dropout, linear_start, linear_end, n_epoch, lambda loss, val_freq, and lr.

To train our model for 2D image registration on MNIST dataset, run this command:

```train
python main_2D.py -p train -c config/diffuseMorph_train_2D_mnist.json
```

## Test

To test the trained our model for 2D image registration on Brain MR dataset, run:

```eval
python main_2D.py -p test -c config/diffuseMorph_test_2D_brainMR.json
```
To test the trained our model for 2D image registration on Google Draw dataset, run:

```eval
python main_2D.py -p test -c config/diffuseMorph_test_2D_googleDraw.json
```

## Pre-trained Models

Access the Google Drive link and download the experiments zip folder and place in the main DiffuseMorph project directory. Here is the link: https://drive.google.com/file/d/1xtqshoA1g9t7HqySHj9RAc_uPy4OF1E2/view?usp=sharing. The pre-trained models are the following: 

- Brain MR pre-trained model destination:
```
./experiments/DiffuseMorph_2D_251211_201614
```
- MNIST pre-trained model destination:
```
./experiments/DiffuseMorph_2D_251211_201945
```

## Viewing Image Output Results

To view the image outputs of the training, go to your pre-trained model folder and then
```
./checkpoint/images
```
The source image is the *_M.png, the fixed image is the *_F.png, the generated sample from diffusion is the *_MF.png, the registration image is the *_out_M.png, and the registration field visualization is the *_flow.png.

To view the image outputs (i.e., continuous generation and continuous registration), go to your tested model folder (which you can always double check the checkpoint folder whether it has checkpoint paths whether the folder is training or testing) and then
```
./results
```
The images with the sub-string "sample" are representations of the continuous generation. Meanwhile, the images with the sub-string "regist" are representations of the continuous registration.

## Citation

```    
@inproceedings{kim2022diffusemorph,
  title={DiffuseMorph: Unsupervised Deformable Image Registration Using Diffusion Model},
  author={Kim, Boah and Han, Inhwa and Ye, Jong Chul},
  booktitle={European Conference on Computer Vision},
  pages={347--364},
  year={2022},
  organization={Springer}
}
```

