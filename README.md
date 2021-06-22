# Deep Learning Based Segmentation of Multicontrast TNBC PDX MR Images
This repository provides an automated pipeline for automatic localization and segmentation of multicontrast MR images in small animals. In this project we have tried using UNet <sup>1</sup>, Residual UNet <sup>2</sup>, Dense UNet <sup>3</sup>, Recurrent Residual UNet (R2UNet) <sup>4</sup> and Dense Recurrent Residual UNet (DR2UNet) <sup>5</sup>.


## Steps to Test the Deep Learning Pipeline for Segmentation
1. Git Clone the repository
2. Install `python 3.7.10` and the necessary packages by running `pip install -r requirements.txt`
3. Download the trained weights from https://wustl.box.com/s/hntgs87p91fscx624vpajru6xa789p6m into a `weights_new` folder inside the folder where you have cloned the other codes from the repository.
4. Download `images_test.npy` , `images_train.npy`, `images_train_T1.npy` and `mask_train.npy`.
5. Run `main_testing.py`. Have to manually load which model you want to learn. By default the DR2UNet is loaded and running the `main_testing.py` gives the performance score for the selected network on the test dataset. It automatically calculates the optimum threshold based on the precision recall curves for the network. The `main_testing.py` automatically calls the model architecture and the `data2D.py` to load the data.
6. You can view the output probability maps in `output` folder under each model folder name (which you have to change each time you run a model).

##Different Network Model and their architecture files:
| Network Model | Architecture File | Weight File |
| --- | --- | --- |
| DR2UNet | new_r2udensenet.py | model_r2udensenet.hdf5 |
| R2UNet | new_r2unet.py | model_r2unet.hdf5 |
| Dense-UNet | dense_unet.py | model_denseunet.hdf5 |
| Res-UNet | res_unet.py | model_resunet.hdf5 |
| UNet | unet.py | model_unet.hdf5 |

## Steps for Training and Five Fold Cross Validation
1. Git Clone the repository.
2. Install `python 3.7.10` and the necessary packages by running `pip install -r requirements.txt`
3. For five fold cross validation run `main2D.py` after selecting the network model you want to train the data on. The training considers data augmentation due to the limited dataspace available to us. The `main2D.py ` outputs the mean performance of the network model for 5 folds.
4. For standalone training on the whole dataset after the cross-validation run `main_training.py` to generate the training-weights.


## References
1. Ronneberger, O.;  Fischer, P.; Brox, T., U-net: Convolutional networks for biomedical image segmentation. Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics) 2015, 9351, 234-241.
2. He, K.;  Zhang, X.;  Ren, S.; Sun, J., Deep residual learning for image recognition. Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition 2016, 2016-Decem, 770-778.
3. Kolařík, M.;  Burget, R.;  Uher, V.;  Říha, K.; Dutta, M. K., Optimized high resolution 3D dense-U-Net network for brain and spine segmentation. Applied Sciences (Switzerland) 2019, 9 (3).
4. Alom, M. Z.;  Hasan, M.;  Yakopcic, C.;  Taha, T. M.; Asari, V. K., Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net) for Medical Image Segmentation. 2018.
5. Dutta, K., Densely Connected Recurrent Residual (Dense R2UNet) Convolutional Neural Network for Segmentation of Lung CT Images. arXiv preprint arXiv:2102.00663 2021.
