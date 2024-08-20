[![DOI](https://zenodo.org/badge/149826542.svg)](https://doi.org/10.1101/2020.01.17.910562)

[bioRxiv](https://www.biorxiv.org/content/10.1101/2024.07.18.601886v1) and [Dropbox](https://www.dropbox.com/scl/fo/ealblchspq427pfmhtg7h/ALZ7AE5o3bT0VUQ8TTeR1As?rlkey=1e6tseyuwd04rbj7wmn2n6ij7&e=4&st=ybsvv0ry&dl=0)


# pytorch-3dunet

PyTorch implementation 3D U-Net and its variants:

- Standard 3D U-Net based on [3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation](https://arxiv.org/abs/1606.06650) 
Özgün Çiçek et al.

- Residual 3D U-Net based on [Superhuman Accuracy on the SNEMI3D Connectomics Challenge](https://arxiv.org/pdf/1706.00120.pdf) Kisuk Lee et al.

The code allows for training the U-Net for: **semantic segmentation** (binary and multi-class), **instance segmentation** (typically, by using binary training data configured to have background regions between adjacent foreground regions) and **regression** problems (e.g. de-noising, learning deconvolutions).

The following networks are implemented using this package:
- SegmentationNet as used in the [ANTSUN pipeline](https://www.sciencedirect.com/science/article/pii/S0092867423008504?via%3Dihub) Adam Atanas & Jungsoo Kim et al.

- AutoCellLabeler based on [Deep Neural Networks to Register and Annotate the Cells of the *C. elegans* Nervous System](https://www.biorxiv.org/content/10.1101/2024.07.18.601886v1) Adam Atanas et al.

## AutoCellLabeler

### Parameter files

The parameter files used to train and evaluate the AutoCellLabeler network and its variants are available in the [`AutoCellLabeler_parameters`](AutoCellLabeler_parameters) section. `train` parameter files are used for training and `test` are used for evaluation. The network names are:

- `all` is the full network
- `all_red` is the TagRFP-only network
- `OFP` is the OFP+TagRFP-only network
- `BFP` is the BFP+TagRFP-only network
- `mNeptune` is the mNeptune+TagRFP-only network
- `nocustomaug` is the network with our custom augmentations disabled
- `noweight` is the network but with all pixels and channels weighted equally

### Training, validation, and testing data

Please see the [AutoCellLabeler package](https://github.com/flavell-lab/AutoCellLabeler) for instructions on downloading and formatting the training, validation, and testing data for AutoCellLabeler.

### Model weights

The AutoCellLabeler weights are available [here](https://www.dropbox.com/scl/fo/ealblchspq427pfmhtg7h/ALZ7AE5o3bT0VUQ8TTeR1As?rlkey=1e6tseyuwd04rbj7wmn2n6ij7&e=3&st=ybsvv0ry&dl=0), under the `AutoCellLabeler/model_weights` directory.

### Post-processing

Please see the [AutoCellLabeler package](https://github.com/flavell-lab/AutoCellLabeler) for post-processing the network outputs.

## SegmentationNet

The segmentation network used for instance segmentation of neuronal ROIs as part of the [original ANTSUN pipeline](https://www.sciencedirect.com/science/article/pii/S0092867423008504?via%3Dihub) and the [ANTSUN 2.0 pipeline, AutoCellLabeler, and ANTSUN 2U](https://www.biorxiv.org/content/10.1101/2024.07.18.601886v1) is implemented via this package.

### Parameter files

This package contains the parameter files used for [training](SegmentationNet/instance_segmentation_train.yaml) and [evaluating](SegmentationNet/instance_segmentation_test.yaml) the SegmentationNet.

### Training and validation data

To download datasets used to train and validate the SegmentationNet, please see [our Dropbox](https://www.dropbox.com/scl/fo/ealblchspq427pfmhtg7h/ALZ7AE5o3bT0VUQ8TTeR1As?rlkey=1e6tseyuwd04rbj7wmn2n6ij7&e=4&st=ybsvv0ry&dl=0) under the `SegmentationNet` directory. The `hdf5_train` and `hdf5_val` subdirectories are training and validation datasets ready to use for network training. Alternatively, some of the raw labels are available at `img_binned_uncropped` and `label_binned_uncropped`. For instructions on using the raw labels and converting them into network-compatible `h5` labels, please see the [training notebook](SegmentationNet/SegmentationNet_training_data.ipynb).

## Setup

### Prerequisites
- Linux
- NVIDIA GPU
- CUDA CuDNN

### Running on Windows
The package has not been tested on Windows, however some reported using it on Windows. One thing to keep in mind:
when training with `CrossEntropyLoss`: the label type in the config file should be change from `long` to `int64`,
otherwise there will be an error: `RuntimeError: Expected object of scalar type Long but got scalar type Int for argument #2 'target'`.

### Installation
The easiest way to install `pytorch-3dunet` package is via `pip`:
```
pip install torch torchvision torchaudio matplotlib nd2reader hdbscan tensorboard tensorboardX h5py simpleitk pyyam
git clone git@github.com:flavell-lab/pytorch-3dunet
cd pytorch-3dunet
pip install .
```

This has been tested to work with Python 3.10.14 and PyTorch version `2.3.1+cu121`.

### Installation tips
Make sure that the installed `pytorch` is compatible with your CUDA version, otherwise the training/prediction will fail to run on GPU. You can re-install `pytorch` compatible with your CUDA by:

```
conda install -c pytorch torchvision cudatoolkit=<YOU_CUDA_VERSION> pytorch
```

## Train
One can train the network by simply invoking:
```
python pytorch3dunet/train.py --config <CONFIG>
```
where `CONFIG` is the path to a YAML configuration file, which specifies all aspects of the training procedure. See e.g. [train_config_ce.yaml](resources/train_config_ce.yaml) which describes how to train a standard 3D U-Net on a randomly generated 3D volume and random segmentation mask ([random_label3D.h5](resources/random3D.h5)) with cross-entropy loss (just a demo). Configuration files for specific networks like the SegmentationNet and AutoCellLabeler are also given above.

In order to train on your own data just edit the `checkpoint_dir` to the directory where you want to save model checkpoints, the `h5_dir` to the directory where you want to save individual training images (most useful for double-checking that the augmented images look reasonable - this is NOT where you put the training/validation/testing data), and the corresponding instances of `file_paths` to point to the directories containing your training and validation HDF5 files. These HDF5 files should contain the raw/label data sets in the following axis order: `DHW` (in case of 3D) `CDHW` (in case of 4D). They are expected to have `raw` and `label` keys containing raw and labeled data, and for certain loss functions the HDF5 files should contain a `weight` key containing data for how much each pixel should be weighted in the loss function.

One can monitor the training progress with Tensorboard `tensorboard --logdir <checkpoint_dir>/logs/` (you need `tensorflow` installed in your conda env), where `checkpoint_dir` is the path to the checkpoint directory specified in the config.

## Prediction
Given that `pytorch-3dunet` package was installed via conda as described above, one can run the prediction via:
```
python pytorch3dunet/predict.py --config <CONFIG>
```

Here `CONFIG` is the path to a YAML configuration file that specifies the network architecture. Configuration files are specific to the network that was trained, so if you modify the training configuration file you will also need to modify the evaluation configuration file as appropriate. You will also need to modify `model_path` to point to the model checkpoint to use, and `file_paths` to point to the path to the HDF5 files to predict on. These files only need to have the `raw` key.


### Prediction tips
In order to avoid checkerboard artifacts in the output prediction masks the patch predictions are averaged, so make sure that `patch/stride` params lead to overlapping blocks, e.g. `patch: [64 128 128] stride: [32 96 96]` will give you a 'halo' of 32 voxels in each direction.

## Supported Loss Functions

### Semantic Segmentation
- _BCEWithLogitsLoss_ (binary cross-entropy)
- _DiceLoss_ (standard `DiceLoss` defined as `1 - DiceCoefficient` used for binary semantic segmentation; when more than 2 classes are present in the ground truth, it computes the `DiceLoss` per channel and averages the values).
- _BCEDiceLoss_ (Linear combination of BCE and Dice losses, i.e. `alpha * BCE + beta * Dice`, `alpha, beta` can be specified in the `loss` section of the config)
- _CrossEntropyLoss_ (one can specify class weights via `weight: [w_1, ..., w_k]` in the `loss` section of the config)
- _PixelWiseCrossEntropyLoss_ (one can specify not only class weights but also per pixel weights in order to give more gradient to important or under-represented regions in the ground truth)
- _WeightedCrossEntropyLoss_ (see 'Weighted cross-entropy (WCE)' in the below paper for a detailed explanation; one can specify class weights via `weight: [w_1, ..., w_k]` in the `loss` section of the config)
- _GeneralizedDiceLoss_ (see 'Generalized Dice Loss (GDL)' in the below paper for a detailed explanation; one can specify class weights via `weight: [w_1, ..., w_k]` in the `loss` section of the config). 
Note: use this loss function only if the labels in the training dataset are very imbalanced e.g. one class having at least 3 orders of magnitude more voxels than the others. Otherwise use standard _DiceLoss_.


For a detailed explanation of some of the supported loss functions see:
[Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations](https://arxiv.org/pdf/1707.03237.pdf)
Carole H. Sudre, Wenqi Li, Tom Vercauteren, Sebastien Ourselin, M. Jorge Cardoso

### Regression
- _MSELoss_
- _L1Loss_
- _SmoothL1Loss_
- _WeightedSmoothL1Loss_ - extension of the _SmoothL1Loss_ which allows to weight the voxel values above (below) a given threshold differently
- _PSNR_ - peak signal to noise ratio


## Supported Evaluation Metrics

### Semantic Segmentation
- _MeanIoU_ - Mean intersection over union
- _PixelWiseMeanIoU_ - Mean intersection over union, but weighted per-pixel.
- _DiceCoefficient_ - Dice Coefficient (computes per channel Dice Coefficient and returns the average)
If a 3D U-Net was trained to predict cell boundaries, one can use the following semantic instance segmentation metrics
(the metrics below are computed by running connected components on thresholded boundary map and comparing the resulted instances to the ground truth instance segmentation): 
- _BoundaryAveragePrecision_ - Average Precision applied to the boundary probability maps: thresholds the boundary maps given by the network, runs connected components to get the segmentation and computes AP between the resulting segmentation and the ground truth
- _AdaptedRandError_ - Adapted Rand Error (see http://brainiac2.mit.edu/SNEMI3D/evaluation for a detailed explanation)

If not specified `MeanIoU` will be used by default.




## 2D U-Net
Training the standard 2D U-Net is also possible, see [train_config_2d](resources/train_config_2d_boundary.yml) for example configuration. Just make sure to keep the singleton z-dimension in your H5 dataset (i.e. `(1, Y, X)` instead of `(Y, X)`) , cause data loading / data augmentation requires tensors of rank 3 always.


## Data Parallelism
By default, if multiple GPUs are available training/prediction will be run on all the GPUs using [DataParallel](https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html).
If training/prediction on all available GPUs is not desirable, restrict the number of GPUs using `CUDA_VISIBLE_DEVICES`, e.g.
```bash
CUDA_VISIBLE_DEVICES=0,1 train3dunet --config <CONFIG>
``` 
or
```bash
CUDA_VISIBLE_DEVICES=0,1 predict3dunet --config <CONFIG>
```

## Sample configuration files

### AutoCellLabeler

* [train full AutoCellLabeler network](AutoCellLabeler_parameters/train_paper_all.yaml) / [predict using full AutoCellLabeler network](AutoCellLabeler_parameters/test_paper_all.yaml)

### SegmentationNet
* [train SegmentationNet](SegmentationNet/instance_segmentation_train.yaml) / [predict using SegmentationNet](SegmentationNet/instance_segmentation_test.yaml)

### Semantic segmentation
* [train with cross-entropy loss](resources/train_config_ce.yaml) / [predict using the network trained with cross-entropy loss](resources/test_config_ce.yaml)
* [train with Dice loss](resources/train_config_dice.yaml) / [predict using the network trained with Dice loss](resources/test_config_dice.yaml)
* [train using 4D input](resources/train_config_4d_input.yaml) / [predict on the 4D input](resources/test_config_4d_input.yaml)
* [train to predict cell boundaries from the confocal microscope](resources/train_config_boundary.yaml) / [predict using the network on the boundary classification task](resources/test_config_boundary.yaml)

### Regression
* [train on a random noise sample](resources/train_config_regression.yaml) / [predict using the network trained on a regression problem](resources/test_config_regression.yaml)

### 2D (semantic segmentation)
* [train to predict cell boundaries in 2D](resources/train_config_2d_boundary.yml) / [predict cell boundaries in 2D](resources/test_config_2d_boundary.yml)

## Contribute
If you want to contribute back, please make a pull request.

## Cite
If you use this code for your research, please cite the following papers:
```
@article {Wolny2020.01.17.910562,
	author = {Wolny, Adrian and Cerrone, Lorenzo and Vijayan, Athul and Tofanelli, Rachele and Barro,
              Amaya Vilches and Louveaux, Marion and Wenzl, Christian and Steigleder, Susanne and Pape, 
              Constantin and Bailoni, Alberto and Duran-Nebreda, Salva and Bassel, George and Lohmann,
              Jan U. and Hamprecht, Fred A. and Schneitz, Kay and Maizel, Alexis and Kreshuk, Anna},
	title = {Accurate And Versatile 3D Segmentation Of Plant Tissues At Cellular Resolution},
	elocation-id = {2020.01.17.910562},
	year = {2020},
	doi = {10.1101/2020.01.17.910562},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2020/01/18/2020.01.17.910562}, 
	eprint = {https://www.biorxiv.org/content/early/2020/01/18/2020.01.17.910562.full.pdf},
	journal = {bioRxiv}
}
```

```
@article{ATANAS20234134,
title = {Brain-wide representations of behavior spanning multiple timescales and states in C. elegans},
journal = {Cell},
volume = {186},
number = {19},
pages = {4134-4151.e31},
year = {2023},
issn = {0092-8674},
doi = {https://doi.org/10.1016/j.cell.2023.07.035},
url = {https://www.sciencedirect.com/science/article/pii/S0092867423008504},
author = {Adam A. Atanas and Jungsoo Kim and Ziyu Wang and Eric Bueno and McCoy Becker and Di Kang and Jungyeon Park and Talya S. Kramer and Flossie K. Wan and Saba Baskoylu and Ugur Dag and Elpiniki Kalogeropoulou and Matthew A. Gomes and Cassi Estrem and Netta Cohen and Vikash K. Mansinghka and Steven W. Flavell},
keywords = {brain-wide activity, behavior, cell atlas, , neural circuits, internal states},
abstract = {Summary
Changes in an animal’s behavior and internal state are accompanied by widespread changes in activity across its brain. However, how neurons across the brain encode behavior and how this is impacted by state is poorly understood. We recorded brain-wide activity and the diverse motor programs of freely moving C. elegans and built probabilistic models that explain how each neuron encodes quantitative behavioral features. By determining the identities of the recorded neurons, we created an atlas of how the defined neuron classes in the C. elegans connectome encode behavior. Many neuron classes have conjunctive representations of multiple behaviors. Moreover, although many neurons encode current motor actions, others integrate recent actions. Changes in behavioral state are accompanied by widespread changes in how neurons encode behavior, and we identify these flexible nodes in the connectome. Our results provide a global map of how the cell types across an animal’s brain encode its behavior.}
}
```

```
@article {Atanas2024.07.18.601886,
	author = {Atanas, Adam A. and Lu, Alicia Kun-Yang and Kim, Jungsoo and Baskoylu, Saba and Kang, Di and Kramer, Talya S. and Bueno, Eric and Wan, Flossie K. and Flavell, Steven W.},
	title = {Deep Neural Networks to Register and Annotate the Cells of the C. elegans Nervous System},
	elocation-id = {2024.07.18.601886},
	year = {2024},
	doi = {10.1101/2024.07.18.601886},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {Aligning and annotating the heterogeneous cell types that make up complex cellular tissues remains a major challenge in the analysis of biomedical imaging data. Here, we present a series of deep neural networks that allow for automatic non-rigid registration and cell identification in the context of the nervous system of freely-moving C. elegans. A semi-supervised learning approach was used to train a C. elegans registration network (BrainAlignNet) that aligns pairs of images of the bending C. elegans head with single pixel-level accuracy. When incorporated into an image analysis pipeline, this network can link neuronal identities over time with 99.6\% accuracy. A separate network (AutoCellLabeler) was trained to annotate \&gt;100 neuronal cell types in the C. elegans head based on multi-spectral fluorescence of genetic markers. This network labels \&gt;100 different cell types per animal with 98\% accuracy, exceeding individual human labeler performance by aggregating knowledge across manually labeled datasets. Finally, we trained a third network (CellDiscoveryNet) to perform unsupervised discovery and labeling of \&gt;100 cell types in the C. elegans nervous system by analyzing unlabeled multi-spectral imaging data from many animals. The performance of CellDiscoveryNet matched that of trained human labelers. These tools will be useful for a wide range of applications in C. elegans research and should be straightforward to generalize to many other applications requiring alignment and annotation of dense heterogeneous cell types in complex tissues.Competing Interest StatementThe authors have declared no competing interest.},
	URL = {https://www.biorxiv.org/content/early/2024/07/22/2024.07.18.601886},
	eprint = {https://www.biorxiv.org/content/early/2024/07/22/2024.07.18.601886.full.pdf},
	journal = {bioRxiv}
}
```



