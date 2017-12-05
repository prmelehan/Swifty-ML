# Swifty-ML
### Train and convert Keras models to Core ML models easily

## Features

* Train a CNN with only images
* Convert pre-trained keras models to Core ML models
* Train and Convert a CNN into a Core ML Model
* Edit attributes of existing Core ML Models

## Why does this exist?

Swifty-ML allows developers with no experience in ML to create classifiers and improve their apps. It's also good for playing around!



## Requirements

* [Keras](https://keras.io/#installation)
* [coremltools](https://github.com/apple/coremltools#installation)
* Python 3.6

## Usage

To train a model using Swifty-ML, run the following command in terminal.
```
python modeler.py train path/to/data inceptionv3
```

This usually takes a very long time, even with a good laptop, but once finished, there will be two files. A `Core ML Model` and a `Keras .h5` model. Saved in `exported_models/coreml` and `exported_models/keras` respectively. The system also saves the state of the model during training after each epoch.

To edit a coreml model...
```
python modeler.py edit path/to/coreml/model.mlmodel
```

To convert a Keras model (.h5) to CoreML format

```
python modeler.py convert-pre-trained path/to/keras/model.h5
```

## Data Directory Format

The format for training and validation images must be the following

```
.
├── training
│   ├── class1
│   ├── class2
│   └── class3
└── validation
    ├── class1
    ├── class2
    └── class3
```

If I was training a network on images of the different flags, I would have a structure like so...

```
.
├── training
│   ├── belgium
│   ├── canada
│   └── unites_states
└── validation
    ├── belgium
    ├── canada
    └── unites_states
```

Note: `training` and `validation` cannot change, but the sub folders' names can, and will be the class labels for the images inside them.

## Installing Core ML Tools on Python 3.6

As of now, Core ML Tools does not provide support for Python 3.6 via `pip`. Instead, you will need to download the latest version from their GitHub and install it manually.

https://github.com/apple/coremltools

After downloading the project, `cd` into the project and run
```
python setup.py install
```

## Supported models

As of now, only a few model architectures are supported. In the future, all models supported by both `coremltools` and `keras` will be supported.

Future models include

- [ ] Xception
- [x] VGG16
- [x] VGG19
- [ ] ResNet50
- [x] InceptionV3
- [ ] InceptionResNetV2
- [x] MobileNet
