
# credit https://github.com/DeepLearningSandbox/DeepLearningSandbox/blob/master/transfer_learning/fine-tune.py

# importing our dependencies
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model, load_model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping, Callback
from keras.optimizers import SGD
import coremltools
import glob
import os
import sys
import argparse
import ntpath
import datetime


# helpers
# https://stackoverflow.com/questions/8384737/extract-file-name-from-path-no-matter-what-the-os-path-format
def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def get_nb_files(directory):
  """Get number of files by searching directory recursively"""
  if not os.path.exists(directory):
    return 0
  cnt = 0
  for r, dirs, files in os.walk(directory):
    for dr in dirs:
      cnt += len(glob.glob(os.path.join(r, dr + "/*")))
  return cnt

def convert_pre_trained(args):
    model_path, output_name = args.model_path, args.output_name
    # we need to get the model file, and convert it using the coremltools
    if(os.path.exists(model_path)):
        # grab the file and convert it
        model = load_model(model_path)
        # convert the model to coreML
        coreml_model = coremltools.converters.keras.convert(model)
        # save the model somewhere and give the path
        coreml_model.save(output_name + ".mlmodel")
        print("Converted model saved at: " + os.getcwd() + "/" + output_name + ".mlmodel")
    else:
        # otherwise lets say that there was an error
        print("Oops. There is no model at '" + model_path + "'. Did you make a typo?")


def edit_pre_trained(args):
    model_path = args.model_path
    # get the model file and make an MLModel instance to edit
    if(os.path.exists(model_path)):
        # grab the file and convert it
        print("Loading model... (This may take a while)")
        model = coremltools.models.MLModel(model_path)
        author = input("Author: ")
        license = input("License: ")
        description = input("Description: ")
        # update those values
        model.author = author
        model.license = license
        model.short_description = description
        print("Saving model...")
        model.save(os.getcwd() + "/exported_models/coreml/" + path_leaf(model_path))
        print("Converted model saved at: " + os.getcwd() + "/exported_models/coreml/" + path_leaf(model_path))
    else:
        # otherwise lets say that there was an error
        print("Oops. There is no model at '" + model_path + "'. Did you make a typo?")

def train_and_convert(args):
    image_directory, architecture = args.image_directory, args.architecture

    architecture = architecture.lower()

    epochs = 3
    validation_steps = 50
    steps_per_epoch = 256

    ## TESTING
    # epochs = 1
    # validation_steps = 2
    # steps_per_epoch = 5

    # make sure the training and validation directories exist
    if(os.path.exists(image_directory + "/training") == False):
        # die right now
        sys.exit("Invalid data directory configuration")
    if(os.path.exists(image_directory + "/validation") == False):
        # die right now
        sys.exit("Invalid data directory configuration")


    # check to see what architecture we're using
    if(architecture == "inceptionv3"):

        image_width, image_height = 299, 299
        training_samples, validation_samples = get_nb_files(image_directory + "/training"), get_nb_files(image_directory + "/validation")
        classes = len(glob.glob(image_directory + "/training/*"))
        # perform data augmentation on the incoming images
        training_data_generator = ImageDataGenerator(
            preprocessing_function=applications.inception_v3.preprocess_input,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )

        validation_data_generator = ImageDataGenerator(
            preprocessing_function=applications.inception_v3.preprocess_input,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )

        training_generator = training_data_generator.flow_from_directory(
            image_directory + "/training",
            target_size=(image_height, image_width),
            batch_size=32
        )

        validation_generator = validation_data_generator.flow_from_directory(
            image_directory + "/validation",
            target_size=(image_height, image_width),
            batch_size=32
        )

        # get the class names so that core ml can use them later on
        class_names = list(training_generator.class_indices.keys())

        model = applications.inception_v3.InceptionV3(weights = 'imagenet', include_top=False, input_shape=(image_width, image_height, 3))
        print("")
        print("Training model using InceptionV3... (This will take a while)")
        print("")
        x = model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x) #new FC layer, random init
        predictions = Dense(classes, activation='softmax')(x) #new softmax layer
        model = Model(inputs=model.input, outputs=predictions)
        # freeze the layers
        for layer in model.layers:
            layer.trainable = False

        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        for layer in model.layers[:172]:
            layer.trainable = False
        for layer in model.layers[172:]:
            layer.trainable = True
        model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])


        checkpoint = ModelCheckpoint("./saved_states/inceptionv3_1.h5", monitor="val_acc", verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
        early = EarlyStopping(monitor="val_acc", min_delta=0, patience=10, verbose=1, mode='auto')

        model.fit_generator(
            training_generator,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=validation_generator,
            validation_steps=validation_steps,
            class_weight="auto",
            callbacks=[checkpoint, early]
        )

        # save the model weights for later use in Keras
        date = datetime.datetime.now()
        print("Saving model to: " + os.getcwd() + "/exported_models/keras/" + str(date) + ".h5")
        model.save(os.getcwd() + "/exported_models/keras/" + str(date) + ".h5")
        # convert the model now to coreml
        print("Coverting model to CoreML model...")
        coreml_model = coremltools.converters.keras.convert(
            model,
            input_names='image',
            output_names='classProbabilities',
            image_input_names='image',
            class_labels=class_names
        )
        print("Saving CoreML model to: " + os.getcwd() + "/exported_models/coreml/" + str(date) + ".mlmodel")
        coreml_model.save(os.getcwd() + "/exported_models/coreml/" + str(date) + ".mlmodel")

    # elif(architecture == "vgg19"):
    #     image_width, image_height = 256, 256
    #     training_samples, validation_samples = get_nb_files(image_directory + "/training"), get_nb_files(image_directory + "/validation")
    #     classes = len(glob.glob(image_directory + "/training/*"))
    #
    #
    #     # perform data augmentation on the incoming images
    #     training_data_generator = ImageDataGenerator(
    #         preprocessing_function=applications.vgg19.preprocess_input,
    #         rotation_range=30,
    #         width_shift_range=0.2,
    #         height_shift_range=0.2,
    #         shear_range=0.2,
    #         zoom_range=0.2,
    #         horizontal_flip=True
    #     )
    #
    #     validation_data_generator = ImageDataGenerator(
    #         preprocessing_function=applications.vgg19.preprocess_input,
    #         rotation_range=30,
    #         width_shift_range=0.2,
    #         height_shift_range=0.2,
    #         shear_range=0.2,
    #         zoom_range=0.2,
    #         horizontal_flip=True
    #     )
    #
    #     training_generator = training_data_generator.flow_from_directory(
    #         image_directory + "/training",
    #         target_size=(image_height, image_width),
    #         batch_size=32
    #     )
    #
    #     validation_generator = validation_data_generator.flow_from_directory(
    #         image_directory + "/validation",
    #         target_size=(image_height, image_width),
    #         batch_size=32
    #     )



    else:
        sys.exit(architecture + " is not supported.")




if __name__ == '__main__':
    # here we will get our command line arguments
    parser = argparse.ArgumentParser(description="Training and Converting CNNs to Core ML Models easily")
    subparsers = parser.add_subparsers()

    convert_trained_parser = subparsers.add_parser("convert-pre-trained")
    convert_trained_parser.add_argument('model_path', type=str, help='Pre-trained model path')
    convert_trained_parser.add_argument('output_name', type=str, help='The name of the converted model. i.e. flowers, or animals...')
    convert_trained_parser.set_defaults(func=convert_pre_trained)

    edit_parser = subparsers.add_parser("edit")
    edit_parser.add_argument('model_path', type=str, help='Path to CoreML model')
    edit_parser.set_defaults(func=edit_pre_trained)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument('image_directory', type=str, help="Directory that contains 1 training directory, and 1 validation directory")
    train_parser.add_argument('architecture', type=str, help="Type of pre-trained network to use (VGG19, )")
    train_parser.set_defaults(func=train_and_convert)

    args = parser.parse_args()
    args.func(args)
