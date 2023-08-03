import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from skimage import exposure
from keras.models import Sequential
from keras.layers import Cropping2D
from keras.layers.core import Lambda, Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.utils.visualize_util import plot as plot_model
import os.path

#
# hyper parameters to tune
#
myCLIP_LIMIT = 0.5 # for clahe transform
myEPOCHS = 100
myTEST_SIZE = 0.1
STEERING_ANGLE_CORRECTION = 0.2 


#
# directories for training data, which must contain:
# - driving_log.csv
# - IMG directory with images.
#
FILES_TOP_DIRS = ['./data/data_track_2_left','./data/data_track_2_right',
                  './data/data_track_2_left_hug_left','./data/data_track_2_left_hug_right',
                  './data/data_track_1_left','./data/data_track_1_right',
                  './data/data_track_1_left_hug_left','./data/data_track_1_left_hug_right'
                  ]

# hugging drive style --> correct it to center
HUG_CORRECTION = {}
HUG_CORRECTION['./data/data_track_2_left_hug_left']  =  0.2         
HUG_CORRECTION['./data/data_track_2_right_hug_right'] = -0.2 
HUG_CORRECTION['./data/data_track_1_left_hug_left']  =  0.2         
HUG_CORRECTION['./data/data_track_1_right_hug_right'] = -0.2  

def read_log_files(top_dirs):
    """Reads all the log files, and returns the lines after shuffling.
    """
    lines = []
    for FILES_TOP_DIR in top_dirs:
        log_file_name = FILES_TOP_DIR+'/driving_log.csv'
        with open(log_file_name) as csvfile:
            reader = csv.reader(csvfile)
            skipped_header = False
            for line in reader:
                if skipped_header:
                    line.insert(0,FILES_TOP_DIR) # insert top directory as first token
                    lines.append(line) 
                else:
                    skipped_header=True    # skip header line
    return sklearn.utils.shuffle(lines)

def plot_image(image, filename=None, txt='---'):
    """Plots an image with description

    | txt | image |

    If filename is specified, it will write the plot to file, else to screen.

    The image can be either grayscale or RGB.
    """    
    nrows       = 2  # always need 2 rows at minimum for indexing into axes...
    ncols       = 2            
    axes_width  = 6            
    axes_height = 6            
    width       = ncols * axes_width    
    height      = nrows * axes_height  
    fontsize    = 15 
    fig, axes   = plt.subplots(nrows, ncols, figsize = (width, height) )

    # turn off:
    #  - all tick marks and tick labels
    #  - frame of each axes
    for row in range(nrows):
        for ncol in range(ncols):
            axes[row,ncol].xaxis.set_visible(False)
            axes[row,ncol].yaxis.set_visible(False)
            axes[row,ncol].set_frame_on(False)

    row=0
    axes[row, 0].text(0.0, 0.25, 
                      (txt),
                      fontsize=fontsize)    

    if image.ndim == 3 and image.shape[2] == 3:
        axes[row,1].imshow(image)
    else:
        axes[row,1].imshow(image.squeeze(), cmap='gray')

    if filename == None:      
        plt.show()  
    else:  
        fig.savefig(filename)
        print ('Written the file: '+ filename)

    plt.close(fig)

def plot_convergence(history_object, filename=None, txt='---'):
    """Plots the convergence with description

    | txt | plot |

    If filename is specified, it will write the plot to file, else to screen.

    """    
    nrows       = 2  # always need 2 rows at minimum for indexing into axes...
    ncols       = 2            
    axes_width  = 6            
    axes_height = 6            
    width       = ncols * axes_width    
    height      = nrows * axes_height  
    fontsize    = 15 
    fig, axes   = plt.subplots(nrows, ncols, figsize = (width, height) )

    # turn off:
    #  - all tick marks and tick labels
    #  - frame of each axes
    for row in range(nrows):
        for ncol in range(ncols):
            # except for right upper corner
            if row==0 and ncol==1:
                break
            axes[row,ncol].xaxis.set_visible(False)
            axes[row,ncol].yaxis.set_visible(False)
            axes[row,ncol].set_frame_on(False)

    row=0
    axes[row, 0].text(0.0, 0.25, 
                      (txt),
                      fontsize=fontsize)    

    axes[row, 1].plot(history_object.history['loss'])
    axes[row, 1].plot(history_object.history['val_loss'])
    axes[row, 1].set_title('model mean squared error loss')
    axes[row, 1].set_ylabel('mean squared error loss')
    axes[row, 1].set_xlabel('epoch')
    axes[row, 1].legend(['training set', 'validation set'], loc='upper right')     

    if filename == None:      
        plt.show()  
    else:  
        fig.savefig(filename)
        print ('Written the file: '+ filename)

    plt.close(fig)    

def save_augmented_images_to_disk(save_to_dir, save_prefix, save_format, X_batch, y_batch,
                                  txt='---'):
    """Save images in a batch to disk, for debug purposes""" 
    for image in X_batch:
        fname = '{dir}/{prefix}_{hash}.{format}'.format(dir=save_to_dir,
                                                        prefix=save_prefix,
                                                         hash=np.random.randint(1e4),
                                                         format=save_format)
        plot_image(image, filename=fname, txt=txt)  

def rgb_to_grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')

    input: image"""
    x = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # shape (160, 320)
    return x.reshape(x.shape[0], x.shape[1], 1) # shape (160, 320, 1) required in rest
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   


def apply_clahe(img, clip_limit=0.01):
    """Applies a Contrast Limited Adaptive Histogram Equalization (CLAHE)
    for description:
    http://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.equalize_adapthist
    """
    x = exposure.equalize_adapthist(img.squeeze(), clip_limit=clip_limit) # shape (160, 320)
    return x.reshape(x.shape[0], x.shape[1], 1) # shape (160, 320, 1) required in rest


def prep_for_yield(save_to_dir, save_prefix, save_format, images, angles):
    # yield the batch as numpy arrays
    X_batch = np.array(images)
    y_batch = np.array(angles)
    # write it to disk for debug purposes if requested
    if save_prefix:
        save_augmented_images_to_disk(save_to_dir, save_prefix, save_format,
                                      X_batch, y_batch,'Image yielded from generator')
    # return shuffled -> the calling function will yield it
    return sklearn.utils.shuffle(X_batch, y_batch)

def generator(lines, batch_size=32,
              flip_horizontal=False,
              grayscale=False,
              clahe=False,
              save_to_dir=None,
              save_prefix=None,
              save_format='jpeg'):
    """
    Yields batches of augmented images & steering angles:

    -> for images of 'hugging' style, the steering angle is adjusted with: HUG_CORRECTION
    -> for left and right images, the steering angle is further adjusted with STEERING_ANGLE_CORRECTION
    -> each image is also flipped
    ==> so, each line in the log files results in 6 images !

    -> if requested, each image is grayscaled & a clahe transform is applied

    -> the augmented data is yielded in batches.


    inputs:
    - lines: the content of the log file created during data acquisition
    - batch_size: the size of batches to be yielded
    - flip_horizontal: True -> each image (center, left, right) will be also flipped
    - grayscale: True -> each image is grayscaled as well
    - clahe: True -> a clahe transform is applied to each image
    - save_to_dir: If specified, triggers writing of generated images to this folder.
    This is handy for debugging.
    - save_prefix: prefix for image files.
    - save_format: format of image files.

    returns: 
    X_batch, y_batch: lists of batch_size images & steering angles
    """
    assert(len(lines)>0)

    num_samples = len(lines) * 6  # center, left, right, and each one flipped

    while True: # Loop forever so the generator never terminates
        images = []
        angles = []    
        count = 0   
        total_count = 0
        fnames = ['']*3
        for line in lines:
            FILES_TOP_DIR = line[0].strip()
            fnames[0]  = FILES_TOP_DIR + '/' + line[1].strip() # center

            if line[2].strip() == 'EMPTY':
                fnames[1] = 'EMPTY'
            else:
                fnames[1]  = FILES_TOP_DIR + '/' + line[2].strip() # left

            if line[3].strip() == 'EMPTY':
                fnames[2] = 'EMPTY'
            else:
                fnames[2]  = FILES_TOP_DIR + '/' + line[3].strip() # left                

            if FILES_TOP_DIR in HUG_CORRECTION:
                hug_correction = HUG_CORRECTION[FILES_TOP_DIR]
            else:
                hug_correction = 0.0

            steering  = float(line[4]) + hug_correction

            for f_index in range(3):
                f = fnames[f_index]

                if f_index == 0:    # center
                    angle = steering
                elif f == 'EMPTY':
                    # no left or right image provided. Just duplicate the center image.
                    angle = steering 
                    f = fnames[0]
                elif f_index == 1:  # left
                    angle = steering + STEERING_ANGLE_CORRECTION
                else:               # right
                    angle = steering - STEERING_ANGLE_CORRECTION

                if os.path.isfile(f):
                    image = cv2.imread(f)

                    # apply image manipulations
                    if grayscale:
                        image = rgb_to_grayscale(image)

                    if clahe:
                        image = apply_clahe(image, clip_limit=myCLIP_LIMIT)


                    images.append(image)
                    angles.append(angle)
                    count += 1
                    total_count += 1
                    if count == batch_size or total_count == num_samples:
                        yield prep_for_yield(save_to_dir, save_prefix, save_format, 
                                             images, angles)

                        # reset for next batch
                        images, angles, count = [], [], 0

                    if flip_horizontal:          
                        # augment image by horizontal flip
                        x = cv2.flip(image,1) # returns (160, 320)
                        image_flipped = x.reshape(x.shape[0], x.shape[1], 1) # (160, 320, 1)

                        images.append(image_flipped)
                        angles.append(angle*-1.0)
                        count += 1 
                        total_count += 1
                        if count == batch_size or total_count == num_samples:
                            yield prep_for_yield(save_to_dir, save_prefix, save_format, 
                                                 images, angles)

                            # reset for next batch
                            images, angles, count = [], [], 0

                else:
                    msg = 'Image file does not exist: '+str(f)
                    raise Exception(msg)

if __name__ == '__main__':
    # ======================================================================================
    # define Keras network

    model = Sequential()

    #
    # Some pre-processing as part of Keras model
    #
    # crop:
    # 70 rows pixels from the top of the image
    # 25 rows pixels from the bottom of the image
    # 0 columns of pixels from the left of the image
    # 0 columns of pixels from the right of the image
    #rgb 
    #model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3), name='layer_crop'))
    #grayscale
    model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,1), name='layer_crop'))
    #
    # normalize and mean-center the data
    #without clahe:
    #model.add(Lambda(lambda x: x / 255.0 - 0.5, name='layer_normalize'))
    #with clahe
    model.add(Lambda(lambda x: x / 1.0 - 0.5, name='layer_normalize'))

    # Nvidia CNN 
    # - https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
    # - added a dropout layer to avoid over-fitting
    model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Dropout(0.5, noise_shape=None, seed=None)) # ab: attempt to avoid overfitting
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')

    # plot a graph of the model
    #
    # note: this required pydot_ng (or pydot), and Graphviz software on system path.
    #       --> had to update the carnd-term1 environment with:
    #       % python -m pip install pydot_ng 
    #
    print ("Writing graph of CNN to model.gif")
    plot_model(model, to_file='model.gif', show_shapes=True, show_layer_names=True)

    # ======================================================================================
    # Trial of the generator & model --> Write images to disk for debug purposes

    print ('Trial of the data generator and model to preview images...')

    trial_images_top_dir = 'data/data_failure-track2-a/'
    lines_trial = read_log_files([trial_images_top_dir])

    trial_generator  = generator(lines_trial, batch_size=1,
                                 flip_horizontal=False,
                                grayscale=True,
                                clahe=True,
                                save_to_dir=trial_images_top_dir+'/preview',
                                save_prefix='0_grayscale_and_clahe',
                                save_format='jpeg')

    # run the model using the trial_generator.
    i = 0
    for batch in trial_generator:  # Note that upon yield, images are written to disk
        i += 1

        # In addition, get images from image processing layers of model
        # see: https://keras.io/getting-started/faq/#how-can-i-visualize-the-output-of-an-intermediate-layer

        X_batch         = batch[0]    

        # get cropped immage
        layer_name = 'layer_crop'
        layer_model = Model(input=model.input,
                            output=model.get_layer(layer_name).output)
        layer_images = layer_model.predict(X_batch)
        plot_image(layer_images[0],            # plot the first image, cropped
                   filename=trial_images_top_dir+'/preview/1_crop.jpeg',
                   txt='Cropped image')    

        if i > 0: 
            break  # otherwise the generator would loop indefinitely

    # ======================================================================================
    print ('Train & Validate the network...')

    # 
    # read log files of labeled training images
    lines = read_log_files(FILES_TOP_DIRS)

    train_samples, validation_samples = train_test_split(lines, test_size=myTEST_SIZE)

    #
    # Define training and validation generators
    # -> during validation, we do NOT generate additional images.
    #
    train_generator      = generator(train_samples, batch_size=32,
                                     flip_horizontal=True,
                                     grayscale=True,
                                     clahe=True,
                                     save_to_dir=None,
                                     save_prefix=None,
                                     save_format='jpeg')

    validation_generator = generator(validation_samples, batch_size=32,
                                     flip_horizontal=False, # do NOT generate new images during validation
                                     grayscale=True,
                                     clahe=True,
                                     save_to_dir=None,
                                     save_prefix=None,
                                     save_format='jpeg')

    # early stopping: 
    # -> stop training when loss on validation set doesn't improve for 3 epochs
    callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')]

    # train & test the model    
    history_object = model.fit_generator( train_generator, 
                                          samples_per_epoch=len(train_samples*6), # center, left, right, flipped*3
                         validation_data=validation_generator,
                         nb_val_samples=len(validation_samples),
                         nb_epoch=myEPOCHS, verbose=1,
                         callbacks = callbacks)

    # save the model
    model.save('model.h5')

    # plot the training and validation loss for each epoch
    plot_convergence(history_object, 
                     filename='convergence.jpeg', txt='Convergence during training')      
