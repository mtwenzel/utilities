from tensorflow.keras.preprocessing.image import ImageDataGenerator

def provide_PPMI_dataset(paths_dict={}, target_size=(109,91), batch_size=32):

    def get_data_generators(paths_dict, batch_size, target_size):
        '''
        Returns generators for batches of train/val/test data collected from the folders given to the function. 
    
        Params: 
            paths_dict: Dictionary with three paths in 'train', 'val', 'test' pointing to folders with images to use. Subfolders are assumed to contain classes. Mandatory.
            batch_size: Batch size. Default=32
            target_size: 2D size tuple to rescale the images to. Default=(109,91)
        Returns:
            train_generator, val_generator, test_generator: Generator functions to retrieve a batch of data
        '''
    train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, dtype=tf.float32)
    test_datagen =  ImageDataGenerator(rescale=1./255, dtype=tf.float32)
    train_generator = train_datagen.flow_from_directory(paths_dict['train'], 
                                                 target_size=target_size,
                                                 color_mode='grayscale',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)

    val_generator = train_datagen.flow_from_directory(paths_dict['val'],
                                                 target_size=target_size,
                                                 color_mode='grayscale',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)

    test_generator = test_datagen.flow_from_directory(paths_dict['test'], 
                                                 target_size=target_size,
                                                 color_mode='grayscale',
                                                 batch_size=566,
                                                 class_mode='categorical',
                                                 shuffle=False) # don't shuffle so that the file names and indices are in sync.
    

    return train_generator, val_generator, test_generator

    from urllib.request import urlopen
    from zipfile import ZipFile

    zipurl = 'https://github.com/mtwenzel/parkinson-classification/raw/master/data/PPMI-classification.zip'
    zipresp = urlopen(zipurl)
    tempzip = open("PPMI-classification.zip", "wb")
    tempzip.write(zipresp.read())
    tempzip.close()
    print("download complete, extracting...")

    zf = ZipFile("PPMI-classification.zip")
    zf.extractall(path = 'data/')
    zf.close()
    print("... done")

    if paths_dict == {}:
        paths_dict = {'train': '/content/data/PPMI-classification/all_2d_train',
                    'val': '/content/data/PPMI-classification/all_2d_val',
                    'test': '/content/data/PPMI-classification/all_2d_val'}

    train_generator, val_generator, test_generator = get_data_generators(paths_dict, target_size=target_size)
    
    return train_generator, val_generator, test_generator