# TensorFlow and TF-Hub modules.
from absl import logging
import tensorflow as tf
# from tensorflow_docs.vis import embed
from tensorflow import keras
from tensorflow_docs.vis import embed
logging.set_verbosity(logging.ERROR)


# Some modules to help with reading the dataset.
import random
import cv2
import imageio
import os 
from sklearn.model_selection import train_test_split
import pandas as pd


TVL1 = cv2.optflow.DualTVL1OpticalFlow_create()

import numpy as np
import os 
import pickle

# Some modules to display an animation using imageio.
# import imageio
# from IPython import display

# from urllib import request  # requires python3

from sklearn.preprocessing import OneHotEncoder


class DataLoader:

    #Dictionary where input is label and value is its One Hot Encoding
    label_to_ohe_dict = {}
    #Dictionary where input is index of the value 1 in One Hot Encoding value is corresponding label to that
    index_to_label_dict = {}
    #Labels 
    labels = set()
    
    

    def __init__(self,data_path,save_ohe_dicts = False, 
                                cache_dir_name = 'datastream_cache', 
                                save_video_paths = False, 
                                train_size = 0.9):

        #current path
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        #path to the data
        self.data_path = self.get_path(data_path)
        #Cache directory
        self.cache_directory = self.get_path(cache_dir_name)
        #train videos paths
        self.train_paths = os.path.join(self.cache_directory,'train.txt')
        #test videos paths 
        self.test_paths  = os.path.join(self.cache_directory,'test.txt')
        #excel path that will contain information about how data splitted
        self.data_split_report_path = os.path.join(self.cache_directory,'data_split')
        #create cache directory if do not exist
        DataLoader.create_cache_dir(self.cache_directory)
        #number of the classes 
        self.num_classes = int()
        #video paths dataset
        self.video_paths_ds = tf.data.Dataset.list_files(self.data_path+r"/*/*")
        # Train test ratio 
        self.train_size = train_size

        if not (os.path.exists(self.train_paths) and
                os.path.exists(self.test_paths)) or save_video_paths:
            

            # self.save_test_train_video_paths()
            DataLoader.split_and_save(self.data_path,self.cache_directory,split_report_path = self.data_split_report_path)

            print('New train test video paths saved succesfully!')
        
        else:
            print('We have presaved test train video paths!')
            
        self.label_to_ohe_path  = self.get_path(self.cache_directory+'/label_to_ohe.pkl')
        self.index_to_label_path= self.get_path(self.cache_directory+'/index_to_label.pkl')

        # If we have dictionaries assigned to the class variable or we have saved dict directory 
        # which is in this case we give load_ohe_dicts = True when we define DataLoader data type
        if  save_ohe_dicts == False and ((DataLoader.label_to_ohe_dict != {} and 
                                      DataLoader.index_to_label_dict != {})   or   (os.path.exists(self.index_to_label_path) and
                                                                                    os.path.exists(self.label_to_ohe_path))) :       
            print('We have saved dictionaries to map labels to ohes')   
            self.load_ohe_dicts()
            self.num_classes = len(DataLoader.index_to_label_dict)  
            del self.video_paths_ds

        else:   
            print('We will get new OHE dicts')
            self.fill_ohe_dicts()
            self.save_ohe_dicts()
            self.num_classes = len(DataLoader.index_to_label_dict)
            del self.video_paths_ds



    
    def get_path(self,path):
        """
        Static Method that Concatenate given path with current path

        Args:
            path: path of the file to load
            
        Return: 
            Concatenated full path of the file
        """
        
        return os.path.join(self.current_dir,path)
    
    @staticmethod
    def split_dataset(dataset,train_size = 0.80):
        """
        This function takes a dataset as input and splits it into two parts for training and testing respectively.
        The train_size parameter determines the size of the training dataset, which is set to 80% of the total number of videos by default.
        The function returns the training and testing datasets as output. It is helper function for need. If you need it within DataLoader you add it to the init function.

        Params:

        dataset: The dataset to be split into training and testing sets.
        train_size: The proportion of the dataset to be used for training.
        By default, it is set to 0.80, which means that 80% of the dataset is used for training.
        
        Returns:

        train_ds: The training dataset consisting of a portion of the input dataset.
        test_ds: The testing dataset consisting of the remaining portion of the input dataset.
        
        """
        num_of_videos = len(dataset)
        train_size = int(num_of_videos*train_size)
        train_ds = dataset.take(train_size)
        test_ds = dataset.skip(train_size)

        return train_ds,test_ds
    
    def compute_class_weights(self):
        """
        Description:
        This function computes class weights based on the distribution of samples in each class. 
        It reads a data split report from an Excel file, calculates the maximum number of samples in any class, 
        and then iterates through each label to determine the class weight.

        Parameters:
        self: The current instance of the class.

        Returns:
        class_weights: A dictionary containing the computed class weights, where the keys are the class labels and the values are the corresponding weights."""

        #initialization 
        class_weights = dict()

        #After you initialize the class you will get data_split_report. Reading it.
        data_split_df = pd.read_excel(self.data_split_report_path+'.xlsx')
        
        # Getting number of the samples in the largest class
        max_sample = max(data_split_df.train.values)

        # Calculating the class weights for each class  =>  max_sample/( number of samples in certain class ) 
        for label in list(DataLoader.label_to_ohe_dict.keys()):
            class_weights[int(np.argmax(DataLoader.label_to_ohe_dict[label]))] =  max_sample/int(data_split_df[data_split_df.labels == label].train.iloc[0])

        return class_weights
    
    @staticmethod
    def split_and_save(directory,output_dir,split_report_path,test_size):
        """
        Description:
        This static method is responsible for splitting video paths into training and test sets,
        saving the split paths into text files, and generating a split report in Excel format.
        It takes a directory path containing video files, an output directory path for saving the split files, and a split report path as input.

        Parameters:
        directory: The directory path containing the video files to be split.
        output_dir: The directory path where the split files and the split report will be saved.
        split_report_path: The file path for saving the split report in Excel format.
        test_size = number in between 0,1 that represent percentage of the data we want for validation set

        Returns:
        None
        """

        # Reading the path containing the video files to be split.
        paths = os.listdir(directory)
        video_paths= []

        # Access to the each class
        for folder in paths:
            class_path = os.path.join(directory,folder)
            # access to the each sample within class
            for video in os.listdir(class_path):
                # adding to the list 
                if video.endswith(".mp4"):
                    video_path = os.path.join(class_path,video)
                    video_paths.append(video_path)

                else:
                    # if there is none supported file report it 
                    print(f'There is unexpected format inside the folder: {folder} with the name: {video}.\n skipping this sample')

        # Getting the labels from the video paths
        labels = [video_path.split('/')[-2] for video_path in video_paths]
        # Turning the gotten data into pd.DataFrame to split and further proceed easily
        data_paths = pd.DataFrame({'paths':video_paths,'labels':labels})

        # Splitting each class seperately based on test_size
        X_train, X_val, y_train, y_val = train_test_split(data_paths.index.values, 
                                                        data_paths.labels.values, 
                                                        test_size=test_size, 
                                                        random_state=42, 
                                                        stratify= data_paths.labels.values)
        
        # New column to assign for each data sample to show is it belong to the train set or test set 
        data_paths['data_type'] = ['not_set']*data_paths.shape[0]
        
        # Assigning the data type for each sample
        data_paths.loc[X_train, 'data_type'] = 'train'
        data_paths.loc[X_val, 'data_type'] = 'test'

        # Create a report df where we show how many sample we gave to the train set and to the test set for each label 
        report_df = pd.DataFrame({'labels':[],'train':[],'test':[]})

        # Filling report DataFrame
        for label in data_paths.labels.unique():    
            num_train = sum(data_paths.loc[data_paths.labels == label ].data_type == 'train')
            num_test = sum(data_paths.loc[data_paths.labels == label ].data_type == 'test')
            report_df.loc[len(report_df.index)] = [label,num_train,num_test]

        #saving report df
        report_df.to_excel(split_report_path+'.xlsx',index = False)

        #Get and save splitted video files to the txt files inside the directory given as an input to this method
        train_paths = data_paths.loc[data_paths['data_type']== 'train'].paths.values
        test_paths = data_paths.loc[data_paths['data_type'] == 'test'].paths.values

        #Shuffling the data for the model do not memorize the pattern
        random.shuffle(train_paths)
        random.shuffle(test_paths)

        with open(os.path.join(output_dir,'train.txt'), 'w', encoding='utf-8') as f:
            f.write(train_paths[0])
            for path in train_paths[1:]:
                f.write('\n' + path)
        
        with open(os.path.join(output_dir,'test.txt'), 'w', encoding='utf-8') as f:
            f.write(test_paths[0])
            for path in test_paths[1:]:
                f.write('\n' + path)

        print('Video paths splitted and saved succesfully!')

        return

    @staticmethod
    def create_cache_dir(dir_name):
        """
        Description:
        This static method is used to create a cache directory with the specified name if it does not already exist.

        Parameters:
        dir_name: The name of the cache directory to be created.

        Returns:
        None
        """
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        return  


    @staticmethod
    def get_labels(tensor_file_path):
        """
        Static Method that Seperate tensor type path to the dirs and get label from dirname
        Args:
            tensor_file_path: File path in tensor type. When we list files to create streaming data pipeline it gives us paths in the tensor type

        Return:
            Label seperated from path
        """
        dir_list = tf.strings.split(tensor_file_path, '/')
        return dir_list[-2]
    
    @staticmethod
    def format_frames(frame, output_size):
        """
        Static Method that Pad and resize an image from a video.

        Args:
            frame: Image that needs to resized and padded. 
            output_size: Pixel size of the output frame image.

        Return:
            Formatted frame with padding of specified output size.
        """
        frame = tf.image.central_crop(frame, 0.8)
        # frame = tf.image.per_image_standardization(frame)
        frame = tf.image.convert_image_dtype(frame, tf.float32)
        frame = tf.image.resize_with_pad(frame, *output_size)
        # frame = np.array(frame, dtype=np.uint8)
        frame = np.array(frame)
        return frame
    
    @staticmethod
    def compute_TVL1(prev, curr, bound=15):
        """
        Description:
        This static method computes the TV-L1 optical flow between two input frames (prev and curr) using the OpenCV TV-L1 optical flow algorithm. The computed optical flow represents the apparent motion of pixels between the two frames.

        Parameters:
        prev: The previous frame (numpy array or OpenCV image) from which the optical flow will be computed.
        curr: The current frame (numpy array or OpenCV image) to which the optical flow will be computed.
        bound (optional): The value used for constraining the optical flow values. It sets the maximum absolute value of the flow. The default value is 15.
        Returns:
        flow: The computed optical flow between the prev and curr frames, represented as a numpy array.
        
        Overall:
        This method provides a convenient way to compute the TV-L1 optical flow between two frames. 
        Optical flow estimation is widely used in computer vision tasks, such as motion tracking, video analysis, and object detection, 
        to capture the apparent motion of objects in a sequence of frames.
        
        """
        flow = TVL1.calc(prev, curr, None)
        # The computed optical flow is expected to have a dtype of np.float32, and an assertion is made to verify this.
        assert flow.dtype == np.float32
        # to map them into the desired range
        flow = (flow + bound) * (255.0 / (2*bound))
        # The flow values are rounded to the nearest integer using np.round and then converted to int type.
        flow = np.round(flow).astype(int)
        # Flow values greater than or equal to 255 are clipped to 255, and values less than or equal to 0 are clipped to 0, ensuring that they stay within the valid range
        flow[flow >= 255] = 255
        flow[flow <= 0] = 0

        return flow
    
    @staticmethod
    def to_gif(images):
        """
        Description:
        This static method converts a sequence of images into an animated GIF and saves it as animation.gif. It utilizes the imageio library to perform the conversion.

        Parameters:
        images: A sequence of images represented as a numpy array or a list of numpy arrays. Each image should have values in the range [0, 1].
        
        Returns:
        The method returns an embedded file representation of the generated GIF, which can be displayed or accessed in an appropriate context.
        """

        # The images are expected to be in a numpy array format, where each image has values normalized to the range [0, 1]
        # converted_images = np.clip(images*255 , 0, 255).astype(np.uint8)
        # The imageio.mimsave function is used to save the converted_images as an animated GIF named animation.gif with a frame rate of 25 frames per second (fps).
        # imageio.mimsave('./animation.gif', converted_images, fps=25)
        # return embed.embed_file('./animation.gif')
    
    @staticmethod
    def frames_from_video_file(video_path, n_frames, output_size = (224,224), frame_step = 5,output_format = ['RGB']):
        """
        Static Method that Creates frames from each video file present for each category.

        Args:
            video_path: File path to the video.
            n_frames: Number of frames to be created per video file.
            output_size: Pixel size of the output frame image.
            frame_step: Size of the steps in the time axis to get frame image
            output_format: List that define what to return. If it is just ['RGB'] it will return video array, if it is ['RGB','TVL1'] it will return 
            video tensor and optical flow. Note that it should be either length of 1 or 2.
            
            the function will extract 5 frames from the video with a step of 15 frames,
            meaning that it will skip 14 frames after the first frame has been extracted, 
            then skip 14 more frames after the second frame has been extracted,
            and so on until it extracts the desired number of frames.
            This parameter is used to control the number of frames extracted from the video,
            as well as the temporal resolution of the resulting frames. A larger frame_step value will result in fewer frames being extracted, 
            but each frame will be further apart in time. Conversely, a smaller frame_step value will result in more frames being extracted, 
            but each frame will be closer together in time. The choice of frame_step value will depend on the desired balance between 
            temporal resolution and computational efficiency.

        Return:
            An NumPy array of frames in the shape of (n_frames, height, width, channels).
        """

        if len(output_format) > 2:
            raise ValueError(f'Invalid output format. We are expecting output format with length 1 or 2 given length: {len(output_format)}')
        # Read each video frame by frame
        frames = []
        print('this is video path',video_path)
        src = cv2.VideoCapture(str(video_path))  

        video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)

        need_length = 1 + (n_frames - 1) * frame_step

        if need_length > video_length:
            start = 0
        else:
            max_start = video_length - need_length
            start = random.randint(0, max_start + 1)
        
        src.set(cv2.CAP_PROP_POS_FRAMES, start)

        #because we cannot calculate optical flow for the first frame when we create list while defining first element zero matrix 
        for format in output_format:
            #list that we will store optical flow for each frame
            if format == 'TVL1':
                optical_flows = [np.zeros((output_size[0], output_size[1], 2))]

            elif format == 'Farneback':
                optical_flows = [np.zeros((output_size[0], output_size[1], 3))]

        # ret is a boolean indicating whether read was successful, frame is the image itself
        ret, frame = src.read()
        # format and add first frame to the result list
        frames.append(DataLoader.format_frames(frame, output_size))
        # convert formated first frame to gray color space and assign it as previous to calculate optical flow
        prvs = cv2.cvtColor(frames[-1],cv2.COLOR_BGR2GRAY)

        hsv = np.zeros_like(frames[-1])
        hsv[..., 1] = 255

        for _ in range(n_frames - 1):
            for _ in range(frame_step):
                ret, frame = src.read()
            if ret:
                #processing the current frame of the video 
                frame = DataLoader.format_frames(frame, output_size)
                #defining the current frame so as to compare with previous one and calculate the optical flow
                curr = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                # next = cv2.resize(next, (0,0), fx=0.5, fy=0.5)
                #calculating the optical flow based on current and previous frame

                if "TVL1" in output_format:
                    #note: bound: limit the maximum movement of one pixel. It's an optional setting.
                    flow = DataLoader.compute_TVL1(prvs,curr,bound = 20) #compute_TVL1 function normalizing the flow itself
                    optical_flows.append(flow)
                    # print(f'the shape prvs {prvs.shape} and the shape of the current: {curr.shape}')

                elif 'farneback' in output_format:
                    flow = cv2.calcOpticalFlowFarneback(prvs, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    hsv[..., 0] = ang*180/np.pi/2
                    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                    # print('this is the shape of the flow',flow.shape,'\n type is: ',type(flow))
                    optical_flows.append(bgr)

                
                frames.append(frame)
                
                prvs = curr
            else:
                frames.append(np.zeros_like(frames[0]))

                if ('TVL1' in output_format) or ('Farneback' in output_format):
                    optical_flows.append(np.zeros_like(optical_flows[0]))

            
            
        src.release()
        # rearrange the frames array
        frames = np.array(frames)[..., [2, 1, 0]]
        output_list = []
        ## add all the results into list
        for format in output_format:
            if format == "RGB":
                output_list.append(frames)

            elif format == 'TVL1' or format=="Farneback":
                optical_flows = np.array(optical_flows)
                output_list.append(optical_flows)
            
        return output_list

          

    
    def load_ohe_dicts(self):
        """
        Function to load dictionaries from static defined path and assign it to the variables
        """
        with open(self.label_to_ohe_path,'rb') as jd:
            DataLoader.label_to_ohe_dict = pickle.load(jd)
        
        with open(self.index_to_label_path,'rb') as jd:
            DataLoader.index_to_label_dict = pickle.load(jd)


    def save_ohe_dicts(self):
        """
        Save created ohe dicts to the static defined path 
        label_to_ohe - > current_dir//label_to_ohe.txt
        label_to_ohe_dict - > current_dir//label_to_ohe_dict.txt
        """
        #save label to ohe dict
        with open(self.label_to_ohe_path,'wb') as jd:
            pickle.dump(self.label_to_ohe_dict,jd)
        print('Label to one hot encoding dict saved!')

        #saving dictionary where key is index of 1 in ohe and value is label corresponds to that
        with open(self.index_to_label_path,'wb') as jd:
            pickle.dump(self.index_to_label_dict,jd)
        print('index to label dict saved!')
    
    def fill_ohe_dicts(self):
        """
        Get labels from given datapath, create ohe dicts, assign them to the class variables with same name as below
        Created ohe dicts:

            label_to_ohe_dict -> Dictionary where input is label and value is its One Hot Encoding

            index_to_label_dict -> Dictionary where input is index of the value 1 in One Hot Encoding
            value is corresponding label to that
        """

        for video_path in self.video_paths_ds.map(DataLoader.get_labels):
            self.labels.add(video_path.numpy().decode('utf8'))
        

        #make labels static
        self.labels = list(self.labels)
        # Create OneHotEncoder instance
        encoder = OneHotEncoder()
        # Fit and transform the labels to one-hot encoding
        ohe_labels = encoder.fit_transform(np.array(self.labels).reshape(-1, 1))

        for i, label in enumerate(self.labels):
            self.label_to_ohe_dict[label] = ohe_labels.toarray()[i]

        # Create a dictionary mapping the index of the 1 in the one-hot encoding to the corresponding string label
        for i, label in enumerate(self.labels):
            self.index_to_label_dict[np.where(ohe_labels.toarray()[i]==1)[0][0]] = label

        print('Label maping dicts gotten succesfully!')

    
    def save_path_dataset(self,dataset):
        """
        This function written in case you want to change streaming pipeline to the simpler form where you dont split data and train with all the data
        """
        # Create a new dataset that contains only the file paths
        file_paths_ds = dataset.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))

        # Map each file path to a string
        string_ds = file_paths_ds.map(lambda x: tf.strings.join([x, '\n']))

        # Reduce the dataset to a single string
        file_paths_string = string_ds.reduce(
            initial_state=b'',
            reducer=lambda x, y: tf.strings.join([x, y])
        )

        # Save the string to a file
        with open(self.get_path(self.cache_directory,'video_paths.txt'), 'wb') as f:
            f.write(file_paths_string.numpy())

        return 
    
    @staticmethod
    def load_video_paths_ds(file_path):
        """
        Description:
        This static method loads video paths from a text file and returns them as a TensorFlow Dataset object.

        Parameters:
        file_path: The file path to the text file containing the video paths.

        Returns:
        paths: A TensorFlow Dataset object containing the loaded video paths.
        """
        paths = tf.data.TextLineDataset(file_path)
        return paths
    
    def process_videos_ds(self,filepath):
        """
        Main mapping function in the streaming pipeline that calls required other maping functions
        Sequence of the mapping for each sample from dataset where just filepath of certain sample provided : 
            1. Get label from filepath. Name of the folder that contains videos for specific class is label of the class
            2. Get corresponding one hot encoding representation of the gotten label
            3. Get video or tvl1 tensor with filepath
            4. Reshape video tensor for not losing its dimensions
            5. Return video_tensor,label_ohe 

        Args: 
            filepath: filepath of the certain video in the tensor string type  

        Returns:
            tf.convert_to_tensor(video) - >  tensor of the video (one data sample), 
            tf.convert_to_tensor(label) - >  one hot encoding of the label in tensor type
        """
        label = DataLoader.get_labels(filepath)
        label = label.numpy().decode('utf8')
        label = tf.reshape(self.label_to_ohe_dict[label], shape = (self.num_classes,))
        

        results = DataLoader.frames_from_video_file(filepath.numpy().decode('utf8'),
                                                  n_frames = self.n_frames,
                                                  output_size = self.output_size,
                                                  frame_step = self.frame_step,
                                                  output_format = self.mode)
        
        format_mapping = {}

        for i,format in enumerate(self.mode):
            format_mapping[format] = results[i]

        if len(self.mode) == 1:
            mode_value = self.mode[0]  # Get the value from the list
            if mode_value in format_mapping:     return tf.convert_to_tensor(format_mapping[mode_value]),tf.convert_to_tensor(label)

        elif len(self.mode) == 2:
            x_results =  [tf.convert_to_tensor(format_mapping[format], dtype=tf.float32) for format in self.mode if format in format_mapping]
            return          x_results[0],x_results[1],tf.convert_to_tensor(label)
                    
    
    def stream_line(self, n_frames, output_size = (224,224), frame_step = 15,stream = ['RGB']):
        """
        Method to stream the data to the tf dataset variable 

        Args:
            video_path: File path to the video.
            n_frames: Number of frames to be created per video file.
            output_size: Pixel size of the output frame image.
            frame_step: Number of the frame to pass in each step
            stream: Could be RGB, RGB_optical_flow, optical_flow

        Return:
            Dataset in tuples (x,y) where x is array of the video sample and y is the corresponding ohe label 
        """

        # we declare the mode that data will stream here and it will be given as an parameter to the func: frames_from_video_file 
        self.mode = stream
        self.n_frames = n_frames
        self.output_size = output_size
        self.frame_step = frame_step
        tf.config.run_functions_eagerly(True)
        # we use this dict while mapping from @tf.function process_videos_ds function 
        self.tensor_formats = {
            'RGB' :tf.float32,
            'TVL1':tf.float64,
            'Farneback':tf.float64
        }

        self.tensor_sizes = {
            'RGB' :((n_frames,) + output_size + (3,)),
            'TVL1':((n_frames,) + output_size + (2,)),
            'Farneback':((n_frames,) + output_size + (3,))
        }

        # get the tf.data data structure from saved test and train txts 
        train_ds = DataLoader.load_video_paths_ds(self.train_paths)
        test_ds  = DataLoader.load_video_paths_ds(self.test_paths)

        # Two type of mapping based on number of the 
        if len(self.mode) == 1:
            mode_value = self.mode[0]

            train_ds = train_ds.map(lambda filepath:
                                                    (tf.py_function(self.process_videos_ds, [filepath],
                                                                    [self.tensor_formats[mode_value], tf.float64])))
        
            test_ds = test_ds.map(lambda filepath:
                                                    (tf.py_function(self.process_videos_ds, [filepath],
                                                                    [self.tensor_formats[mode_value], tf.float64])))
            train_ds = train_ds.map(lambda x,y: (tf.reshape(x,self.tensor_sizes[mode_value]),tf.reshape(y,(self.num_classes,))))
            test_ds  = test_ds.map( lambda x,y: (tf.reshape(x,self.tensor_sizes[mode_value]),tf.reshape(y,(self.num_classes,))))
            return train_ds,test_ds
        
        elif len(self.mode) == 2:

            mode_value = self.mode[0]
            train_ds = train_ds.map(lambda filepath:
                                                    (tf.py_function(self.process_videos_ds, [filepath],
                                                                    [tf.float32,tf.float32,tf.float64])))
                                                    
            test_ds = test_ds.map(lambda filepath:
                                                    (tf.py_function(self.process_videos_ds, [filepath],
                                                                    [tf.float32,tf.float32,tf.float64])))

            return train_ds,test_ds
        

data = DataLoader('WL_AzSL/videos_by_labels')

train,test = data.stream_line(n_frames = 20,output_size=(224,224),frame_step = 1,stream = ['RGB','TVL1'])

train_1,test_1 = data.stream_line(n_frames = 20,output_size=(224,224),frame_step = 2,stream = ['TVL1'])


# for result in test.batch(5).take(1):
#     for sample in result:
#         print(sample.shape)


