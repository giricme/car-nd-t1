import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
import pickle
import glob
import time
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split
from scipy.ndimage.measurements import label
from scipy.ndimage import generate_binary_structure
from collections import deque

class Pipeline:
    def __init__(self, svc, scaler, confidence, heat_threshold, heatmap_history_count=0):
        self.svc = svc
        self.scaler = scaler
        self.confidence = confidence
        self.heat_threshold = heat_threshold
        self.heatmap_history_count = heatmap_history_count
        self.heatmap_history = deque([])
        self.debug = []
        self.img_count = 0
        
    def apply_labels(self, heatmap):
        labels = None
        if self.heatmap_history_count > 0:
            # standardise heatmap -
            heatmap_std = heatmap.std(ddof=1)
            if heatmap_std != 0.0:
                heatmap = (heatmap-heatmap.mean())/heatmap_std
            heatmap = apply_threshold(heatmap, np.max([heatmap.std(), 1]))
            self.heatmap_history.append(heatmap)
            if len(self.heatmap_history) > self.heatmap_history_count:
                self.heatmap_history.popleft()
            hh = np.array(self.heatmap_history)
            # create a structure for connectivity
            s = generate_binary_structure(hh.ndim, hh.ndim)
            labels = label(hh, s)
        else:
            labels = label(heatmap)
        return labels

    def detect_bboxes(self, labels):
        bboxes = []
        # Iterate through all detected cars
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroz = np.array(nonzero[0])
            nonzeroy = np.array(nonzero[1])
            nonzerox = np.array(nonzero[2])

            nonzerox_min = np.min(nonzerox)
            nonzerox_max = np.max(nonzerox)
            nonzeroy_min = np.min(nonzeroy)
            nonzeroy_max = np.max(nonzeroy)
            nonzeroz_min = np.min(nonzeroz)
            nonzeroz_max = np.max(nonzeroz)

            # only add if they appear in contiguous planes
            nplane_min_threshold = self.heatmap_history_count - 2
            # planes connected via label function and ndims of heatmap
            # they start at 0 so add 1
            nplanes = nonzeroz_max-nonzeroz_min+1
            if nplanes >= nplane_min_threshold:
                bbox = ((nonzerox_min, nonzeroy_min),
                        (nonzerox_max, nonzeroy_max))
                bboxes.append(bbox)
        return bboxes

    def draw_detected_bboxes(self, img, bbox_list):
        for bbox in bbox_list:
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
        # Return the image
        return img

    def process_image(self, image):
        self.img_count = self.img_count + 1
        #if self.img_count < 1100 or self.img_count > 1150:
        #    return np.zeros_like(image).astype(np.float)
        """
        1. Pick windows in the image.
        2. Check if those windows have cars.
        3. Draw bounding boxes on those.
        """
        windows = []
        windows = slide_window(image, x_start_stop=[300, None], y_start_stop=[380, 480],
                                           xy_window=(96, 96), xy_overlap=(0.9, 0.9))
        windows += slide_window(image, x_start_stop=[300, None], y_start_stop=[400, 500],
                                           xy_window=(128, 128), xy_overlap=(0.8, 0.8)) 
        windows += slide_window(image, x_start_stop=[300, None], y_start_stop=[430, 550],
                                           xy_window=(144, 144), xy_overlap=(0.8, 0.8))  
        windows += slide_window(image, x_start_stop=[300, None], y_start_stop=[460, 580],
                                           xy_window=(192, 192), xy_overlap=(0.8, 0.8))
        car_windows = find_cars(image, windows, self.svc, self.scaler, self.confidence)
        # Add heat to each box in box list
        heat = add_heat(image, car_windows)  
        # Apply threshold to help remove false positives
        heat = apply_threshold(heat, self.heat_threshold)
        labels = self.apply_labels(heat)
        bboxes = self.detect_bboxes(labels)
        # draw_img = draw_labeled_bboxes(np.copy(image), labels)
        draw_img = self.draw_detected_bboxes(np.copy(image), bboxes)
        self.debug.append(draw_img)
        return draw_img

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Define a function to compute color histogram features  
def color_hist(image, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the channels separately
    # Take histograms of each channel
    c1, c2, c3 = image[:,:,0], image[:,:,1], image[:,:,2]
    c1hist = np.histogram(c1, bins=nbins, range=bins_range)
    c2hist = np.histogram(c2, bins=nbins, range=bins_range)
    c3hist = np.histogram(c3, bins=nbins, range=bins_range)
    # Generating bin centers
    bin_edges = c1hist[1]
    bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((c1hist[0], c2hist[0], c3hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return c1hist, c2hist, c3hist, bin_centers, hist_features

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features
        
# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space='YCrCb', spatial_size=(32, 32),
                        hist_bins=32, orient=8, 
                        pix_per_cell=8, cell_per_block=2, hog_channel="ALL",
                        spatial_feat=True, hist_feat=True, hog_feat=True):   
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: 
        feature_image = np.copy(img) 
    #3) Normalize image so that values are between 0 and 1
    image_min = np.min(feature_image)
    image_max = np.max(feature_image)
    feature_image = (feature_image - image_min) / (image_max - image_min)
    #4) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #5) Append features to list
        img_features.append(spatial_features)
    #6) Compute histogram features if flag is set
    if hist_feat == True:
        c1hist, c2hist, c3hist, bin_centers, hist_features = color_hist(feature_image, nbins=hist_bins)
        #7) Append features to list
        img_features.append(hist_features)
    #8) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))      
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #9) Append features to list
        img_features.append(hog_features)

    #10) Return concatenated array of features
    return np.concatenate(img_features)

def extract_features(images, cspace='YCrCb', spatial_feat=True, hist_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in images:   
        # Read in each one by one
        image = cv2.imread(file)
        img_features = single_img_features(image, color_space=cspace, 
                                           spatial_feat=spatial_feat, 
                                           hist_feat=hist_feat)
        features.append(img_features)
    return features

def load_model():
    with open('svc.pickle', 'rb') as f:
        pickle_data = pickle.load(f)
        svc = pickle_data['svc']
        X_scaler = pickle_data['X_scaler']
        del pickle_data
        return svc, X_scaler

# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_windows = np.int(xspan/nx_pix_per_step) - 1
    ny_windows = np.int(yspan/ny_pix_per_step) - 1
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

def find_cars(img, windows, svc, feature_scaler, confidence):
    result = []
    for window in windows:
        window_image = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        img_features = single_img_features(window_image)
        test_features = feature_scaler.transform(img_features)
        prediction = svc.predict(test_features)
        conf = svc.decision_function(test_features)
        if prediction == 1 and conf > confidence:
            result.append(window)
    return result

def add_heat(image, bbox_list):
    heatmap = np.zeros_like(image[:,:,0]).astype(np.float)
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    # Return updated heatmap
    return heatmap

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

def process_images():
    svc, scaler = load_model()
    pipeline = Pipeline(svc, scaler, 0.4, 2)
    test_images = glob.iglob('./test_images/*.jpg', recursive=True)

    test_output = []
    for image in test_images:
        test_image = cv2.imread(image)
        test_output.append(pipeline.process_image(test_image))

    for output_image in test_output:
        plt.imshow(output_image)
        plt.show()    

def process_video():
    svc, scaler = load_model()
    white_output = 'project_video_out.mp4'
    clip1 = VideoFileClip('project_video.mp4')
    pipeline = Pipeline(svc, scaler, 0.6, 2, 10)
    white_clip = clip1.fl_image(pipeline.process_image)
    white_clip.write_videofile(white_output, audio=False)
    for dbg_img in pipeline.debug:
        plt.imshow(dbg_img)
        plt.show() 

process_video()	