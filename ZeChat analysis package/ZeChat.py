import cv2
import os
from os import path
import re
import numpy as np
import imutils
import scipy
import math
import glob
from keras.models import model_from_json
import matplotlib.pyplot as plt
from time import time
import pickle
from sklearn.metrics import pairwise_distances
from sklearn.utils import shuffle
import pywt
import matplotlib.colors as colors
from sklearn.decomposition import PCA
from collections import Counter
from mpl_toolkits.axes_grid1 import make_axes_locatable


### import models:
os.chdir(r'C:\Folder\Containing\Analysis\Package\Files') # address of the folder containing the analysis package files

json_file = open('autoencoder model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
autoencoder_model = model_from_json(loaded_model_json)
autoencoder_model.load_weights("autoencoder model.h5")
print("Loaded autoencoder model from disk")

json_file = open('encoder model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
encoder_model = model_from_json(loaded_model_json)
encoder_model.load_weights("encoder model.h5")
print("Loaded encoder model from disk")

pca_model_encoded = pickle.load(open('pca_model_encoded.sav', 'rb'))

X_train = np.loadtxt(open("cwtmatr_random_shuffle_3000to1000_X_train.csv", "rb"), delimiter=",")
print('load X_train')

Y_train = np.loadtxt(open("cwtmatr_tsne_3000to1000_Y_train.csv", "rb"), delimiter=",")
print('load Y_train')

watershed = np.loadtxt(open("watershed.csv", "rb"), delimiter=",")


### pre-calculations for ktSNE

## define JS Distance (square root of JS Divergence, a metric)

def jsd(p, q):
    m = (p + q) / 2
    return (np.nan_to_num(np.nan_to_num(scipy.stats.entropy(p, m)) + np.nan_to_num(scipy.stats.entropy(q, m))) / 2)**0.5


## calculate pairwise distances:
D_train = pairwise_distances(X_train, metric=jsd)
print('precomputed data')


## determin sigma based on D_train:
D_train_no_zero = np.ma.masked_values(D_train, 0.0)
D_train_min_distance = D_train_no_zero.min(axis=0)

print(D_train_min_distance)

scaling_factor = 0.1  


def calculate_K_train(scaling_factor):
    sigma = scaling_factor * D_train_min_distance 
    sigma_squared = np.square(sigma)
    
    # calculate K_train:
    D_train_squared = np.square(D_train)
    k_train_ij = np.exp(-0.5*D_train_squared/sigma_squared)
    sum_k_train_il = np.sum(k_train_ij, axis=1)
    K_train = k_train_ij / sum_k_train_il[:,None]
    
    return K_train, sigma_squared


K_train, sigma_squared = calculate_K_train(scaling_factor)


## calculate A:
K_train_pseudo_inverse = np.linalg.pinv(K_train)
A = np.dot(K_train_pseudo_inverse, Y_train)

print('pre-calculations for ktSNE completed')


### set parameters:
jump_to_frame = 7490 # 50fps*60sec*2.5min=7500, 7500-10=7490. Jump to ~ 2.5min into the recording
frame_cutoff = 10  # precalculate 10 frames before starting the analysis to prevent fgmasked images being blank

## set parameters for locating each ZeChat unit as ROIs for cropping
y_orig = 25
x_orig = 20

wall = 20
window = 15

box_dim = 220 # dimension for cropping out each fish

img_dim = 56 # reduce image to this dimension for autoencoder

sample_size = 5000 # sample this many frames out of 15000 frames (5min)
batch_size = 2500 # set testing sample batch size for ktSNE

cmap = colors.LinearSegmentedColormap.from_list("", ["white","royalblue","yellow","red"])



### iterate through all .avi files in path containing the data
for filename in glob.glob('C:/Folder/Containing/Data/*.avi'): # folder containing the data
    for ro in range(8): # iterate through the rows, 8 rows total
        for col in range(10): # interate through the columns, 10 columns total

            ### preprocessing
            t0 = time()

            splitfilename = re.split(r'\.|\\', filename)
            # first split '.' using \., then split '\\' which separates the folder names with the *.avi filename
            # note that the filename contains the full path of the directory

            if not os.path.exists('%s/%s' % (splitfilename[0], splitfilename[1])):
                os.makedirs('%s/%s' % (splitfilename[0], splitfilename[1]))
                os.makedirs('%s/%s/ktSNE' % (splitfilename[0], splitfilename[1]))
                os.makedirs('%s/%s/watershed' % (splitfilename[0], splitfilename[1]))
                os.makedirs('%s/%s/watershed_counts' % (splitfilename[0], splitfilename[1]))
                
            if os.path.exists('%s/%s/watershed_counts/%s_row%d_column%d_watershed_counts.csv' % (splitfilename[0], splitfilename[1], splitfilename[1], ro+1, col+1)):
                continue
            
            temp = []

            cap = cv2.VideoCapture(filename)
            cap.set(1,jump_to_frame)

            fgbg = cv2.createBackgroundSubtractorKNN()

            success = True
            count = 0

            ## locate the roi:
            y_roi = y_orig+(box_dim*(ro))+(window*int(math.ceil(ro/2)))+(wall*(int(math.ceil((ro+1)/2))-1))
            x_roi = x_orig+(box_dim*(col))+(wall*col)

            
            for frames_to_analyze in range(15000 + frame_cutoff): # set the total number of frames to analyze
                # 50fps*60sec*10min = 30000 frames for a total of 10 mins of recording;
                # To analyze 5 mins of recording from 2.5 min to 7.5 min: 15000 frames
                
                success, frame = cap.read()
                if not success:
                    break

                ## isolate the roi:
                roi = frame[y_roi:y_roi+box_dim, x_roi:x_roi+box_dim]

                if (ro+1) % 2 == 1:
                    roi = cv2.flip(roi, -1) # if the row is oddly numbered (top fish), flip frame
                
                roi_shape = roi
                if count == 0:
                    hsv = np.zeros_like(roi_shape)
                    hsv[...,1] = 255
                
                count += 1

                roi = cv2.bilateralFilter(roi, 7, 150, 150)
             
                fgmask = fgbg.apply(roi)

                if count == (frame_cutoff - 1):
                    prev_roi = roi

                if count >= frame_cutoff:
                    ## CLOSE closes small holes inside the foreground objects
                    kernel_close = np.ones((5,5),np.uint8)
                    img = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel_close)

                    ## remove small blobs; OPEN removes salt-and-pepper noise
                    kernel_open = np.ones((5,5),np.uint8)
                    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_open)

                    ## create mask for the fish and compute dense optical flow
                    ret, thresh = cv2.threshold(img, 0, 255, 0)
                    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cnts = imutils.grab_contours(cnts)
                    mask = np.zeros(thresh.shape, dtype="uint8")
                    cv2.drawContours(mask, cnts, -1, 255, -1)
                    masked_image = cv2.bitwise_and(roi, roi, mask=mask)
                    masked_image = cv2.cvtColor(masked_image,cv2.COLOR_BGR2GRAY)
                    
                    if count == frame_cutoff:
                        prvs = masked_image
                        continue
                    
                    elif count > frame_cutoff: 
                        current = masked_image

                    flow = cv2.calcOpticalFlowFarneback(prvs,
                                                        current,
                                                        flow = None,
                                                        pyr_scale = 0.5, 
                                                        levels = 3,
                                                        winsize = 15,    
                                                        iterations = 3,
                                                        poly_n = 5,
                                                        poly_sigma = 1.1,
                                                        flags = 0
                                                        )

                    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
                    hsv[...,0] = ang*180/np.pi/2
                    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
                    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)


                    ## create mask using the subtracted image
                    subtracted_image = cv2.absdiff(roi, prev_roi)
                    prev_roi = roi
                    subtracted_image = cv2.cvtColor(subtracted_image,cv2.COLOR_BGR2GRAY) 

                    ret, threshold_image = cv2.threshold(subtracted_image,18,255,cv2.THRESH_BINARY)
                 
                    contours = cv2.findContours(threshold_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contours = imutils.grab_contours(contours)
                    mask_subtracted = np.zeros(threshold_image.shape, dtype="uint8")
                    cv2.drawContours(mask_subtracted, contours, -1, 255, -1)
                    masked_subtracted_image = cv2.bitwise_and(roi, roi, mask = mask_subtracted)
                    masked_subtracted_image = cv2.cvtColor(masked_subtracted_image,cv2.COLOR_BGR2GRAY)

                    ## apply subtracted image mask to the optical flow image
                    subtracted_image_masked_opticalflow = cv2.bitwise_and(rgb, rgb, mask = mask_subtracted)
                    
                    temp_img = cv2.resize(subtracted_image_masked_opticalflow, (img_dim, img_dim))
                    temp_img = temp_img.astype('float32') / 255.
                    temp_img = np.array(temp_img)
                    temp.append(temp_img)


                    prvs = current


            cap.release()

            t1 = time()

            ### feature extraction
            
            temp_data = np.stack(temp)

            print("Loaded test data for row%d column%d in %s" % (ro+1, col+1, filename))

            encoded_imgs = encoder_model.predict(temp_data)
            decoded_imgs = autoencoder_model.predict(temp_data)


            for i in range(len(encoded_imgs)):

                encoded_flatten = encoded_imgs[i].reshape((1,784))
                
                if i == 0:
                    combine_flatten = encoded_flatten
                else:
                    combine_flatten = np.concatenate((combine_flatten, encoded_flatten), axis=0)


            ## extract pca principal components using a saved pca model
            pca_model_encoded.fit(combine_flatten)

            print(str(sum(pca_model_encoded.explained_variance_ratio_)) + " variance is kept within 40 components")

            encoded_pca = pca_model_encoded.transform(combine_flatten)

            t2 = time()

            ## complex morlet wavelet analysis

            sig = encoded_pca

            t = np.arange(0, len(sig))
            widths = np.arange(10, 131, 5)

            no_Nan = True
            
            for i in range(len(sig.T)):
                cwtmatr, freqs = pywt.cwt(sig[:, i], widths, 'cmor1.5-1.0')

                abs_cwtmatr = abs(cwtmatr)
                
                abs_cwtmatr = abs_cwtmatr.T

                if np.all(np.isfinite(abs_cwtmatr)) == False:
                    no_Nan= False
                    break
                
                if i == 0:
                    combine_cwtmatr = abs_cwtmatr
                else:
                    combine_cwtmatr = np.column_stack((combine_cwtmatr, abs_cwtmatr))  # rows: frames; columns: 1000 (40x25) Morlet wavelet amplitudes 

            if no_Nan == False:
                continue

            if np.count_nonzero(np.sum(combine_cwtmatr, axis=1) == 0) > 0: # count zeros
                continue
            combine_cwtmatr /= np.sum(combine_cwtmatr, axis=1)[:,None] # normalize: dividing each element by the sum of its row 
            
            if np.all(np.isfinite(combine_cwtmatr)) == False:
                continue

            
            np.savetxt("%s/%s/%s_row%d_column%d_cwtmatr.csv" % (splitfilename[0], splitfilename[1], splitfilename[1], ro+1, col+1), combine_cwtmatr, delimiter=",")
            print("Saved cwtmatr40x25 for row%d column%d in %s" % (ro+1, col+1, filename))
            

            t3 = time()

            ## random sample cwtmatr
            idx = np.random.randint(len(combine_cwtmatr), size=sample_size)
            combine_cwtmatr = combine_cwtmatr[idx,:]
            

            ### ktSNE:
            
            ktsne_count = 0
            
            for i in range(0, len(combine_cwtmatr), batch_size):

                ## batch process X_test
                X_test = combine_cwtmatr[i:i+batch_size, :]

                ## calculate pairwise distances:
                D_test = pairwise_distances(X_test, X_train, metric=jsd)

                ## calculate K_test:
                D_test_squared = np.square(D_test)
                k_test_ij = np.exp(-0.5*D_test_squared/sigma_squared)
                sum_k_test_il = np.sum(k_test_ij, axis=1)
                K_test = k_test_ij / sum_k_test_il[:,None]

                ## calculate Y_test:
                Y_test = np.dot(K_test, A)

                ## combine batched tested Y_test
                if ktsne_count == 0:
                    ktsne = Y_test
                    ktsne_count = 1
                else:
                    ktsne = np.concatenate((ktsne, Y_test), axis=0)

                print(i)

            np.savetxt("%s/%s/%s_row%d_column%d_ktSNE.csv" % (splitfilename[0], splitfilename[1], splitfilename[1], ro+1, col+1), ktsne, delimiter=",")
            print("Saved ktSNE data for row%d column%d in %s" % (ro+1, col+1, filename))

            
            train_x = Y_train[:, 0]
            train_y = Y_train[:, 1]

            test_x = ktsne[:, 0]
            test_y = ktsne[:, 1]

            plt.figure(figsize=(10, 10))

            plt.scatter(test_x,
                        test_y,
                        s=3,
                        )

            plt.scatter(train_x,
                        train_y,
                        s=1,
                        marker='x',
                        )

            plt.savefig("%s/%s/ktSNE/%s_row%d_column%d_ktSNE.png" % (splitfilename[0], splitfilename[1], splitfilename[1], ro+1, col+1))
            print("Saved ktSNE plot for row%d column%d in %s" % (ro+1, col+1, filename))
            plt.close()

            t4 = time()

            ## export watershed labels

            xmin, xmax = 0, 1000
            ymin, ymax = 0, 1000

            x = 1000 * (ktsne[:, 0] + 140) / 260 # x axis coordinates: minVal = -140, range = 260
            y = 1000 * (ktsne[:, 1] + 100) / 200 # y axis coordinates: minVal = -100, range = 200

            ktsne_watershed_label = np.zeros(np.shape(x))

            watershed_T = watershed.T

            for i in range(len(x)):
                # background label is 68
                if watershed_T[int(y[i]), int(x[i])] == 68:
                    ktsne_watershed_label[i] = 0
                if watershed_T[int(y[i]), int(x[i])] < 68:
                    ktsne_watershed_label[i] = watershed_T[int(y[i]), int(x[i])]
                if watershed_T[int(y[i]), int(x[i])] > 68:
                    ktsne_watershed_label[i] = watershed_T[int(y[i]), int(x[i])] - 1

            np.savetxt("%s/%s/watershed/%s_row%d_column%d_watershed.csv" % (splitfilename[0], splitfilename[1], splitfilename[1], ro+1, col+1), ktsne_watershed_label, delimiter=",")
            print("Saved watershed label for row%d column%d in %s" % (ro+1, col+1, filename))


            ## count and sort watershed labels

            dict_label_count = Counter(ktsne_watershed_label)

            temp_label_count = []
            label_count = []

            for key, value in dict_label_count.items():
                temp_label_count = [key,value]
                label_count.append(temp_label_count)

            temp = []
            
            for j in range(81): # 81 is the total number of watershed labels; value zero now labels the background
                check = 0
                for k in range(len(label_count)):
                    if label_count[k][0] == j:
                        temp.extend([label_count[k][1]])
                        check = 1
                if check == 0:
                    temp.extend([0])

            temp = np.array(temp)

            np.savetxt("%s/%s/watershed_counts/%s_row%d_column%d_watershed_counts.csv" % (splitfilename[0], splitfilename[1], splitfilename[1], ro+1, col+1), temp, delimiter=",")
            print("Saved watershed label count for row%d column%d in %s" % (ro+1, col+1, filename))

            t5=time()
            
            print(t1-t0) # preprocessing time
            print(t2-t1) # encode time
            print(t3-t2) # wavelet analysis time
            print(t4-t3) # ktSNE analysis time
            print(t5-t4) # watershed analysis time
            print(t5-t0) # total duration for analyzing a fish













