import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.misc import imread,imresize
from skimage.segmentation import clear_border
from skimage.morphology import label
from skimage.measure import regionprops
import matplotlib.cm as cm
from sklearn.neighbors import KNeighborsClassifier
from math import sqrt
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, InputLayer
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
from keras.models import Sequential, load_model
from keras.utils import to_categorical
import pandas as pd

def extract_letters(filename):
    image = imread(filename,1)

    #apply threshold in order to make the image binary
    bw = image < 120

    # remove artifacts connected to image border
    cleared = bw.copy()
    clear_border(cleared)

    # label image regions
    label_image = label(cleared,neighbors=8)
    borders = np.logical_xor(bw, cleared)
    label_image[borders] = -1

    print label_image.max()

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(7, 7))
    ax.imshow(bw, cmap='jet')


    # array initialised with length as zero in order to avoid duplication with append
    # 20x20 respecting the image normalised width.
    letter_images = np.empty((0,28,28))

    letters = list()
    order = list()

    for region in regionprops(label_image):
        minc, minr, maxc, maxr = region.bbox
            # skip small images
        if maxc - minc > len(image)/250: # better to use height rather than area.
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            order.append(region.bbox)
            # print(maxc-minc)
            # print(maxr-minr)


    plt.show()
    lines = list()
    first_in_line = ''
    counter = 0

        #worst case scenario there can be 1 character per line
    for x in range(len(order)):
        lines.append([])

    for character in order:
        if first_in_line == '':
            first_in_line = character
            lines[counter].append(character)
        elif abs(character[0] - first_in_line[0]) < (first_in_line[2] - first_in_line[0]):
            lines[counter].append(character)
        elif abs(character[0] - first_in_line[0]) > (first_in_line[2] - first_in_line[0]):
            first_in_line = character
            counter += 1
            lines[counter].append(character)


    for x in range(len(lines)):
        lines[x].sort(key=lambda tup: tup[1])

    final = list()
    prev_tr = 0
    prev_line_br = 0

    for i in range(len(lines)):
        for j in range(len(lines[i])):
            tl_2 = lines[i][j][1]
            bl_2 = lines[i][j][0]
            if tl_2 > prev_tr and bl_2 > prev_line_br:
                tl,tr,bl,br = lines[i][j]
                letter_raw = label_image[tl:bl,tr:br]
                letter_norm = imresize(letter_raw ,(28 ,28))
                final.append(letter_norm)
                prev_tr = lines[i][j][3]
            if j == (len(lines[i])-1):
                prev_line_br = lines[i][j][2]
        prev_tr = 0
        tl_2 = 0

    letter_train = np.empty_like(final)
    letter_train[:] = final

    return letter_train;

a_author = extract_letters("1234567.jpg")
b_author = extract_letters("1222.jpg")

def cluster_this(training_files):
    hyp = list()

    for i in range(len(training_files)):
        x = imread(training_files[i], 1)
        test_image = x.reshape(x.shape[0], x.shape[1])
        hist = np.zeros(256)

        for (x,y),value in np.ndenumerate(test_image):
        	hist[int(value)] = hist[int(value)]+1

        cdf = hist.cumsum()
        z = dimensions[i,0] * dimensions[i,1]
        hyp.append([z,np.std(test_image) ])
        # print('Person --------> ', i % 3)
        # print(z)
        # print(np.std(test_image))

    theory = np.empty_like(hyp)
    theory[:] = hyp

    return theory



theory = cluster_this(training_files)
kmeans = KMeans(n_clusters=2)
clusters = kmeans.fit_predict(theory)
centers = kmeans.cluster_centers_

labels = KMeans(2, random_state=0).fit_predict(theory)

plt.scatter(theory[:,0], theory[:,1], c=labels, s=50, alpha=0.5)
# plt.scatter(centers[:,0], centers[:,1], c='black', s=50, alpha=0.5)
plt.xlabel('Area')
plt.ylabel('Standard Deviation')

plt.show()


# imhist,_ = np.histogram(moon.flatten(),256)
# #compute the cumulative distribution
# cdf = imhist.cumsum()
#
# #create the pixel equalization transform
# cdf = 255 * cdf / cdf[255]
# print (cdf[200])
# #create a new image to hold the equalized
# moon_equalized = np.empty_like(moon)
#
# for (x,y), pixel_value in np.ndenumerate(moon):
#     print(pixel_value)
#     moon_equalized[x,y] = cdf[pixel_value]
#
# #get the info for the equalized image
# imhist_eq,_ = np.histogram(moon_equalized.flatten(),256)
# cdf_eq = imhist_eq.cumsum()
#
# #plot the images
# plt.figure(0)
# plt.subplot(1,2,1)
# plt.xticks([])
# plt.yticks([])
# plt.imshow(moon.astype(np.uint8),
# cmap=cm.Greys_r,aspect='equal',vmin=0,vmax=255)
# plt.subplot(1,2,2)
# plt.xticks([])
# plt.yticks([])
# plt.imshow(moon_equalized.astype(np.uint8),
# cmap=cm.Greys_r,aspect='equal',vmin=0,vmax=255)
#
# plt.show()
