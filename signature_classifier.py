from scipy.misc import imread,imresize
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
import numpy as np
from sklearn.cluster import KMeans
from skimage import io, img_as_float

def readImage(filenames):

    signatures = list()
    dimensions = list()

    total_width, total_length = 0, 0

    for i in range(len(filenames)):
        signature = imread(filenames[i],1)
        width, length = signature.shape
        total_width += width
        total_length += length
        dimensions.append([width, length])
        signatures.append(signature)

    images = np.empty_like(signatures)
    images[:] = signatures

    x = np.empty_like(dimensions)
    x[:] = dimensions
    return images, total_width/len(filenames), total_length/len(filenames), x

def resizeImage(images, width, length):

    training_images = np.empty((len(images), width, length))

    for i in range(len(images)):
        training_images[i] = imresize(images[i] ,(width ,length))


    return training_images

training_files = ['signatures/Training/NFI-00101001.png', 'signatures/Training/NFI-00102001.png', 'signatures/Training/NFI-00103001.png',
                    'signatures/Training/NFI-00401004.png', 'signatures/Training/NFI-00402004.png', 'signatures/Training/NFI-00403004.png',
                    'signatures/Training/NFI-00501005.png', 'signatures/Training/NFI-00502005.png', 'signatures/Training/NFI-00503005.png',
                    'signatures/Training/NFI-00801008.png', 'signatures/Training/NFI-00802008.png', 'signatures/Training/NFI-00803008.png',
                    'signatures/Training/NFI-00901009.png', 'signatures/Training/NFI-00902009.png', 'signatures/Training/NFI-00903009.png']

training_images, width, length, dimensions = readImage(training_files)

# training_images = resizeImage(training_images, width, length)
# training_images = training_images.reshape(30, width*length)

# for i in range(30):
#     print(training_images[i].shape)

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
kmeans = KMeans(n_clusters=5)
clusters = kmeans.fit_predict(theory)
centers = kmeans.cluster_centers_

labels = KMeans(5, random_state=0).fit_predict(theory)

plt.scatter(theory[:,0], theory[:,1], c=labels, s=50, alpha=0.5)
# plt.scatter(centers[:,0], centers[:,1], c='black', s=50, alpha=0.5)
plt.xlabel('Area')
plt.ylabel('Standard Deviation')

plt.show()


testing_files = ['signatures/Testing/NFI-00104001.png', 'signatures/Testing/NFI-00105001.png',
                    'signatures/Testing/NFI-00404004.png', 'signatures/Testing/NFI-00405004.png',
                    'signatures/Testing/NFI-00504005.png', 'signatures/Testing/NFI-00505005.png',
                    'signatures/Testing/NFI-00804008.png', 'signatures/Testing/NFI-00805008.png',
                    'signatures/Testing/NFI-00904009.png', 'signatures/Testing/NFI-00905009.png']

plt.figure(0)
plt.subplot(1,2,1)
plt.xticks([])
plt.yticks([])
plt.imshow(training_images[0].astype(np.uint8),
cmap=cm.Greys_r,aspect='equal',vmin=0,vmax=255)
plt.subplot(1,2,2)
plt.xticks([])
plt.yticks([])
plt.imshow(training_images[4].astype(np.uint8),
cmap=cm.Greys_r,aspect='equal',vmin=0,vmax=255)

plt.show()

def predicting(testing_files):
    for i in range(len(testing_files)):
        testi = imread(testing_files[i], 1)
        w, l = testi.shape
        test = np.empty((1, 2))
        test[0, 0] = w*l
        test[0, 1] = np.std(testi)
        print(kmeans.predict(test))

predicting(testing_files)
#
#
