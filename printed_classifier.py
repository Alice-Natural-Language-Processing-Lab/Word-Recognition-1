# Sample code for AI Coursework
# Extracts the letters from a printed page.

# Keep in mind that after you extract each letter, you have to normalise the size.
# You can do that by using scipy.imresize. It is a good idea to train your classifiers
# using a constast size (for example 20x20 pixels)
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



# function to preprocess the image
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

    # fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(7, 7))
    # ax.imshow(bw, cmap='jet')


    # array initialised with length as zero in order to avoid duplication with append
    # 20x20 respecting the image normalised width.
    letter_images = np.empty((0,20,20))

    letters = list()
    order = list()

    for region in regionprops(label_image):
        minc, minr, maxc, maxr = region.bbox
            # skip small images
        if maxc - minc > len(image)/250: # better to use height rather than area.
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=2)
            order.append(region.bbox)


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
    raw = list()

    prev_tr = 0
    prev_line_br = 0

    for i in range(len(lines)):
        for j in range(len(lines[i])):
            tl_2 = lines[i][j][1]
            bl_2 = lines[i][j][0]
            if tl_2 > prev_tr and bl_2 > prev_line_br:
                tl,tr,bl,br = lines[i][j]
                letter_raw = label_image[tl:bl,tr:br]
                letter_norm = imresize(letter_raw ,(20 ,20))
                raw.append(letter_raw)
                final.append(letter_norm)
                prev_tr = lines[i][j][3]
            if j == (len(lines[i])-1):
                prev_line_br = lines[i][j][2]
        prev_tr = 0
        tl_2 = 0

    letter_train = np.empty_like(final)
    letter_train[:] = final

    # plt.imshow(final[0],cmap=cm.Greys_r,aspect='equal')
    # print(final[0].shape)
    #
    # plt.ylabel('Length')
    # plt.xlabel('Width')
    # plt.show()




    return letter_train;

def target_letters(order):

    # array full of A objects as target
    # 1026 A objects passed
    letter_target = np.full((220, 1), order)
    letter_target = to_categorical(letter_target, 37)

    return letter_target

def target_letterss(order):

    # array full of A objects as target
    # 1026 A objects passed
    letter_target = np.full((40, 1), order)
    letter_target = to_categorical(letter_target, 37)

    return letter_target


# image = imread('./ocr/testing/shazam.png',1)
imageA = 'letters-fonts/letters-fonts-01.png'
imageB = 'letters-fonts/letters-fonts-02.png'
imageC = 'letters-fonts/letters-fonts-03.png'
imageD = 'letters-fonts/letters-fonts-04.png'
imageE = 'letters-fonts/letters-fonts-05.png'
imageF = 'letters-fonts/letters-fonts-06.png'
imageG = 'letters-fonts/letters-fonts-07.png'
imageH = 'letters-fonts/letters-fonts-08.png'
imageI = 'letters-fonts/letters-fonts-09.png'
imageJ = 'letters-fonts/letters-fonts-10.png'
imageK = 'letters-fonts/letters-fonts-11.png'
imageL = 'letters-fonts/letters-fonts-12.png'
imageM = 'letters-fonts/letters-fonts-13.png'
imageN = 'letters-fonts/letters-fonts-14.png'
imageO = 'letters-fonts/letters-fonts-15.png'
imageP = 'letters-fonts/letters-fonts-16.png'
imageQ = 'letters-fonts/letters-fonts-17.png'
imageR = 'letters-fonts/letters-fonts-18.png'
imageS = 'letters-fonts/letters-fonts-19.png'
imageT = 'letters-fonts/letters-fonts-20.png'
imageU = 'letters-fonts/letters-fonts-21.png'
imageV = 'letters-fonts/letters-fonts-22.png'
imageW = 'letters-fonts/letters-fonts-23.png'
imageX = 'letters-fonts/letters-fonts-24.png'
imageY = 'letters-fonts/letters-fonts-25.png'
imageZ = 'letters-fonts/letters-fonts-26.png'


imageA2 = 'letters-fonts/letters-fonts-27.png'
imageB2 = 'letters-fonts/letters-fonts-28.png'
imageD2 = 'letters-fonts/letters-fonts-29.png'
imageE2 = 'letters-fonts/letters-fonts-30.png'
imageF2 = 'letters-fonts/letters-fonts-31.png'
imageG2 = 'letters-fonts/letters-fonts-32.png'
imageH2 = 'letters-fonts/letters-fonts-33.png'
imageN2 = 'letters-fonts/letters-fonts-34.png'
imageQ2 = 'letters-fonts/letters-fonts-35.png'
imageR2 = 'letters-fonts/letters-fonts-36.png'
imageT2 = 'letters-fonts/letters-fonts-37.png'



imageAtest = 'letters-fonts-test/letters-fonts-test-01.png'
imageBtest = 'letters-fonts-test/letters-fonts-test-02.png'
imageCtest = 'letters-fonts-test/letters-fonts-test-03.png'
imageDtest = 'letters-fonts-test/letters-fonts-test-04.png'
imageEtest = 'letters-fonts-test/letters-fonts-test-05.png'
imageFtest = 'letters-fonts-test/letters-fonts-test-06.png'
imageGtest = 'letters-fonts-test/letters-fonts-test-07.png'
imageHtest = 'letters-fonts-test/letters-fonts-test-08.png'
imageItest = 'letters-fonts-test/letters-fonts-test-09.png'
imageJtest = 'letters-fonts-test/letters-fonts-test-10.png'
imageKtest = 'letters-fonts-test/letters-fonts-test-11.png'
imageLtest = 'letters-fonts-test/letters-fonts-test-12.png'
imageMtest = 'letters-fonts-test/letters-fonts-test-13.png'
imageNtest = 'letters-fonts-test/letters-fonts-test-14.png'
imageOtest = 'letters-fonts-test/letters-fonts-test-15.png'
imagePtest = 'letters-fonts-test/letters-fonts-test-16.png'
imageQtest = 'letters-fonts-test/letters-fonts-test-17.png'
imageRtest = 'letters-fonts-test/letters-fonts-test-18.png'
imageStest = 'letters-fonts-test/letters-fonts-test-19.png'
imageTtest = 'letters-fonts-test/letters-fonts-test-20.png'
imageUtest = 'letters-fonts-test/letters-fonts-test-21.png'
imageVtest = 'letters-fonts-test/letters-fonts-test-22.png'
imageWtest = 'letters-fonts-test/letters-fonts-test-23.png'
imageXtest = 'letters-fonts-test/letters-fonts-test-24.png'
imageYtest = 'letters-fonts-test/letters-fonts-test-25.png'
imageZtest = 'letters-fonts-test/letters-fonts-test-26.png'

imageA2test = 'letters-fonts-test/letters-fonts-test-27.png'
imageB2test = 'letters-fonts-test/letters-fonts-test-28.png'
imageD2test = 'letters-fonts-test/letters-fonts-test-29.png'
imageE2test = 'letters-fonts-test/letters-fonts-test-30.png'
imageF2test = 'letters-fonts-test/letters-fonts-test-31.png'
imageG2test = 'letters-fonts-test/letters-fonts-test-32.png'
imageH2test = 'letters-fonts-test/letters-fonts-test-33.png'
imageN2test = 'letters-fonts-test/letters-fonts-test-34.png'
imageQ2test = 'letters-fonts-test/letters-fonts-test-35.png'
imageR2test = 'letters-fonts-test/letters-fonts-test-36.png'
imageT2test = 'letters-fonts-test/letters-fonts-test-37.png'

trainA, targetA = extract_letters(imageA), target_letters(0)
trainB, targetB = extract_letters(imageB), target_letters(1)
trainC, targetC = extract_letters(imageC), target_letters(2)
trainD, targetD = extract_letters(imageD), target_letters(3)
trainE, targetE = extract_letters(imageE), target_letters(4)
trainF, targetF = extract_letters(imageF), target_letters(5)
trainG, targetG = extract_letters(imageG), target_letters(6)
trainH, targetH = extract_letters(imageH), target_letters(7)
trainI, targetI = extract_letters(imageI), target_letters(8)
trainJ, targetJ = extract_letters(imageJ), target_letters(9)
trainK, targetK = extract_letters(imageK), target_letters(10)
trainL, targetL = extract_letters(imageL), target_letters(11)
trainM, targetM = extract_letters(imageM), target_letters(12)
trainN, targetN = extract_letters(imageN), target_letters(13)
trainO, targetO = extract_letters(imageO), target_letters(14)
trainP, targetP = extract_letters(imageP), target_letters(15)
trainQ, targetQ = extract_letters(imageQ), target_letters(16)
trainR, targetR = extract_letters(imageR), target_letters(17)
trainS, targetS = extract_letters(imageS), target_letters(18)
trainT, targetT = extract_letters(imageT), target_letters(19)
trainU, targetU = extract_letters(imageU), target_letters(20)
trainV, targetV = extract_letters(imageV), target_letters(21)
trainW, targetW = extract_letters(imageW), target_letters(22)
trainX, targetX = extract_letters(imageX), target_letters(23)
trainY, targetY = extract_letters(imageY), target_letters(24)
trainZ, targetZ = extract_letters(imageZ), target_letters(25)


trainA2, targetA2 = extract_letters(imageA2), target_letters(26)
trainB2, targetB2 = extract_letters(imageB2), target_letters(27)
trainD2, targetD2 = extract_letters(imageD2), target_letters(28)
trainE2, targetE2 = extract_letters(imageE2), target_letters(29)
trainF2, targetF2 = extract_letters(imageF2), target_letters(30)
trainG2, targetG2 = extract_letters(imageG2), target_letters(31)
trainH2, targetH2 = extract_letters(imageH2), target_letters(32)
trainN2, targetN2 = extract_letters(imageN2), target_letters(33)
trainQ2, targetQ2 = extract_letters(imageQ2), target_letters(34)
trainR2, targetR2 = extract_letters(imageR2), target_letters(35)
trainT2, targetT2 = extract_letters(imageT2), target_letters(36)



testA, targetAtest = extract_letters(imageAtest), target_letterss(0)
testB, targetBtest = extract_letters(imageBtest), target_letterss(1)
testC, targetCtest = extract_letters(imageCtest), target_letterss(2)
testD, targetDtest = extract_letters(imageDtest), target_letterss(3)
testE, targetEtest = extract_letters(imageEtest), target_letterss(4)
testF, targetFtest = extract_letters(imageFtest), target_letterss(5)
testG, targetGtest = extract_letters(imageGtest), target_letterss(6)
testH, targetHtest = extract_letters(imageHtest), target_letterss(7)
testI, targetItest = extract_letters(imageItest), target_letterss(8)
testJ, targetJtest = extract_letters(imageJtest), target_letterss(9)
testK, targetKtest = extract_letters(imageKtest), target_letterss(10)
testL, targetLtest = extract_letters(imageLtest), target_letterss(11)
testM, targetMtest = extract_letters(imageMtest), target_letterss(12)
testN, targetNtest = extract_letters(imageNtest), target_letterss(13)
testO, targetOtest = extract_letters(imageOtest), target_letterss(14)
testP, targetPtest = extract_letters(imagePtest), target_letterss(15)
testQ, targetQtest = extract_letters(imageQtest), target_letterss(16)
testR, targetRtest = extract_letters(imageRtest), target_letterss(17)
testS, targetStest = extract_letters(imageStest), target_letterss(18)
testT, targetTtest = extract_letters(imageTtest), target_letterss(19)
testU, targetUtest = extract_letters(imageUtest), target_letterss(20)
testV, targetVtest = extract_letters(imageVtest), target_letterss(21)
testW, targetWtest = extract_letters(imageWtest), target_letterss(22)
testX, targetXtest = extract_letters(imageXtest), target_letterss(23)
testY, targetYtest = extract_letters(imageYtest), target_letterss(24)
testZ, targetZtest = extract_letters(imageZtest), target_letterss(25)

testA2, targetA2test = extract_letters(imageA2test), target_letterss(26)
testB2, targetB2test = extract_letters(imageB2test), target_letterss(27)
testD2, targetD2test = extract_letters(imageD2test), target_letterss(28)
testE2, targetE2test = extract_letters(imageE2test), target_letterss(29)
testF2, targetF2test = extract_letters(imageF2test), target_letterss(30)
testG2, targetG2test = extract_letters(imageG2test), target_letterss(31)
testH2, targetH2test = extract_letters(imageH2test), target_letterss(32)
testN2, targetN2test = extract_letters(imageN2test), target_letterss(33)
testQ2, targetQ2test = extract_letters(imageQ2test), target_letterss(34)
testR2, targetR2test = extract_letters(imageR2test), target_letterss(35)
testT2, targetT2test = extract_letters(imageT2test), target_letterss(36)


# for i in range(len(trainI)):
#     print(trainI[i])


train = np.concatenate((trainA, trainB, trainC, trainD, trainE, trainF, trainG, trainH,trainI,trainJ,  trainK, trainL, trainM,
                        trainN, trainO, trainP, trainQ, trainR, trainS, trainT, trainU, trainV, trainW, trainX, trainY, trainZ,
                        trainA2, trainB2, trainD2, trainE2, trainF2, trainG2, trainH2, trainN2, trainQ2, trainR2, trainT2))

train_target = np.concatenate((targetA, targetB, targetC, targetD, targetE, targetF, targetG, targetH, targetI,targetJ,
                                targetK, targetL, targetM, targetN, targetO, targetP, targetQ, targetR, targetS, targetT,
                                targetU, targetV, targetW, targetX, targetY, targetZ, targetA2, targetB2, targetD2, targetE2,
                                targetF2, targetG2, targetH2, targetN2, targetQ2, targetR2, targetT2))


test = np.concatenate((testA, testB, testC, testD, testE, testF, testG, testH,testI,testJ,  testK, testL, testM, testN,
                        testO, testP, testQ, testR, testS, testT, testU, testV, testW, testX, testY, testZ, testA2, testB2, testD2,
                        testE2, testF2, testG2, testH2, testN2, testQ2, testR2, testT2))

test_target = np.concatenate((targetAtest, targetBtest, targetCtest, targetDtest, targetEtest, targetFtest, targetGtest,
                                targetHtest, targetItest,targetJtest, targetKtest, targetLtest, targetMtest,  targetNtest,
                                targetOtest, targetPtest, targetQtest, targetRtest, targetStest, targetTtest, targetUtest,
                                targetVtest, targetWtest, targetXtest, targetYtest, targetZtest, targetA2test, targetB2test,
                                targetD2test, targetE2test, targetF2test, targetG2test, targetH2test, targetN2test, targetQ2test,
                                targetR2test,targetT2test))

train = train.reshape(-1,20,20,1)
train = train/255.

test = test.reshape(-1,20,20,1)
test = test/255.

print(trainA.shape)
print(targetA.shape)

print(targetA)

print(trainA2.shape)
print(targetA2.shape)


def create_model2():
    model = Sequential()
    model.add(Conv2D(20, kernel_size=3, padding='same',activation="relu", input_shape=(20,20,1)))
    model.add(Conv2D(40, kernel_size=5, padding='same',activation="relu"))
    model.add(Conv2D(40, kernel_size=3, padding='same',activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(370))
    model.add(Dropout(0.25))
    model.add(Dense(37, activation="softmax"))

    return model

def create_model():
    model = Sequential()
    model.add(Conv2D(20, kernel_size=3, padding='same',activation="relu", input_shape=(20,20,1)))
    model.add(Conv2D(40, kernel_size=5, padding='same',activation="relu"))
    model.add(Conv2D(40, kernel_size=3, padding='same',activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(200))
    model.add(Dropout(0.25))
    model.add(Dense(37, activation="softmax"))

    return model

opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)
opt2 = keras.optimizers.rmsprop(lr=0.003, decay=1e-6)


from keras.utils import plot_model


model = create_model()
model2 = create_model2()

plot_model(model, to_file='model.png')

# model2.summary()
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
# model2.compile(optimizer=opt2, loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train, train_target,validation_data=(test, test_target), epochs=5, batch_size=20)
# history2 = model2.fit(train, train_target,validation_data=(test, test_target), epochs=4, batch_size=20)

model.save('printed_recogniser.h5')


# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

model = load_model('printed_recogniser.h5')
folder_string = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt'
folder_string = 'abcdefghijklmnopqrstuvwxyzABDEFGHNQRT'


im = extract_letters('ocr/testing/shazam.png')
fr = np.empty_like(im)

fr[::] = im

fr = fr.reshape(-1,20,20,1)
fr = fr/255.

ans = list()

for i in range(len(fr)):
    prediction = model.predict(fr)[i]

    most_conf_index = np.argmax(prediction)
    answer_confidence = prediction[most_conf_index]

    print("Model classified image as", str(folder_string[most_conf_index]), "with", answer_confidence,"confidence")
    ans.append(folder_string[most_conf_index])




print(ans)


# display the letter A image
# plt.imshow(,cmap=cm.Greys_r,aspect='equal')
# plt.show()
