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

train_data_path = 'emnist/emnist-balanced-train.csv'
test_data_path = 'emnist/emnist-balanced-test.csv'


num_classes = 47
img_size = 28


def img_label_load(data_path, num_classes=None):
    data = pd.read_csv(data_path, header=None)
    # data = pd.read_csv(data_path, header=None)[:20000]

    data_rows = len(data)
    if not num_classes:
        num_classes = len(data[0].unique())

    # this assumes square imgs. Should be 28x28
    img_size = int(np.sqrt(len(data.iloc[0][1:])))


    # Images need to be transposed. This line also does the reshaping needed.
    imgs = np.transpose(data.values[:,1:].reshape(data_rows, img_size, img_size, 1), axes=[0,2,1,3]) # img_size * img_size arrays

    labels = to_categorical(data.values[:,0], num_classes) # one-hot encoding vectors

    plt.imshow(data.values[3, 1:].reshape([28, 28]), cmap='Greys_r')
    plt.show()

    img_flip = np.transpose(data.values[3,1:].reshape(28, 28), axes=[1,0]) # img_size * img_size arrays
    plt.imshow(img_flip, cmap='Greys_r')

    plt.show()

    return imgs/255., labels

X, y = img_label_load(train_data_path)

print(X.shape)
print(y.shape)




# print(train_data[1])
# for i in range(len(train_labels)):
#     print(train_labels[i])

def create_model():
    model = Sequential()
    model.add(Conv2D(10, kernel_size=3, padding='same',activation="relu", input_shape=(28,28,1)))
    model.add(Conv2D(20, kernel_size=5, padding='same',activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(40, kernel_size=3, padding='same',activation="relu"))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(470))
    model.add(Dense(47, activation="softmax"))

    return model

def create_model2():
    model = Sequential()
    model.add(Conv2D(10, kernel_size=3, padding='same',activation="relu", input_shape=(28,28,1)))
    model.add(Conv2D(20, kernel_size=5, padding='same',activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(40, kernel_size=3, padding='same',activation="relu"))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(470))
    model.add(Dense(47, activation="softmax"))

    return model


model = create_model()

model.summary()

opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X[:100000], y[:100000],validation_data=(X[100000:],y[100000:]), epochs=3, batch_size=50)
# model2.fit(X[:100000], y[:100000],validation_data=(X[100000:],y[100000:]), epochs=3, batch_size=50)



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



model.save('my_model5.h5')

model = load_model('my_model5.h5')
folder_string = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt'

im = extract_letters('ahmed-1.png')
fr = np.empty_like(im)

fr[::] = im

fr = fr.reshape(-1,28,28,1)
fr = fr/255.

ans = list()

for i in range(len(fr)):
    prediction = model.predict(fr)[i]

    most_conf_index = np.argmax(prediction)
    answer_confidence = prediction[most_conf_index]

    print("Model classified image as", str(folder_string[most_conf_index]), "with", answer_confidence,"confidence")
    ans.append(folder_string[most_conf_index])
