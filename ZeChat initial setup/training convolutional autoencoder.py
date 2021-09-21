
import os
import glob
import cv2

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation
from keras.models import Model
from keras.models import model_from_json

import numpy as np
import matplotlib.pyplot as plt



os.chdir(r'C:\Folder\Containing\Preprocessed\Images') # folder containing preprocessed images

temp = []

img_dim = 56

for img_name in glob.glob('C:/Folder/Containing/Preprocessed/Images/*.jpg'): # folder containing preprocessed images
    img = cv2.imread(img_name, cv2.IMREAD_COLOR)
    if not np.count_nonzero(img):
        print("empty image")
        continue
    img = cv2.resize(img, (img_dim, img_dim))
    img = img.astype('float32') / 255.
    img = np.array(img)
    temp.append(img)

print(img)


temp_data = np.stack(temp)


input_img = Input(shape=(img_dim, img_dim, 3))

encoded = Conv2D(64, (3, 3), padding='same')(input_img)
encoded = BatchNormalization()(encoded)
encoded = Activation('relu')(encoded)
encoded = MaxPooling2D((2, 2), padding='same')(encoded)

encoded = Conv2D(32, (3, 3), padding='same')(encoded)
encoded = BatchNormalization()(encoded)
encoded = Activation('relu')(encoded)
encoded = MaxPooling2D((2, 2), padding='same')(encoded)

encoded = Conv2D(16, (3, 3), padding='same')(encoded)
encoded = BatchNormalization()(encoded)
encoded = Activation('relu')(encoded)
encoded = MaxPooling2D((2, 2), padding='same')(encoded)

decoded = Conv2D(16, (3, 3), padding='same')(encoded)
decoded = BatchNormalization()(decoded)
decoded = Activation('relu')(decoded)
decoded = UpSampling2D((2, 2))(decoded)

decoded = Conv2D(32, (3, 3), padding='same')(decoded)
decoded = BatchNormalization()(decoded)
decoded = Activation('relu')(decoded)
decoded = UpSampling2D((2, 2))(decoded)

decoded = Conv2D(64, (3, 3), padding='same')(decoded)
decoded = BatchNormalization()(decoded)
decoded = Activation('relu')(decoded)
decoded = UpSampling2D((2, 2))(decoded)

decoded = Conv2D(3, (3, 3), padding='same')(decoded)
decoded = BatchNormalization()(decoded)
decoded = Activation('sigmoid')(decoded)


encoder = Model(input_img, encoded)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

batch_size = 32
epochs = 50

autoencoder_train = autoencoder.fit(temp_data, temp_data,
                                    epochs=epochs,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    validation_split=0.2)

encoded_imgs = encoder.predict(temp_data)
decoded_imgs = autoencoder.predict(temp_data)

autoencoder.summary()

plt.figure(figsize=(20, 6))

rows = 3
columns = 10

nudge = 10

for i in range(10):
    # original
    plt.subplot(rows, columns, i + 1)
    plt.imshow(temp_data[i+nudge].reshape(img_dim, img_dim, 3))
    plt.axis('off')

    # encoded
    plt.subplot(rows, columns, i + 1 + 10)
    plt.imshow(encoded_imgs[i+nudge].reshape(28, 28))
    plt.gray()
    plt.axis('off')
    
    # reconstruction
    plt.subplot(rows, columns, i + 1 + 20)
    plt.imshow(decoded_imgs[i+nudge].reshape(img_dim, img_dim, 3))
    plt.axis('off')

plt.tight_layout()
plt.savefig('autoencoder.png')


loss = autoencoder_train.history['loss']
val_loss = autoencoder_train.history['val_loss']
epochs = range(epochs)
plt.figure(figsize=(10, 10))
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('autoencoder_loss.png')


model_json = autoencoder.to_json()
with open("autoencoder model.json", "w") as json_file:
    json_file.write(model_json)
autoencoder.save_weights("autoencoder model.h5")
print("Saved autoencoder model to disk")


model_json = encoder.to_json()
with open("encoder model.json", "w") as json_file:
    json_file.write(model_json)
encoder.save_weights("encoder model.h5")
print("Saved encoder model to disk")














