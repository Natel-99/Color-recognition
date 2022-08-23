from keras.layers import Input, Add, Activation, ZeroPadding2D, BatchNormalization, Flatten, AveragePooling2D
from keras.models import Model
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import pickle

ROWS = 64
COLS = 64
CHANNELS = 3
def load_data():
    print("Bắt đầu load dữ liệu từ file....")
    file = open('dataA.pkl', 'rb')
    (pixels, labels) = pickle.load(file)
    file.close()

    print("Đã load xong dữ liệu từ file. Kích thước input và output:")
    print(pixels.shape)
    print(labels.shape)

    num_class = labels.shape[1]

    return pixels, labels, num_class
print("Bắt đầu thực hiện chia dữ liệu train,test....")
X, y, num_class = load_data()
CLASSES = num_class
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Đã chia xong dữ liệu. Kích thước dữ liệu train:")
print(X_train.shape)
print(y_train.shape)
def identity_block(X, f, filters):
    # Retrieve Filters
    F1, F2, F3 = filters
    # Save the input value. We'll need this later to add back to the main path. 
    X_shortcut = X
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1),padding = 'valid')(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)
    # Second component of main path
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1),padding = 'same')(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)
    # Third component of main path
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1),padding = 'valid')(X)
    X = BatchNormalization(axis = 3)(X)
    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    return X


def convolutional_block(X, f, filters, s=2):
    # Retrieve Filters
    F1, F2, F3 = filters
    # Save the input value
    X_shortcut = X
    ##### MAIN PATH #####
    # First component of main path
    X = Conv2D(F1, (1, 1), strides=(s, s))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    # Second component of main path
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    # Third component of main path
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid')(X)
    X = BatchNormalization(axis=3)(X)
    ##### SHORTCUT PATH ####
    X_shortcut = Conv2D(F3, (1, 1), strides=(s, s))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3)(X_shortcut)
    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    return X


def ResNet50(input_shape=(64, 64, 3), classes=2):
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)
    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    # Stage 1
    X = Conv2D(64, (7, 7), strides=(2, 2))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)
    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], s=1)
    X = identity_block(X, 3, [64, 64, 256])
    X = identity_block(X, 3, [64, 64, 256])
    # Stage 3
    X = convolutional_block(X, f=3, filters=[128, 128, 512], s=2)
    X = identity_block(X, 3, [128, 128, 512])
    X = identity_block(X, 3, [128, 128, 512])
    X = identity_block(X, 3, [128, 128, 512])
    # Stage 4
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], s=2)
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    # Stage 5
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], s=2)
    X = identity_block(X, 3, [512, 512, 2048])
    X = identity_block(X, 3, [512, 512, 2048])
    # AVGPOOL.
    X = AveragePooling2D((2, 2))(X)
    # Lớp đầu ra
    X = Flatten()(X)
    X = Dense(classes, activation='softmax')(X)
    # Tạo model mới
    model = Model(inputs=X_input, outputs=X)
    return model


model = ResNet50(input_shape = (ROWS, COLS, CHANNELS), classes = CLASSES)
#adam = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
filepath = 'color_weightsA.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
train_datagen =ImageDataGenerator(rotation_range=20, zoom_range=0.1,
                         rescale=1,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         horizontal_flip=True,
                         brightness_range=[0.2, 1.5], fill_mode="nearest")
test_datagen = ImageDataGenerator(rescale=1)

model.fit_generator(train_datagen.flow(X_train, y_train, batch_size=16),
                                 epochs=100,
                                 validation_data=train_datagen.flow(X_test, y_test,batch_size=len(X_test)),
                                 callbacks=callbacks_list)
model.save('model_categorical_complex_7_7.h5')
print('da train xong')