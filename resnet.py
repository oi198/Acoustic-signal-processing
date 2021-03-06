from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Layer, BatchNormalization
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Flatten, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler

batch_size = 16
num_classes = 8
epochs = 10
lr = 0.01

df1 = pd.read_csv('drive/My Drive/Colab Notebooks/traindata.csv',header=None)
df2 = pd.read_csv('drive/My Drive/Colab Notebooks/testdata.csv',header=None)

y_train = df1.iloc[:,[0]]
x_train = df1.iloc[:,1:]

y_test = df2.iloc[:,[0]]
x_test = df2.iloc[:,1:]

x_train = x_train.to_numpy().reshape(720, 20,50,1)
x_test = x_test.to_numpy().reshape(80, 20,50,1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

class ResidualBlock(Layer):
  def __init__(self, filters, strides, identity=True):
    super(ResidualBlock, self).__init__()
    self.identity = identity

    # 必要なレイヤーを事前定義
    self.conv1 = Conv2D(filters // 4, (1, 1), strides=strides, padding='same', kernel_initializer='he_normal')
    self.bn1 = BatchNormalization()

    self.conv2 = Conv2D(filters // 4, (3, 3), padding='same', kernel_initializer='he_normal')
    self.bn2 = BatchNormalization()

    self.conv3 = Conv2D(filters, (1, 1), padding='same', kernel_initializer='he_normal')
    self.bn3 = BatchNormalization()

    # 出力のチャネル数やシェイプが途中で変化する場合の調整用の畳み込み層
    if not self.identity:
      self.skip_conv = Conv2D(filters, (1, 1), strides=strides, padding='same', kernel_initializer='he_normal')

  def call(self, inputs):

    # residual path
    x = self.conv1(inputs)
    x = self.bn1(x)
    x = Activation('relu')(x)

    x = self.conv2(x)
    x = self.bn2(x)
    x = Activation('relu')(x)

    x = self.conv3(x)
    _residual = self.bn3(x)

    # shortcut path
    if self.identity:
      _shortcut = inputs
    else:
      _shortcut =self.skip_conv(inputs)

    outputs = _residual + _shortcut
    return outputs

# ResNet56の実装

block_nums = 6

model = Sequential()
model.add(Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal', input_shape=(20,50,1)))

# depthが64のResidualBlockの塊。block_nums個のResidualBlockで構成される
model.add(ResidualBlock(64, strides=(1, 1), identity=False))
for _ in range(block_nums - 1):
  model.add(ResidualBlock(64, strides=(1, 1), identity=True))

# 同様にdepthが128のResidualBlockの塊
model.add(ResidualBlock(128, strides=(2, 2), identity=False))
for _ in range(block_nums - 1):
  model.add(ResidualBlock(128, strides=(1, 1), identity=True))

# 同様にdepthが256のResidualBlockの塊
model.add(ResidualBlock(256, strides=(2, 2), identity=False))
for _ in range(block_nums - 1):
  model.add(ResidualBlock(256, strides=(1, 1), identity=True))

model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(GlobalAveragePooling2D())
model.add(Dense(8, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'])

#model.summary()

datagen = ImageDataGenerator(  
    height_shift_range=4,
    width_shift_range=4,
    horizontal_flip=True)

model.fit_generator(
    datagen.flow(x_train, y=y_train, batch_size=batch_size, shuffle=True),
    steps_per_epoch=(x_train.shape[0] // batch_size),
    epochs=epochs,
    verbose=1,
    validation_data=(x_test, y_test),
    callbacks=[
        LearningRateScheduler(lambda epoch: float(lr / 3 ** (epoch * 4 // epochs)))  
    ])

model.save('/content/drive/My Drive/model2.h5')
