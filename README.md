### Residual Networkを使って音響信号処理モデルを作りました！

コード：[google colabで開く](https://colab.research.google.com/drive/1hEXoFEK_z_HRq-quTmOZVT62IMXy70Go?usp=sharing)<br>
※google colabで開く場合は以下のファイルをgoogleドライブ上「drive/My Drive/Colab Notebooks/traindata.csv」「drive/My Drive/Colab Notebooks/testdata.csv」にアップロードしてください。
<br>
<br>
訓練用データ：[googleスプレッドシートで開く](https://drive.google.com/file/d/1zlQFh_jN4yOs188_GQ0hY4pLB5UlSAxv/view?usp=sharing)<br>
検証用データ：[googleスプレッドシートで開く](https://drive.google.com/file/d/1qlaZw2J2fjxv5pVTxT4Y_CRjFF7vSukO/view?usp=sharing)<br>


### 概要

#### CNN(畳み込みニューラルネットワーク)とResNet(Residual Network)について簡単に説明

CNNとは、畳み込みニューラルネットワークのことで、主に画像認識に特化したニューラルネットワークです。画像をn行m列の2次元配列とすると、2次元配列を入力としてニューラルネットワークに読み込むには、1次元配列に変形する必要があります。しかし、画像のデータすべて、つまり配列の要素すべてを入力してしまうと、パラメータが膨大な数になり、過学習を引き起こしやすくなります。そこで、まず入力画像に対して畳み込みを行います。畳み込みとは、画像に小さいサイズのフィルタを当てたり、プーリングを行うことで、画像を大まかに認識していくことです。<br>
今回はさらに精度の高いモデルを作るために、CNNではセットでよく用いられるResidual Networkを実装しました。
Resnetとは、畳み込み層へ入力させるものと何もしないものの二つに入力を分割し、畳み込み層からの出力と何もしなかった入力を足し合わせる手法です。これにより、勾配降下法における勾配消失が起きにくくなり、またランダムサンプリング効果もあるため過学習が起きにくくなります。<br>

#### 音声信号データについて簡単に説明

今回は以下のような8種類の音声データを使用します。<br>

(例)
||波形|スペクトログラム|
|:---:|:---:|:---:|
|サンプル1|<img src="https://uploda3.ysklog.net/482c0137098d111e20a94c6f469f5030.png" width="300">|<img src="https://uploda2.ysklog.net/d4521ff2d22eee6d1ef55babfa7901fe.png" width="300">|
|サンプル2|<img src="https://uploda3.ysklog.net/86f6b3573257bd89737f6eaaf2e4c76d.png" width="300">|<img src="https://uploda2.ysklog.net/53d5a8100f3b331b30b312b268ed77a8.png" width="300">|
|サンプル3|<img src="https://uploda3.ysklog.net/d9af6850f38b0391bef5a605dee83aff.png" width="300">|<img src="https://uploda2.ysklog.net/ffdee3669860d8cc085c30add1aedd57.png" width="300">|
<表1 : 音データの波形とスペクトログラムを3つ紹介><br>

上記のような8種類のスペクトログラムをそれぞれ(20×50)の配列データに変換し、さらにそれらを(1000×1)の配列データに変形させます。<br>

[訓練用データ](https://drive.google.com/file/d/1zlQFh_jN4yOs188_GQ0hY4pLB5UlSAxv/view?usp=sharing)には、先頭列に音の種類のラベル(1～8)とそれ以降の列に配列データ(1000×1)が、各音の種類に対して90個ずつ、合計720個のデータが保存されています。(720×1001)<br>
<img src="https://uploda1.ysklog.net/a75e5af2acaf358a3a07560f91d276f7.png" width="800"><br>
<図1 : 訓練用データのデータフレーム><br>

[検証用データ](https://drive.google.com/file/d/1qlaZw2J2fjxv5pVTxT4Y_CRjFF7vSukO/view?usp=sharing)には、先頭列に音の種類のラベル(1～8)とそれ以降の列に配列データ(1000×1)が、各音の種類に対して10個ずつ、合計80個のデータが保存されています。(80×1001)<br>
<img src="https://uploda1.ysklog.net/1e9d2ab4fe3c459d9aab7aedebbd4db5.png" width="800"><br>
<図2 : 検証用データのデータフレーム><br>
<br>


### データ処理

今回はCNNを使って上記の音声データを認識するため、まず1001カラムの配列データを1列の正解ラベルと(20×50×1)の入力データに変形します。<br>
※今回は音声データでRGBなどは関係ないためチャンネル数は1にします。<br>
<br>

### Residual Blockの作成

```
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
```
<図3 : Residual Block実装コード><br>

このResidualBlockは、入力inputsに対して畳み込み層conv1,conv2,conv3によって畳み込みされた_residualと、畳み込みされていない_shortcutを合成したoutputsを返すクラスです。<br>
またidentityがTrueの時はinputsがそのまま_shortcutとなるが、Falseの時は_residualと合成可能にする形にするために、inputsは一度畳み込み層skip_convによって畳み込みされて_shortcutとなります。<br>
<br>

```
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

model.summary()
```
<図4 : CNNモデルの実装><br>

ResidualBlockを並べ、出力されたものをGlobalAveragePooling用いて1次元配列とし、活性化関数softmaxの出力層から出力します。
最適化アルゴリズムにはAdam、
モデル評価のための損失関数は多クラス交差エントロピー(categorical cross entropy)を設定しました。<br>
複雑に見えますが、パラメータの数は約60万個と深層学習モデルにしては計算量は抑えられています。<br>
<br>

### 学習データを入力

CNNでは画像を上下左右に反転させるなどして、より正確に画像を認識させるオーグメンテーションという処理があります。今回はオーグメンテーションを行うのに便利なImageDataGeneratorという関数があるため、それを使用しました。また、LearningRateSchedulerを使うことで、学習が進むほど学習率を下げるという処理も行い、より早く最適解を見つけるように施しました。

