### Residual Networkを使って音響信号処理モデルを作りました！

コード：[google colabで開く](https://colab.research.google.com/drive/1hEXoFEK_z_HRq-quTmOZVT62IMXy70Go?usp=sharing)<br>
※google colabで開く場合は以下のファイルをgoogleドライブ上「drive/My Drive/Colab Notebooks/traindata.csv」「drive/My Drive/Colab Notebooks/testdata.csv」にアップロードしてください。
<br>
<br>
訓練用データ：[googleスプレッドシートで開く](https://drive.google.com/file/d/1zlQFh_jN4yOs188_GQ0hY4pLB5UlSAxv/view?usp=sharing)<br>
検証用データ：[googleスプレッドシートで開く](https://drive.google.com/file/d/1qlaZw2J2fjxv5pVTxT4Y_CRjFF7vSukO/view?usp=sharing)<br>


### 概要

以下のような8種類の音声データを認識し、分類するモデルを作成します。<br>

(例)
||波形|スペクトログラム|
|:---:|:---:|:---:|
|サンプル1|<img src="https://uploda3.ysklog.net/482c0137098d111e20a94c6f469f5030.png" width="300">|<img src="https://uploda2.ysklog.net/d4521ff2d22eee6d1ef55babfa7901fe.png" width="300">|
|サンプル2|<img src="https://uploda3.ysklog.net/86f6b3573257bd89737f6eaaf2e4c76d.png" width="300">|<img src="https://uploda2.ysklog.net/53d5a8100f3b331b30b312b268ed77a8.png" width="300">|
|サンプル3|<img src="https://uploda3.ysklog.net/d9af6850f38b0391bef5a605dee83aff.png" width="300">|<img src="https://uploda2.ysklog.net/ffdee3669860d8cc085c30add1aedd57.png" width="300">|

上記のような8種類のスペクトログラムをそれぞれ(20×50)の配列データに変換し、さらにそれらを(1000×1)の配列データに変形させます。<br>

[訓練用データ](https://drive.google.com/file/d/1zlQFh_jN4yOs188_GQ0hY4pLB5UlSAxv/view?usp=sharing)には、先頭列に音の種類のラベル(1～8)とそれ以降の列に配列データ(1000×1)が、各音の種類に対して90個ずつ、合計720個のデータが保存されています。(720×1001)<br>
<img src="https://uploda1.ysklog.net/a75e5af2acaf358a3a07560f91d276f7.png" width="800"><br>

[検証用データ](https://drive.google.com/file/d/1qlaZw2J2fjxv5pVTxT4Y_CRjFF7vSukO/view?usp=sharing)には、先頭列に音の種類のラベル(1～8)とそれ以降の列に配列データ(1000×1)が、各音の種類に対して10個ずつ、合計80個のデータが保存されています。(80×1001)<br>
<img src="https://uploda1.ysklog.net/1e9d2ab4fe3c459d9aab7aedebbd4db5.png" width="800"><br>
