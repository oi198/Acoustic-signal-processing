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
|サンプル2|<img src="https://uploda3.ysklog.net/86f6b3573257bd89737f6eaaf2e4c76d.png" width="300">|<img src="https://uploda2.ysklog.net/53d5a8100f3b331b30b312b268ed77a8.png
" width="300">|
|サンプル3|<img src="https://uploda3.ysklog.net/d9af6850f38b0391bef5a605dee83aff.png" width="300">|<img src="https://uploda2.ysklog.net/ffdee3669860d8cc085c30add1aedd57.png" width="300">|
