## ResNet50バックボーンの構築
RetinaNet は ResNet ベースのバックボーンを使用し、それを利用して特徴ピラミッドネットワークが構築されます。このサンプルでは、バックボーンとして ResNet50 を使用し、そしてストライド 8, 16 と 32 で特徴マップを返します。

![image](https://github.com/user-attachments/assets/e55f7456-1a02-4173-aceb-a410ed122721)

![image](https://github.com/user-attachments/assets/d258ccfa-d973-4b92-a5ea-539f8accb5b0)
![image](https://github.com/user-attachments/assets/bf7d0865-c9d3-4469-8cfe-24669db46ea6)

#### 各処理ごとの意味と詳細
1. ResNet50の構築:
  - ResNet50は、深層学習の画像認識モデルで、多くの特徴を自動的に抽出する能力があります。事前学習済みのモデルを使用することで、トレーニング時間を短縮し、良い精度を得ることができます。
2. 中間層の出力を取得:
  - 各中間層から出力を取得することで、異なる解像度の特徴マップを得られ、マルチスケールの特徴を学習するのに役立ちます。
3. 新しいモデルの構築:
  - 新しいモデルは、入力画像に対して複数のスケールで特徴マップを抽出できるため、物体検出などのタスクで多用されます。

各処理ごとの入力の型、形状、出力の型、形状について詳しく解説

1. ResNet50の構築
  - 入力の型: tf.Tensor
  - 入力の形状: [batch_size, height, width, 3]
  - 出力の型: Kerasモデル
  - 出力の形状: [batch_size, height//32, width//32, channels]（最終的な出力層の形状）

2. 中間層の出力を取得
  - 入力の型: Kerasモデル
  - 入力の形状: None（内部的に取得される）
  - 出力の型: List[tf.Tensor]
  - 出力の形状: [batch_size, height//8, width//8, channels], [batch_size, height//16, width//16, channels], [batch_size, height//32, width//32, channels]

3. 新しいモデルの構築
  - 入力の型: tf.Tensor
  - 入力の形状: [batch_size, height, width, 3]
  - 出力の型: List[tf.Tensor]
  - 出力の形状: [batch_size, height//8, width//8, channels], [batch_size, height//16, width//16, channels], [batch_size, height//32, width//32, channels]





