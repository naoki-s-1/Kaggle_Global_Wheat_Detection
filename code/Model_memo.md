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


## 特徴ピラミッドネットワークをカスタム層として構築する

![image](https://github.com/user-attachments/assets/82c2f407-b22d-4bdc-b4a0-fd9e818f7434)

![image](https://github.com/user-attachments/assets/d55a65b3-61b3-4f37-bdb4-22d1ce153f0c)

call メソッド

- 処理内容:
  - このメソッドは、入力画像からバックボーン（ResNet50）を使用して得られた中間層の特徴マップを元に、FPNを構築します。
  - 
- 具体的な処理:
  1. c3_output, c4_output, c5_output = self.backbone(images, training=training)
    - 入力画像をバックボーンに通し、ResNet50の中間層から3つの特徴マップ（c3, c4, c5）を取得します。
  2. p3_output = self.conv_c3_1x1(c3_output)
    - c3_outputに対して1x1の畳み込みを適用し、チャンネル数を256に変換します。
  3. p4_output = self.conv_c4_1x1(c4_output)
    - 同様に、c4_outputに対して1x1の畳み込みを適用します。
  4. p5_output = self.conv_c5_1x1(c5_output)
    - c5_outputに対して1x1の畳み込みを適用します。
  5. p4_output = p4_output + self.upsample_2x(p5_output)
    - p5_outputをアップサンプリングして解像度を2倍にし、p4_outputに加算します。この操作により、解像度が異なる特徴マップを統合します。
  6. p3_output = p3_output + self.upsample_2x(p4_output)
    - 同様に、p4_outputをアップサンプリングしてp3_outputに加算します。
  7. p3_output = self.conv_c3_3x3(p3_output)
    - これらの統合された特徴マップに対して、3x3の畳み込みを適用して最終的なp3_outputを生成します。
  8. p4_output = self.conv_c4_3x3(p4_output)
    - 同様に、p4_outputに3x3の畳み込みを適用します。
  9. p5_output = self.conv_c5_3x3(p5_output)
    - p5_outputに3x3の畳み込みを適用します。
  10. p6_output = self.conv_c6_3x3(c5_output)
    - c5_outputを使用して、新しい特徴マップp6_outputを生成します。この操作は2xダウンサンプリングを伴います。
  11. p7_output = self.conv_c7_3x3(tf.nn.relu(p6_output))
    - p6_outputにReLUを適用して活性化し、その後3x3の畳み込みを行い、最終的なp7_outputを生成します。

- 入力の型と形状:
  - 入力の型: tf.Tensor
  - 入力の形状: [batch_size, height, width, 3]

- 出力の型と形状:
  - 出力の型: Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]
  - 出力の形状:
    - p3_output: [batch_size, height//8, width//8, 256]
      - p4_output: [batch_size, height//16, width//16, 256]
      - p5_output: [batch_size, height//32, width//32, 256]
      - p6_output: [batch_size, height//64, width//64, 256]
      - p7_output: [batch_size, height//128, width//128, 256]

各処理ごとの意味と詳細
1. ResNet50バックボーンからの特徴抽出 (self.backbone):
  - ResNet50から特徴マップを抽出します。これらは物体検出やセマンティックセグメンテーションで使用されるマルチスケールの特徴マップです。
2. 1x1の畳み込み (self.conv_c3_1x1, self.conv_c4_1x1, self.conv_c5_1x1):
  - 各特徴マップのチャンネル数を256に削減し、他の解像度の特徴マップと統合しやすくします。
3. アップサンプリング (self.upsample_2x):
  - 解像度を2倍にすることで、異なる解像度の特徴マップを統合できるようにします。
4. 3x3の畳み込み (self.conv_c3_3x3, self.conv_c4_3x3, self.conv_c5_3x3):
  - 統合された特徴マップに対して、3x3の畳み込みを適用し、最終的な出力を生成します。
5. 新しい特徴マップの生成 (self.conv_c6_3x3, self.conv_c7_3x3):
  - c5_outputからさらなる特徴マップp6_outputとp7_outputを生成します。これにより、より低解像度のマップも含めたマルチスケールの表現が可能になります。





