# RetinaNetモデル


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

### 処理内容:

このメソッドは、入力画像からバックボーン（ResNet50）を使用して得られた中間層の特徴マップを元に、FPNを構築します。

1. c3_output, c4_output, c5_output = self.backbone(images, training=training):
  - 入力画像をバックボーンに通し、ResNet50の中間層から3つの特徴マップ（c3, c4, c5）を取得します。
2. p3_output = self.conv_c3_1x1(c3_output):
  - c3_outputに対して1x1の畳み込みを適用し、チャンネル数を256に変換します。
3. p4_output = self.conv_c4_1x1(c4_output):
  - 同様に、c4_outputに対して1x1の畳み込みを適用します。
4. p5_output = self.conv_c5_1x1(c5_output):
  - c5_outputに対して1x1の畳み込みを適用します。
5. p4_output = p4_output + self.upsample_2x(p5_output):
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
1 ResNet50バックボーンからの特徴抽出 (self.backbone):
  
  - ResNet50から特徴マップを抽出します。これらは物体検出やセマンティックセグメンテーションで使用されるマルチスケールの特徴マップです。
2. 1x1の畳み込み (self.conv_c3_1x1, self.conv_c4_1x1, self.conv_c5_1x1):
  - 各特徴マップのチャンネル数を256に削減し、他の解像度の特徴マップと統合しやすくします。
3. アップサンプリング (self.upsample_2x):
  - 解像度を2倍にすることで、異なる解像度の特徴マップを統合できるようにします。
4. 3x3の畳み込み (self.conv_c3_3x3, self.conv_c4_3x3, self.conv_c5_3x3):
  - 統合された特徴マップに対して、3x3の畳み込みを適用し、最終的な出力を生成します。
5. 新しい特徴マップの生成 (self.conv_c6_3x3, self.conv_c7_3x3):
  - c5_outputからさらなる特徴マップp6_outputとp7_outputを生成します。これにより、より低解像度のマップも含めたマルチスケールの表現が可能になります。

![image](https://github.com/user-attachments/assets/75482f6f-f1e0-4178-bd62-1edde01a9407)


### 分類と回帰ヘッドの構築
RetinaNet は境界ボックス回帰のためと、物体のクラス確率を予測するための個別のヘッドを持ちます。これらのヘッドは特徴ピラミッドの総ての特徴マップの間で共有されます。

![image](https://github.com/user-attachments/assets/b6f0aeda-bc8a-451e-9406-5792aca45984)

1. 関数の定義:

  - build_head 関数は2つの引数を取ります:
    - output_filters: 最後の畳み込み層におけるフィルターの数を指定します。これにより、空間的な各位置に対して出力される値の数が決まります。
    - bias_init: 最後の畳み込み層のバイアス初期化方法を指定します。

2. Sequentialモデルの作成:

  - Kerasの Sequential モデルである head が初期化されます。最初に、入力層が追加され、形状 [None, None, 256] のテンソルを期待します。
  - None は高さと幅が可変であることを示し、256 は入力テンソルのチャンネル数を表します。

3. 畳み込み層とReLU層の追加（ループ処理）:

  - 4回ループが実行され、4つの畳み込み層とReLU活性化層がモデルに追加されます。各畳み込み層は次の設定で作成されます:
    - フィルター数: 256
    - カーネルサイズ: 3x3
    - パディング: "same" （出力のサイズが入力と同じになります）
    - カーネル初期化: 平均 0.0、標準偏差 0.01 の正規分布に基づく初期化
  - 各畳み込み層の後に、非線形性を導入するためにReLU活性化関数が続きます。

4. 最後の畳み込み層:
   
  - ループが終了した後、最後の畳み込み層がモデルに追加されます。この層の設定は次の通りです:
フィルター数: output_filters（関数の引数として渡されます）
    - カーネルサイズ: 3x3
    - ストライド: 1
    - パディング: "same"
    - カーネル初期化: 先ほどと同じ設定
    - バイアス初期化: bias_init が使用されます

5. モデルの返却:

  - 最後に、構築されたKeras Sequential モデルが返されます。

入力の型、形状、出力の型、形状について詳しく解説

- 入力:
  - このモデルへの入力は、形状 [batch_size, height, width, 256] の4次元テンソルです。
  - height と width は可変 (None) で、モデルは異なる空間解像度の特徴マップを処理することができます。
  - 256 は入力特徴マップのチャンネル数で、Feature Pyramid Network (FPN) から供給されます。

- 出力:

  - モデルの出力形状は [batch_size, height, width, output_filters] です。
  - output_filters は関数に渡された引数で、各空間位置に対して予測される値の数を表します（例えば、クラス数やバウンディングボックスの座標数など）。
  - 例えば、このヘッドがクラス分類用で、クラスが80個ある場合、output_filters は80になります。バウンディングボックスの回帰用であれば、output_filters はおそらく4になります。

各処理ごとの意味について詳しく解説

- Sequentialモデルの作成:
  - Sequential モデルは、レイヤーを線形に積み重ねるのに適しており、このヘッドのように各レイヤーが前のレイヤーの出力を直接受け取る場合に最適です。

- 畳み込み層とReLUの追加:
  - 一連の畳み込み層は、入力特徴マップから複雑なパターンを学習するために使用されます。
  - ReLU 活性化関数は非線形性を導入し、モデルがより複雑な関数を学習できるようにします。

- 最後の畳み込み層:
  - この層は、出力チャンネル数を調整し、必要な出力数に一致させます（例: クラスのスコアやバウンディングボックスの座標）。

![image](https://github.com/user-attachments/assets/b3bfbfa4-c7b5-420a-b697-23d567dc88d1)


## サブクラス化モデルを使用して RetinaNetを構築する

![image](https://github.com/user-attachments/assets/7572066e-84ab-4f6f-a77e-897e24a5fd0d)
![image](https://github.com/user-attachments/assets/3a0150ce-09c2-4183-88b0-aa06b63caa37)
![image](https://github.com/user-attachments/assets/b2d3666a-ad10-4a91-97d8-ee15038503d5)

![image](https://github.com/user-attachments/assets/84fa37b6-27ea-4075-8437-7b3565a6cc87)

1. for feature in features:

  - 入力: features は、FeaturePyramid から出力された複数の特徴マップのリストです。それぞれの特徴マップは異なる解像度を持っています。
  - 意味: このループは、各特徴マップに対して、バウンディングボックスとクラス予測を個別に処理します。

2. self.box_head(feature)

  - 入力: feature は、特徴マップの1つであり、形状は [batch_size, height, width, num_channels]（通常、num_channels は256）です。
  - 処理内容: box_head は、build_head 関数で構築されたサブネットワークです。このサブネットワークは、各特徴マップに対して、バウンディングボックスの座標 (x, y, width, height) を予測します。
  - 出力: 出力は形状 [batch_size, height, width, 36] のテンソルで、ここで36は各位置での4つの座標を9つのアンカーボックスに対して予測した結果です。

3. tf.reshape(self.box_head(feature), [N, -1, 4])

  - 入力: self.box_head(feature) の出力形状は [batch_size, height, width, 36] です。
  - 処理内容: このテンソルを [batch_size, num_boxes, 4] に再整形します。num_boxes は各特徴マップの解像度に基づく全てのアンカーボックスの数です。-1 は自動計算されるため、具体的な値は height * width * 9 になります。
  - 出力: box_outputs に追加される形状 [batch_size, num_boxes, 4] のテンソルで、各バウンディングボックスの4つの座標（x, y, width, height）を表します。

4. self.cls_head(feature)

  - 入力: feature は、特徴マップの1つであり、形状は [batch_size, height, width, num_channels]（通常、num_channels は256）です。
  - 処理内容: cls_head は、クラス予測を行うサブネットワークです。このサブネットワークは、各特徴マップの各位置でクラスごとの確信度を予測します。
  - 出力: 出力は形状 [batch_size, height, width, 81] のテンソルで、81は各位置でのクラス数（num_classes）を9つのアンカーボックスに対して予測した結果です。

5. tf.reshape(self.cls_head(feature), [N, -1, self.num_classes])

  - 入力: self.cls_head(feature) の出力形状は [batch_size, height, width, 81] です。
  - 処理内容: このテンソルを [batch_size, num_boxes, num_classes] に再整形します。num_boxes は height * width * 9 に相当します。
  - 出力: cls_outputs に追加される形状 [batch_size, num_boxes, num_classes] のテンソルで、各アンカーボックスに対するクラス確信度を表します。

![image](https://github.com/user-attachments/assets/f4079371-dd40-40c6-b09e-8d6c7f5fb1e9)


## 予測をデコードするカスタム層の実装

DecodePredictionsクラスは、RetinaNetモデルの予測をデコードし、予測バウンディングボックスを実際の画像空間に変換し、不要な検出を削除するための処理を行います。特に、ボックス予測のデコードと、非最大抑制 (NMS) を適用して、信頼性の低い予測をフィルタリングします。

### 各メソッドの詳細

![image](https://github.com/user-attachments/assets/ea8e1ecf-5ff2-4c18-bae1-2b8d3c0e80c0)
![image](https://github.com/user-attachments/assets/7664fd1f-f965-4356-8e73-904a4cfb6d7e)

![image](https://github.com/user-attachments/assets/e7d7b080-2a79-4147-9b5e-7ed4e8c27c13)
![image](https://github.com/user-attachments/assets/5d431a93-4adc-481a-9ed7-4a2ffd2eef29)
![image](https://github.com/user-attachments/assets/9c1632b8-ff15-41e8-a347-2cb1b8a2f199)
![image](https://github.com/user-attachments/assets/de86878c-aaf7-41c7-8cfb-9ffdcb65a383)

![image](https://github.com/user-attachments/assets/e6d2238b-56a8-4959-81c1-85eb8958c020)
![image](https://github.com/user-attachments/assets/daeffeea-fac4-4b74-a7da-e2000cdefcfc)

### 各処理ごとの詳細

![image](https://github.com/user-attachments/assets/a0b15fe3-8588-4ef7-a3d3-1fda0c290f41)

![image](https://github.com/user-attachments/assets/e7928b6b-d093-4513-b18a-b95f3ca7ff49)

- boxes:

  - 型と形状: tf.Tensor (形状: [batch_size, num_boxes, q, 4])
  - 説明: 各検出のバウンディングボックスの座標を表すテンソルです。q は通常 1 ですが、q が 1 の場合はクラスごとに異なるボックスがあることを意味します。

- scores:

  - 型と形状: tf.Tensor (形状: [batch_size, num_boxes, num_classes])
  - 説明: 各ボックスに対する各クラスの確信度（スコア）を表すテンソルです。

- max_output_size_per_class:

  - 型: int
  - 説明: クラスごとに保持される最大検出数です。

- max_total_size:

  - 型: int
  - 説明: 全体で保持される最大検出数です。

- iou_threshold:

  - 型: float
  - 説明: NMSを適用する際のインターセクション・オーバー・ユニオン (IoU) の閾値です。この値を超える重複率のボックスは抑制されます。

- score_threshold:

  - 型: float
  - 説明: この閾値を下回るスコアを持つ検出は無視されます。

- clip_boxes:

  - 型: bool
  - 説明: 出力されたボックスを画像の境界内にクリップするかどうかを指定します。

![image](https://github.com/user-attachments/assets/eaf35859-a00d-4e31-a135-ffc6f69c1ace)
![image](https://github.com/user-attachments/assets/85428961-56e3-465b-898a-b3aad0f93151)

まとめ

- tf.image.combined_non_max_suppression は、オブジェクト検出モデルの出力を精緻化するための重要な関数です。
- スコアとNMSによって重複するボックスが除去され、信頼度の高い検出結果が得られます。
- 戻り値は、NMS後の検出結果のボックス、スコア、クラス、および有効な検出数です。
