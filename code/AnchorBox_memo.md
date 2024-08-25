## アンカーgeneratorの実装

アンカーボックスは、モデルが物体に対する境界ボックスを予測するために使用する固定サイズのボックスです。それは物体の中心の位置とアンカーボックスの中心の位置の間のオフセットを回帰することでこれを行ない、そして物体の相対的なスケールを予測するためにアンカーボックスの幅と高さを使用します。RetinaNet の場合は、与えられた特徴マップの各位置は (3 つのスケールと 3 つの比率の) 9 つのアンカーボックスを持ちます。

![image](https://github.com/user-attachments/assets/c2162428-12aa-4a56-9431-94d08070c62c)

1. __init__ メソッド

詳細解説
- self.aspect_ratios = [0.5, 1.0, 2.0]:

  - 3つのアスペクト比を定義します。これにより、アンカーボックスの幅と高さの比率が異なる3種類のボックスが生成されます。
- self.scales = [2 ** x for x in [0, 1 / 3, 2 / 3]]:

  - スケールリストを生成します。これにより、同じアスペクト比のボックスでも異なるサイズのアンカーボックスが生成されます。具体的には、1.0、1.26、1.587のスケールが作られます。
- self._num_anchors = len(self.aspect_ratios) * len(self.scales):

  - 各特徴マップの位置に対して生成されるアンカーボックスの総数を計算します。この例では、3つのアスペクト比と3つのスケールがあるため、1つの位置につき9つのアンカーボックスが生成されます。
- self._strides = [2 ** i for i in range(3, 8)]:

  - 特徴マップのストライドを定義します。これは、各特徴マップが元の画像に対してどのくらい縮小されているかを示します。[8, 16, 32, 64, 128]のストライドが設定されます。
- self._areas = [x ** 2 for x in [32.0, 64.0, 128.0, 256.0, 512.0]]:

  - 各特徴マップレベルでのアンカーボックスのエリアを設定します。これらは、アンカーボックスのサイズが異なる特徴ピラミッドレベルでどのように変化するかを決定します。
- self._anchor_dims = self._compute_dims():

  - _compute_dimsメソッドを呼び出し、各レベルでのアンカーボックスの幅と高さを計算し、self._anchor_dimsに保存します。

型と形状
- 入力: なし
- 出力: なし

![image](https://github.com/user-attachments/assets/ef4d60a5-52fc-4a4b-a57e-3b939ac73156)

2. _compute_dims メソッド

詳細解説

このメソッドは、各アスペクト比とスケールに対してアンカーボックスの幅と高さを計算します。

- anchor_dims_all = []:

  - 全てのレベルのアンカーボックスの寸法を保存するリストを初期化します。
- for area in self._areas::

  - 各特徴マップレベル（_areas）ごとにループを回します。ここでは、アンカーボックスのエリアが異なる特徴マップレベルごとに変化します。
- for ratio in self.aspect_ratios::

  - 各アスペクト比に対してループを回します。これにより、指定されたエリアに対するアンカーボックスの高さと幅を計算します。
- anchor_height = tf.math.sqrt(area / ratio):

  - アンカーボックスの高さを計算します。エリアをアスペクト比で割り、その平方根をとることで高さを求めます。
- anchor_width = area / anchor_height:

  - 幅はエリアを高さで割ることで求めます。これにより、指定されたアスペクト比に基づく幅が計算されます。
- dims = tf.reshape(tf.stack([anchor_width, anchor_height], axis=-1), [1, 1, 2]):

  - 幅と高さを結合して2次元のテンソルにし、それをリシェイプして形状[1, 1, 2]にします。これは後の処理でブロードキャストするために必要です。
- for scale in self.scales::

  - スケールごとにループを回し、各スケールに対してアンカーボックスの寸法を計算します。スケールをかけた寸法をanchor_dimsリストに追加します。
- anchor_dims_all.append(tf.stack(anchor_dims, axis=-2)):

  - 全てのスケールとアスペクト比に対応する寸法を一つのテンソルにまとめ、これをanchor_dims_allリストに追加します。

型と形状
- 入力: なし
- 出力:
  - anchor_dims_all: 各レベルごとのアンカーボックスの寸法を含むリスト。リストの各要素は形状が [num_anchors_per_level, 2] のテンソル。

![image](https://github.com/user-attachments/assets/ea385a71-2d52-4a51-afdf-dd602ec81aa0)

3. _get_anchors メソッド

詳細解説

このメソッドは、特定の特徴マップサイズとレベルに対してアンカーボックスを生成します。

- rx = tf.range(feature_width, dtype=tf.float32) + 0.5 と ry = tf.range(feature_height, dtype=tf.float32) + 0.5:

  - 特徴マップの各位置に対してx座標とy座標を生成します。+0.5を加えることで、各アンカーボックスの中心がピクセルの中心になるように調整します。
- centers = tf.stack(tf.meshgrid(rx, ry), axis=-1) * self._strides[level - 3]:

  - x座標とy座標のメッシュグリッドを作成し、それを結合して中心座標を計算します。これにより、アンカーボックスの中心が画像のどこに配置されるかが決まります。さらに、self._strides[level - 3]を掛けて、元の画像に対するスケールを調整します。
- centers = tf.expand_dims(centers, axis=-2):

  - 中心座標のテンソルに次元を追加し、アンカーボックスの数に対応できるようにします。これにより、中心座標の形状が[feature_height, feature_width, 1, 2]になります。
- centers = tf.tile(centers, [1, 1, self._num_anchors, 1]):

  - 中心座標のテンソルをアンカーボックスの数self._num_anchorsに応じて繰り返します。これにより、各特徴マップ位置に対して複数のアンカーボックスが生成されるようになります。形状は[feature_height, feature_width, num_anchors, 2]になります。
- dims = tf.tile(self._anchor_dims[level - 3], [feature_height, feature_width, 1, 1]):

  - _compute_dimsで計算したアンカーボックスの寸法を、特徴マップの全ての位置に複製します。これにより、形状が[feature_height, feature_width, num_anchors, 2]の寸法テンソルが作成されます。
- anchors = tf.concat([centers, dims], axis=-1):

  - 中心座標と寸法を結合して、アンカーボックスの形状[x, y, width, height]を持つテンソルを生成します。形状は[feature_height, feature_width, num_anchors, 4]になります。
- return tf.reshape(anchors, [feature_height * feature_width * self._num_anchors, 4]):

  - アンカーボックスを平坦化し、形状[total_anchors, 4]のテンソルを返します。ここで、total_anchorsはfeature_height * feature_width * num_anchorsです。

型と形状
- 入力:
  - feature_height: int (特徴マップの高さ)
  - feature_width: int (特徴マップの幅)
  - level: int (特徴ピラミッドのレベル)
- 出力:

  - 形状: [feature_height * feature_width * num_anchors, 4] のテンソル。各要素は [x, y, width, height] 形式のアンカーボックス。

![image](https://github.com/user-attachments/assets/ced5b316-a919-4193-9180-09865312dcd2)

4. get_anchors メソッド

詳細解説

このメソッドは、入力画像の高さと幅に基づいて全ての特徴マップレベルのアンカーボックスを生成します。

- anchors = [  
            self._get_anchors(  
                tf.math.ceil(image_height / 2 ** i),  
                tf.math.ceil(image_width / 2 ** i),  
                i,  
            )  
            for i in range(3, 8)  
        ]  

  - _get_anchorsを使用して、全ての特徴マップレベルに対してアンカーボックスを生成します。
- self._get_anchors(tf.math.ceil(image_height / 2 ** i), tf.math.ceil(image_width / 2 ** i), i):

  - 画像の高さと幅を特徴マップのストライドに応じてスケールダウンし、各特徴マップのサイズを計算します。iは特徴マップのレベルを示し、3から7までの値を取ります。tf.math.ceilを使用して、サイズを整数に切り上げます。
- return tf.concat(anchors, axis=0):

  - 各レベルで生成されたアンカーボックスを結合し、1つのテンソルとして返します。最終的なテンソルは全てのアンカーボックスを含みます。

型と形状
- 入力:

  - image_height: int (入力画像の高さ)
  - image_width: int (入力画像の幅)
- 出力:

  - 形状: [total_anchors, 4] のテンソル。total_anchorsは全ての特徴マップレベルにおけるアンカーボックスの総数。

まとめ

AnchorBoxクラスは、異なるスケールとアスペクト比を持つアンカーボックスを効率的に生成するためのツールです。各メソッドは、特徴マップ上の位置に応じて正確なアンカーボックスを計算するために必要な処理を行っています。これにより、オブジェクト検出モデルで使用される候補ボックスが生成されます。各処理は、異なるサイズや形状を持つオブジェクトに対応できるようにするために重要です。


## ラベルのエンコーディング
境界ボックスとクラス id から成る raw ラベルは訓練のためにターゲットに変換される必要があります。変換は以下のステップで構成されます :

- 与えられた画像の大きさのためのアンカーボックスの生成
- 正解ボックスをアンカーボックスに割り当てる
- どの物体にも割当てられないアンカーボックスは IOU に依存して背景クラスに割当てられるか無視されます。
- アンカーボックスを使用して分類と回帰ターゲットを生成する

![image](https://github.com/user-attachments/assets/d8ff9603-9510-4e8c-8a86-87a8f3ae1a70)
![image](https://github.com/user-attachments/assets/55b7aa27-6487-48ee-9d68-4ca68c516753)

![image](https://github.com/user-attachments/assets/6bdd4591-5fbe-4ca4-b049-d6533a6d7170)
![image](https://github.com/user-attachments/assets/5d7f6b44-9178-459c-839f-6eb2670fc349)


アンカーボックスとグラウンドトゥルースボックスの間のIoU（Intersection over Union）を計算し、それに基づいてアンカーボックスをマッチングします。

処理の内容
1. compute_iou: アンカーボックスとグラウンドトゥルースボックスのペアごとにIoUを計算します。
2. max_iou = tf.reduce_max(iou_matrix, axis=1): 各アンカーボックスに対する最大IoUを取得します。
3. matched_gt_idx = tf.argmax(iou_matrix, axis=1): 最大IoUを持つグラウンドトゥルースボックスのインデックスを取得します。
4. positive_mask = tf.greater_equal(max_iou, match_iou): IoUが閾値以上であればポジティブマスクを作成します。
5. negative_mask = tf.less(max_iou, ignore_iou): IoUが低い場合、ネガティブマスクを作成します。
6. ignore_mask = tf.logical_not(tf.logical_or(positive_mask, negative_mask)): ポジティブでもネガティブでもないアンカーボックスを無視するマスクを作成します。

入力の型、形状
- anchor_boxes: tf.Tensor、形状(total_anchors, 4)。アンカーボックスを表します。
- gt_boxes: tf.Tensor、形状(num_objects, 4)。グラウンドトゥルースボックスを表します。
- match_iou: float、マッチングするためのIoU閾値。
- ignore_iou: float、バックグラウンドクラスに割り当てるためのIoU閾値。

出力の型、形状
- matched_gt_idx: tf.Tensor、形状(total_anchors,)。各アンカーボックスにマッチするグラウンドトゥルースボックスのインデックス。
- positive_mask: tf.Tensor、形状(total_anchors,)。ポジティブなアンカーボックスのマスク。
- ignore_mask: tf.Tensor、形状(total_anchors,)。無視するアンカーボックスのマスク。

各処理の意味と詳細
- compute_iou: アンカーボックスとグラウンドトゥルースボックスの類似度を計算するために使用します。
- positive_mask: モデルが学習すべきポジティブな例を示します。
- ignore_mask: 学習に使用しないアンカーボックスを示します。

![image](https://github.com/user-attachments/assets/30a370ea-54a1-4969-a43e-a7a092ee2c3d)

概要

アンカーボックス（予測の基準となるボックス）とマッチングされたグランドトゥルースボックス（正解ラベルとして与えられるボックス）をもとに、モデルが学習するためのターゲットボックスを計算します。

目的: 

この関数の主な目的は、アンカーボックスとマッチングされたグランドトゥルースボックスの差異を計算し、それを正規化してターゲットボックスとして返すことです。このターゲットボックスは、モデルが学習中に予測すべき目標値になります。

処理の内容
1. (matched_gt_boxes[:, :2] - anchor_boxes[:, :2]) / anchor_boxes[:, 2:]: アンカーボックスに対するグラウンドトゥルースボックスのオフセットを計算します。
2. tf.math.log(matched_gt_boxes[:, 2:] / anchor_boxes[:, 2:]): ボックスのスケールを計算し、ログ変換を行います。
3. box_target = box_target / self._box_variance: スケーリングファクターでターゲットを正規化します。

入力の型、形状
- anchor_boxes: tf.Tensor、形状(total_anchors, 4)。アンカーボックスを表します。
- matched_gt_boxes: tf.Tensor、形状(total_anchors, 4)。マッチングされたグラウンドトゥルースボックスを表します。

出力の型、形状
- box_target: tf.Tensor、形状(total_anchors, 4)。トレーニング用のターゲットボックス。

各処理の意味と詳細
- オフセットとスケールの計算: モデルが適切なボックス調整を学習できるようにするためのターゲットを生成します。

![image](https://github.com/user-attachments/assets/07e9fc11-9932-4e35-bfc8-41051bdd24ac)

概要

単一のサンプルに対して、バウンディングボックスと分類ターゲットを作成します。

処理の内容
1. self._anchor_box.get_anchors(image_shape[1], image_shape[2]): 画像サイズに応じてアンカーボックスを生成します。
2. matched_gt_idx, positive_mask, ignore_mask = self._match_anchor_boxes(anchor_boxes, gt_boxes): アンカーボックスとグラウンドトゥルースボックスをマッチングします。
3. box_target = self._compute_box_target(anchor_boxes, matched_gt_boxes): マッチングされたボックスに対するターゲットを計算します。
4. cls_target = tf.where(tf.not_equal(positive_mask, 1.0), -1.0, matched_gt_cls_ids): ポジティブでないアンカーボックスには-1.0を割り当てます。
5. cls_target = tf.where(tf.equal(ignore_mask, 1.0), -2.0, cls_target): 無視するアンカーボックスには-2.0を割り当てます。
6. label = tf.concat([box_target, cls_target], axis=-1): ボックスターゲットとクラスターゲットを結合してラベルを作成します。

入力の型、形状
- image_shape: tf.Tensor、形状(3,)。入力画像の形状を表します。
- gt_boxes: tf.Tensor、形状(num_objects, 4)。グラウンドトゥルースボックスを表します。
- cls_ids: tf.Tensor、形状(num_objects,)。オブジェクトのクラスIDを表します。

出力の型、形状
- label: tf.Tensor、形状(total_anchors, 5)。トレーニング用のラベルで、各アンカーボックスに対して[dx, dy, dw, dh, class_id]の形式を持ちます。

各処理の意味と詳細
- 各サンプルごとに、アンカーボックスに対して適切なターゲットを生成します。これにより、モデルは異なるスケールや位置のオブジェクトに対応できるようになります。

![image](https://github.com/user-attachments/assets/5d7a050c-6e79-4b7e-b658-6e9f0a824e71)
![image](https://github.com/user-attachments/assets/2540f699-7bb0-4063-9d2e-71ac6d4f9d31)

![image](https://github.com/user-attachments/assets/2fabd843-b670-45c9-a38e-e6674d5f6b69)

概要

バッチ内の各サンプルに対して、ボックスと分類ターゲットを作成します。

処理の内容
1. batch_size = images_shape[0]: バッチサイズを取得します。
2. labels = tf.TensorArray(dtype=tf.float32, size=batch_size, dynamic_size=True): ラベルを格納するためのテンソル配列を作成します。
3. label = self._encode_sample(images_shape, gt_boxes[i], cls_ids[i]): 各サンプルに対してターゲットをエンコードします。
4. batch_images = tf.keras.applications.resnet.preprocess_input(batch_images): バッチ画像をResNetの前処理関数で処理します。

入力の型、形状
- batch_images: tf.Tensor、形状(batch_size, height, width, channels)。バッチ内の画像を表します。
- gt_boxes: tf.Tensor、形状(batch_size, num_objects, 4)。バッチ内のグラウンドトゥルースボックスを表します。
- cls_ids: tf.Tensor、形状(batch_size, num_objects)。バッチ内のオブジェクトのクラスIDを表します。

出力の型、形状
- batch_images: tf.Tensor、形状(batch_size, height, width, channels)。前処理されたバッチ画像。
- labels: tf.Tensor、形状(batch_size, total_anchors, 5)。各バッチに対するラベル。

各処理の意味と詳細
- バッチ内の各サンプルに対して、アンカーボックスに適したターゲットを一括で生成し、効率的なトレーニングを実現します。
