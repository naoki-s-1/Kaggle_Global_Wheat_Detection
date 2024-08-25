# 前処理についてのメモ

## バウンディングボックスの形式の変換

![image](https://github.com/user-attachments/assets/fa83c043-b88f-4268-a21f-d3ffe4ea39b3)

1. swap_xy(boxes)

処理の内容

swap_xy関数は、バウンディングボックスの座標のうち、x座標とy座標の順序を入れ替える関数です。具体的には、バウンディングボックスの左上の(xmin, ymin)と右下の(xmax, ymax)のxとyを交換します。この処理は、座標系の変更や、特定のフォーマットに合わせたデータ変換を行う際に使用されます。

入力

- 型: tf.Tensor（または互換性のあるもの）
- 形状: (num_boxes, 4)  
ここで、num_boxesはバウンディングボックスの数を示し、各ボックスは[xmin, ymin, xmax, ymax]の形式で表されています。

出力

- 型: tf.Tensor
- 形状: (num_boxes, 4)
出力も同じ形状を持ちますが、座標の順序が[ymin, xmin, ymax, xmax]に変更されています。

詳細解説

この関数は、tf.stackを用いて入力テンソルの特定のスライスを選択し、軸を指定して新しいテンソルを作成します。boxes[:, 1]やboxes[:, 0]のようなスライスを使用して、yとxの座標を入れ替えています。

![image](https://github.com/user-attachments/assets/d7e954e5-0e57-4e29-90d7-17ec1b3be5a4)

2. convert_to_xywh(boxes)
   
処理の内容

convert_to_xywh関数は、バウンディングボックスの表現形式を[xmin, ymin, xmax, ymax]から[x_center, y_center, width, height]に変換します。x_centerとy_centerはボックスの中心座標、widthとheightはボックスの幅と高さを示します。

入力

- 型: tf.Tensor（または互換性のあるもの）
- 形状: (..., num_boxes, 4)
ここで、先頭の...は任意の次元を示し、num_boxesはバウンディングボックスの数を示します。各ボックスは[xmin, ymin, xmax, ymax]の形式で表されています。

出力

- 型: tf.Tensor
- 形状: 入力と同じ形状(..., num_boxes, 4)
出力は、[x_center, y_center, width, height]の形式になっています。

詳細解説

この関数では、まずボックスの左上と右下の座標の平均を取り、ボックスの中心座標を計算します。その後、右下と左上の座標差分を取ることで、ボックスの幅と高さを計算します。tf.concatを使用して、これらの計算結果を新しいテンソルとして結合しています。

![image](https://github.com/user-attachments/assets/86f58691-6f38-433d-8c7e-477177418e77)

3. convert_to_corners(boxes)
   
処理の内容

convert_to_corners関数は、バウンディングボックスの表現形式を[x_center, y_center, width, height]から[xmin, ymin, xmax, ymax]に変換します。xminとyminはボックスの左上の座標、xmaxとymaxは右下の座標を示します。

入力

- 型: tf.Tensor（または互換性のあるもの）
- 形状: (..., num_boxes, 4)
ここで、先頭の...は任意の次元を示し、num_boxesはバウンディングボックスの数を示します。各ボックスは[x_center, y_center, width, height]の形式で表されています。

出力
- 型: tf.Tensor
- 形状: 入力と同じ形状(..., num_boxes, 4)
出力は、[xmin, ymin, xmax, ymax]の形式になっています。

詳細解説

この関数では、まずボックスの中心座標から幅と高さの半分を引いて左上の座標を計算し、次に中心座標に幅と高さの半分を加えて右下の座標を計算します。tf.concatを使用して、これらの座標を新しいテンソルとして結合しています。

## ペアごとの Intersection Over Union (IOU) 計算

![image](https://github.com/user-attachments/assets/0fdbb742-5bc1-4640-9b5e-cebab7e2b50b)

1. compute_iou(boxes1, boxes2)

処理の内容

compute_iou関数は、2つのバウンディングボックスセット間のペアワイズIOU（Intersection over Union）行列を計算します。IOUは、2つのボックスがどれだけ重なっているかを示す指標で、物体検出の精度評価などに使用されます。

入力
- 型: tf.Tensor
- 形状:
  - boxes1: (N, 4) - N個のバウンディングボックスを含むテンソル。
  - boxes2: (M, 4) - M個のバウンディングボックスを含むテンソル。
  - 各ボックスは [x, y, width, height] の形式です。

出力
- 型: tf.Tensor
- 形状: (N, M) - boxes1の各ボックスとboxes2の各ボックス間のIOUを格納した行列。

各処理の詳細解説

1. convert_to_corners(boxes1)とconvert_to_corners(boxes2)

  - 内容: boxes1とboxes2を[x, y, width, height]形式から[xmin, ymin, xmax, ymax]形式に変換します。
  - 入力:
    - 型: tf.Tensor
    - 形状: (N, 4)および(M, 4)
  -出力:
    - 型: tf.Tensor
    - 形状: (N, 4)および(M, 4) - それぞれのボックスが[xmin, ymin, xmax, ymax]形式に変換されます。

2. lu = tf.maximum(boxes1_corners[:, None, :2], boxes2_corners[:, :2])

  - 内容: boxes1とboxes2の各ペア間で、交差部分の左上角(lu)を計算します。
  - 入力:
    - 型: tf.Tensor
    - 形状: (N, 4)および(M, 4)
  - 出力:
    - 型: tf.Tensor
    - 形状: (N, M, 2) - 各ボックスペアの交差部分の左上角の座標を持つテンソル。

3. rd = tf.minimum(boxes1_corners[:, None, 2:], boxes2_corners[:, 2:])

  - 内容: boxes1とboxes2の各ペア間で、交差部分の右下角(rd)を計算します。
  - 入力:
    - 型: tf.Tensor
    - 形状: (N, 4)および(M, 4)
  - 出力:
    - 型: tf.Tensor
    - 形状: (N, M, 2) - 各ボックスペアの交差部分の右下角の座標を持つテンソル。

4. intersection = tf.maximum(0.0, rd - lu)

  - 内容: 交差部分の幅と高さ（すなわちintersection）を計算します。交差がない場合には、幅や高さが負の値になることを防ぐために0.0と比較して取ります。
  - 入力:
    - 型: tf.Tensor
    - 形状: (N, M, 2) - 左上と右下の座標
  - 出力:
    - 型: tf.Tensor
    - 形状: (N, M, 2) - 交差部分の幅と高さを持つテンソル。

5. intersection_area = intersection[:, :, 0] * intersection[:, :, 1]

  - 内容: 交差部分の面積を計算します。
  - 入力:
    - 型: tf.Tensor
    - 形状: (N, M, 2) - 交差部分の幅と高さ
  - 出力:
    - 型: tf.Tensor
    - 形状: (N, M) - 交差部分の面積を持つテンソル。

6. boxes1_area = boxes1[:, 2] * boxes1[:, 3]

  - 内容: boxes1の各ボックスの面積を計算します。
  - 入力:
    - 型: tf.Tensor
    - 形状: (N, 4) - 各ボックスの幅と高さを持つテンソル。
  - 出力:
    - 型: tf.Tensor
    - 形状: (N,) - 各ボックスの面積を持つテンソル。

7. boxes2_area = boxes2[:, 2] * boxes2[:, 3]

  - 内容: boxes2の各ボックスの面積を計算します。
  - 入力:
    - 型: tf.Tensor
    - 形状: (M, 4) - 各ボックスの幅と高さを持つテンソル。
  - 出力:
    - 型: tf.Tensor
    - 形状: (M,) - 各ボックスの面積を持つテンソル。

8. union_area = tf.maximum(boxes1_area[:, None] + boxes2_area - intersection_area, 1e-8)

  - 内容: 2つのボックスの合計面積から交差部分を引いて、合計面積（union）を計算します。小さな数値 1e-8 を使って、ゼロ除算を防ぎます。
  - 入力:
    - 型: tf.Tensor
    - 形状: (N,), (M,), および (N, M) - 各ボックスの面積と交差部分の面積を持つテンソル。
  - 出力:
    - 型: tf.Tensor
    - 形状: (N, M) - 合計面積を持つテンソル。

9. return tf.clip_by_value(intersection_area / union_area, 0.0, 1.0)

  - 内容: IOU（Intersection over Union）を計算し、結果を0.0から1.0の範囲にクリップして返します。
  - 入力:
    - 型: tf.Tensor
    - 形状: (N, M) - 交差面積と合計面積を持つテンソル。
  - 出力:
    - 型: tf.Tensor
    - 形状: (N, M) - IOU行列。


## データの前処理

![image](https://github.com/user-attachments/assets/b26a1733-e185-4f0d-9927-a767cecffcc5)

概要

この関数は、画像とそのバウンディングボックスを水平方向にランダムに反転させる処理を行います。反転する確率は50%です。

処理の内容

- tf.random.uniform(()) > 0.5: 一様分布からランダムな数を生成し、その値が0.5より大きければ反転します。
- image = tf.image.flip_left_right(image): 画像を水平方向に反転させます。
- boxes = tf.stack([1 - boxes[:, 2], boxes[:, 1], 1 - boxes[:, 0], boxes[:, 3]], axis=-1): バウンディングボックスの座標を反転させるために、左側のx座標を1 - x_maxに、右側のx座標を1 - x_minに変更します。

入力の型、形状
- image: tf.Tensor、形状(height, width, channels)の3次元テンソル。これは画像を表します。
- boxes: tf.Tensor、形状(num_boxes, 4)の2次元テンソル。バウンディングボックスを正規化された座標系で表し、各ボックスは[x_min, y_min, x_max, y_max]の形式を持ちます。

出力の型、形状
- image: tf.Tensor、形状は入力と同じ(height, width, channels)。
- boxes: tf.Tensor、形状は入力と同じ(num_boxes, 4)。

各処理の意味と詳細
- tf.random.uniform(()) > 0.5: 50%の確率で反転を行うための条件式です。
- tf.image.flip_left_right(image): 画像を水平方向に反転させ、対象物が右から左に移動するようにします。
- boxesの反転処理: 反転後のバウンディングボックスの座標を適切に変換し、オブジェクトの位置が正確に反映されるようにします。

![image](https://github.com/user-attachments/assets/3e16a911-297b-4db6-b147-2b1ad62d8eef)

概要

この関数は、画像のアスペクト比を維持しつつ、短辺を指定されたサイズにリサイズし、必要に応じて長辺が制限を超えないようにリサイズした後、ゼロパディングを行います。

処理の内容
- 短辺をmin_sideにリサイズします。
- リサイズ後の画像で長辺がmax_sideを超える場合、長辺をmax_sideに合わせて再度リサイズします。
-画像の幅と高さがstrideの倍数になるように、右側と下側にゼロパディングを行います。

入力の型、形状
- image: tf.Tensor、形状(height, width, channels)の3次元テンソル。これは画像を表します。
- min_side: float、画像の短辺をリサイズするターゲットサイズ。
- max_side: float、画像の長辺の最大サイズ。
- jitter: List[float]、リサイズ時に短辺をランダムに変更する範囲を示します。
- stride: float、画像サイズを最終的にstrideの倍数にするためのパラメータ。

出力の型、形状
- image: tf.Tensor、リサイズ・パディングされた画像。形状は(padded_height, padded_width, channels)になります。
- image_shape: tf.Tensor、リサイズ後の画像の実際の形状。形状は(2,)で、(height, width)を表します。
- ratio: float、画像がリサイズされたスケーリングファクター。

各処理の意味と詳細
- min_side: 画像の短辺を指定された値にリサイズし、統一された入力サイズにするためです。
- max_side: 長辺が指定された最大値を超えないように調整し、過度なスケーリングを防ぎます。
- ゼロパディング: モデルが固定サイズの入力を期待する場合に、画像をそのサイズに合わせます。

![image](https://github.com/user-attachments/assets/70edb733-5c5d-4cde-ae68-fde707d7f041)
![image](https://github.com/user-attachments/assets/082bb415-7229-4ace-9127-2b6bb4f936bc)
![image](https://github.com/user-attachments/assets/15bde025-255d-4175-94bb-24751bff912e)


![image](https://github.com/user-attachments/assets/b5e4ba8b-488d-4e05-8336-41cad00e1f14)

概要
この関数は、データセットの1つのサンプルに対して前処理を行い、モデルへの入力として適切な形式に変換します。具体的には、画像の読み込み、デコード、バウンディングボックスの座標変換、リサイズ・パディングを行います。

処理の内容
1. 画像をファイルパスから読み込み、JPEG形式からデコードします。
2. バウンディングボックスの座標をswap_xy関数で変換します（[x_min, y_min, x_max, y_max]形式）。
3. クラスラベルをint32型にキャストします。
4. 画像をresize_and_pad_image関数でリサイズし、バウンディングボックスの座標もスケーリングします。
5. バウンディングボックスの形式を[x, y, width, height]に変換します。

入力の型、形状
- sample: Dict、1つのトレーニングサンプルを表す辞書。辞書には以下のキーが含まれます。
   - "image": 画像ファイルへのパス。
   - "objects": バウンディングボックスとクラスラベルを含む辞書。

出力の型、形状
- image: tf.Tensor、リサイズ・パディングされた画像。形状は(padded_height, padded_width, channels)。
- bbox: tf.Tensor、リサイズされたバウンディングボックス。形状は(num_objects, 4)。
- class_id: tf.Tensor、オブジェクトのクラスID。形状は(num_objects,)。

各処理の意味と詳細
- 画像の読み込みとデコード: 画像をモデルが処理できる形式に変換します。
- バウンディングボックスの変換: バウンディングボックスをリサイズやその他の操作に対応できるようにします。
- 画像とバウンディングボックスのリサイズ: モデルが期待する入力サイズに合わせるための処理です。

![image](https://github.com/user-attachments/assets/fd04efa9-53a4-4834-ba5f-f787b78ec411)
![image](https://github.com/user-attachments/assets/374f6d04-f36b-44ef-aefb-02a17f194c67)
