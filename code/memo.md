### 参考  
* 変換器  
![image](https://github.com/user-attachments/assets/52b09053-c96c-4173-a10e-45c945934cf1)

def castF(x):  
- castF(x): x を浮動小数点型（float32 または float64）にキャストします。Kerasバックエンド（K）のfloatx()メソッドを使用して、現在のデフォルト浮動小数点データ型にキャストします。

    return K.cast(x, K.floatx())

def castB(x):  
- castB(x): x をブール型にキャストします。値が0でない場合にTrue、0の場合にFalseになります。

    return K.cast(x, bool)

def iou_loss_core(true,pred):  
#this can be used as a loss if you make it negative  
iou_loss_core(true, pred)
- 説明: IoU（Intersection over Union）を計算するための関数です。これを負にすると損失関数としても使用できます。

- 詳細:

  - true * pred: 真のラベル（true）と予測ラベル（pred）の積を取ることで、両方が1のピクセルの数、すなわち交差部分を計算します。
1 - true: 真のラベルの反転を取ることで、true が0の部分（背景ピクセル）を得ます。
  - true + (notTrue * pred): 合計部分を計算します。真のラベルが1かつ予測ラベルが0、または真のラベルが0かつ予測ラベルが1のピクセルの和を求めます。
  - 最終的に、交差部分を合計部分で割ることでIoUを計算します。K.epsilon()は、ゼロ除算を防ぐために小さな定数を加えます。

    intersection = true * pred
    notTrue = 1 - true
    union = true + (notTrue * pred)

    return (K.sum(intersection, axis=-1) + K.epsilon()) / (K.sum(union, axis=-1) + K.epsilon())

def competitionMetric2(true, pred):  
#### competitionMetric2(true, pred)
- 説明: 平均IoUを計算するためのカスタムメトリック関数です。これは、複数の閾値を適用して真陽性（True Positives）の割合を計算し、最終的に全体のスコアを求めます。

- 詳細:

  - 閾値の設定: tresholds には、0.5から0.7までの範囲で0.05刻みの閾値が5つ設定されています。これらの閾値は、IoUを測定する際の基準値として使用されます。

  1. フラット化: K.batch_flatten により、入力のラベルと予測をフラット化し、1次元のピクセル列に変換します。

  2. 予測の二値化: 予測されたピクセルが0.5以上であれば、それを1（真）として扱います。castFにより、ブール値を浮動小数点型にキャストします。

  3. マスク有無の判定: trueSum と predSum により、各画像内でマスク（白いピクセル）が存在するかどうかを確認します。1つ以上の白いピクセルがあれば、その画像にはマスクがあると判断します。

  4. 真陽性マスクの抽出: 両方にマスクが存在する画像を選び出します（true1 * pred1）。tf.boolean_mask を使用して、これらの画像をIoU計算の対象にします。

  5. IoUの計算と閾値比較: 各閾値に対して、IoUが閾値を超えるかどうかを確認し、その結果を truePositives に格納します。最後に、各閾値に対する真陽性率の平均を求め、総和を取ります。

  6. 真陰性の計算: マスクが存在しない画像（trueNegatives）を計算します。これもスコアに加えます。

  7. スコアの計算: 最後に、真陽性と真陰性の合計を、画像のバッチ数で割って平均スコアを出します。

    tresholds = [0.5 + (i * 0.05)  for i in range(5)]

    #flattened images (batch, pixels)
    true = K.batch_flatten(true)
    pred = K.batch_flatten(pred)
    pred = castF(K.greater(pred, 0.5))

    #total white pixels - (batch,)
    trueSum = K.sum(true, axis=-1)
    predSum = K.sum(pred, axis=-1)

    #has mask or not per image - (batch,)
    true1 = castF(K.greater(trueSum, 1))    
    pred1 = castF(K.greater(predSum, 1))

    #to get images that have mask in both true and pred
    truePositiveMask = castB(true1 * pred1)

    #separating only the possible true positives to check iou
    testTrue = tf.boolean_mask(true, truePositiveMask)
    testPred = tf.boolean_mask(pred, truePositiveMask)

    #getting iou and threshold comparisons
    iou = iou_loss_core(testTrue,testPred) 
    truePositives = [castF(K.greater(iou, tres)) for tres in tresholds]

    #mean of thressholds for true positives and total sum
    truePositives = K.mean(K.stack(truePositives, axis=-1), axis=-1)
    truePositives = K.sum(truePositives)

    #to get images that don't have mask in both true and pred
    trueNegatives = (1-true1) * (1 - pred1) # = 1 -true1 - pred1 + true1*pred1
    trueNegatives = K.sum(trueNegatives) 

    return (truePositives + trueNegatives) / castF(K.shape(true)[0])


## tf.stackとtf.concatとの動作の違い

![image](https://github.com/user-attachments/assets/cd1e03d0-c8bd-4bc2-a5ee-7344130b043f)

![image](https://github.com/user-attachments/assets/a8beb3e9-bc3e-4003-8d9c-2e418edda868)


<details open><summary>## スライスする時のNoneについて</summary>  



Noneをスライスに使う場合、テンソルの次元を拡張するために使用されます。具体的には、Noneを挿入することで、指定した位置に新しい次元が追加されます。この新しい次元のサイズは1になります。

以下に、lu = tf.maximum(boxes1_corners[:, None, :2], boxes2_corners[:, :2]) と union_area = tf.maximum(boxes1_area[:, None] + boxes2_area - intersection_area, 1e-8) のそれぞれについて解説します。

1. lu = tf.maximum(boxes1_corners[:, None, :2], boxes2_corners[:, :2])

役割
- boxes1_corners[:, None, :2] では、boxes1_cornersテンソルの2番目の次元に新しい次元を追加しています。
- これにより、boxes1_cornersはもともと (N, 4) の形状でしたが、[:, None, :2] の操作により (N, 1, 2) になります。

なぜ必要か
- tf.maximum関数で boxes1_corners と boxes2_corners を比較する際、形状を揃える必要があります。boxes1_corners が (N, 1, 2) になり、boxes2_corners[:, :2] が (M, 2) のとき、tf.maximum が要素ごとに N x M の比較を行うことが可能になります。
- つまり、これにより lu の形状は (N, M, 2) となり、boxes1_corners の各要素が boxes2_corners の各要素とペアごとに比較されるようになります。

2. union_area = tf.maximum(boxes1_area[:, None] + boxes2_area - intersection_area, 1e-8)

役割
- boxes1_area[:, None] では、boxes1_areaテンソルの1番目の次元に新しい次元を追加しています。
- これにより、boxes1_areaはもともと (N,) の形状でしたが、[:, None] の操作により (N, 1) になります。

なぜ必要か
- boxes1_area[:, None] と boxes2_area を足し合わせる際、形状を (N, M) にする必要があります。
- boxes1_area の形状が (N, 1) であり、boxes2_area が (M,) の場合、boxes1_area[:, None] + boxes2_area の結果は (N, M) になります。
- こうして、boxes1_area の各要素が boxes2_area の各要素とペアごとに加算されるようになり、結果として union_area は (N, M) となります。

まとめ

Noneを使うことで、次元が拡張され、ブロードキャストが可能になります。ブロードキャストは、異なる形状のテンソル間で演算を行う際に非常に重要です。これにより、テンソルの形状が揃えられ、要素ごとの計算が正しく行われるようになります。
</details>


## バウンディングボックス

![image](https://github.com/user-attachments/assets/8a1958ff-de90-4404-ae6b-9e71b1830daf)


## 比率に対してlogをとる
- 比率の安定化: 物体検出では、オブジェクトのサイズが非常に異なる可能性があります。たとえば、アンカーボックスが小さい場合でも、グランドトゥルースボックスが大きい場合があります。このような場合に、単純な比率を使用すると、非常に大きな値や非常に小さな値が生じてしまい、これがモデルの学習を難しくすることがあります。対数（log）を取ることで、これらの極端な値を圧縮し、より安定したスケールで表現できます。

- スケールの変化に対する対応: 比率を直接扱うよりも、比率の対数を取ることで、モデルが異なるスケールに対応しやすくなります。たとえば、あるボックスのサイズが2倍になる場合、その比率の対数は log(2) の変化として表現されます。これは、スケールの変化に対するモデルの感度を抑える効果があります。

- 線形変換への適応: ニューラルネットワークは通常、線形変換を適用する層（たとえば、全結合層や畳み込み層）を持っています。対数変換されたデータは、こうした層で扱いやすく、学習が効率的になります。

box_targetのスケーリングに self._box_variance を使う理由

self._box_variance は、ターゲットボックスをスケーリングするために使用されます。このスケーリングにはいくつかの重要な理由があります。

1. 勾配の制御: box_target をスケーリングすることで、勾配の大きさを制御し、勾配爆発（gradients explosion）や勾配消失（gradients vanishing）を防ぎます。特に、スケーリングしないと非常に大きな値を持つターゲットがモデルに入力され、これが学習を不安定にする可能性があります。

2. スケールの標準化: self._box_variance を使用して box_target をスケーリングすることで、各ターゲットの値が適切な範囲に収まるようにします。これは、モデルが異なるスケールのボックスサイズに対して一貫したパフォーマンスを発揮するのに役立ちます。

3. 学習の安定化: self._box_variance = tf.convert_to_tensor([0.1, 0.1, 0.2, 0.2], dtype=tf.float32) は、通常、学習を安定させるために経験的に選ばれた値です。これにより、モデルがボックスのサイズや位置の変化に対して適切に対応するようになります。具体的には、x_offset や y_offset の値は 0.1 でスケーリングされ、log_width_scale や log_height_scale の値は 0.2 でスケーリングされます。

スケーリング後に取る値の範囲について

スケーリング後に box_target が取り得る値の範囲は、self._box_variance の値に依存しますが、基本的に以下のような範囲に収まります。

1. オフセット (x_offset, y_offset) の範囲:
    - self._box_variance が [0.1, 0.1, 0.2, 0.2] の場合、オフセット部分は 0.1 で割られるので、値が 10 倍になります。これにより、オフセットが [0.0, 1.0] の範囲にあると仮定すると、スケーリング後の範囲は [0.0, 10.0] になります。
2. スケーリング (log_width_scale, log_height_scale) の範囲:
    - log_width_scale や log_height_scale の部分は 0.2 で割られるので、値が 5 倍になります。これにより、ログ変換された値が通常の範囲内（たとえば、[-2, 2]）であれば、スケーリング後の範囲は [-10, 10] となります。

![image](https://github.com/user-attachments/assets/97806815-f2a5-4c9d-8539-c3ccdb536cae)

![image](https://github.com/user-attachments/assets/a5594bc8-1f5d-42b1-b348-9ed0ad9610fe)
#### マスク生成のループ

1. masksという辞書を初期化します。ここにマスク画像を保存します。
2. データフレーム df を image_id ごとにグループ化し、各グループごとに処理を行います。
3. gp['polygons'] に各バウンディングボックスをポリゴンに変換したリストを格納します。gp['bbox'] は文字列として保存されているため、まず eval でリストに変換し、その後 make_polygon 関数を適用します。
4. 画像サイズ (IMG_WIDTH, IMG_HEIGHT) で新しい画像（マスク画像）を作成し、全てのピクセルを0（黒）で初期化します。
5. ポリゴンをマスク画像に描画し、ポリゴンの内部を1（白）で塗りつぶします。
6. マスク画像をNumPy配列に変換し、masks 辞書に img_id をキーとして追加します。
![image](https://github.com/user-attachments/assets/f62d4b1f-e1f8-4df8-b423-773d63797564)
