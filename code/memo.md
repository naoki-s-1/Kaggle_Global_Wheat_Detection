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

_役割_
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
