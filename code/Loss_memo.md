# Loss関数、クラス

## Smooth L1 lossとFocal Lossをkerasカスタム損失として実装する

![image](https://github.com/user-attachments/assets/8ccf79f4-7333-49d4-8e52-c01b0f475dba)

クラス概要

RetinaNetBoxLoss クラスは、Smooth L1損失（Huber損失とも呼ばれる）を実装しています。これは、バウンディングボックスの座標予測の誤差を計算するために使用されます。

メソッドの処理内容

- __init__(self, delta):
  - このメソッドは、クラスの初期化を行います。
  - delta は、L1損失とL2損失の間を区切る閾値です。

- call(self, y_true, y_pred):
  - y_true と y_pred の差を計算し、絶対値 (absolute_difference) と二乗値 (squared_difference) を求めます。
  - absolute_difference が delta より小さい場合は、L2損失 (0.5 * squared_difference) を使用し、それ以外の場合はL1損失 (absolute_difference - 0.5) を使用します。
  - 最後に、損失の総和を計算して返します。

入力と出力の型・形状

- 入力:
  - y_true: バウンディングボックスの真の値 (型: tf.Tensor, 形状: [batch_size, num_boxes, 4])
  - y_pred: バウンディングボックスの予測値 (型: tf.Tensor, 形状: [batch_size, num_boxes, 4])

- 出力:
  - 損失の総和 (型: tf.Tensor, 形状: [batch_size, num_boxes])

- 処理の意味と入力・出力の形状
  - 差の計算:
    - difference = y_true - y_pred
    - 意味: 真のバウンディングボックス座標と予測座標の誤差を計算します。
    - 形状: [batch_size, num_boxes, 4]

- 絶対値と二乗値の計算:
  - absolute_difference = tf.abs(difference)
  - squared_difference = difference ** 2
  - 意味: 誤差の絶対値と二乗値を計算します。
  - 形状: 両方とも [batch_size, num_boxes, 4]

- 損失の計算:
  - loss = tf.where(tf.less(absolute_difference, self._delta), 0.5 * squared_difference,         absolute_difference - 0.5)
  - 意味: Smooth L1損失を計算します。
  - 形状: [batch_size, num_boxes, 4]

- 損失の総和:
  - return tf.reduce_sum(loss, axis=-1)
  - 意味: ボックスごとの損失を合計し、1つのスカラー値にします。
  - 形状: [batch_size, num_boxes]

![image](https://github.com/user-attachments/assets/40155bc0-ff1f-40db-8362-b8bdd344cce7)

クラス概要

RetinaNetClassificationLoss クラスは、Focal Lossを実装しています。Focal Lossは、クラスの不均衡を処理するために設計された損失関数であり、難しい例により多くの重みを与えます。

メソッドの処理内容

- __init__(self, alpha, gamma):
  - alpha と gamma はFocal Lossのハイパーパラメータです。
  - alpha はクラスのバランスを調整し、gamma は困難な例に対する重み付けを制御します。

- call(self, y_true, y_pred):
  - y_true と y_pred を使用して、まず交差エントロピー損失 (cross_entropy) を計算します。
  - 次に、予測された確率 (probs) を計算し、クラスごとの重み (alpha) と補正された確率 (pt) を計算します。
  - 最後に、Focal Lossの計算式に基づいて損失 (loss) を計算し、合計を返します。

入力と出力の型・形状

- 入力:
  - y_true: 真のラベル (型: tf.Tensor, 形状: [batch_size, num_boxes, num_classes])
  - y_pred: 予測されたロジット (型: tf.Tensor, 形状: [batch_size, num_boxes, num_classes])

- 出力:
  - 損失の総和 (型: tf.Tensor, 形状: [batch_size, num_boxes])

処理の意味と入力・出力の形状

- 交差エントロピーの計算:
  - cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
  - 意味: 予測されたロジットと真のラベルとの間の交差エントロピーを計算します。
  - 形状: [batch_size, num_boxes, num_classes]

- 確率の計算:
  - probs = tf.nn.sigmoid(y_pred)
  - 意味: 予測されたロジットをシグモイド関数を通じて確率に変換します。
  - 形状: [batch_size, num_boxes, num_classes]

- Focal Lossの計算:
  - loss = alpha * tf.pow(1.0 - pt, self._gamma) * cross_entropy
  - 意味: Focal Lossを計算します。これは、困難な例に対してより大きな重みを与えるために設計されています。
  - 形状: [batch_size, num_boxes, num_classes]

- 損失の総和:
  - return tf.reduce_sum(loss, axis=-1)
  - 意味: クラスごとの損失を合計し、1つのスカラー値にします。
  - 形状: [batch_size, num_boxes]

![image](https://github.com/user-attachments/assets/1009631f-e995-44d3-965e-d5333d047c5c)

クラス概要

RetinaNetLoss クラスは、RetinaNetBoxLoss と RetinaNetClassificationLoss を組み合わせ、総合的な損失を計算するためのラッパーです。

メソッドの処理内容

- __init__(self, num_classes=2, alpha=0.25, gamma=2.0, delta=1.0):
  - このメソッドは、クラスの初期化を行います。
  - num_classes、alpha、gamma、delta は、それぞれ分類損失とボックス損失のハイパーパラメータです。

- call(self, y_true, y_pred):
  - y_true と y_pred を使用して、ボックス損失 (box_loss) と分類損失 (clf_loss) を計算します。
  - 計算された損失を合計して最終的な損失 (loss) として返します。

入力と出力の型・形状

- 入力:
  - y_true: 真のラベルとボックス (型: tf.Tensor, 形状: [batch_size, num_boxes, 5])
  - y_pred: 予測されたボックスとクラススコア (型: tf.Tensor, 形状: [batch_size, num_boxes, num_classes + 4])

- 出力:
  - 総合損失 (型: tf.Tensor, 形状: [batch_size])

処理の意味と入力・出力の形状

- ボックス損失の計算:
  - box_loss = self._box_loss(box_labels, box_predictions)
  - 意味: バウンディングボックスの予測と真の値の間の誤差を計算します。
  - 形状: [batch_size, num_boxes]

- 分類損失の計算:
  - clf_loss = self._clf_loss(cls_labels, cls_predictions)
  - 意味: 予測されたクラスと真のクラスの間の誤差を計算します。
  - 形状: [batch_size, num_boxes]

損失の正規化:
  - clf_loss = tf.math.divide_no_nan(tf.reduce_sum(clf_loss, axis=-1), normalizer)
  - box_loss = tf.math.divide_no_nan(tf.reduce_sum(box_loss, axis=-1), normalizer)
  - 意味: 正のサンプル数で分類損失とボックス損失を正規化します。
  - 形状: [batch_size]

- 最終損失の計算:
  - loss = clf_loss + box_loss
  - 意味: 分類損失とボックス損失を合計し、最終的な損失を計算します。
  - 形状: [batch_size]





