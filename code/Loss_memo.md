# Loss関数、クラス

## Smooth L1 lossとFocal Lossをkerasカスタム損失として実装する

![image](https://github.com/user-attachments/assets/8ccf79f4-7333-49d4-8e52-c01b0f475dba)

クラス概要

RetinaNetBoxLoss クラスは、Smooth L1損失（Huber損失とも呼ばれる）を実装しています。これは、バウンディングボックスの座標予測の誤差を計算するために使用されます。

### 各メソッドの処理の内容について
__init__ メソッド

- 内容: クラスの初期化メソッドで、deltaというパラメータを受け取ります。このパラメータはSmooth L1 lossの閾値となります。
- 引数:
  - delta: Smooth L1 lossの閾値。
- 出力: なし（クラスのインスタンスを初期化する）

call メソッド

- 内容: Smooth L1 lossを計算します。具体的には、予測値 (y_pred) と真値 (y_true) の間の差分を計算し、その差分に基づいて損失を計算します。
- 引数:
  - y_true: 真値のテンソル。
  - y_pred: 予測値のテンソル。
- 出力: 損失のテンソル（各サンプルごとの損失）

### 入力の型、形状、出力の型、形状について

call メソッドの詳細

- 入力の型と形状:
  - y_true: tf.Tensor、形状は(batch_size, num_boxes, 4)（ボックスの真値座標）
  - y_pred: tf.Tensor、形状は(batch_size, num_boxes, 4)（ボックスの予測座標）
- 出力の型と形状:
  - loss: tf.Tensor、形状は(batch_size, num_boxes)（各ボックスごとの損失）

各処理ごとの意味

1. 差分の計算 (difference = y_true - y_pred):

    - 意味: 予測値と真値の間の差を計算します。
    - 入力の型と形状: y_trueとy_pred、形状は(batch_size, num_boxes, 4)
    - 出力の型と形状: difference、形状は(batch_size, num_boxes, 4)

2. 絶対差分の計算 (absolute_difference = tf.abs(difference)):

    - 意味: 差分の絶対値を計算します。
    - 入力の型と形状: difference、形状は(batch_size, num_boxes, 4)
    - 出力の型と形状: absolute_difference、形状は(batch_size, num_boxes, 4)

3. 差分の二乗の計算 (squared_difference = difference ** 2):

    - 意味: 差分の二乗を計算します。
    - 入力の型と形状: difference、形状は(batch_size, num_boxes, 4)
    - 出力の型と形状: squared_difference、形状は(batch_size, num_boxes, 4)

4. Smooth L1 lossの計算 (loss = tf.where(tf.less(absolute_difference, self._delta), 0.5 * squared_difference, absolute_difference - 0.5)):

    - 意味: Smooth L1 lossを計算します。絶対差分がdelta未満の場合、二乗和の0.5倍を使用し、それ以外の場合は絶対差分から0.5を引いた値を使用します。
    - 入力の型と形状: absolute_differenceとsquared_difference、形状は(batch_size, num_boxes, 4)
    - 出力の型と形状: loss、形状は(batch_size, num_boxes, 4)

5. 損失の合計 (return tf.reduce_sum(loss, axis=-1)):

    - 意味: 各ボックスごとの損失を合計します。
    - 入力の型と形状: loss、形状は(batch_size, num_boxes, 4)
    - 出力の型と形状: 合計された損失のテンソル、形状は(batch_size, num_boxes)

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

1. クロスエントロピー損失の計算 (cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred):

    - 意味: 各クラスごとのクロスエントロピー損失を計算します。
    - 入力の型と形状: y_trueとy_pred、形状は(batch_size, num_boxes, num_classes)
    - 出力の型と形状: cross_entropy、形状は(batch_size, num_boxes, num_classes)

2. 予測確率の計算 (probs = tf.nn.sigmoid(y_pred)):

    - 意味: シグモイド関数を使用して、予測値を確率に変換します。
    - 入力の型と形状: y_pred、形状は(batch_size, num_boxes, num_classes)
    - 出力の型と形状: probs、形状は(batch_size, num_boxes, num_classes)

3. alphaの計算 (alpha = tf.where(tf.equal(y_true, 1.0), self._alpha, (1.0 - self._alpha))):

    - 意味: 各クラスに対して、真値が1の場合にalphaを、それ以外の場合に1 - alphaを適用します。
    - 入力の型と形状: y_true、形状は(batch_size, num_boxes, num_classes)
    - 出力の型と形状: alpha、形状は(batch_size, num_boxes, num_classes)

4. ptの計算 (pt = tf.where(tf.equal(y_true, 1.0), probs, 1 - probs)):

    - 意味: 真値が1の場合に予測確率を、それ以外の場合に1から予測確率を引いた値を使用します。
    - 入力の型と形状: y_trueとprobs、形状は(batch_size, num_boxes, num_classes)
    - 出力の型と形状: pt、形状は(batch_size, num_boxes, num_classes)

5. Focal lossの計算 (loss = alpha * tf.pow(1.0 - pt, self._gamma) * cross_entropy):

    - 意味: Focal lossを計算します。クロスエントロピー損失に、クラス不均衡を補正するためのalphaと、難易度の高いサンプルを強調するための(1 - pt)^gammaを乗じます。
    - 入力の型と形状: alpha, pt, cross_entropy、形状は(batch_size, num_boxes, num_classes)
    - 出力の型と形状: loss、形状は(batch_size, num_boxes, num_classes)

5. 損失の合計 (return tf.reduce_sum(loss, axis=-1)):

    - 意味: 各ボックスごとの損失を合計します。
    - 入力の型と形状: loss、形状は(batch_size, num_boxes, num_classes)
    - 出力の型と形状: 合計された損失のテンソル、形状は(batch_size, num_boxes)

![image](https://github.com/user-attachments/assets/1009631f-e995-44d3-965e-d5333d047c5c)

クラス概要

RetinaNetLoss クラスは、RetinaNetBoxLoss と RetinaNetClassificationLoss を組み合わせ、総合的な損失を計算するためのラッパーです。

### メソッドの処理内容

__init__(self, num_classes=2, alpha=0.25, gamma=2.0, delta=1.0):

- 内容: クラスの初期化メソッドで、分類損失 (RetinaNetClassificationLoss) と回帰損失 (RetinaNetBoxLoss) をインスタンス化します。
- 引数:
  - num_classes: クラス数。
  - alpha: Focal lossのalphaパラメータ。
  - gamma: Focal lossのgammaパラメータ。
  - delta: Smooth L1 lossのdeltaパラメータ。

call(self, y_true, y_pred):

- 内容: 分類損失と回帰損失を計算し、それらを合計します。具体的には、予測値 (y_pred) と真値 (y_true) からクラスラベルとボックス座標を抽出し、各損失関数を適用します。無視すべきボックスや正のボックスを考慮し、最終的な損失を計算します。
- 引数:
  - y_true: 真値のテンソル。
  - y_pred: 予測値のテンソル。
- 出力: 総合損失のテンソル

入力と出力の型・形状

- 入力:
  - y_true: 真のラベルとボックス (型: tf.Tensor, 形状: [batch_size, num_boxes, 5])
  - y_pred: 予測されたボックスとクラススコア (型: tf.Tensor, 形状: [batch_size, num_boxes, num_classes + 4])

- 出力:
  - 総合損失 (型: tf.Tensor, 形状: [batch_size])

### 処理の意味と入力・出力の形状

1. 予測値の型変換 (y_pred = tf.cast(y_pred, dtype=tf.float32)):

    - 意味: 予測値をfloat32型に変換します。
    - 入力の型と形状: y_pred、形状は(batch_size, num_boxes, num_classes + 4)
    - 出力の型と形状: y_pred、形状は(batch_size, num_boxes, num_classes + 4)

2. ボックスラベルの抽出 (box_labels = y_true[:, :, :4]):

    - 意味: 真値からボックス座標を抽出します。
    - 入力の型と形状: y_true、形状は(batch_size, num_boxes, 5)
    - 出力の型と形状: box_labels、形状は(batch_size, num_boxes, 4)

3. ボックス予測値の抽出 (box_predictions = y_pred[:, :, :4]):

    - 意味: 予測値からボックス座標を抽出します。
    - 入力の型と形状: y_pred、形状は(batch_size, num_boxes, num_classes + 4)
    - 出力の型と形状: box_predictions、形状は(batch_size, num_boxes, 4)

4. クラスラベルのone-hotエンコーディング (cls_labels = tf.one_hot(tf.cast(y_true[:, :, 4], dtype=tf.int32), depth=self._num_classes, dtype=tf.float32)):

    - 意味: 真値のクラスラベルをone-hotエンコーディングします。
    - 入力の型と形状: y_true、形状は(batch_size, num_boxes, 5)
    - 出力の型と形状: cls_labels、形状は(batch_size, num_boxes, num_classes)

5. クラス予測値の抽出 (cls_predictions = y_pred[:, :, 4:]):

    - 意味: 予測値からクラスラベルを抽出します。
    - 入力の型と形状: y_pred、形状は(batch_size, num_boxes, num_classes + 4)
    - 出力の型と形状: cls_predictions、形状は(batch_size, num_boxes, num_classes)

6. ポジティブマスクの作成 (positive_mask = tf.cast(tf.greater(y_true[:, :, 4], -1.0), dtype=tf.float32)):

    - 意味: クラスラベルが-1より大きい（有効なボックス）の場合、ポジティブマスクを1に設定します。
    - 入力の型と形状: y_true、形状は(batch_size, num_boxes, 5)
    - 出力の型と形状: positive_mask、形状は(batch_size, num_boxes)

7. 無視マスクの作成 (ignore_mask = tf.cast(tf.equal(y_true[:, :, 4], -2.0), dtype=tf.float32)):

    - 意味: クラスラベルが-2（無視すべきボックス）の場合、無視マスクを1に設定します。
    - 入力の型と形状: y_true、形状は(batch_size, num_boxes, 5)
    - 出力の型と形状: ignore_mask、形状は(batch_size, num_boxes)

8. 分類損失の計算 (clf_loss = self._clf_loss(cls_labels, cls_predictions)):

    - 意味: クラスラベルと予測値に基づいて分類損失を計算します。
    - 入力の型と形状: cls_labelsとcls_predictions、形状は(batch_size, num_boxes, num_classes)
    - 出力の型と形状: clf_loss、形状は(batch_size, num_boxes)

9. 回帰損失の計算 (box_loss = self._box_loss(box_labels, box_predictions)):

    - 意味: ボックスの真値と予測値に基づいて回帰損失を計算します。
    - 入力の型と形状: box_labelsとbox_predictions、形状は(batch_size, num_boxes, 4)
    - 出力の型と形状: box_loss、形状は(batch_size, num_boxes)

10. 無視するボックスの損失を0に設定 (clf_loss = tf.where(tf.equal(ignore_mask, 1.0), 0.0, clf_loss)):

    - 意味: 無視すべきボックスの分類損失を0に設定します。
    - 入力の型と形状: ignore_maskとclf_loss、形状は(batch_size, num_boxes)
    - 出力の型と形状: 更新されたclf_loss、形状は(batch_size, num_boxes)

11. 正のボックスでない損失を0に設定 (box_loss = tf.where(tf.equal(positive_mask, 1.0), box_loss, 0.0)):

    - 意味: 正のボックスでない損失を0に設定します。
    - 入力の型と形状: positive_maskとbox_loss、形状は(batch_size, num_boxes)
    - 出力の型と形状: 更新されたbox_loss、形状は(batch_size, num_boxes)

12. 正のボックスの数を正規化 (normalizer = tf.reduce_sum(positive_mask, axis=-1)):

    - 意味: 正のボックスの数を計算し、それを正規化に使用します。
    - 入力の型と形状: positive_mask、形状は(batch_size, num_boxes)
    - 出力の型と形状: normalizer、形状は(batch_size,)

13. 分類損失の正規化 (clf_loss = tf.math.divide_no_nan(tf.reduce_sum(clf_loss, axis=-1), normalizer)):

    - 意味: 正のボックスの数で分類損失を正規化します。
    - 入力の型と形状: clf_loss、形状は(batch_size, num_boxes)
    - 出力の型と形状: 正規化されたclf_loss、形状は(batch_size,)

14. 回帰損失の正規化 (box_loss = tf.math.divide_no_nan(tf.reduce_sum(box_loss, axis=-1), normalizer)):

    - 意味: 正のボックスの数で回帰損失を正規化します。
    - 入力の型と形状: box_loss、形状は(batch_size, num_boxes)
    - 出力の型と形状: 正規化されたbox_loss、形状は(batch_size,)

15. 総合損失の計算 (loss = clf_loss + box_loss):

    - 意味: 正規化された分類損失と回帰損失を合計して総合損失を計算します。
    - 入力の型と形状: clf_lossとbox_loss、形状は(batch_size,)
    - 出力の型と形状: loss、形状は(batch_size,)

16. 総合損失の返却 (return loss):

    - 意味: 総合損失を返却します。
    - 入力の型と形状: loss、形状は(batch_size,)
    - 出力の型と形状: loss、形状は(batch_size,)





