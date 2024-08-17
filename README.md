# Kaggle_Global_Wheat_Detection
Global Wheat Detectionコンペの記録

![](https://storage.googleapis.com/kaggle-media/competitions/UofS-Wheat/descriptionimage.png)

### データセットの説明
データ取得とプロセスの詳細については、https://arxiv.org/abs/2005.02162 をご覧ください

データ形式はどうなると思いますか?
データは小麦畑の画像であり、識別された各小麦の頭に境界ボックスがあります。すべての画像に小麦の頭/バウンディングボックスが含まれているわけではありません。画像は世界中の多くの場所で記録されました。

CSV データは単純です - 画像 ID は特定の画像のファイル名と一致し、画像の幅と高さがバウンディングボックス (以下を参照) と共に含まれます。各バウンディング ボックスには の行があります。すべての画像にバウンディングボックスがあるわけではありません。train.csv

ほとんどのテストセット画像は非表示になっています。テスト画像の小さなサブセットは、コードの記述に使用するために含まれています。

私は何を予測しているのか?
各小麦の頭の周りの境界ボックスを、それらを含む画像で予測しようとしています。小麦の頭がない場合は、バウンディング ボックスがないと予測する必要があります。

#### ファイル  
- train.csv - トレーニングデータ  
- sample_submission.csv - 正しい形式のサンプル提出ファイル  
- train.zip - トレーニング画像  
- test.zip - テストイメージ  

#### 列  
- image_id- 一意の画像 ID  
- width, - 画像の幅と高さheight  
- bbox- Python スタイルの [xmin, ymin, width, height] のリストとして書式設定されたバウンディングボックス  
- 等。

### 2024-08-11 日曜日
yolov5を使ってテストしたがsubmission.csvがうまく作成されず提出できなかった

### 2024-08-12 月曜日
- ImageOps.fit 関数の構造
  - ImageOps.fit 関数は、Pillowライブラリに含まれる ImageOps モジュールのメソッドで、指定された画像を指定のサイズに収めるためにリサイズする機能を持っています。この処理では、アスペクト比（縦横比）を保持しつつ、画像の中央部分が希望のサイズに収まるようにトリミングされます。

### 2024-08-14 水曜日
チャットGPTを駆使するも複雑でベースモデルのサブミットまでたどり着けず。

### 2024-08-16 金曜日
進展なし

### 2024-08-17 土曜日
#### 便利な関数
- 文字列表現をPythonのオブジェクトに変換する  
  import ast ライブラリ
  df.bbox=df.bbox.apply(lambda x: ast.literal_eval(x)): 使用例    

- ast.literal_eval(x):
  - これは ast モジュール（Abstract Syntax Treesの略）に含まれる literal_eval 関数です。
literal_eval は、文字列で表現されたPythonリテラルを、実際のPythonオブジェクト（リスト、タプル、辞書など）に変換します。通常、文字列が安全に評価されることを保証するために使用されます。
例えば、x が "[1, 2, 3, 4]" のような文字列の場合、ast.literal_eval(x) はこれを実際のリスト [1, 2, 3, 4] に変換します。
- numpyを使ったbboxの要素の分割
  - bbox = np.stack(df['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))
  - array([[834., 222.,  56.,  36.],  
           [226., 548., 130.,  58.],  
           [377., 504.,  74., 160.],  
           ...,  
           [134., 228., 141.,  71.],  
           [430.,  13., 184.,  79.],  
           [875., 740.,  94.,  61.]])

- CV予測をする際に役立ちそうだった  
gfk = GroupKFold(n_splits=4)  
for fold, (_, va_idx) in enumerate(gfk.split(X=df, y=df['source'],  
                                                  groups=df['image_id']),  
                                   start=1):  
    df.loc[va_idx, 'fold'] = fold  

