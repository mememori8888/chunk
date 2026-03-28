# Python ビッグデータ処理 実践チートシート（完全版）

巨大なデータ（数百万行〜テラバイト級、あるいは無限に続くセンサーデータ）をPythonで処理するための、実務レベルのアプローチ全7種をまとめたチートシートです。
「データが重すぎてPCが固まる」「処理が遅すぎる」という状況に合わせて、最適な手法を選択してください。

---

## 第1章：データの「流し方」を変える（処理アーキテクチャ）

メモリの限界を超えたり、無限のデータを扱うための根本的なアプローチです。

### 1. 遅延評価（Lazy Evaluation） / Polars
* **適した場面:** 数GB〜数十GBの「終わりがある巨大データ」を一番賢く高速に集計したい時。
* **ミソ:** 処理の指示だけを蓄積し、実行直前にシステムが「読まない列・捨てる行・最適な処理順」を自動計算してから一気に実行します。

```python
import polars as pl

# scan_csvで遅延評価（準備だけ）
query = pl.scan_csv("massive_data.csv")

# collect()が呼ばれた瞬間に最適化された計算が走る
result = (
    query.filter(pl.col("店舗名") == "東京")
    .group_by("商品名").agg(pl.col("売上").sum())
).collect()
```

### 2. チャンク処理（Chunking / 分割処理） / pandas
* **適した場面:** `pandas` しか使えない環境で、PCのメモリ容量を超える巨大なCSVを処理したい時。
* **ミソ:** 一気に飲み込まず、1万行ずつ切り分けて「読んで・処理して・捨てる」を繰り返します。メモリ消費が一定になります。

```python
import pandas as pd

# chunksizeを指定して小分けに読み込む
chunk_reader = pd.read_csv("massive_data.csv", chunksize=10000)

for chunk in chunk_reader:
    # 10000行分だけを処理（メモリがパンクしない）
    print(f"処理中... {len(chunk)}行")
    # ここに集計処理やDBへの保存処理を書く
```

### 3. ストリーム処理（Stream Processing）
* **適した場面:** IoTセンサーやSNSのログなど、24時間365日「無限に生まれ続けるデータ」をリアルタイムでさばきたい時。
* **ミソ:** データが全部揃うのを待たず、流れてきた端から瞬時に処理して次へ送る「ベルトコンベア方式」です。

```python
# ジェネレータを使った簡易的なストリーム処理の概念
def sensor_stream():
    while True:
        # 絶えずデータを取得して生み出す想定
        yield {"temp": 105, "status": "active"} 

for data in sensor_stream():
    # 流れてきた瞬間に異常検知などを行う
    if data["temp"] > 100:
        print("異常アラート！")
```

---

## 第2章：データの「持ち方」を変える（フォーマットとDB）

重いCSVから卒業し、データ分析に特化した形に変換するアプローチです。

### 4. プロの世界の常識「脱CSV」（Parquet形式）
* **適した場面:** 巨大なデータを何度も読み込んで分析するプロジェクトの初期段階。
* **ミソ:** データを横（行）ではなく縦（列）に圧縮して保存する「列指向フォーマット」。特定の列だけを爆速で読み込めます。

```python
import pandas as pd

# 一度だけCSVをParquetに変換（以降はParquetを使う）
df = pd.read_csv("data.csv")
df.to_parquet("data.parquet", index=False)

# 次回からは圧倒的な速度で読み込める
df_fast = pd.read_parquet("data.parquet")
```

### 5. 分析用DBの直接埋め込み（DuckDB）
* **適した場面:** SQLが得意で、重いデータベースサーバーを立てずに巨大ファイルを超高速集計したい時。
* **ミソ:** Pythonプログラム内に超高速DBエンジンを組み込み、ParquetやCSVファイルに直接SQLをぶつけます。

```python
import duckdb

# ファイルをメモリに読み込まず、直接SQLで集計
query = """
    SELECT 店舗名, SUM(売上) AS 総売上
    FROM 'data.parquet' 
    GROUP BY 店舗名
"""
result_df = duckdb.query(query).to_df()
print(result_df)
```

---

## 第3章：ハードウェアの「暴力」で解決する（インフラ拡張）

1台のPCの限界を、物理的なパワーでねじ伏せる最終手段です。

### 6. 複数台でタコ殴りにする「分散処理」（PySpark）
* **適した場面:** 1台のPCでは絶対に無理な、テラバイト（1000GB）級のデータを扱う時。
* **ミソ:** データを100台のサーバーに分割し、「いっせーのーで」で並列計算させます。大企業のビッグデータ基盤の標準です。

```python
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

# Sparkエンジンの起動
spark = SparkSession.builder.appName("BigDataApp").getOrCreate()

# 100台のサーバー群（クラスタ）で分散処理される想定
df = spark.read.parquet("data.parquet")
result = (
    df.filter(F.col("売上") >= 5000)
    .groupBy("店舗名")
    .agg(F.sum("売上").alias("総売上"))
)
result.show()
```

### 7. AIの脳みそで強制計算（RAPIDS cuDF）
* **適した場面:** 手元のPCやサーバーにNVIDIA製の強力なGPU（グラボ）が積まれている時。
* **ミソ:** 普通のCPUではなく、数千個のコアを持つGPUの並列処理能力をデータ集計に転用し、pandasの数十倍〜100倍の速度を出します。

```python
import cudf

# pandasと全く同じ書き方で、GPUのメモリ(VRAM)上で爆速処理される
gdf = cudf.read_parquet("data.parquet")

result_gdf = (
    gdf[gdf["売上"] >= 5000]
    .groupby("店舗名")
    .agg({"売上": "sum"})
)
print(result_gdf)
```
