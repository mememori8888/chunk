import os
import time
import pandas as pd
import numpy as np

# ==========================================
# 準備：100万行のダミーデータを一瞬で作る
# ==========================================
print("--- データ準備開始 ---")
num_rows = 1_000_000
data = {
    "店舗名": np.random.choice(["東京", "大阪", "名古屋", "福岡", "札幌"], num_rows),
    "商品ID": np.random.randint(1, 100, num_rows),
    "売上金額": np.random.randint(100, 10000, num_rows)
}
df_dummy = pd.DataFrame(data)

# CSVとParquetの両方の形式で保存する
csv_path = "dummy_1M.csv"
parquet_path = "dummy_1M.parquet"

if not os.path.exists(csv_path):
    df_dummy.to_csv(csv_path, index=False)
if not os.path.exists(parquet_path):
    df_dummy.to_parquet(parquet_path, index=False)
print("100万行のデータ作成・保存完了！\n")


# ==========================================
# 1. プロの世界の常識「脱CSV」（Parquet形式の恩恵）
# ==========================================
print("--- 1. CSV vs Parquet 読み込み速度対決 ---")
# CSVの読み込み
start = time.time()
df_csv = pd.read_csv(csv_path)
csv_time = time.time() - start
print(f"CSV読み込み時間: {csv_time:.3f} 秒")

# Parquetの読み込み
start = time.time()
df_parquet = pd.read_parquet(parquet_path)
parquet_time = time.time() - start
print(f"Parquet読み込み時間: {parquet_time:.3f} 秒 (圧倒的に速いはずです)\n")


# ==========================================
# 2. 分析用データベースを直接埋め込む（DuckDB）
# ==========================================
print("--- 2. DuckDB による直接SQL集計 ---")
try:
    import duckdb
    start = time.time()
    # メモリに読み込まず、Parquetファイルに直接SQLをぶつけて集計する
    query = f"""
        SELECT 店舗名, SUM(売上金額) AS 総売上
        FROM '{parquet_path}'
        WHERE 売上金額 >= 5000
        GROUP BY 店舗名
        ORDER BY 総売上 DESC
    """
    result_duckdb = duckdb.query(query).to_df()
    print(f"DuckDB処理時間: {time.time() - start:.3f} 秒")
    print(result_duckdb.head(3).to_string(), "\n")
except ImportError:
    print("DuckDBがインストールされていません (pip install duckdb)\n")


# ==========================================
# 3. パソコン複数台でタコ殴り（PySpark）
# ==========================================
print("--- 3. PySpark による分散処理（ローカルモード） ---")
try:
    from pyspark.sql import SparkSession
    import pyspark.sql.functions as F
    
    # Sparkエンジンの立ち上げ
    spark = SparkSession.builder.appName("BigDataTest").getOrCreate()
    
    start = time.time()
    # Parquetを読み込み、遅延評価で集計プランを組み立てる
    df_spark = spark.read.parquet(parquet_path)
    result_spark = (
        df_spark.filter(F.col("売上金額") >= 5000)
        .groupBy("店舗名")
        .agg(F.sum("売上金額").alias("総売上"))
        .orderBy(F.col("総売上").desc())
    )
    # collect()やshow()で一気に計算実行
    result_spark.show(3)
    print(f"PySpark処理時間: {time.time() - start:.3f} 秒")
    spark.stop()
    print()
except ImportError:
    print("PySpark環境がありません（実務で巨大サーバーを使う際に活躍します）\n")
except Exception as e:
    print("PySparkの実行にはJava(JVM)環境が必要です。今回はスキップします。\n")


# ==========================================
# 4. AIの脳みそ（GPU）で強制計算（RAPIDS cuDF）
# ==========================================
print("--- 4. RAPIDS cuDF によるGPU爆速計算 ---")
try:
    import cudf
    start = time.time()
    # GPUのメモリ(VRAM)にデータを読み込む
    gdf = cudf.read_parquet(parquet_path)
    
    # 処理の書き方は通常のpandasと全く同じ
    result_cudf = (
        gdf[gdf["売上金額"] >= 5000]
        .groupby("店舗名")
        .agg({"売上金額": "sum"})
        .sort_values("売上金額", ascending=False)
    )
    print(result_cudf.head(3))
    print(f"cuDF(GPU)処理時間: {time.time() - start:.3f} 秒\n")
except ImportError:
    print("cuDF環境（NVIDIA GPUとLinux環境）がありません。今回はスキップします。\n")
