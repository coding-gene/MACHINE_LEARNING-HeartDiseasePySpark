# todo: https://www.kaggle.com/ronitf/heart-disease-uci
# todo: https://towardsdatascience.com/predicting-presence-of-heart-diseases-using-machine-learning-36f00f3edb2c
# todo: https://www.youtube.com/watch?v=1a7bB1ZcZ3k
# todo: https://www.youtube.com/watch?v=3LTSSzBZvXE
# todo: Konvertirat sve logging-e na engleski jezik
from pyspark.sql import SparkSession
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import time
import os
import pandas as pd

os.environ['HADOOP_HOME'] = r'C:\_SPARK\Hadoop'
os.environ["JAVA_HOME"] = r'C:\_JAVA'

try:
    start_time = time.time()
    logging.basicConfig(filename='logs.txt',
                        filemode='a',
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    logging.info(f'Pocetak izvrsavanja zadatka.')
    # Initializing spark session
    spark = SparkSession.builder.appName('HeartDisease').config('spark.some.config.option', 'some-value').getOrCreate()
    df = spark.read.csv(r'dataset.csv', mode='DROPMALFORMED', inferSchema=True, header=True)
    logging.info(f'Podatci uspjesno ucitani u pyspark.')
    # Saving csv to parquet dataformat if doesn't exists
    if not os.path.exists('df.parquet'):
        df.write.parquet('df.parquet')
    else:
        pass
    # Read parquet into pyspark DataFrame
    df_parquet = spark.read.parquet('df.parquet')
    df_pandas = df_parquet.toPandas()
    logging.info(f'PySpark DataFrame uspjesno inicijaliziran u parquet i pandas data format.')
    # Elementary data explanation
    #df_parquet.printSchema()
    #df_parquet.show(10)
    # todo: CORRELATION MATRIX
        # convert to vector column
    vector_col = "corr_features"
    assembler = VectorAssembler(inputCols=df_parquet.columns, outputCol=vector_col)
    df_vector = assembler.transform(df_parquet).select(vector_col)
        # get a list of lists
    matrix = Correlation.corr(df_vector, vector_col).collect()[0][0]
    corrmatrix = matrix.toArray().tolist()
        # corrmatrix to Pandas DataFrame
    df_corr = pd.DataFrame(corrmatrix, columns=df_parquet.columns, index=df_parquet.columns)
        # plot a heatmap
    sns.heatmap(df_corr, annot=True, fmt=".2f", cmap='viridis', vmin=-1, vmax=1)
    figure = plt.gcf()
    figure.set_size_inches(20, 10)
    plt.savefig(r'plots\1. correlationMatrix.png', dpi=300)
    plt.close()
    logging.info(f'CorrelationMatrix uspjesno proveden.')
    # todo: HISTOGRAM
    df_pandas.hist()
    figure = plt.gcf()
    figure.set_size_inches(20, 10)
    plt.savefig(r'plots\2. histogram.png', dpi=300)
    plt.close()
    logging.info(f'Histogram uspjesno proveden.')
    # todo: BAR-PLOT TARGET COLUMN
    sns.countplot(x='target', data=df_pandas)
    figure = plt.gcf()
    figure.set_size_inches(20, 10)
    plt.savefig(r'plots\3. barPlot.png', dpi=300)
    plt.close()
    logging.info(f'Bar-Plot uspjesno proveden.')
    # todo: Prepare Data for Machine Learning algorithms
    train, test = df_parquet.randomSplit([0.7, 0.3], seed=7)
    logging.info(f'Podatci uspjesno podjeljeni na train (ukupno: {train.count()}), i test (ukupno: {test.count()}).')

    # todo: Logistic Regression
    # todo: Linear Regression


except Exception:
    logging.exception(f'Dogodila se greska sljedeceg sadrzaja:')
else:
    logging.info(f'Uspjesno izvrsen zadatak.')
finally:
    logging.info(f'Obrada trajala: {time.strftime("%H sati, %M minuta i %S sekundi.", time.gmtime(time.time() - start_time))}\n')