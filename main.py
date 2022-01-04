# todo: https://www.kaggle.com/ronitf/heart-disease-uci
# todo: https://towardsdatascience.com/predicting-presence-of-heart-diseases-using-machine-learning-36f00f3edb2c
# todo: https://www.youtube.com/watch?v=1a7bB1ZcZ3k
# todo: https://www.youtube.com/watch?v=3LTSSzBZvXE
from pyspark.sql import SparkSession
import logging
import time
import os

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
    spark = SparkSession.builder.appName("HeartDisease").config("spark.some.config.option", "some-value").getOrCreate()
    #spark.sparkContext.setLogLevel("WARN")
    df = spark.read.csv(r'dataset.csv', mode='DROPMALFORMED', inferSchema=True, header=True)
    # Saving csv to parquet dataformat
    if not os.path.exists('df.parquet'):
        df.write.parquet('df.parquet')
    else:
        pass

    df_parquet = spark.read.parquet('df.parquet')
    df_parquet.printSchema()
    df_parquet.show(10)
    # Split dataset
    train, test = df_parquet.randomSplit([0.7, 0.3], seed=7)

    print(f'Train: {train.count()}')
    print(f'Test: {test.count()}')

except Exception:
    logging.exception(f'Dogodila se greska sljedeceg sadrzaja:')
else:
    logging.info(f'Uspjesno izvrsen zadatak.')
finally:
    logging.info(f'Obrada trajala: {time.strftime("%H sati, %M minuta i %S sekundi.", time.gmtime(time.time() - start_time))}\n')