# todo: https://www.kaggle.com/ronitf/heart-disease-uci
# todo: https://towardsdatascience.com/predicting-presence-of-heart-diseases-using-machine-learning-36f00f3edb2c
# todo: https://www.youtube.com/watch?v=1a7bB1ZcZ3k
from pyspark.sql import SparkSession
import logging
import time


try:
    start_time = time.time()
    logging.basicConfig(filename='logs.txt',
                        filemode='a',
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    logging.info(f'Pocetak izvrsavanja zadatka.')

    spark = SparkSession.builder.appName("HeartDisease").getOrCreate()
    df = spark.read.csv(r'dataset.csv', inferSchema=True, header=True)
    #df.write.parquet(r'dataset.parquet')
    #df = spark.read.parquet(r'C:\Users\Ivana\PycharmProjects\HeartDiseasePySpark\dataset.parquet')
    # Split dataset
    train, test = df.randomSplit([0.7, 0.3], seed=7)

    print(f'Train: {train.count()}')
    print(f'Test: {test.count()}')

    df.printSchema()
    df.show(20)
    #print(df.dtypes)

except Exception:
    logging.exception(f'Dogodila se greska sljedeceg sadrzaja:')
else:
    logging.info('Uspjesno izvrsen zadatak.')
finally:
    logging.info(f'Obrada trajala: {time.strftime("%H sati, %M minuta i %S sekundi.", time.gmtime(time.time() - start_time))}\n')