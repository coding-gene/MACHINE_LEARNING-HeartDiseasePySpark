# todo: https://www.kaggle.com/ronitf/heart-disease-uci
# todo: https://towardsdatascience.com/predicting-presence-of-heart-diseases-using-machine-learning-36f00f3edb2c
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

    spark = SparkSession.builder.master("local[1]").appName("HeartDisease").getOrCreate()
    df = spark.read.csv('dataset.csv', header=True)

    df.printSchema()
    df.show(20)
    print(df.dtypes)

except Exception:
    logging.exception(f'Dogodila se greska sljedeceg sadrzaja:')
else:
    logging.info('Uspjesno izvrsen zadatak.')
finally:
    logging.info(f'Obrada trajala: {time.strftime("%H sati, %M minuta i %S sekundi.", time.gmtime(time.time() - start_time))}\n')