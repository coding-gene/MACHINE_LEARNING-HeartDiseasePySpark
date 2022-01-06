# todo: https://www.kaggle.com/ronitf/heart-disease-uci
# todo: https://towardsdatascience.com/predicting-presence-of-heart-diseases-using-machine-learning-36f00f3edb2c
# todo: https://www.youtube.com/watch?v=1a7bB1ZcZ3k
# todo: https://www.youtube.com/watch?v=3LTSSzBZvXE
# todo: Konvertirat sve logging-e na engleski jezik
from pyspark.sql import SparkSession
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import isnull, when, count, col
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, DecisionTreeClassifier, GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
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
    # Elementary data exploration
    #df_parquet.printSchema()
    #df_parquet.show(10)
    # print(df_parquet.columns)
    #df_parquet.select([count(when(isnull(c), c)).alias(c) for c in df_parquet.columns]).show()
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
    required_features = [
                            'age', 'sex',
                            'cp', 'trestbps',
                            'chol', 'fbs',
                            'restecg', 'thalach',
                            'exang', 'oldpeak',
                            'slope', 'ca', 'thal'
                        ]
    ml_assembler = VectorAssembler(inputCols=required_features, outputCol='features')
    ml_dataset = ml_assembler.transform(df_parquet)
    #ml_dataset.show(10)
    train, test = ml_dataset.randomSplit([0.7, 0.3], seed=7)
    logging.info(f'Podatci uspjesno podjeljeni na train ({train.count()}), i test ({test.count()}).')
    logging.info(f'Resaults of Machine Learning algorithms:')
    # todo: Logistic Regression
    lor = LogisticRegression(featuresCol='features', labelCol='target', maxIter=10)
    lorModel = lor.fit(train)
    lor_predictions = lorModel.transform(test)
    multi_evaluator = MulticlassClassificationEvaluator(labelCol='target', metricName='accuracy')
    logging.info(f'\tLogistic Regression Accuracy: {multi_evaluator.evaluate(lor_predictions)}')
    # todo: Random Forest Classifier
    rf = RandomForestClassifier(featuresCol='features', labelCol='target', maxDepth=5)
    rfModel = rf.fit(train)
    rf_predictions = rfModel.transform(test)
    multi_evaluator = MulticlassClassificationEvaluator(labelCol='target', metricName='accuracy')
    logging.info(f'\tRandom Forest Classifier Accuracy: {multi_evaluator.evaluate(rf_predictions)}')
    # todo: Decision Tree Classifier
    dt = DecisionTreeClassifier(featuresCol='features', labelCol='target', maxDepth=5)
    dtModel = dt.fit(train)
    dt_predictions = dtModel.transform(test)
    multi_evaluator = MulticlassClassificationEvaluator(labelCol='target', metricName='accuracy')
    logging.info(f'\tDecision Tree Classifier Accuracy: {multi_evaluator.evaluate(dt_predictions)}')
    # todo: Gradient-boosted Tree classifier
    gb = GBTClassifier(featuresCol='features', labelCol='target', maxDepth=5)
    gbModel = gb.fit(train)
    gb_predictions = gbModel.transform(test)
    multi_evaluator = MulticlassClassificationEvaluator(labelCol='target', metricName='accuracy')
    logging.info(f'\tGradient-boosted Tree classifier Accuracy: {multi_evaluator.evaluate(gb_predictions)}')
except Exception:
    logging.exception(f'Dogodila se greska sljedeceg sadrzaja:')
else:
    logging.info(f'Uspjesno izvrsen zadatak.')
finally:
    logging.info(f'Obrada trajala: {time.strftime("%H sati, %M minuta i %S sekundi.", time.gmtime(time.time() - start_time))}\n')