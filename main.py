from pyspark.sql import SparkSession
from pyspark.sql.functions import isnull, when, count
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.classification import LinearSVC
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import time
import os

os.environ['HADOOP_HOME'] = r'C:\_SPARK\Hadoop'
os.environ["JAVA_HOME"] = r'C:\_JAVA'
start_time = time.time()
# noinspection PyBroadException
try:
    logging.basicConfig(filename='logs.txt',
                        filemode='a',
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    logging.info('Job started.')
    # Initializing spark session
    spark = SparkSession.builder.appName('HeartDisease').config('spark.some.config.option', 'some-value').getOrCreate()
    df = spark.read.csv('dataset.csv', mode='DROPMALFORMED', inferSchema=True, header=True)
    logging.info('Dataset successfully loaded to pyspark.')
    # Saving csv to parquet dataformat if doesn't exists
    if not os.path.exists('df.parquet'):
        df.write.parquet('df.parquet')
    else:
        pass
    # Read parquet into pyspark DataFrame
    df_parquet = spark.read.parquet('df.parquet')
    df_pandas = df_parquet.toPandas()
    logging.info('PySpark DataFrame loaded into parquet and pandas data format.')
    # Elementary data exploration
    df_parquet.printSchema()
    df_parquet.show(10)
    df_parquet.describe()
    df_parquet.select([count(when(isnull(c), c)).alias(c) for c in df_parquet.columns]).show()
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
    sns.heatmap(df_corr, annot=True, fmt='.2f', cmap='viridis', vmin=-1, vmax=1)
    figure = plt.gcf()
    figure.set_size_inches(20, 10)
    plt.savefig(r'plots/1. correlationMatrix.png', dpi=300)
    plt.close()
    logging.info('CorrelationMatrix successfully done.')
    # todo: HISTOGRAM
    df_pandas.hist()
    figure = plt.gcf()
    figure.set_size_inches(20, 10)
    plt.savefig(r'plots/2. histogram.png', dpi=300)
    plt.close()
    logging.info('Histogram successfully done.')
    # todo: BAR-PLOT TARGET COLUMN
    sns.countplot(x='target', data=df_pandas)
    figure = plt.gcf()
    figure.set_size_inches(20, 10)
    plt.savefig(r'plots/3. barPlot.png', dpi=300)
    plt.close()
    logging.info('Bar-Plot successfully done.')
    # todo: Prepare Data for Machine Learning algorithms
    required_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                         'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    ml_assembler = VectorAssembler(inputCols=required_features, outputCol='features')
    ml_dataset = ml_assembler.transform(df_parquet)
    # ml_dataset.show(10)
    train, test = ml_dataset.randomSplit([0.7, 0.3], seed=7)
    logging.info(f'Data successfully split into train ({train.count()}), test ({test.count()}) group.')
    logging.info('Results of Machine Learning Algorithms:')
    # todo: Logistic Regression
    lor = LogisticRegression(featuresCol='features', labelCol='target', maxIter=10)
    lorModel = lor.fit(train)
    lor_predictions = lorModel.transform(test)
    multi_evaluator = MulticlassClassificationEvaluator(labelCol='target', metricName='accuracy')
    logging.info(f'\tLogistic Regression Accuracy: '
                 f'{round((multi_evaluator.evaluate(lor_predictions)*100), 1)}%')
    # todo: Random Forest Classifier
    rf = RandomForestClassifier(featuresCol='features', labelCol='target', maxDepth=5)
    rfModel = rf.fit(train)
    rf_predictions = rfModel.transform(test)
    multi_evaluator = MulticlassClassificationEvaluator(labelCol='target', metricName='accuracy')
    logging.info(f'\tRandom Forest Classifier Accuracy: '
                 f'{round((multi_evaluator.evaluate(rf_predictions)*100), 1)}%')
    # todo: Decision Tree Classifier
    dt = DecisionTreeClassifier(featuresCol='features', labelCol='target', maxDepth=5)
    dtModel = dt.fit(train)
    dt_predictions = dtModel.transform(test)
    multi_evaluator = MulticlassClassificationEvaluator(labelCol='target', metricName='accuracy')
    logging.info(f'\tDecision Tree Classifier Accuracy: '
                 f'{round((multi_evaluator.evaluate(dt_predictions)*100), 1)}%')
    # todo: Gradient-boosted Tree classifier
    gb = GBTClassifier(featuresCol='features', labelCol='target', maxDepth=5)
    gbModel = gb.fit(train)
    gb_predictions = gbModel.transform(test)
    multi_evaluator = MulticlassClassificationEvaluator(labelCol='target', metricName='accuracy')
    logging.info(f'\tGradient-boosted Tree classifier Accuracy: '
                 f'{round((multi_evaluator.evaluate(gb_predictions)*100), 1)}%')
    # todo: Linear Support Vector Machine
    sv = LinearSVC(featuresCol='features', labelCol='target', maxIter=10)
    svModel = sv.fit(train)
    sv_predictions = svModel.transform(test)
    multi_evaluator = MulticlassClassificationEvaluator(labelCol='target', metricName='accuracy')
    logging.info(f'\tLinear Support Vector Machine Accuracy: '
                 f'{round((multi_evaluator.evaluate(sv_predictions)*100), 1)}%')

    # todo: Multilayer perceptron classifier
    train = train.withColumnRenamed('target', 'label')
    test = test.withColumnRenamed('target', 'label')
    layers = [4, 2, 2]
    mp = MultilayerPerceptronClassifier(featuresCol='features', labelCol='label', maxIter=10,
                                        layers=layers, blockSize=128, seed=1234)
    mpModel = mp.fit(train)
    print(mpModel)
    mp_predictions = mpModel.transform(test)
    # mp_predictions.show()
    mp_predictionAndLabels = mp_predictions.select('prediction', 'label')
    # mp_predictionAndLabels.show()
    multi_evaluator = MulticlassClassificationEvaluator(metricName='accuracy')
    # logging.info(f'\tMultilayer perceptron classifier Accuracy: '
    #             f'{multi_evaluator.evaluate(mp_predictionAndLabels)}%')
except Exception:
    logging.exception('An error occurred during job performing:')
else:
    logging.info('Job ended.')
finally:
    logging.info(
        f'Job duration: {time.strftime("%H hours, %M minutes, %S seconds.", time.gmtime(time.time() - start_time))}\n')
