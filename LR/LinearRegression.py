#mount('/mnt/spark-mooc')
#dbutils.fs.mount('s3n://AKIAJUIYIBOAUUTJ3G5A:SU%2FsifB7wuzeWexDJCMTBVxG7MLIbZkk4ZcB+qzd@spark-mooc', '/mnt/spark-mooc/')
# check if it is mounted
dbutils.fs.ls('/mnt/spark-mooc/')

import os.path
baseDir = os.path.join('databricks-datasets', 'cs190')
inputPath = os.path.join('data-001', 'millionsong.txt')
fileName = os.path.join(baseDir, inputPath)

numPartitions = 2
rawData = sc.textFile(fileName, numPartitions)

numPoints = rawData.count()
print (numPoints)
samplePoints = rawData.take(1)
print (samplePoints)

from pyspark.mllib.regression import LabeledPoint
import numpy as np
def parsePoint(line):
    """Converts a comma separated unicode string into a `LabeledPoint`.

    Args:
        line (unicode): Comma separated unicode string where the first element is the label and the
            remaining 12 elements are features.

    Returns:
        LabeledPoint: The line is converted into a `LabeledPoint`, which consists of a label and
            features. where features is a list.
    """
    splitted = str.split(line, ',')
    features = splitted[1:]
    label = splitted[0]
    #LabeledPoint = (label, features)
    return LabeledPoint(label, features)

firstPoint = samplePoints[0]
firstPointParsed = parsePoint(firstPoint)
#print firstPointParsed
firstPointFeatures = firstPointParsed.features
print ('features: ',firstPointFeatures)
firstPointLabel = firstPointParsed.label
print ('label: ',firstPointLabel)

d = len(firstPointFeatures)
print (d)

parsedDataInit = rawData.map(lambda point: parsePoint(point)) #parsedDataInit is of the form LabeledPoint(label, features) where features is a list of features
onlyLabels = parsedDataInit.map(lambda parsed_point: parsed_point.label)
# arrange onlyLabels rdd in ascending order and first element is minimum
minYear = onlyLabels.takeOrdered(numPoints)[0]
# arrange onlyLabels rdd in ascending order and last element is maximum
maxYear = onlyLabels.takeOrdered(numPoints)[-1]
print (minYear, maxYear)

parsedData = parsedDataInit.map(lambda lp : LabeledPoint(lp.label - minYear, lp.features))

# Should be a LabeledPoint
print (type(parsedData.take(1)[0]))
# View the first point
print ('\n{0}'.format(parsedData.take(1)))

# get data for plot
oldData = (parsedDataInit
           .map(lambda lp: (lp.label, 1))
           .reduceByKey(lambda x, y: x + y)
           .collect()) # oldData has tuples of form (year1, count1), (year2, count2), (year3, count3),...
x, y = zip(*oldData) # zip(*oldData) unzips oldData and returns: (year1, year2, year3, ...) and (count1, count2, count3, ...) that are X-values and Y-values

# generate layout and plot data
fig, ax = preparePlot(np.arange(1920, 2050, 20), np.arange(0, 150, 20))
plt.scatter(x, y, s=14**2, c='#d6ebf2', edgecolors='#8cbfd0', alpha=0.75)
ax.set_xlabel('Year'), ax.set_ylabel('Count')
display(fig) 
pass

# get data for plot
newData = (parsedData
           .map(lambda lp: (lp.label, 1))
           .reduceByKey(lambda x, y: x + y)
           .collect())
x, y = zip(*newData)

# generate layout and plot data
fig, ax = preparePlot(np.arange(0, 120, 20), np.arange(0, 120, 20))
plt.scatter(x, y, s=14**2, c='#d6ebf2', edgecolors='#8cbfd0', alpha=0.75)
ax.set_xlabel('Year (shifted)'), ax.set_ylabel('Count')
display(fig) 
pass

weights = [.8, .1, .1]
seed = 42
parsedTrainData, parsedValData, parsedTestData = parsedData.randomSplit(weights, seed) # parsedData is of type LabeledPoint
parsedTrainData.cache()
parsedValData.cache()
parsedTestData.cache()
nTrain = parsedTrainData.count()
nVal = parsedValData.count()
nTest = parsedTestData.count()

print (nTrain, '+' ,nVal, '+' ,nTest, '+' , '=' ,nTrain + nVal + nTest)
print (parsedData.count())

averageTrainYear = (parsedTrainData
                    .map(lambda lp: float(lp.label))
                    .reduce(lambda a,b: a+b))/float(parsedTrainData.count())        
print (averageTrainYear)

def squaredError(label, prediction):
    """Calculates the squared error for a single prediction.

    Args:
        label (float): The correct value for this observation.
        prediction (float): The predicted value for this observation.

    Returns:
        float: The difference between the `label` and `prediction` squared.
    """
    squared_error = (label - prediction)**2
    return squared_error

def calcRMSE(labelsAndPreds):
    """Calculates the root mean squared error for an `RDD` of (label, prediction) tuples.

    Args:
        labelsAndPred (RDD of (float, float)): An `RDD` consisting of (label, prediction) tuples.

    Returns:
        float: The square root of the mean of the squared errors.
    """
    squaredErrorRDD = labelsAndPreds.map(lambda label_prediction: squaredError(label_prediction[0], label_prediction[1]))
    RMSE = ((squaredErrorRDD.reduce(lambda a,b: a+b))/float(squaredErrorRDD.count()))**(0.5)
    return RMSE

# run the function for a simple example
labelsAndPreds = sc.parallelize([(3., 1.), (1., 2.), (2., 2.)])
# RMSE = sqrt[((3-1)^2 + (1-2)^2 + (2-2)^2) / 3] = 1.291
exampleRMSE = calcRMSE(labelsAndPreds)
print (exampleRMSE)

labelsAndPredsTrain = parsedTrainData.map(lambda lp: (lp.label, averageTrainYear))
rmseTrainBase = calcRMSE(labelsAndPredsTrain)

labelsAndPredsVal = parsedValData.map(lambda lp: (lp.label, averageTrainYear))
rmseValBase = calcRMSE(labelsAndPredsVal)

labelsAndPredsTest = parsedTestData.map(lambda lp: (lp.label, averageTrainYear))
rmseTestBase = calcRMSE(labelsAndPredsTest)
print ('Baseline Train RMSE = {0:.3f}'.format(rmseTrainBase))
print ('Baseline Validation RMSE = {0:.3f}'.format(rmseValBase))
print ('Baseline Test RMSE = {0:.3f}'.format(rmseTestBase))

from matplotlib.colors import ListedColormap, Normalize
from matplotlib.cm import get_cmap
cmap = get_cmap('YlOrRd')
norm = Normalize()

actual = np.asarray(parsedValData
                    .map(lambda lp: lp.label)
                    .collect())
error = np.asarray(parsedValData
                   .map(lambda lp: (lp.label, lp.label))
                   .map(lambda l_p: squaredError(l_p[0], l_p[1]))
                   .collect())
clrs = cmap(np.asarray(norm(error)))[:,0:3]
fig, ax = preparePlot(np.arange(0, 100, 20), np.arange(0, 100, 20))
plt.scatter(actual, actual, s=14**2, c=clrs, edgecolors='#888888', alpha=0.75, linewidths=0.5)
ax.set_xlabel('actual'), ax.set_ylabel('Actual')
display(fig) 
pass

predictions = np.asarray(parsedValData
                         .map(lambda lp: averageTrainYear)
                         .collect())
error = np.asarray(parsedValData
                   .map(lambda lp: (lp.label, averageTrainYear))
                   .map(lambda l_p: squaredError(l_p[0], l_p[1]))
                   .collect())
norm = Normalize()
clrs = cmap(np.asarray(norm(error)))[:,0:3]

fig, ax = preparePlot(np.arange(53.0, 55.0, 0.5), np.arange(0, 100, 20))
ax.set_xlim(53, 55)
plt.scatter(predictions, actual, s=14**2, c=clrs, edgecolors='#888888', alpha=0.75, linewidths=0.3)
ax.set_xlabel('Predicted'), ax.set_ylabel('Actual')
display(fig) 

from pyspark.mllib.linalg import DenseVector
# 
def gradientSummand(weights, lp):
    """Calculates the gradient summand for a given weight and `LabeledPoint`.

    Note:
        `DenseVector` behaves similarly to a `numpy.ndarray` and they can be used interchangably
        within this function.  For example, they both implement the `dot` method.

    Args:
        weights (DenseVector): An array of model weights (betas).
        lp (LabeledPoint): The `LabeledPoint` for a single observation.

    Returns:
        DenseVector: An array of values the same length as `weights`.  The gradient summand.
    """
    #floated_features = map(lambda feature: float(feature) , lp.features)
    x = DenseVector(lp.features)
    y = float(lp.label)
  
    first = (weights.dot(x)) - y # this is a float number
    second = first * x # this is a dencevector
    return second

exampleW = DenseVector([1, 1, 1])
exampleLP = LabeledPoint(2.0, [3, 1, 4])
#gradientSummand = (dot([1 1 1], [3 1 4]) - 2) * [3 1 4] = (8 - 2) * [3 1 4] = [18 6 24]
summandOne = gradientSummand(exampleW, exampleLP)
print (summandOne)

exampleW = DenseVector([.24, 1.2, -1.4])
exampleLP = LabeledPoint('3.0', [-1.4, 4.2, 2.1])
summandTwo = gradientSummand(exampleW, exampleLP)
print (summandTwo)

def getLabeledPrediction(weights, observation):
    """Calculates predictions and returns a (label, prediction) tuple.

    Note:
        The labels should remain unchanged as we'll use this information to calculate prediction
        error later.

    Args:
        weights (np.ndarray): An array with one weight for each features in `trainData`.
        observation (LabeledPoint): A `LabeledPoint` that contain the correct label and the
            features for the data point.

    Returns:
        tuple: A (label, prediction) tuple.
    """
    return (observation.label, weights.dot(observation.features))
weights = np.array([1.0, 1.5])
predictionExample = sc.parallelize([LabeledPoint(2, np.array([1.0, .5])),LabeledPoint(1.5, np.array([.5, .5]))])
labelsAndPredsExample = predictionExample.map(lambda lp: getLabeledPrediction(weights, lp))
print (labelsAndPredsExample.collect())

def linregGradientDescent(trainData, numIters):
    """Calculates the weights and error for a linear regression model trained with gradient descent.
    Note:
        `DenseVector` behaves similarly to a `numpy.ndarray` and they can be used interchangably
        within this function.  For example, they both implement the `dot` method.

    Args:
        trainData (RDD of LabeledPoint): The labeled data for use in training the model.
        numIters (int): The number of iterations of gradient descent to perform.

    Returns:
        (np.ndarray, np.ndarray): A tuple of (weights, training errors).  Weights will be the
            final weights (one weight per feature) for the model, and training errors will contain
            an error (RMSE) for each iteration of the algorithm.
    """
    # The length of the training data
    n = trainData.count()
    # The number of features in the training data
    d = len(trainData.take(1)[0].features)
    # weight initialization:
    w = np.zeros(d)
    alpha = 1.0 # learning rate
    # We will compute and store the training error after each iteration
    errorTrain = np.zeros(numIters)
    for i in range(numIters):
        # Use getLabeledPrediction from (3b) with trainData to obtain an RDD of (label, prediction)
        # tuples.  Note that the weights all equal 0 for the first iteration, so the predictions will
        # have large errors to start.
        labelsAndPredsTrain = trainData.map(lambda lp: getLabeledPrediction(w, lp)) # labelsAndPredsTrain is a (label, prediction) tuple.
        errorTrain[i] = calcRMSE(labelsAndPredsTrain)

        # Calculate the `gradient`.  Make use of the `gradientSummand` function you wrote in (3a).
        # Note that `gradient` should be a `DenseVector` of length `d`.
        gradient = trainData.map(lambda lp: gradientSummand(w, lp)).reduce(lambda summand1, summand2:  summand1 + summand2)

        # Update the weights
        alpha_i = alpha / (n * np.sqrt(i+1))
        w -= alpha_i * gradient
    return w, errorTrain

# create a toy dataset with n = 10, d = 3, and then run 5 iterations of gradient descent
# note: the resulting model will not be useful; the goal here is to verify that
# linregGradientDescent is working properly
exampleN = 10
exampleD = 3
exampleData = (sc
               .parallelize(parsedTrainData.take(exampleN))
               .map(lambda lp: LabeledPoint(lp.label, lp.features[0:exampleD])))
print (exampleData.take(2))
exampleNumIters = 5
exampleWeights, exampleErrorTrain = linregGradientDescent(exampleData, exampleNumIters)
print (exampleWeights)

numIters = 50
weightsLR0, errorTrainLR0 = linregGradientDescent(parsedTrainData , numIters)
# using the weights in our training phase, we predict labels for validation set and find RMSE error on the validation set
labelsAndPreds = parsedValData.map(lambda lp: getLabeledPrediction(weightsLR0, lp)) # labelsAndPreds is a (label, prediction) tuple
rmseValLR0 = calcRMSE(labelsAndPreds)

print ('Validation RMSE:\n\tBaseline = {0:.3f}\n\tLR0 = {1:.3f}'.format(rmseValBase, rmseValLR0))

norm = Normalize()
clrs = cmap(np.asarray(norm(np.log(errorTrainLR0))))[:,0:3]

fig, ax = preparePlot(np.arange(0, 60, 10), np.arange(2, 6, 1))
ax.set_ylim(2, 6)
plt.scatter(range(0, numIters), np.log(errorTrainLR0), s=14**2, c=clrs, edgecolors='#888888', alpha=0.75)
ax.set_xlabel('Iteration'), ax.set_ylabel(r'$\log_e(errorTrainLR0)$')
display(fig) 
pass

norm = Normalize()
clrs = cmap(np.asarray(norm(errorTrainLR0[6:])))[:,0:3]
fig, ax = preparePlot(np.arange(0, 60, 10), np.arange(17, 22, 1))
ax.set_ylim(17.8, 21.2)
plt.scatter(range(0, numIters-6), errorTrainLR0[6:], s=14**2, c=clrs, edgecolors='#888888', alpha=0.75)
ax.set_xticklabels(map(str, range(6, 66, 10)))
ax.set_xlabel('Iteration'), ax.set_ylabel(r'Training Error')
display(fig) 
pass

from pyspark.mllib.regression import LinearRegressionWithSGD
# Values to use when training the linear regression model with LinearRegressionWithSGD
numIters = 500  # iterations
alpha = 1.0  # step
miniBatchFrac = 1.0  # miniBatchFraction
reg = 1e-1  # regParam
regType = 'l2'  #  L2 regularization
useIntercept = True  # intercept

# d is number of features already defined in phase 1b
firstModel = LinearRegressionWithSGD.train(parsedTrainData, iterations=numIters,miniBatchFraction=miniBatchFrac, initialWeights=np.zeros(d), regParam=reg, regType="l2", intercept=True)

# weightsLR1 stores the model weights; interceptLR1 stores the model intercept
weightsLR1 = firstModel.weights
interceptLR1 = firstModel.intercept
print (weightsLR1, interceptLR1)

samplePoint = parsedTrainData.take(1)[0]
samplePrediction = firstModel.predict(samplePoint.features) 
print (samplePrediction)

labelsAndPreds = parsedValData.map(lambda lp: (lp.label, firstModel.predict(lp.features)))
#labelsAndPreds = firstModel.predict(first_element.features) 
rmseValLR1 = calcRMSE(labelsAndPreds)bestRMSE = rmseValLR1
bestRegParam = reg
bestModel = firstModel
numIters = 500
alpha = 1.0 # The step parameter used in SGD
miniBatchFrac = 1.0
for reg in [1, 1e-10, 1e-5]:
    model = LinearRegressionWithSGD.train(parsedTrainData, numIters, alpha, miniBatchFrac, regParam=reg, regType='l2', intercept=True)
    labelsAndPreds = parsedValData.map(lambda lp: (lp.label, model.predict(lp.features)))
    rmseValGrid = calcRMSE(labelsAndPreds)
    print (rmseValGrid)
    if rmseValGrid < bestRMSE:
        bestRMSE = rmseValGrid
        bestRegParam = reg
        bestModel = model
rmseValLRGrid = bestRMSE
print ('Validation RMSE: Baseline=', rmseValBase)
print ('Validation RMSE: LR0=', rmseValLR0)
print ('Validation RMSE: LR1=', rmseValLR1)
print ('Validation RMSE: LRGrid=', rmseValLRGrid)

predictions = np.asarray(parsedValData
                         .map(lambda lp: bestModel.predict(lp.features))
                         .collect())
actual = np.asarray(parsedValData
                    .map(lambda lp: lp.label)
                    .collect())
error = np.asarray(parsedValData
                   .map(lambda lp: (lp.label, bestModel.predict(lp.features)))
                   .map(lambda l_p: squaredError(l_p[0], l_p[1]))
                   .collect())

norm = Normalize()
clrs = cmap(np.asarray(norm(error)))[:,0:3]

fig, ax = preparePlot(np.arange(0, 120, 20), np.arange(0, 120, 20))
ax.set_xlim(15, 82), ax.set_ylim(-5, 105)
plt.scatter(predictions, actual, s=14**2, c=clrs, edgecolors='#888888', alpha=0.75, linewidths=.5)
ax.set_xlabel('Predicted'), ax.set_ylabel(r'Actual')
display(fig) 
pass






