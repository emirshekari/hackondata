display(dbutils.fs.ls('/databricks-datasets/cs100/lab4/data-001/'))
import sys
import os

baseDir = os.path.join('databricks-datasets')
inputPath = os.path.join('cs100', 'lab4', 'data-001')

ratingsFilename = os.path.join(baseDir, inputPath, 'ratings.dat.gz')
moviesFilename = os.path.join(baseDir, inputPath, 'movies.dat')

numPartitions = 2
rawRatings = sc.textFile(ratingsFilename).repartition(numPartitions)
rawMovies = sc.textFile(moviesFilename)

def get_ratings_tuple(entry):
    """ Parse a line in the ratings dataset
    Args:
        entry (str): a line in the ratings dataset in the form of UserID::MovieID::Rating::Timestamp
    Returns:
        tuple: (UserID, MovieID, Rating)
    """
    items = entry.split('::')
    return int(items[0]), int(items[1]), float(items[2])

def get_movie_tuple(entry):
    """ Parse a line in the movies dataset
    Args:
        entry (str): a line in the movies dataset in the form of MovieID::Title::Genres
    Returns:
        tuple: (MovieID, Title)
    """
    items = entry.split('::')
    return int(items[0]), items[1]

ratingsRDD = rawRatings.map(get_ratings_tuple).cache()
moviesRDD = rawMovies.map(get_movie_tuple).cache()

ratingsCount = ratingsRDD.count()
moviesCount = moviesRDD.count()

print ('There are %s ratings and %s movies in the datasets' % (ratingsCount, moviesCount))
print ('Ratings: %s' % ratingsRDD.take(3))
print ('Movies: %s' % moviesRDD.take(3))

def sortFunction(tuple):
    """ Construct the sort string (does not perform actual sorting)
    Args:
        tuple: (rating, MovieName)
    Returns:
        sortString: the value to sort with, 'rating MovieName'
    """
    # movie id could be a float number with 3 decimal points like 3.141
    key = str('%.3f' % tuple[0]) # in python 2 use 'unicode()' instead of 'str()'
    value = tuple[1]
    return (key + ' ' + value)
   
# First, implement a helper function `getCountsAndAverages` using only Python
def getCountsAndAverages(IDandRatingsTuple):
    """ Calculate average rating
    Args:
        IDandRatingsTuple: a single tuple of (MovieID, (Rating1, Rating2, Rating3, ...))
    Returns:
        tuple: a tuple of (MovieID, (number of ratings, averageRating))
    """
    rating_sum = 0
    rating_count = 0
    for rating in IDandRatingsTuple[1]:
      rating_sum = rating_sum + rating
      rating_count = rating_count + 1
    return (IDandRatingsTuple[0],(rating_count, rating_sum/ float(rating_count)))
    
    
movieIDsWithRatingsRDD = (ratingsRDD
                          .map(lambda x_y_z: (x_y_z[1],x_y_z[2]))
                           .groupByKey())
print ('movieIDsWithRatingsRDD: %s\n' % movieIDsWithRatingsRDD.take(3))

# Using `movieIDsWithRatingsRDD`, compute the number of ratings and average rating for each movie to
# yield tuples of the form (MovieID, (number of ratings, average rating))
movieIDsWithAvgRatingsRDD = movieIDsWithRatingsRDD.map(lambda x: getCountsAndAverages(x))
print ('movieIDsWithAvgRatingsRDD: %s\n' % movieIDsWithAvgRatingsRDD.take(3))

# To `movieIDsWithAvgRatingsRDD`, apply RDD transformations that use `moviesRDD` to get the movie
# names for `movieIDsWithAvgRatingsRDD`, yielding tuples of the form
# (average rating, movie name, number of ratings)

# example of join : on two pair RDDs (rdd = {(1, 2), (3, 4), (3, 6)} other = {(3, 9)}) join results in : {(3, (4, 9)), (3, (6, 9))}
movieNameWithAvgRatingsRDD = (moviesRDD  # moviesRDD has (MovieID,Title,Genres)
                              .join(movieIDsWithAvgRatingsRDD)
                              .map(lambda x_y: (x_y[1][1][1], x_y[1][0], x_y[1][1][0])))
print ('movieNameWithAvgRatingsRDD: %s\n' % movieNameWithAvgRatingsRDD.take(3))

# Apply an RDD transformation to `movieNameWithAvgRatingsRDD` to limit the results to movies with
# ratings from more than 500 people. We then use the `sortFunction()` helper function to sort by the
# average rating to get the movies in order of their rating (highest rating first)
movieLimitedAndSortedByRatingRDD = (movieNameWithAvgRatingsRDD
                                    .filter(lambda avgRating_movieName_numberOfRating: avgRating_movieName_numberOfRating[2] > 500)
                                    .sortBy(sortFunction, False))
# sortBy(self, keyfunc, ascending=True, numPartitions=None)
print ('Movies with highest ratings: %s' % movieLimitedAndSortedByRatingRDD.take(20))

trainingRDD, validationRDD, testRDD = ratingsRDD.randomSplit([6, 2, 2], seed=0)
# [6, 2, 2] are weights that should be normalized i.e. add up to 1 if they are not normal they will be normalized automatically
# we chose 6,2,2 because 60% of data should be for training, 20% for cross validation, 20% for testing
print ('Training: %s, validation: %s, test: %s\n' % (trainingRDD.count(),
                                                    validationRDD.count(),
                                                    testRDD.count()))
print (ratingsRDD.count())
print (trainingRDD.take(3))
print (validationRDD.take(3))
print (testRDD.take(3))

import math
def computeError(predictedRDD, actualRDD):
    """ Compute the root mean squared error between predicted and actual
    Args:
        predictedRDD: predicted ratings for each movie and each user where each entry is in the form
                      (UserID, MovieID, Rating)
        actualRDD: actual ratings where each entry is in the form (UserID, MovieID, Rating)
    Returns:
        RSME (float): computed RSME value
    """
    # Transform predictedRDD into the tuples of the form ((UserID, MovieID), Rating)
    predictedReformattedRDD = predictedRDD.map(lambda userID_movieID_rating: ((userID_movieID_rating[0],userID_movieID_rating[1]), userID_movieID_rating[2]))
    # Transform actualRDD into the tuples of the form ((UserID, MovieID), Rating)
    actualReformattedRDD = actualRDD.map(lambda userID_movieID_rating: ((userID_movieID_rating[0],userID_movieID_rating[1]), userID_movieID_rating[2]))
    # Compute the squared error for each matching entry (i.e., the same (User ID, Movie ID) in each
    # RDD) in the reformatted RDDs using RDD transformtions - do not use collect()
    squaredErrorsRDD = (predictedReformattedRDD
                        .join(actualReformattedRDD)
                        .filter(lambda uID_mID_pair_Pred_actl_rate_list: uID_mID_pair_Pred_actl_rate_list[1] is not None )#.filter(lambda (uID_mID_pair, pred_actl_rate_list): pred_actl_rate_list is not None )
                        .map(lambda uID_mID_pair_Pred_actl_rate_list: (uID_mID_pair_Pred_actl_rate_list[0], (uID_mID_pair_Pred_actl_rate_list[1][0] - uID_mID_pair_Pred_actl_rate_list[1][1])**2)))#.map(lambda (uID_mID_pair, pred_actl_rate_list): (uID_mID_pair, (pred_actl_rate_list[0] - pred_actl_rate_list[1])**2)))


    # Compute the total squared error - do not use collect()
    totalError = (squaredErrorsRDD
                  .map(lambda x_Squared_difference_rating: x_Squared_difference_rating[1])
                  .reduce(lambda squared_difference_rating1,squared_difference_rating2: squared_difference_rating1 + squared_difference_rating2))

    # Count the number of entries for which you computed the total squared error
    numRatings = squaredErrorsRDD.count()

    # Using the total squared error and the number of entries, compute the RSME
    return (totalError/float(numRatings))**(0.5)
    
# sc.parallelize turns a Python list into a Spark RDD.
testPredicted = sc.parallelize([
    (1, 1, 5),
    (1, 2, 3),
    (1, 3, 4),
    (2, 1, 3),
    (2, 2, 2),
    (2, 3, 4)])
testActual = sc.parallelize([
     (1, 2, 3),
     (1, 3, 5),
     (2, 1, 5),
     (2, 2, 1)])
testPredicted2 = sc.parallelize([
     (2, 2, 5),
     (1, 2, 5)])
testError = computeError(testPredicted, testActual)
print ('Error for test dataset (should be 1.22474487139): %s' % testError)

testError2 = computeError(testPredicted2, testActual)
print ('Error for test dataset2 (should be 3.16227766017): %s' % testError2)

testError3 = computeError(testActual, testActual)
print ('Error for testActual dataset (should be 0.0): %s' % testError3)

from pyspark.mllib.recommendation import ALS

validationForPredictRDD = validationRDD.map(lambda userID_movieID_rating: (userID_movieID_rating[0], userID_movieID_rating[1]))

seed = 5
iterations = 5
regularizationParameter = 0.1
ranks = [4, 8, 12]
errors = [0, 0, 0]
err = 0
tolerance = 0.03

minError = float('inf')
bestRank = -1
bestIteration = -1
for rank in ranks:
    model = ALS.train(trainingRDD, rank, seed=seed, iterations=iterations,
                      lambda_=regularizationParameter)
    predictedRatingsRDD = model.predictAll(validationForPredictRDD)
    error = computeError(predictedRatingsRDD, validationRDD)
    errors[err] = error
    err += 1
    print ('For rank %s the RMSE is %s' % (rank, error))
    if error < minError:
        minError = error
        bestRank = rank

print ('The best model was trained with rank %s' % bestRank)

myModel = ALS.train(trainingRDD, 12, seed=5, iterations=5, lambda_=0.1)
testForPredictingRDD = testRDD.map(lambda userID_movieID_rating: (userID_movieID_rating[0], userID_movieID_rating[1]))
predictedTestRDD = myModel.predictAll(testForPredictingRDD)
testRMSE = computeError(testRDD, predictedTestRDD)
print ('The model had a RMSE on the test set of %s' % testRMSE)

print ('Most rated movies:')
print ('(average rating, movie name, number of reviews)')
for ratingsTuple in moviesRDD.take(250):
    print (ratingsTuple)
    
myUserID = 0

# Note that the movie IDs are the *last* number on each line. A common error was to use the number of ratings as the movie ID.
myRatedMovies = [
     (myUserID, 260, 1),
     (myUserID, 36, 4),
     (myUserID, 93, 1),
     (myUserID, 165, 3),
     (myUserID, 187, 2),
     (myUserID, 175, 4),
     (myUserID, 163, 4),
     (myUserID, 153, 1),
     (myUserID, 111, 5),
     (myUserID, 251, 3)
     # The format of each line is (myUserID, movie ID, your rating)
     # For example, to give the movie "Star Wars: Episode IV - A New Hope (1977)" a five rating, you would add the following line:
     #   (myUserID, 260, 5),
    ]
myRatingsRDD = sc.parallelize(myRatedMovies)
print ('My movie ratings: %s' % myRatingsRDD.take(10))

# TODO: Replace <FILL IN> with appropriate code
trainingWithMyRatingsRDD = trainingRDD.union(myRatingsRDD)

print ('The training dataset now has %s more entries than the original training dataset' %
       (trainingWithMyRatingsRDD.count() - trainingRDD.count()))
assert (trainingWithMyRatingsRDD.count() - trainingRDD.count()) == myRatingsRDD.count()

myRatingsModel = ALS.train(trainingWithMyRatingsRDD, bestRank,  seed=5, iterations=5, lambda_=0.1)

predictedTestMyRatingsRDD = myRatingsModel.predictAll(testForPredictingRDD)
testRMSEMyRatings = computeError(testRDD, predictedTestMyRatingsRDD)
print ('The model had a RMSE on the test set of %s' % testRMSEMyRatings)

# Use the Python list myRatedMovies to transform the moviesRDD into an RDD with entries that are pairs of the form (myUserID, Movie ID) and that does not contain any movies that you have rated.
myUnratedMoviesRDD = (moviesRDD
                      .filter(lambda movieID_title: movieID_title[0] not in myRatedMovies)
                      .map(lambda movieID_title: (myUserID, movieID_title[0])))

# Use the input RDD, myUnratedMoviesRDD, with myRatingsModel.predictAll() to predict your ratings for the movies
predictedRatingsRDD = myRatingsModel.predictAll(myUnratedMoviesRDD)

# Transform movieIDsWithAvgRatingsRDD from part (1b), which has the form (MovieID, (number of ratings, average rating)), into and RDD of the form (MovieID, number of ratings)
movieCountsRDD = movieIDsWithAvgRatingsRDD.map(lambda movieID_numRating__avgRating: (movieID_numRating__avgRating[0], movieID_numRating__avgRating[1][0]))

# Transform predictedRatingsRDD into an RDD with entries that are pairs of the form (Movie ID, Predicted Rating)
predictedRDD = predictedRatingsRDD.map(lambda userID_movieID_rating: (userID_movieID_rating[1], userID_movieID_rating[2]))

# Use RDD transformations with predictedRDD and movieCountsRDD to yield an RDD with tuples of the form (Movie ID, (Predicted Rating, number of ratings))
predictedWithCountsRDD  = (predictedRDD
                           .join(movieCountsRDD))

# Use RDD transformations with PredictedWithCountsRDD and moviesRDD to yield an RDD with tuples of the form (Predicted Rating, Movie Name, number of ratings), for movies with more than 75 ratings
ratingsWithNamesRDD = (predictedWithCountsRDD
                       .join(moviesRDD)
                       .map(lambda movieID_predicted_rating_number_of_rating_title: (movieID_predicted_rating_number_of_rating_title[1][0][0], movieID_predicted_rating_number_of_rating_title[1][1], movieID_predicted_rating_number_of_rating_title[1][0][1]))
                       #.map(lambda (movieID, ((predicted_rating, number_of_rating), title)): (predicted_rating, title, number_of_rating))
                       .filter(lambda predicted_rating_title_number_of_rating: predicted_rating_title_number_of_rating[2] > 75))

predictedHighestRatedMovies = ratingsWithNamesRDD.takeOrdered(20, key=lambda x: -x[0])
print ('My highest rated movies as predicted (for movies with more than 75 reviews):\n%s' %
        '\n'.join(map(str, predictedHighestRatedMovies)))
        


