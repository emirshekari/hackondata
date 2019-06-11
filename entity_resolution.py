# Text Analysis and Entity Resolution using Apache Spark (in Python)
# I perform entity resolution across two datasets of commercial products
# Entity Resolution (ER) refers to the task of finding records in a dataset that refer to the same entity across different data sources (e.g., data files, books, websites, databases)
# [wiki]: https://en.wikipedia.org/wiki/Record_linkage

# this code contains the following files:
# Google.csv, the Google Products dataset
# Amazon.csv, the Amazon dataset
# Google_small.csv, 200 records sampled from the Google data
# Amazon_small.csv, 200 records sampled from the Amazon data
# Amazon_Google_perfectMapping.csv, the "gold standard" mapping
# stopwords.txt, a list of common English words

# part 0: read in each of the files and create an RDD consisting of lines.
# We read in each of the files and create an RDD consisting of lines.
# For each of the data files ("Google.csv", "Amazon.csv", and the samples), we want to parse the IDs out of each record. 
# The IDs are the first column of the file (they are URLs for Google, and alphanumeric strings for Amazon). 
# Omitting the headers, we load these data files into pair RDDs where the *mapping ID* is the key, and the value is 
# a string consisting of the name/title, description, and manufacturer from the record.
# The file format of an Amazon line is:
#  `"id","title","description","manufacturer","price"`
# The file format of a Google line is:
# `"id","name","description","manufacturer","price"`


import re
DATAFILE_PATTERN = '^(.+),"(.+)",(.*),(.*),(.*)'

def removeQuotes(s):
    """ Remove quotation marks from an input string
    Args:
        s (str): input string that might have the quote "" characters
    Returns:
        str: a string without the quote characters
    """
    
    match = re.search("^\"(.*?)\"$", s)
    
    if match is None:
      return s
    else:
      s_filtered = match.group(1)
      return s_filtered


def parseDatafileLine(datafileLine):
    """ Parse a line of the data file using the specified regular expression pattern
    Args:
        datafileLine (str): input string that is a line from the data file
    Returns:
        str: a string parsed using the given regular expression and without the quote characters
    """
    match = re.search(DATAFILE_PATTERN, datafileLine)
    if match is None:
        print 'Invalid datafile line: %s' % datafileLine
        return (datafileLine, -1)
    elif match.group(1) == '"id"':
        print 'Header datafile line: %s' % datafileLine
        return (datafileLine, 0)
    else:
        product = '%s %s %s' % (match.group(2), match.group(3), match.group(4))
        return ((removeQuotes(match.group(1)), product), 1)
        
        

import sys
import os
from test_helper import Test

baseDir = os.path.join('databricks-datasets')
inputPath = os.path.join('cs100', 'lab3', 'data-001')

GOOGLE_PATH = 'Google.csv'
GOOGLE_SMALL_PATH = 'Google_small.csv'
AMAZON_PATH = 'Amazon.csv'
AMAZON_SMALL_PATH = 'Amazon_small.csv'
GOLD_STANDARD_PATH = 'Amazon_Google_perfectMapping.csv'
STOPWORDS_PATH = 'stopwords.txt'

def parseData(filename):
    """ Parse a data file
    Args:
        filename (str): input file name of the data file
    Returns:
        RDD: a RDD of parsed lines
    """
    return (sc
            .textFile(filename, 4, 0)
            .map(parseDatafileLine))

def loadData(path):
    """ Load a data file
    Args:
        path (str): input file name of the data file
    Returns:
        RDD: a pair RDD of parsed valid lines with 1st element removeQuotes(match.group(1)) and 2nd element product (it's a string consits of           "title","description","manufacturer")

    """
    filename = os.path.join(baseDir, inputPath, path)
    #raw below is a pair RDD whose first element is a pair of (removeQuotes(match.group(1)), product) and second element is either 1,-1 or 0.
    raw = parseData(filename).cache()
    failed = (raw
              .filter(lambda s: s[1] == -1)
              .map(lambda s: s[0]))
    for line in failed.take(10):
        print '%s - Invalid datafile line: %s' % (path, line)
    valid = (raw
             .filter(lambda s: s[1] == 1)
             .map(lambda s: s[0])
             .cache())
    print '%s - Read %d lines, successfully parsed %d lines, failed to parse %d lines' % (path,
                                                                                        raw.count(),
                                                                                        valid.count(),
                                                                                        failed.count())
    assert failed.count() == 0
    assert raw.count() == (valid.count() + 1)
    return valid

googleSmall = loadData(GOOGLE_SMALL_PATH)
google = loadData(GOOGLE_PATH)
amazonSmall = loadData(AMAZON_SMALL_PATH)
amazon = loadData(AMAZON_PATH)


# Let's examine the lines that were just loaded in the two subset (small) files - one from Google and one from Amazon
# line is a pair RDD (output of function loadData)
# line[0] is removeQuotes(match.group(1)) 
# line[1] is a string consits of ("title","description","manufacturer")

for line in googleSmall.take(3):
    print 'google: %s: %s\n' % (line[0], line[1])

for line in amazonSmall.take(3):
    print 'amazon: %s: %s\n' % (line[0], line[1])
    
# Part 1: ER as Text Similarity  **Bags of Words**
# (1a) Tokenize a String

quickbrownfox = 'A quick brown fox jumps over the lazy dog.'
# \W+ matches any non-word character [^a-zA-Z0-9_]
split_regex = r'\W+'

def simpleTokenize(string):
    """ A simple implementation of input string tokenization
    Args:h
        string (str): input string
    Returns:
        list: a list of tokens
    """
    list_tokens = re.split(split_regex, string.lower())
    token_empty_removed = filter(lambda string: string != '', list_tokens)
    return token_empty_removed
    
print simpleTokenize(quickbrownfox) # Should give ['a', 'quick', 'brown', ... ]

#(1b) Removing stopwords: Stopwords add noise to bag-of-words comparisons, so they are usually excluded. Using the included file "stopwords.txt", implement tokenize, an improved tokenizer that does not emit stopwords.

stopfile = os.path.join(baseDir, inputPath, STOPWORDS_PATH)
stopwords = set(sc.textFile(stopfile).collect())
print 'These are the stopwords: %s' % stopwords

def tokenize(string):
    """ An implementation of input string tokenization that excludes stopwords
    Args:
        string (str): input string
    Returns:
        list: a list of tokens without stopwords
    """
    # tokenize and lower case:
    list_tokens = re.split(split_regex, string.lower())
    # remove empty tokens
    token_empty_removed = filter(lambda string: string != '', list_tokens)
    #filter out stopwords
    stopWords_removed = filter(lambda word: word not in stopwords, token_empty_removed)

    return stopWords_removed

print tokenize(quickbrownfox) # Should give ['quick', 'brown', ... ]

# (1c) Tokenizing the small datasets
# amazonSmall is a pair RDD of parsed valid lines with 1st element removeQuotes(match.group(1)) and 2nd element product (it's a string consits of # "title","description","manufacturer")
amazonRecToToken = amazonSmall.map(lambda (id_key, product_value): (id_key, tokenize(product_value)))
googleRecToToken = googleSmall.map(lambda (id_key, product_value): (id_key, tokenize(product_value)))

def countTokens(vendorRDD):
    """ Count and return the number of tokens
    Args:
        vendorRDD (RDD of (recordId, tokenizedValue)): Pair tuple of record ID to tokenized output
    Returns:
        count: count of all tokens
    """
    count_each_tuple = vendorRDD.map(lambda (id_key, tokenizedProduct_value): (1, len(tokenizedProduct_value)))
    total_RDD = count_each_tuple.reduceByKey(lambda key1, key2: key1+key2)
    total = total_RDD.map(lambda (x,y): y).collect()
    return total[0]
totalTokens = countTokens(amazonRecToToken) + countTokens(googleRecToToken)
print 'There are %s tokens in the combined datasets' % totalTokens

#(1d) Amazon record with the most tokens: sort the records and get the one with the largest count of tokens.
def findBiggestRecord(vendorRDD):
    """ Find and return the record with the largest number of tokens
    Args:
        vendorRDD (RDD of (recordId, tokens)): input Pair Tuple of record ID and tokens
    Returns:
        list: a list of 1 Pair Tuple of record ID and tokens
    """
    # number 1 below as the first parameter of takeOrdered is because we want the first top biggest record:
    id_grouped = vendorRDD.takeOrdered(1, key = lambda x: -1 * len(x[1]))
    return id_grouped
biggestRecordAmazon = findBiggestRecord(amazonRecToToken)
print 'The Amazon record with ID "%s" has the most tokens (%s)' % (biggestRecordAmazon[0][0],
                                                                   len(biggestRecordAmazon[0][1]))

# Part 2: ER as Text Similarity - Weighted Bag-of-Words using TF-IDF
# (2a) Implement a TF function: takes a list of tokens and returns a Python dictionary mapping tokens to TF weights
def tf(tokens):
    """ Compute TF
    Args:
        tokens (list of str): input list of tokens from tokenize
    Returns:
        dictionary: a dictionary of tokens to its TF values
    """
    dict_token_weight = {}
    for x in set(tokens):
      dict_token_weight[x] = tokens.count(x) / float(len(tokens))
    return dict_token_weight
print tf(tokenize(quickbrownfox)) # Should give { 'quick': 0.1666 ... }

# (2b) Create a corpus: Create a pair RDD called corpusRDD, consisting of a combination of the two small datasets, amazonRecToToken and googleRecToToken
corpusRDD = amazonRecToToken.union(googleRecToToken)

# (2c) Implement an IDFs function: The function should return an pair RDD where the key is the unique token and value is the IDF weight for the token.
def idfs(corpus):
    """ Compute IDF
    Args:
        corpus (RDD): input corpus is a pair RDD (id, tokenize(product)) that may have repetitive members
    Returns:
        RDD: a pair RDD of (token, IDF value)
    """
    # in a pair rdd of (id,tokenized_product) there may be two rdds with same id but different tokenized_product
    N = corpus.map(lambda (id,tokenized_product): id).distinct().count()
    uniqueTokens = corpus.flatMap(lambda (id,tokenized_product): set(tokenized_product))
    tokenCountPairTuple = uniqueTokens.map(lambda token: (token,1))
    tokenSumPairTuple = tokenCountPairTuple.reduceByKey(lambda a,b: a+b)
    return (tokenSumPairTuple.map(lambda (token,count): (token,float(N)/count)))
# with union we may have repetitive members
idfsSmall = idfs(amazonRecToToken.union(googleRecToToken))
uniqueTokenCount = idfsSmall.count()

print 'There are %s unique tokens in the small datasets.' % uniqueTokenCount

# (2d) Tokens with the smallest IDF
smallIDFTokens = idfsSmall.takeOrdered(11, lambda s: s[1])

#(2e) IDF Histogram
import matplotlib.pyplot as plt
small_idf_values = idfsSmall.map(lambda s: s[1]).collect()
fig = plt.figure(figsize=(8,3))
plt.hist(small_idf_values, 50, log=True)
display(fig) 
pass

from pyspark.sql import Row
# Create a DataFrame and visualize using display()
idfsToCountRow = idfsSmall.map(lambda (x, y): Row(token=x, value=y))
idfsToCountDF = sqlContext.createDataFrame(idfsToCountRow)
display(idfsToCountDF)

# (2f) Implement a TF-IDF function: Create a Python dictionary where each token maps to the token's frequency times the token's IDF weight
def tfidf(tokens, idfs):
    """ Compute TF-IDF
    Args:
        tokens (list of str): input list of tokens from tokenize
        idfs (dictionary): record to IDF value
    Returns:
        dictionary: a dictionary of records to TF-IDF values
    """
    tfs = tf(tokens)
    # create empty dictionary
    tfIdfDict = {}
    for token in tfs:
      tfIdfDict[token] = tfs[token]*idfs[token]
    return tfIdfDict
recb000hkgj8k = amazonRecToToken.filter(lambda x: x[0] == 'b000hkgj8k').collect()[0][1]
# idfsSmall = idfs(amazonRecToToken.union(googleRecToToken))
# collectAsMap: Return the key-value pairs in this RDD to the master as a dictionary.	
idfsSmallWeights = idfsSmall.collectAsMap()
rec_b000hkgj8k_weights = tfidf(recb000hkgj8k, idfsSmallWeights)
print 'Amazon record "b000hkgj8k" has tokens and weights:\n%s' % rec_b000hkgj8k_weights

# Part 3: ER as Text Similarity - Cosine Similarity: We will treat each document as a vector in some high dimensional space. Then, to compare two documents we compute the cosine of the angle between their two document vectors
# (3a) Implement the components of a cosineSimilarity function
import math

def dotprod(a, b):
    """ Compute dot product
    Args:
        a (dictionary): first dictionary of record to value
        b (dictionary): second dictionary of record to value
    Returns:
        dotProd: result of the dot product with the two input dictionaries
    """
    # create empty dictionary
    intersect = {}
    for item in a.keys(  ):
       if b.has_key(item):
           intersect[item] = a[item]*b[item]
    # now sum all of the values in the dictionary intersect
    return sum(intersect.values())

def norm(a):
    """ Compute square root of the dot product
    Args:
        a (dictionary): a dictionary of record to value
    Returns:
        norm (float): the square root of the dot product value
    """
    return math.sqrt(dotprod(a,a))

def cossim(a, b):
    """ Compute cosine similarity
    Args:
        a (dictionary): first dictionary of record to value
        b (dictionary): second dictionary of record to value
    Returns:
        cossim: dot product of two dictionaries divided by the norm of the first dictionary and
                then by the norm of the second dictionary
    """
    cosine_similarity = dotprod(a,b) / float(norm(a))
    cosine_similarity = cosine_similarity / float(norm(b))
    return cosine_similarity

testVec1 = {'foo': 2, 'bar': 3, 'baz': 5 }
testVec2 = {'foo': 1, 'bar': 0, 'baz': 20 }
dp = dotprod(testVec1, testVec2)
nm = norm(testVec1)
print dp, nm

# (3b) Implement a cosineSimilarity function: takes two strings and a dictionary of IDF weights, and computes their cosine similarity in the context of some global IDF weights.
def cosineSimilarity(string1, string2, idfsDictionary):
    """ Compute cosine similarity between two strings
    Args:
        string1 (str): first string
        string2 (str): second string
        idfsDictionary (dictionary): a dictionary of IDF values
    Returns:
        cossim: cosine similarity value
    """
    w1 = tfidf(tokenize(string1), idfsDictionary)
    w2 = tfidf(tokenize(string2), idfsDictionary)
    return cossim(w1, w2)
cossimAdobe = cosineSimilarity('Adobe Photoshop',
                               'Adobe Illustrator',
                               idfsSmallWeights)
print cossimAdobe

# (3c) Perform Entity Resolution: For every product record in the small Google dataset, use your cosineSimilarity function to compute its similarity to every record in the small Amazon dataset. Then, build a dictionary mapping (Google URL, Amazon ID) tuples to similarity scores between 0 and 1.
crossSmall = (googleSmall
              .cartesian(amazonSmall)
              .cache())

def computeSimilarity(record):
    """ Compute similarity on a combination record
    Args:
        record: a pair, (google record, amazon record)
    Returns:
        pair: a pair, (google URL, amazon ID, cosine similarity value)
    """
    googleRec = record[0]
    amazonRec = record[1]
    googleURL = googleRec[0]
    amazonID = amazonRec[0]
    googleValue = googleRec[1]
    amazonValue = amazonRec[1]
    #idfsSmallWeights is a dictionary containing union of googleRectoToken and AmazonRecToToken with their corresponding idf weights (global var)
    cs = cosineSimilarity(googleValue, amazonValue, idfsSmallWeights)
    return (googleURL, amazonID, cs)

similarities = (crossSmall
                .map(lambda rec: computeSimilarity(rec))
                .cache())

def similar(amazonID, googleURL):
    """ Return similarity value
    Args:
        amazonID: amazon ID
        googleURL: google URL
    Returns:
        similar: cosine similarity value
    """
    return (similarities
            .filter(lambda record: (record[0] == googleURL and record[1] == amazonID))
            .collect()[0][2])
similarityAmazonGoogle = similar('b000o24l3q', 'http://www.google.com/base/feeds/snippets/17242822440574356561')
print 'Requested similarity is %s.' % similarityAmazonGoogle

# (3d) Perform Entity Resolution with Broadcast Variables : First, we'll load the "gold standard" data and use it to answer several questions. We read and parse the Gold Standard data, where the format of each line is "Amazon Product ID","Google URL". The resulting RDD has elements of the form ("AmazonID GoogleURL", 'gold')
# Every row in the gold standard file has a pair of record IDs ("Amazon Product ID","Google URL") that belong to two record that describe the same thing in the real world. We will use the gold standard to evaluate our algorithms.
GOLDFILE_PATTERN = '^(.+),(.+)'
# Parse each line of a data file useing the specified regular expression pattern
def parse_goldfile_line(goldfile_line):
    """ Parse a line from the 'golden standard' data file
    Args:
        goldfile_line: a line of data
    Returns:
        pair: ((key, 'gold', 1 if successful or else 0))
    """
    match = re.search(GOLDFILE_PATTERN, goldfile_line)
    if match is None:
        print 'Invalid goldfile line: %s' % goldfile_line
        return (goldfile_line, -1)
    elif match.group(1) == '"idAmazon"':
        print 'Header datafile line: %s' % goldfile_line
        return (goldfile_line, 0)
    else:
        key = '%s %s' % (removeQuotes(match.group(1)), removeQuotes(match.group(2)))
        return ((key, 'gold'), 1)
goldfile = os.path.join(baseDir, inputPath, GOLD_STANDARD_PATH)
gsRaw = (sc
         .textFile(goldfile)
         .map(parse_goldfile_line)
         .cache())
gsFailed = (gsRaw
            .filter(lambda s: s[1] == -1)
            .map(lambda s: s[0]))
for line in gsFailed.take(10):
    print 'Invalid goldfile line: %s' % line
goldStandard = (gsRaw
                .filter(lambda s: s[1] == 1)
                .map(lambda (s,t): s)
                .cache())
print 'Read %d lines, successfully parsed %d lines, failed to parse %d lines' % (gsRaw.count(),
                                                                                 goldStandard.count(),
                                                                                 gsFailed.count())
assert (gsFailed.count() == 0)
assert (gsRaw.count() == (goldStandard.count() + 1))

# Using the "gold standard" data we can answer the following questions:
# How many true duplicate pairs are there in the small datasets?
# What is the average similarity score for true duplicates?
# What about for non-duplicates? The steps you should perform are:
# Create a new sims RDD from the similaritiesBroadcast RDD, where each element consists of a pair of the form ("AmazonID GoogleURL", cosineSimilarityScore). An example entry from sims is: ('b000bi7uqs http://www.google.com/base/feeds/snippets/18403148885652932189', 0.40202896125621296)
# Combine the sims RDD with the goldStandard RDD by creating a new trueDupsRDD RDD that has the just the cosine similarity scores for those "AmazonID GoogleURL" pairs that appear in both the sims RDD and goldStandard RDD. Hint: you can do this using the join() transformation.
# Count the number of true duplicate pairs in the trueDupsRDD dataset
# Compute the average similarity score for true duplicates in the trueDupsRDD datasets. Remember to use float for calculation
# Create a new nonDupsRDD RDD that has the just the cosine similarity scores for those "AmazonID GoogleURL" pairs from the similaritiesBroadcast RDD that do not appear in both the sims RDD and gold standard RDD.
# Compute the average similarity score for non-duplicates in the last datasets. Remember to use float for calculation

sims = similaritiesBroadcast.map(lambda (google, amazon, cs): ('{0} {1}'.format(amazon, google),cs))

trueDupsRDD = (sims.join(goldStandard))
trueDupsCount = trueDupsRDD.count()
tru_sum_cs = trueDupsRDD.map(lambda (key,cs_gold):cs_gold).map(lambda (cs,gold): (gold, cs)).reduceByKey(lambda cs1, cs2: cs1 + cs2).collect()
#print sim_total_sum_cs[0][1]
avgSimDups = tru_sum_cs[0][1] / float(trueDupsCount)

nonDupsRDD = (sims.subtractByKey(trueDupsRDD))
nonDupsCount = nonDupsRDD.count()
non_sum_cs = nonDupsRDD.map(lambda(key,cs): (1,cs)).reduceByKey(lambda cs1, cs2: cs1 + cs2).collect()
avgSimNon = non_sum_cs[0][1] / float(nonDupsCount)
print 'There are %s true duplicates.' % trueDupsCount
print 'The average similarity of true duplicates is %s.' % avgSimDups
print 'And for non duplicates, it is %s.' % avgSimNon

# Part 4: Scalable ER: implement a more scalable algorithm and use it to do entity resolution on the full dataset
# (4a) Tokenize the full dataset: Tokenize each of the two full datasets for Google and Amazon.
amazonFullRecToToken = amazon.map(lambda (id_key, product_value): (id_key, tokenize(product_value)))
googleFullRecToToken = google.map(lambda (id_key, product_value): (id_key, tokenize(product_value)))
print 'Amazon full dataset is %s products, Google full dataset is %s products' % (amazonFullRecToToken.count(),
                                                                                    googleFullRecToToken.count())
# (4b) Compute IDFs and TF-IDFs for the full datasets
fullCorpusRDD = amazonFullRecToToken.union(googleFullRecToToken)
idfsFull = idfs(fullCorpusRDD)
idfsFullCount = idfsFull.count()
print 'There are %s unique tokens in the full datasets.' % idfsFullCount

# Recompute IDFs for full dataset
idfsFullWeights = idfsFull.collectAsMap()
idfsFullBroadcast = sc.broadcast(idfsFullWeights)

# Pre-compute TF-IDF weights.  Build mappings from record ID weight vector.
amazonWeightsRDD = amazonFullRecToToken.map(lambda (id_key, tokenize): (id_key, tfidf(tokenize, idfsFullBroadcast.value)))
googleWeightsRDD = googleFullRecToToken.map(lambda (url_key, tokenize): (url_key, tfidf(tokenize, idfsFullBroadcast.value)))
print 'There are %s Amazon weights and %s Google weights.' % (amazonWeightsRDD.count(),
                                                              googleWeightsRDD.count())

(4c) Compute Norms for the weights from the full datasets
amazonNorms = amazonWeightsRDD.map(lambda (x,y): (x,norm(y)))
amazonNormsBroadcast = sc.broadcast(amazonNorms.collect())
googleNorms = googleWeightsRDD.map(lambda (x,y): (x,norm(y)))
googleNormsBroadcast = sc.broadcast(googleNorms.collect())

(4d) Create inverted indicies from the full datasets: An inverted index is a data structure that will allow us to avoid making quadratically many token comparisons
def invert(record):
    """ Invert (ID, tokens) to a list of (token, ID)
    Args:
        record: a pair, (ID, token vector) where token vector is a dictionary with key: token, value:wight
    Returns:
        pairs: a list of pairs of token to ID
    """
    list_of_pairs = []
    for key,value in record[1].items():
      list_of_pairs.append((key, record[0]))
    return list_of_pairs
amazonInvPairsRDD = (amazonWeightsRDD
                    .flatMap(lambda rec: invert(rec))
                    .cache())
googleInvPairsRDD = (googleWeightsRDD
                    .flatMap(lambda rec: invert(rec))
                    .cache())
print 'There are %s Amazon inverted pairs and %s Google inverted pairs.' % (amazonInvPairsRDD.count(),
                                                                            googleInvPairsRDD.count())

# (4e) Identify common tokens from the full dataset
def swap(record):
    """ Swap (token, (ID, URL)) to ((ID, URL), token)
    Args:
        record: a pair, (token, (ID, URL))
    Returns:
        pair: ((ID, URL), token)
    """
    token = record[0]
    keys = record[1]
    return (keys, token)
commonTokens = (amazonInvPairsRDD
                .join(googleInvPairsRDD)
                .map(lambda rec: swap(rec))
                .groupByKey()
                .cache())
print 'Found %d common tokens' % commonTokens.count()

# (4f) Identify common tokens from the full dataset: Use the data structures from parts (4a) and (4e) to build a dictionary to map record pairs to cosine similarity scores
amazonWeightsBroadcast = sc.broadcast(amazonWeightsRDD.collectAsMap())
googleWeightsBroadcast = sc.broadcast(googleWeightsRDD.collectAsMap())
def fastCosineSimilarity(record):
    """ Compute Cosine Similarity using Broadcast variables
    Args:
        record: ((ID, URL), token) where token is a list of tokens
    Returns:
        pair: ((KEY, VALUE), cosine similarity value)
    """
    amazonRec = record[0][0]
    googleRec = record[0][1]
    tokens = record[1]
    # s is sum
    s = 0
    for token in tokens:
      products = (amazonWeightsBroadcast.value[amazonRec][token]) * (googleWeightsBroadcast.value[googleRec][token])
      s = s + products
    # below value is similarity or cos (teta)
    value = s/float(amazonNormsBroadcast.value[amazonRec] * googleNormsBroadcast.value[googleRec])
    key = (amazonRec, googleRec)
    return (key, value)
similaritiesFullRDD = (commonTokens.map(lambda rec:fastCosineSimilarity(rec)).cache())
print similaritiesFullRDD.count()

# Part 5: Analysis: Now we have an authoritative list of record-pair similarities, but we need a way to use those similarities to decide if two records are duplicates or not. The simplest approach is to pick a threshold. Pairs whose similarity is above the threshold are declared duplicates, and pairs below the threshold are declared distinct.
# (5a) Counting True Positives, False Positives, and False Negatives
# Create an RDD of ((Amazon ID, Google URL), similarity score)
simsFullRDD = similaritiesFullRDD.map(lambda x: ("%s %s" % (x[0][0], x[0][1]), x[1]))
assert (simsFullRDD.count() == 2441100)

# Create an RDD of just the similarity scores
simsFullValuesRDD = (simsFullRDD
                     .map(lambda x: x[1])
                     .cache())
assert (simsFullValuesRDD.count() == 2441100)

# Look up all similarity scores for true duplicates

# This helper function will return the similarity score for records that are in the gold standard and the simsFullRDD (True positives), and will return 0 for records that are in the gold standard but not in simsFullRDD (False Negatives).
def gs_value(record):
    if (record[1][1] is None):
        return 0
    else:
        return record[1][1]

# Join the gold standard and simsFullRDD, and then extract the similarities scores using the helper function
trueDupSimsRDD = (goldStandard
                  .leftOuterJoin(simsFullRDD)
                  .map(gs_value)
                  .cache())
print 'There are %s true duplicates.' % trueDupSimsRDD.count()
assert(trueDupSimsRDD.count() == 1300)

from pyspark.accumulators import AccumulatorParam
class VectorAccumulatorParam(AccumulatorParam):
    # Initialize the VectorAccumulator to 0
    def zero(self, value):
        return [0] * len(value)

    # Add two VectorAccumulator variables
    def addInPlace(self, val1, val2):
        for i in xrange(len(val1)):
            val1[i] += val2[i]
        return val1

# Return a list with entry x set to value and all other entries set to 0
def set_bit(x, value, length):
    bits = []
    for y in xrange(length):
        if (x == y):
          bits.append(value)
        else:
          bits.append(0)
    return bits

# Pre-bin counts of false positives for different threshold ranges
BINS = 101
nthresholds = 100
def bin(similarity):
    return int(similarity * nthresholds)

# fpCounts[i] = number of entries (possible false positives) where bin(similarity) == i
zeros = [0] * BINS
fpCounts = sc.accumulator(zeros, VectorAccumulatorParam())

def add_element(score):
    global fpCounts
    b = bin(score)
    fpCounts += set_bit(b, 1, BINS)

simsFullValuesRDD.foreach(add_element)

# Remove true positives from FP counts
def sub_element(score):
    global fpCounts
    b = bin(score)
    fpCounts += set_bit(b, -1, BINS)

trueDupSimsRDD.foreach(sub_element)

def falsepos(threshold):
    fpList = fpCounts.value
    return sum([fpList[b] for b in range(0, BINS) if float(b) / nthresholds >= threshold])

def falseneg(threshold):
    return trueDupSimsRDD.filter(lambda x: x < threshold).count()

def truepos(threshold):
    return trueDupSimsRDD.count() - falsenegDict[threshold]

# (5b) Precision, Recall, and F-measures


