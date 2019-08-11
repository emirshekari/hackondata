import matplotlib.pyplot as plt
from numpy.linalg import svd
from pyspark.sql.functions import col, lower
import nltk
from pyspark.sql.functions import UserDefinedFunction
from pyspark.sql.types import StringType
from pyspark.sql.types import *
from stemming.porter2 import stem
import timeit
import numpy
import pandas
from collections import defaultdict
from nltk import word_tokenize, sent_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
# import rake
import os
from collections import Counter
import re
nltk.download('all')
from pyspark.sql.functions import lit
from pyspark.sql.types import StructField
from pyspark.sql.types import StringType
from __future__ import absolute_import
from __future__ import print_function
import operator
import six
from six.moves import range


debug = False
test = True

def is_number(s):
    try:
        float(s) if '.' in s else int(s)
        return True
    except ValueError:
        return False

def load_stop_words(stop_word_file):
    """
    Utility function to load stop words from a file and return as a list of words
    @param stop_word_file Path and file name of a file containing stop words.
    @return list A list of stop words.
    """
    stop_words = []
    MOUNT_NAME = "hackondata-toronto"
    myRDD = (sc.textFile("/mnt/%s/SmartStoplist.txt" % MOUNT_NAME))
    for line in myRDD.toLocalIterator():
        if line.strip()[0:1] != "#":
            for word in line.split():  # in case more than one per line
                stop_words.append(word)
    return stop_words

def separate_words(text, min_word_return_size):
    """
    Utility function to return a list of all words that have a length greater than a specified number of characters.
    @param text The text that must be split in to words.
    @param min_word_return_size The minimum no of characters a word must have to be included.
    """
    splitter = re.compile('[^a-zA-Z0-9_\\+\\-/]')
    words = []
    for single_word in splitter.split(text):
        current_word = single_word.strip().lower()
        # leave numbers in phrase, but don't count as words, since they tend to invalidate scores of their phrases
        if len(current_word) > min_word_return_size and current_word != '' and not is_number(current_word):
            words.append(current_word)
    return words

def split_sentences(text):
    """
    Utility function to return a list of sentences.
    @param text The text that must be split in to sentences.
    output is a list
    """
    sentence_delimiters = re.compile(u'[\\[\\]\n.!?,;:\t\\-\\"\\(\\)\\\'\u2019\u2013]')
    sentences = sentence_delimiters.split(text)
    return sentences

def build_stop_word_regex(stop_word_list):
    stop_word_regex_list = []
    for word in stop_word_list:
        word_regex = '\\b' + word + '\\b'
        stop_word_regex_list.append(word_regex)
    stop_word_pattern = re.compile('|'.join(stop_word_regex_list), re.IGNORECASE)
    return stop_word_pattern

#
# Function that extracts the adjoined candidates from a list of sentences and filters them by frequency
#
def extract_adjoined_candidates(sentence_list, stoplist, min_keywords, max_keywords, min_freq):
    adjoined_candidates = []
    for s in sentence_list:
        # Extracts the candidates from each single sentence and adds them to the list
        adjoined_candidates += adjoined_candidates_from_sentence(s, stoplist, min_keywords, max_keywords)
    # Filters the candidates and returns them
    return filter_adjoined_candidates(adjoined_candidates, min_freq)


# return adjoined_candidates

#
# Function that extracts the adjoined candidates from a single sentence
#
def adjoined_candidates_from_sentence(s, stoplist, min_keywords, max_keywords):
    # Initializes the candidate list to empty
    candidates = []
    # Splits the sentence to get a list of lowercase words
    sl = s.lower().split()
    # For each possible length of the adjoined candidate
    for num_keywords in range(min_keywords, max_keywords + 1):
        # Until the third-last word
        for i in range(0, len(sl) - num_keywords):
            # Position i marks the first word of the candidate. Proceeds only if it's not a stopword
            if sl[i] not in stoplist:
                candidate = sl[i]
                # Initializes j (the pointer to the next word) to 1
                j = 1
                # Initializes the word counter. This counts the non-stopwords words in the candidate
                keyword_counter = 1
                contains_stopword = False
                # Until the word count reaches the maximum number of keywords or the end is reached
                while keyword_counter < num_keywords and i + j < len(sl):
                    # Adds the next word to the candidate
                    candidate = candidate + ' ' + sl[i + j]
                    # If it's not a stopword, increase the word counter. If it is, turn on the flag
                    if sl[i + j] not in stoplist:
                        keyword_counter += 1
                    else:
                        contains_stopword = True
                    # Next position
                    j += 1
                # Adds the candidate to the list only if:
                # 1) it contains at least a stopword (if it doesn't it's already been considered)
                # AND
                # 2) the last word is not a stopword
                # AND
                # 3) the adjoined candidate keyphrase contains exactly the correct number of keywords (to avoid doubles)
                if contains_stopword and candidate.split()[-1] not in stoplist and keyword_counter == num_keywords:
                    candidates.append(candidate)
    return candidates

#
# Function that filters the adjoined candidates to keep only those that appears with a certain frequency
#
def filter_adjoined_candidates(candidates, min_freq):
    # Creates a dictionary where the key is the candidate and the value is the frequency of the candidate
    candidates_freq = Counter(candidates)
    filtered_candidates = []
    # Uses the dictionary to filter the candidates
    for candidate in candidates:
        freq = candidates_freq[candidate]
        if freq >= min_freq:
            filtered_candidates.append(candidate)
    return filtered_candidates

def generate_candidate_keywords(sentence_list, stopword_pattern, stop_word_list, min_char_length=1, max_words_length=5,
                                min_words_length_adj=1, max_words_length_adj=1, min_phrase_freq_adj=2):
    phrase_list = []
    for s in sentence_list:
        tmp = re.sub(stopword_pattern, '|', s.strip())
        phrases = tmp.split("|")
        for phrase in phrases:
            phrase = phrase.strip().lower()
            if phrase != "" and is_acceptable(phrase, min_char_length, max_words_length):
                phrase_list.append(phrase)
    phrase_list += extract_adjoined_candidates(sentence_list, stop_word_list, min_words_length_adj,
                                               max_words_length_adj, min_phrase_freq_adj)
    return phrase_list

def is_acceptable(phrase, min_char_length, max_words_length):
    # a phrase must have a min length in characters
    if len(phrase) < min_char_length:
        return 0
    # a phrase must have a max number of words
    words = phrase.split()
    if len(words) > max_words_length:
        return 0
    digits = 0
    alpha = 0
    for i in range(0, len(phrase)):
        if phrase[i].isdigit():
            digits += 1
        elif phrase[i].isalpha():
            alpha += 1
    # a phrase must have at least one alpha character
    if alpha == 0:
        return 0
    # a phrase must have more alpha than digits characters
    if digits > alpha:
        return 0
    return 1

def calculate_word_scores(phraseList):
    word_frequency = {}
    word_degree = {}
    for phrase in phraseList:
        word_list = separate_words(phrase, 0)
        word_list_length = len(word_list)
        word_list_degree = word_list_length - 1
        # if word_list_degree > 3: word_list_degree = 3 #exp.
        for word in word_list:
            word_frequency.setdefault(word, 0)
            word_frequency[word] += 1
            word_degree.setdefault(word, 0)
            word_degree[word] += word_list_degree  # orig.
            # word_degree[word] += 1/(word_list_length*1.0) #exp.
    for item in word_frequency:
        word_degree[item] = word_degree[item] + word_frequency[item]

    # Calculate Word scores = deg(w)/frew(w)
    word_score = {}
    for item in word_frequency:
        word_score.setdefault(item, 0)
        word_score[item] = word_degree[item] / (word_frequency[item] * 1.0)  # orig.
    # word_score[item] = word_frequency[item]/(word_degree[item] * 1.0) #exp.
    return word_score

def generate_candidate_keyword_scores(phrase_list, word_score, min_keyword_frequency=1):
    keyword_candidates = {}
    for phrase in phrase_list:
        if min_keyword_frequency > 1:
            if phrase_list.count(phrase) < min_keyword_frequency:
                continue
        keyword_candidates.setdefault(phrase, 0)
        word_list = separate_words(phrase, 0)
        candidate_score = 0
        for word in word_list:
            candidate_score += word_score[word]
        keyword_candidates[phrase] = candidate_score
    return keyword_candidates

class Rake(object):
    def __init__(self, stop_words_path, min_char_length=1, max_words_length=5, min_keyword_frequency=1,
                 min_words_length_adj=1, max_words_length_adj=1, min_phrase_freq_adj=2):
        self.__stop_words_path = stop_words_path
        self.__stop_words_list = load_stop_words(stop_words_path)
        self.__min_char_length = min_char_length
        self.__max_words_length = max_words_length
        self.__min_keyword_frequency = min_keyword_frequency
        self.__min_words_length_adj = min_words_length_adj
        self.__max_words_length_adj = max_words_length_adj
        self.__min_phrase_freq_adj = min_phrase_freq_adj

    def run(self, text):
        sentence_list = split_sentences(text)
        stop_words_pattern = build_stop_word_regex(self.__stop_words_list)
        phrase_list = generate_candidate_keywords(sentence_list, stop_words_pattern, self.__stop_words_list,
                                                  self.__min_char_length, self.__max_words_length,
                                                  self.__min_words_length_adj, self.__max_words_length_adj,
                                                  self.__min_phrase_freq_adj)
        word_scores = calculate_word_scores(phrase_list)
        keyword_candidates = generate_candidate_keyword_scores(phrase_list, word_scores, self.__min_keyword_frequency)
        sorted_keywords = sorted(six.iteritems(keyword_candidates), key=operator.itemgetter(1), reverse=True)
        return sorted_keywords
        
        
        def dict_map(dict1,dict2):
  # purpose: takes 2 dicts and merge them in a way that new keys are added and old key's values are summed up.
  # input: 2 dictionaries
  # output: dictionary
  
  for word in dict2.keys():
    if word in dict1.keys():
      dict1[word] = dict1[word]+dict2[word] 
    else:
      dict1[word]=dict2[word]
  return dict1
def word_pairs(review):
    # purpose: derive bigrams
    # input : a list of strings
    # output: list of all of bigrams in a given text
    return [a + " " + b for a, b in zip(review, review[1:])]


def word_triples(review):
    # purpose: derive trigrams
    # input : a list of strings
    # output: list of all of trigrams in a given text
    trigram_ls = []
    for a, b, c in zip(review, review[1:], review[2:]):
        if (a != 'a' and b != 'a' and c != 'a') and (a != 'the' and b != 'the' and c != 'the'):
            trigram_ls.append(a + " " + b + " " + c)
    return trigram_ls

def find_topics(simpsent, whole_text_key_phrase_prob):
    # purpose: takes in a simple sent and look at the list of key phrases of the WHOLE TEXT if some key phrases are in that simple sent then return them in a dictionary with its counts as values
    # input: simpsent is a string, and whole_text_key_phrases is a list of all of the keyphrases obtained in topic_extraction,
    # output: a dictionary,

    # consider the first element of tuples in the list whole_text_key_phrases
    whole_text_key_phrases = [i[0] for i in whole_text_key_phrase_prob]# this is a list of keyphrases from all brands
    # initialize the list of topics
    topics_in_simp_sent = []
    wnl = WordNetLemmatizer()
    # lemmatize the given text
    lemmatized_simpsent = [wnl.lemmatize(i, j[0].lower()) if j[0].lower() in ['r', 'a', 'n', 'v'] else wnl.lemmatize(i) for i, j in pos_tag(word_tokenize(simpsent.lower()))]
    stemmed_ls = [stem(word) for word in lemmatized_simpsent]

    #  whole_text_key_phrases includes keyphrases with a single word or multiple words
    # lets start finding the single word topics >> unigram topics
    for word in stemmed_ls:
        if (word in whole_text_key_phrases):# if a word is a topic and not added tp the list yet, then add it
            topics_in_simp_sent.append(word)

    # next lets find the double word topics >> bigram topics
    # derive bigrams of the input in the form of a single string like 'A B'
    bigrams = word_pairs(stemmed_ls)
    for bigram in bigrams:
        if (bigram in whole_text_key_phrases):
            topics_in_simp_sent.append(bigram)

    # next lets find the triple word topics >> trigram topics
    # derive trigrams of the input in the form of a single string like 'A B C'
    trigrams = word_triples(stemmed_ls)
    for trigram in trigrams:
        if (trigram in whole_text_key_phrases):
            topics_in_simp_sent.append(trigram)
    # finally create a {'keyword':count} dictionary using Counter and return it.
    keyword_count_dict = {}
    for topic in topics_in_simp_sent:
        keyword_count_dict[topic] = topics_in_simp_sent.count(topic)
    return keyword_count_dict

def stem_reverse_to_word(whole_text):
    # purpose: convert a stemmed form of word to its original form where original is considered to be the most frequent form
    # input: whole_text is string
    # output: dictionary {'stem form':'original form'}
    whole_text = re.sub('[^a-zA-Z0-9-?!,.;\']', ' ', whole_text)  # Remove special characters that might cause problems with stemming
    wnl = WordNetLemmatizer()
    lemmatized_ls = [wnl.lemmatize(i, j[0].lower()) if j[0].lower() in ['r', 'a', 'n', 'v'] else wnl.lemmatize(i) for i, j in pos_tag(word_tokenize(whole_text))]
    # create a {'word': count} dictionary using Counter()
    word_count_dict = Counter(lemmatized_ls)
    stemmed_dict = {}# initialization
    for word in word_count_dict.keys():
        if not(stem(word) in stemmed_dict):
            # if the stemmed form of the word is not in the dictionary then add it
            stemmed_dict[stem(word)] = word
        elif word_count_dict[stemmed_dict[stem(word)]] < word_count_dict[word]:
            # if there is a form of the word that has bigger frequency then update the stemmed_dict dictionary
            stemmed_dict[stem(word)] = word
    return stemmed_dict
  
def sent_tokenization(review):
    # purpose: derive simple sentences of a review and convert to lower case
    # input : a single review
    # output: list of all of simple sentences
    sents = sent_tokenize(review.lower())
    return sents
    
    
    class CA(object): # Correspondence Analysis
    def __init__(self, ct):

        self.coor = {}
        self.rows = ct.index.values if hasattr(ct, 'index') else None
        self.cols = ct.columns.values if hasattr(ct, 'columns') else None

        # contingency table
        N = numpy.matrix(ct, dtype=float)# N is the original data matrix i.e. the input in numpy format

        # correspondence matrix from contingency table
        P = N / N.sum() # divide the matrix N by its grand total

        # row and column marginal totals of P as vectors
        r = P.sum(axis=1)# is a column vector with number of brands as number of rows and only 1 col
        c = P.sum(axis=0).T # transpose of c is a column vector with number of attributes as number of rows and only 1 col

        # diagonal matrices of row/column sums (The A1 attribute of a matrix returns a flattened numpy array.)
        D_r_rsq = numpy.diag(1. / numpy.sqrt(r.A1)) # this is (D_r)^(-1/2) in the paper
        D_c_rsq = numpy.diag(1. / numpy.sqrt(c.A1)) # this is (D_c)^(-1/2) in the paper

        # the matrix of standarized residuals
        S = D_r_rsq * (P - r * c.T) * D_c_rsq

        # compute the SVD, in SVD If full_matrices is False, the shapes of U and V are (M,K) and (K,N), where K = min(M,N)
        U, D_a, V = svd(S, full_matrices=False) # D_a is a vector of eigen values
        D_a = numpy.asmatrix(numpy.diag(D_a)) # convert the vector of eigen values to a diagonal matrix of eigen values
        V = V.T

        # principal coordinates of rows
        F = D_r_rsq * U * D_a

        # principal coordinates of columns
        G = D_c_rsq * V * D_a
        
        self.F = F.A
        self.G = G.A

    def plot(self):
        """Plot the first and second dimensions."""
        xmin, xmax = None, None
        ymin, ymax = None, None
        if self.rows is not None:
            for i, t in enumerate(self.rows):
                x, y = self.F[i, 0], self.F[i, 1]
                plt.text(x, y, t, va='center', ha='center', color='b')
                xmin = min(x, xmin if xmin else x)
                xmax = max(x, xmax if xmax else x)
                ymin = min(y, ymin if ymin else y)
                ymax = max(y, ymax if ymax else y)
        else:
            plt.plot(self.F[:, 0], self.F[:, 1], 'ro')

        if self.cols is not None:
            for i, t in enumerate(self.cols):
                x, y = self.G[i, 0], self.G[i, 1]
                plt.text(x, y, t, va='center', ha='center', color='m')
                xmin = min(x, xmin if xmin else x)
                xmax = max(x, xmax if xmax else x)
                ymin = min(y, ymin if ymin else y)
                ymax = max(y, ymax if ymax else y)
        else:
            plt.plot(self.G[:, 0], self.G[:, 1], 'bs')

        if xmin and xmax:
            pad = (xmax - xmin) * 0.1
            plt.xlim(xmin - pad, xmax + pad)
        if ymin and ymax:
            pad = (ymax - ymin) * 0.1
            plt.ylim(ymin - pad, ymax + pad)

        plt.grid()
        plt.xlabel('Dim 1')
        plt.ylabel('Dim 2')

    def output(self):
        for i, t in enumerate(self.rows):
            ## First 2 ##
            self.coor[t] = [self.F[i, 0], self.F[i, 1], 'blue']
        for i, t in enumerate(self.cols):
            self.coor[t] = [self.G[i, 0], self.G[i, 1], 'magenta']
        # print self.coor
        return self.coor

    def distance(self, coor1, coor2):
        distance = 0
        for i in range(len(coor1)):
            distance += (coor1[i] - coor2[i]) ** 2
        return distance
        
        
MOUNT_NAME = "hackondata-toronto"
myRDD = sc.textFile("/mnt/%s/brand_asin_map.csv" % MOUNT_NAME) # a 3M reviews file of all of the 24 amazon products pulled out randomly
# myRDD.cache() # save in memory and don't recompute
# get header
header = myRDD.first()
# remove header  
rdd_header_removed = myRDD.filter(lambda x: x != header)
# The schema is encoded in a string.
schemaString = "asin brand"
fields = [StructField(field_name, StringType(), True) for field_name in schemaString.split()]
print (fields)
schema = StructType(fields)
print (schema)
# Apply the schema to the RDD.
asin_brand_df = sqlContext.createDataFrame(rdd_header_removed.map(lambda x:(x.split(',')[0],x.split(',')[1])),schema)

display(asin_brand_df)

asin_brand_dict = asin_brand_df.toPandas().set_index('asin').T.to_dict('list') 
print (asin_brand_dict)

# lets read the file that has reviews in it, then use the dictionary asin_brand_dict to add a "brand" column to this df
all_lipstick_rdd = sc.textFile("/mnt/%s/reviews_all_lipsticks.csv" % MOUNT_NAME) # a 3M reviews file of all of the 24 amazon products pulled out randomly
# get header
header = all_lipstick_rdd.first()
# remove header  
all_lipstick_header_removed_rdd = all_lipstick_rdd.filter(lambda x: x != header)

# The schema is encoded in a string.
schemaString = "brand Review"
fields = [StructField(field_name, StringType(), True) for field_name in schemaString.split()]
schema = StructType(fields)
# Apply the schema to the RDD.
asin_review_df = sqlContext.createDataFrame(all_lipstick_header_removed_rdd.map(lambda x:(x.split(',')[0],re.sub(r'^"|"$', '', x.split(',')[1]))),schema) 
# around some reviews there are quotations, remove it with re

# change the values in the col brand using the dictionary asin_brand_dict={asin:brand}
def modify_values(r):
    if r in asin_brand_dict.keys():
        return asin_brand_dict[r][0]
    else:
        return 
      
name = 'brand'
udf = UserDefinedFunction(lambda x: modify_values(x), StringType()) #when using UserDefinedFunction, we have to specify type of the output, in this case, StringType()
brand_review_df = asin_review_df.select(*[udf(column).alias(name) if column == name else column for column in asin_review_df.columns])
display(brand_review_df)

# drop any row where its brand is null
na_dropped_df = brand_review_df.where(col("brand").isNotNull())
# lower case a column of the sql dataframe 
def lower_case(column):
    """ lower case the reviews
    Args:
        column (Column): A Column containing a review.
    Returns:
        Column: A Column named 'lower_cased' with clean-up operations applied. """
    return lower(column).alias('lower_cased')
lower_case_df = na_dropped_df.select(lower_case(col('Review')),'brand')

# convert df to rdd to do tokenization
lower_case_rdd = lower_case_df.rdd
def tokenization(sentence):
  return sent_tokenize(sentence)
sent_tokenized_rdd = lower_case_rdd.map(lambda rev_brand:(tokenization(rev_brand[0]),rev_brand[1])) # first element of pair rdd is "review"
print(sent_tokenized_rdd.take(5))

def simp_sent_to_brand_map(sents_brand_tuple):
  #purpose: assign the corresponding brand to each simple sentence of a review
  # input: a tuple like ([simpsent1,simpsent2, ...], brand)
  # output: a list
  ls_of_tuples = []
  for simpsent in sents_brand_tuple[0]:
    ls_of_tuples.append((simpsent, sents_brand_tuple[1]))
  return ls_of_tuples
  
  simpsent_brand_rdd = sent_tokenized_rdd.flatMap(lambda simpsents_ls_brand_tuple:simp_sent_to_brand_map(simpsents_ls_brand_tuple)) # each element of this rdd is a tuple (simpsent,brand)
# convert simpsent_brand_rdd to sql rdd to use later when each brand should be considered separately:
# The schema is encoded in a string.
schemaString = "simp_sent brand"
fields = [StructField(field_name, StringType(), True) for field_name in schemaString.split()]
schema = StructType(fields)
# Apply the schema to the RDD.
simpsent_brand_df = sqlContext.createDataFrame(simpsent_brand_rdd.map(lambda simpsent_brand:(simpsent_brand[0],simpsent_brand[1])),schema)
simpsent_brand_df.show()

only_sentences_rdd = simpsent_brand_rdd.map(lambda simpsent_brand: simpsent_brand[0]) # drop the brand column 
text = " ".join(only_sentences_rdd.collect())
whole_text = re.sub('[^a-zA-Z0-9-?!,.;\']', ' ', text)# Remove special characters that might cause problems with stemming (e.g. *&^%...)
wnl = WordNetLemmatizer()
lemmatized_ls_ = [wnl.lemmatize(i, j[0].lower()) if j[0].lower() in ['r', 'a', 'n', 'v'] else wnl.lemmatize(i) for i, j in pos_tag(word_tokenize(whole_text))]
stemmed_ls_ = [stem(word) for word in lemmatized_ls_]
lemmatized_stemmed_text = ' '.join(stemmed_ls_)

# load the list of stop words
stoppath = sc.textFile("/mnt/%s/SmartStoplist.txt" % MOUNT_NAME).collect() 
# Each word has at least 3 characters/ Each phrase has at most 3 words/ Each keyword appears in the text at least 150 times:
rake = Rake(stoppath, 3,3,150)
keyphrases = rake.run(lemmatized_stemmed_text)
print ('There are', len(keyphrases), 'key phrases in this text!')
print (keyphrases)

# derive a map of stem form of the word to its original form using a dictionary of maps
stem_map = stem_reverse_to_word(text) # stem_map is a dictionary like {'stem form':'original form'}
# how many brands (categories) are there:
brands_rdd = sc.parallelize(asin_brand_dict.values())
print ('There are', brands_rdd.count(), ' brands, in the dictionary that maps asin to brand name, so some of the brands may be repetitive.')
# get unique elements in the rdd
unique_brands_rdd = brands_rdd.map(lambda x: x[0]).distinct()
# unique_brands_rdd = unique_brands_rdd.distinct()
print ('There are', unique_brands_rdd.count(), 'unique brands.')

phrases_whose_words_not_stemmed = []
brands_dict_ls = []# items of this list are dictionaries, one dict for each brand, each time the big for loop below runs, one item is added to this list
for brand in unique_brands_rdd.collect():
    # create the sub df which is only for a specific brand
    brand_df = simpsent_brand_df.where(col("brand").isin({brand}))
    # find the topics for each row of the sentences_df i.e. for each simple sentence ,, if there is any      
    name = 'simp_sent'
    udf1 = UserDefinedFunction(lambda x: find_topics(x, keyphrases), MapType(StringType(), IntegerType()))# keyphrases is a list of all key phrases extracted by Rake.
    df3 = brand_df.select(*[udf1(name).alias('keyword_count_dict')])  
#     if brand == 'Revlon':
#       display(df3)    
    # create a big dictionary of all of the dictionaries in the column "key_phrases" like this : keyword_count_dict={'keyword':[count1, count2, ...]}
    # to do this, convert the df to rdd
    keyword_count_dicts_rdd = df3.rdd # when u convert df to rdd like this the element of dd are in form of Row() lets get rid of Row() tuples:
    cleaned_Row_keyword_count_dicts_rdd = keyword_count_dicts_rdd.map(lambda row: [c for c in row][0])
#     print (cleaned_Row_keyword_count_dicts_rdd.first())
    
    brand_keyword_count_dict = cleaned_Row_keyword_count_dicts_rdd.reduce(dict_map)
#     if brand == 'Revlon':    
#         print(brand_keyword_count_dict)
    # At this point we update keyword_count_dict so that if there is anything in the dictionary keyphrases but not in keyword_count_dict, then add it as a key with value zero.
    for keyword in [i[0] for i in keyphrases]:
        if keyword not in brand_keyword_count_dict.keys():
            brand_keyword_count_dict[keyword] = 0   
    # use the dictionary stem_map to map stem form of the word to its original form
    for key in brand_keyword_count_dict.keys(): # stem_map is a dictionary like {'stem form':'original form'}
        if (key in stem_map.keys()):
            brand_keyword_count_dict[stem_map[key]] = brand_keyword_count_dict.pop(key)# update from stem form to original form
        else:# if a keyphrase is not in stem_map it probably means that it is multi word, split it and find the original form of each token then join
            original_joined = ''
            for token in key.split(' '):
                if token in stem_map.keys():
                    original_joined = original_joined + stem_map[token] + ' '
                elif token in stem_map.values():# if token is already in original form (and not stemmed form)
                    original_joined = original_joined + token + ' '
                else:
                    original_joined = original_joined + token + ' '
                    # stem_not_found_counter += 1
                    phrases_whose_words_not_stemmed.append(key)
            # remove the space after the last word
            original_joined = original_joined.strip()
            brand_keyword_count_dict[original_joined] = brand_keyword_count_dict.pop(key) # update from stem form to original form
#     if brand == 'Revlon':    
#         print(brand_keyword_count_dict)
    brands_dict_ls.append(brand_keyword_count_dict) # in each loop add the dictionary of {keyphrase:count} of a specific brand to this list
# print (brands_dict_ls)

ca_dict ={}
# merge the counts of each keyphrase in the brands dictionaries and save the result as ca_dict ={keyphrase:[count_in_rev, count_in_clnq]}
for keyword in brands_dict_ls[0]:# brands_dict_ls[0] is a dictionary of the first brand, you can put the dict of any brand because they all have whole keyphrases
    ca_dict[keyword] = [brands_dict_ls[i][keyword] for i in range(len(brands_dict_ls))]# [rev_dict[keyword], clnq_dict[keyword],...]=[count_in_rev, count_in_clnq,...]
# add one element to the dictionary : this element will be the first column of the input to ca.py
ca_dict['brand'] = unique_brands_rdd.collect()
#print (ca_dict)
# now convert the dictionary to a pandas df
lipstick_df = pandas.DataFrame({'brand': ca_dict['brand']})
#display(lipstick_df)
# fill the table that is the input to ca.py
for attribute in ca_dict.keys():
    lipstick_df[attribute] = ca_dict[attribute]
display(lipstick_df)
# print (lipstick_df)

price = ['brand', 'buy', 'price', 'purchase']
customer_satisfaction = ['brand','perfect','good','discontinue','great','love','recommend', 'happy','glad','amazing','nice']
customer_service = ['brand','amazon', 'expect', 'review', 'receive', 'packaging', 'store', 'picture', 'product', 'order', 'tube']
lipstick_property = ['brand','moisturize', 'texture', 'color', 'lip', 'lipstick', 'red', 'dry', 'gloss', 'creamy', 'line', 'smooth', 'pink', 'lip color', 'shade']
general = ['brand', 'find', 'bit', 'line', 'work', 'give', 'make', 'able', 'definitely', 'apply', 'eat', 'expect', 'wear', 'put', 'feel', 'top', 'lot', 'exactly']
lasting = ['brand', 'day', 'long time', 'year', 'long', 'hour', 'stay', 'longer', 'time']
comprehensive = ['brand', 'price', 'love', 'recommend', 'tube', 'happy', 'creamy', 'shade', 'smooth', 'expect','color', 'gloss']
############################################################################################################
# remove columns of dataframe that are all 0
# lipstick_df = lipstick_df.loc[:, (lipstick_df != 0).any(axis=0)]
# print(lipstick_df)
customer_satisfaction_df = lipstick_df.loc[[0, 3, 6, 10, 11, 14, 15], customer_satisfaction]
# print (customer_satisfaction_df)
df = customer_satisfaction_df 
df = df.set_index('brand')
ca = CA(df) # CA is a class
plt.figure(1)
ca.plot()
image = plt.show() 
display(image)
############################################################################################################
customer_service_df = lipstick_df.loc[[0, 3, 6, 10, 11, 14, 15], customer_service]
# print (customer_satisfaction_df)
df = customer_service_df 
df = df.set_index('brand')
ca = CA(df)
plt.figure(2)
ca.plot()
image = plt.show() 
display(image)
############################################################################################################

lipstick_property_df = lipstick_df.loc[[0, 3, 6, 10, 11, 14, 15], lipstick_property]
# print (customer_satisfaction_df)
df = lipstick_property_df 
df = df.set_index('brand')
ca = CA(df)
plt.figure(3)
ca.plot()
image = plt.show() 
display(image)
