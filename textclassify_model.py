# STEP 1: rename this file to textclassify_model.py

# feel free to include more imports as needed here
# these are the ones that we used for the base model
import numpy as np
import sys
#from collections import Counter
import pandas as pd
from math import log,e

"""
Your name and file comment here:
Name: Pritish Arora
"""


"""
Cite your sources here:
"""

"""
Implement your functions that are not methods of the TextClassify class here
"""
def generate_tuples_from_file(training_file_path):
  """
  Generates tuples from file formated like:
  id\ttext\tlabel
  Parameters:
    training_file_path - str path to file to read in
  Return:
  a list of tuples of strings formatted [(id, example_text, label), (id, example_text, label)....]
  """
  f = open(training_file_path, "r", encoding="utf8")
  listOfExamples = []
  for review in f:
    if len(review.strip()) == 0:
      continue
    dataInReview = review.split("\t")
    for i in range(len(dataInReview)):
      # remove any extraneous whitespace
      dataInReview[i] = dataInReview[i].strip()
    t = tuple(dataInReview)
    listOfExamples.append(t)
  f.close()
  return listOfExamples

def precision(gold_labels, predicted_labels):
  """
  Calculates the precision for a set of predicted labels give the gold (ground truth) labels.
  Parameters:
      gold_labels (list): a list of labels assigned by hand ("truth")
      predicted_labels (list): a corresponding list of labels predicted by the system
  Returns: double precision (a number from 0 to 1)
  """
  true_pos = 0
  false_pos = 0
  positive_label = '1'
  pre = float()
  for i in range(len(predicted_labels)):
    if (predicted_labels[i] == positive_label) and (gold_labels[i] == positive_label):
      true_pos += 1
    if (predicted_labels[i] == positive_label) and (gold_labels[i] != positive_label):
      false_pos += 1
  pre = true_pos / (true_pos + false_pos)

  return pre


def recall(gold_labels, predicted_labels):
  """
  Calculates the recall for a set of predicted labels give the gold (ground truth) labels.
  Parameters:
      gold_labels (list): a list of labels assigned by hand ("truth")
      predicted_labels (list): a corresponding list of labels predicted by the system
  Returns: double recall (a number from 0 to 1)
  """
  true_pos = 0
  false_neg = 0
  positive_label = '1'
  rec = float()
  for i in range(len(predicted_labels)):
    if (predicted_labels[i] == positive_label) and (gold_labels[i] == positive_label):
      true_pos += 1
    if (predicted_labels[i] != positive_label) and (gold_labels[i] == positive_label):
      false_neg += 1
  rec = true_pos / (true_pos + false_neg)

  return rec

def f1(gold_labels, predicted_labels):
  """
  Calculates the f1 for a set of predicted labels give the gold (ground truth) labels.
  Parameters:
      gold_labels (list): a list of labels assigned by hand ("truth")
      predicted_labels (list): a corresponding list of labels predicted by the system
  Returns: double f1 (a number from 0 to 1)
  """
  p = precision(gold_labels, predicted_labels)
  r = recall(gold_labels, predicted_labels)

  if p == 0 and r == 0:
    score = 0
  else:
    score = (2*p*r)/(p+r)

  return score


"""
Implement any other non-required functions here
"""


def create_bag_of_words(directory: dict ,sent: list) -> dict:
  """Creates a dictionary with frequency as values and strings as keys.
      Args:
      sent (list): a list of words

      Returns:
      dict: dictionary which can be used to look up frequency of a token or an n_gram
  """
  for element in sent:
    if element in directory:
      directory[element][0] += 1
    else:
      directory[element] = [1]
  return directory

def create_bag_of_words_vocab(directory: dict ,sent: list, vocab: list) -> dict:
  """Creates a dictionary with frequency as values and strings as keys.
      Args:
      sent (list): a list of words

      Returns:
      dict: dictionary which can be used to look up frequency of a token or an n_gram
  """
  for element in sent:
    if element in vocab:
      if element in directory:
        directory[element] += 1
      else:
        directory[element] = 1
  return directory

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def load_word_list(filename):

    """
    Loads a lexicon from a plain text file in the format of one word per line.
    Parameters:
    filename (str): path to file

    Returns:
    list: list of words
    """
    with open(filename, 'r') as f:
      # skip the header content
      for line in f:
        if line.strip() == "":
          break
      # read the rest of the lines into a list
      return [line.strip() for line in f]
    # otherwise return an empty list
    return []


"""
implement your TextClassify class here
"""
class TextClassify:


  def __init__(self):
    # do whatever you need to do to set up your class here
    pass

  def train(self, examples):
    """
    Trains the classifier based on the given examples
    Parameters:
      examples - a list of tuples of strings formatted [(id, example_text, label), (id, example_text, label)....]
    Return: None
    """
    # create vocab and bag of words features
    true_pos_labels = 0
    true_neg_labels = 0
    total_obs = 0
    vocab = list()
    bow = dict()
    #sent = list()
    tokens = list()
    for i in examples:
      #sent.append(i[1].split())
      words = i[1].split()
      tokens.extend(words)
      #print(words)
      create_bag_of_words(bow, words)


      #calculate priors for classes
      if i[2] == '1':
        true_pos_labels += 1
      elif i[2] == '0':
        true_neg_labels += 1


    total_obs = len(examples)
    prior_prob_true = true_pos_labels/total_obs
    prior_prob_false =  true_neg_labels/total_obs
    #print(prior_prob_true)

    vocab = list(set(tokens))
    #print(vocab)
    #print(bow)

    w_labels = list()
    for j in examples:
      #w_labels.extend([(a, j[2]) for a in j[1].lower().split()])
      w_labels.extend([(a,j[2]) for a in j[1].split()])

    total_pos_labels = 0
    total_neg_labels = 0

    for i in w_labels:
      if i[1] == '1':
        total_pos_labels += 1
      elif i[1] == '0':
        total_neg_labels += 1

    #print(w_labels)
    for i in bow.keys():
      bow[i].append(w_labels.count((i, '1')))#/total_pos_labels)
      bow[i].append(w_labels.count((i, '0')))#/total_neg_labels)

    self.bow = bow
    self.prior_prob_c1 = prior_prob_true
    self.prior_prob_c0 = prior_prob_false

    self.total_neg_labels = total_neg_labels
    self.total_pos_labels = total_pos_labels



  def score(self, data):
    """    
    Score a given piece of text
    you’ll compute e ^ (log(p(c)) + sum(log(p(w_i | c))) here
    
    Parameters:
      data - str like "I loved the hotel"
    Return: dict of class: score mappings
    return a dictionary of the values of P(data | c)  for each class, 
    as in section 4.3 of the textbook e.g. {"0": 0.000061, "1": 0.000032}
    Parameters:
      data - str like "I loved the hotel"
    Return: dict of class: score mappings
    """
    list_of_words = data.split()
    class_scores = dict()
    class_scores['0'] = 1.0
    class_scores['1'] = 1.0
    for word in list_of_words:
      if word in self.bow:

        class_scores['0'] = class_scores['0'] * ((self.bow[word][2] + 1) / (self.total_neg_labels + len(self.bow)))
        class_scores['1'] = class_scores['1'] * ((self.bow[word][1] + 1) / (self.total_pos_labels + len(self.bow)))

    class_scores['0'] = (class_scores['0'] * self.prior_prob_c0)
    class_scores['1'] = (class_scores['1'] * self.prior_prob_c1)

    return class_scores

  def classify(self, data):
    """
    Label a given piece of text
    Parameters:
      data - str like "I loved the hotel"
    Return: string class label
    """
    class_scores = self.score(data)
    if class_scores['0'] != class_scores['1']:
      return max(class_scores, key= lambda x: class_scores[x])
    else:
      return '0'

  def featurize(self, data):
    """
    we use this format to make implementation of your TextClassifyImproved model more straightforward and to be 
    consistent with what you see in nltk
    Parameters:
      data - str like "I loved the hotel"
    Return: a list of tuples linking features to values
    for BoW, a list of tuples linking every word to True [("I", True), ("loved", True), ("it", True)]
    """
    list_of_words = data.lower().split()
    list_with_true = list()
    for word in list_of_words:
      list_with_true.append((word,True))
    return list_with_true

  def __str__(self):
    return "Naive Bayes - bag-of-words baseline"


class TextClassifyImproved:

  def __init__(self):
    self.learning_rate = 1
    self.epochs = 20
    self.threshold = 0.5

  def train(self, examples):
    """
    Trains the classifier based on the given examples
    Parameters:
      examples - a list of tuples of strings formatted [(id, example_text, label), (id, example_text, label)....]
    Return: None
    """
    labels = list()
    tokens = list()
    vocab = list()
    # load in the positive and negative word lists here
    self.neg_lex = load_word_list("negative_words.txt")
    self.pos_lex = load_word_list("positive_words.txt")

    for i in examples:
      words = i[1].split()
      label = int(i[2])
      labels.append(label)
      tokens.extend(words)
    vocab = list(set(tokens))
    self.vocab = vocab
    df = pd.DataFrame(columns=vocab)
    df_index = 0
    for i in examples:
      words = i[1].split()
      bow = dict()
      create_bag_of_words(bow,words)
      #df = df.append(bow, ignore_index=True)
      df = pd.concat([df,pd.DataFrame(bow,index=[df_index])])
      df_index = df_index + 1
      df = df.fillna(0)

    #adding more features
    df_index = 0
    for i in examples:

      pos_lex_count = 0
      neg_lex_count = 0
      words = i[1].split()


      for word in words:
        if word in self.neg_lex:
          neg_lex_count += 1
        elif word in self.pos_lex:
          pos_lex_count += 1

      # print(i[1])
      # print(pos_lex_count)
      # print(neg_lex_count)

      df.loc[[df_index],['count_pos_lex']] = pos_lex_count
      df.loc[[df_index], ['count_neg_lex']] = neg_lex_count
      df.loc[[df_index], ['log_length']] = log(len(i[1]))


      df_index = df_index + 1


    features = df.to_numpy()

    weights = np.random.rand(1,features.shape[1])


    # Calc error
    true_labels = np.asarray(labels)
    true_labels = true_labels.reshape(len(labels),1)

    while self.epochs != 0:
      preds = sigmoid(np.dot(features, weights.T))

      error = preds - true_labels
      #print(error)

      # calc gradients

      gradient = np.dot(features.T,error)

      gradient = gradient.reshape(1,features.shape[1])

      #update weights
      weights -= self.learning_rate * gradient

      self.epochs = self.epochs - 1

      #print(preds[0][0])

    self.features = features
    self.weights = weights


  def score(self, data):
    """
    Score a given piece of text
    Parameters:
      data - str like "I loved the hotel"
    Return: dict of class: score mappings
    """
    class_scores = dict()
    df = pd.DataFrame(columns= self.vocab)
    bow = dict()
    create_bag_of_words_vocab(bow, data.split(),self.vocab)
    df = pd.concat([df,pd.DataFrame(bow,index=[0])])
    df = df.fillna(0)

    pos_lex_count = 0
    neg_lex_count = 0
    for word in data.split():
      if word in self.neg_lex:
        neg_lex_count += 1
      elif word in self.pos_lex:
        pos_lex_count += 1

    df.loc[[0], ['count_pos_lex']] = pos_lex_count
    df.loc[[0], ['count_neg_lex']] = neg_lex_count
    df.loc[[0], ['log_length']] = log(len(data))

    features = df.to_numpy()

    preds = sigmoid(np.dot(features, (self.weights).T))

    class_scores['1'] = preds[0][0]
    class_scores['0'] = 1 - preds[0][0]

    return class_scores





  def classify(self, data):
    """
    Label a given piece of text
    Parameters:
      data - str like "I loved the hotel"
    Return: string class label
    """
    df = pd.DataFrame(columns=self.vocab)
    bow = dict()
    create_bag_of_words_vocab(bow, data.split(),self.vocab)
    df = pd.concat([df,pd.DataFrame(bow,index=[0])])
    df = df.fillna(0)

    pos_lex_count = 0
    neg_lex_count = 0
    for word in data.split():
      if word in self.neg_lex:
        neg_lex_count += 1
      elif word in self.pos_lex:
        pos_lex_count += 1

    df.loc[[0], ['count_pos_lex']] = pos_lex_count
    df.loc[[0], ['count_neg_lex']] = neg_lex_count
    df.loc[[0], ['log_length']] = log(len(data))

    features = df.to_numpy()

    preds = sigmoid(np.dot(features, (self.weights).T))

    if preds > self.threshold:
      return '1'
    elif preds <= self.threshold:
      return '0'



  def featurize(self, data):
    """
    we use this format to make implementation of this class more straightforward and to be 
    consistent with what you see in nltk
    Parameters:
      data - str like "I loved the hotel"
    Return: a list of tuples linking features to values
    for BoW, a list of tuples linking every word to True [("I", True), ("loved", True), ("it", True)]
    """
    list_of_words = data.lower().split()
    list_with_true = list()
    for word in list_of_words:
      list_with_true.append((word, True))
    return list_with_true

  def __str__(self):
    return "Logistic Regression Classifier"



def main():

  training = sys.argv[1]
  testing = sys.argv[2]

  train_data = generate_tuples_from_file(training)
  test_data = generate_tuples_from_file(testing)

  pred = list()
  gold = list()

  #classifier = TextClassify()
  classifier = TextClassifyImproved()
  classifier.train(train_data)

  for i in test_data:
    gold.append(i[2])
    #print(i[1])
    #print(classifier.score(i[1]))
    print(i[1])
    print(classifier.score(i[1]))
    print(classifier.classify(i[1]))
    pred.append(classifier.classify(i[1]))

  print(f1(gold,pred))
  print(precision(gold,pred))
  print(recall(gold,pred))


  # do the things that you need to with your base class


  # report precision, recall, f1
  

  #improved = TextClassifyImproved()
  #print(improved)
  # do the things that you need to with your improved class


  # report final precision, recall, f1 (for your best model)




if __name__ == "__main__":
  if len(sys.argv) != 3:
    print("Usage:", "python textclassify_model.py training-file.txt testing-file.txt")
    sys.exit(1)

  main()
 








