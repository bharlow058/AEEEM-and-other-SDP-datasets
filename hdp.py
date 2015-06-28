# __author__ = 'WeiFu'
from __future__ import print_function, division
import random, math
from utility import *
import org.apache.commons.math3.stat.inference.KolmogorovSmirnovTest as KS
from bipartite import maxWeightMatching
# from scipy import stats
# import numpy as np
# import networkx as nx

def transform(d, selected=[]):
  """
  the data will be stored by column, not by instance
  :param d : data
  :type d: o
  :return col: the data grouped by each column
  :type col: dict
  """
  col = {}
  for val, attr in zip(d["data"], d["attr"]):
    if len(selected) == 0:
      col[attr] = val
    elif attr in selected:
      col[attr] = val
  return col
  # for row in d["data"]:
  # for attr, cell in zip(d["attr"][:-1], row[:-1]):  # exclude last columm, $bug
  #     if len(selected) != 0 and attr not in selected:  # get rid of name, version columns.
  #       continue  # if this is for feature selected data, just choose those features.
  #     col[attr] = col.get(attr, []) + [cell]
  # return col


def maximumWeighted(match, target_lst, source_lst):
  """
  using max_weighted_bipartite to select a group of matched metrics
  :param match : matched metrics with p values, key is the tuple of matched metrics
  :type match : dict
  :param target_lst : matched target metrics
  :type target_lst: list
  :param source_lst : matched source metcis
  :type source_lst: list
  :return : matched metrics as well as corresponding values
  :rtype: class o
  """
  edges,track =[],[]
  value = 0
  attr_source, attr_target = [], []
  for key , val in match.iteritems():
    edges.append((2*int(source_lst.index(key[0])),2*int(target_lst.index(key[1]))+1,val))
    track+=[key[0],key[1]]
  result = maxWeightMatching(edges)
  print(result)
  for node in result:
    if node != -1:
      if track[node] in source_lst:
        attr_source.append(track[node])
      elif track[node] in target_lst:
        attr_target.append(track[node])
  for source, target in zip(attr_source,attr_target):
    value +=match[(source,target)]
  # pdb.set_trace()
  return o(score=value, attr_source=attr_source, attr_target=attr_target)
  #
  #
  #
  #
  #
  #
  #
  #
  #
  #
  #
  # pdb.set_trace()
  # value = 0
  # attr_source, attr_target = [], []
  # G = nx.Graph()
  # for key, val in match.iteritems():
  #   G.add_edge(key[0] + "source", key[1] + "target", weight=val)  # add suffix to make it unique
  # result = nx.max_weight_matching(G)
  # for key, val in result.iteritems():  # in Results, (A:B) and (B:A) both exist
  #   if key[:-6] in source_lst and val[:-6] in target_lst:
  #     attr_target.append(val[:-6])
  #     attr_source.append(key[:-6])
  #     value += match[(key[:-6], val[:-6])]
  # # pdb.set_trace()
  # return o(score=value, attr_source=attr_source, attr_target=attr_target)


def KStest(d_source, d_target, features, cutoff=0.05):
  """
  Kolmogorov-Smirnov Test
  :param d_source : source data
  :type d_source : o
  :param d_target: target data
  :type d_target: o
  :param features: features selected for the source data set
  :type features : list
  :return : results of maximumWeighted
  :rtype: o
  """
  match = {}
  source = transform(d_source, features)
  target = transform(d_target)
  target_lst, source_lst = [], []
  test = KS()
  for tar_feature, val1 in target.iteritems():
    for sou_feature, val2 in source.iteritems():
      result = test.kolmogorovSmirnovTest(val1,val2)
      print(result)
      # result = mytest.kolmogorovSmirnovTest(val1,val2)
      if result > cutoff:
        # match[sou] = match.get(sou,[])+[(tar,result[1])]
        match[(sou_feature, tar_feature)] = result
        if tar_feature not in target_lst:
          target_lst.append(tar_feature)
        if sou_feature not in source_lst:
          source_lst.append(sou_feature)
  if len(match) < 1:
    return o(score=0)
  return maximumWeighted(match, target_lst, source_lst)


def attributeSelection(data):
  feature_dict = {}
  for key, lst in data.iteritems():
    for source in lst:
      source_name = source["name"]
      A = loadWekaData(source_name)
      A_selected_index = featureSelection(A, int(int(A.classIndex()) * 0.15))
      features_list = [str(attr).split(" ")[1] for i,attr in enumerate(A.enumerateAttributes()) if i in A_selected_index]
      feature_dict[source_name] = features_list
  return feature_dict


def KSanalyzer(cutoff=0.05):
  """
  for each target data set, find a best source data set in terms of p-values
  :param data : read data from arff
  :type data : o
  :return pairs of matched data
  :rtype: list
  """
  data = read()
  best_pairs = []
  selected_features = attributeSelection(data)
  for target_group, targetlst in data.iteritems():
    for target in targetlst:
      for source_group, sourcelst in data.iteritems():
        if target_group != source_group:
          for source in sourcelst:
            source_name = source["name"]
            target_name = target["name"]
            X = KStest(source, target, selected_features[source_name]).update(source_src=source_name,
              group=source_group, target_src=target_name)
            if X["score"] > cutoff:
              best_pairs.append(X)
  pdb.set_trace()
  return best_pairs


def call(source_src, target_src, source_attr, target_attr):
  """
  call weka to perform learning and testing
  :param train: src of training data
  :type train: str
  :param test: src of testing data
  :type test: str
  :param source_attr: matched feature for training data set
  :type source_attr: list
  :param target_attr: matched feature for testing data set
  :type target_attr: list
  :return ROC area value
  :rtype: list
  """
  r = round(wekaCALL(source_src, target_src, source_attr, target_attr, True), 3)
  if not math.isnan(r):
    return [r]
  else:
    return []


def hdp(target_src, source_target_match):
  """
   source_target_match = KSanalyzer()
  :param target_src : src of test(target) data set
  :type target_src : str
  :param source_target_match : matched source and target data test
  :type source_target_match: list
  :return: value of ROC area
  :rtype: list
  """
  result = []
  target_name = target_src
  for i in source_target_match:
    if i.target_src == target_name:  # for all
      source_attr = i.attr_source
      target_attr = i.attr_target
      source_src = i.source_src
      result.append(o(result=call(source_src, "./exp/train.arff", source_attr, target_attr), source_src=source_src))
      result.append(o(result=call(source_src, "./exp/test.arff", source_attr, target_attr), source_src=source_src))
  return result


def testEQ():
  def tofloat(lst):
    for x in lst:
      try:
        yield float(x)
      except ValueError:
        yield x[:-1]

  target_src = "./dataset/Relink/apache.arff"
  source_src = "./dataset/AEEEM/EQ.arff"

  d = open("./datasetcsv/AEEEM/EQ.csv", "r")
  content = d.readlines()
  attr = content[0].split(",")
  inst = [list(tofloat(row.split(","))) for row in content[1:]]
  d1 = o(name="./datasetcsv/AEEEM/EQ.csv", attr=attr, data=inst)

  d = open("./datasetcsv/Relink/apache.csv", "r")
  content = d.readlines()
  attr = content[0].split(",")
  inst = [list(tofloat(row.split(","))) for row in content[1:]]
  d2 = o(name="./datasetcsv/Relink/apache.csv", attr=attr, data=inst)
  Result = KStest(d1, d2, source_src)
  print(Result)
  pdb.set_trace()
  print("DONE")


if __name__ == "__main__":
  random.seed(1)
  np.random.seed(1)
  # wpdp()
  # KSanalyzer()
  # wekaCALL()
  # filter()
  # cpdp()
  # readarff()
  testEQ()



