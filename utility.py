# __author__ = 'WeiFu'
from __future__ import print_function, division
import pdb
import random
from os import listdir
from os.path import isfile, join
import weka.core.jvm as jvm
import weka.core.converters
from weka.core.converters import Loader, Saver
from weka.classifiers import Classifier, Evaluation
from weka.experiments import SimpleCrossValidationExperiment
from weka.filters import Filter
from weka.attribute_selection import ASSearch, ASEvaluation, AttributeSelection


class o:
  ID = 0

  def __init__(i, **d):
    o.ID = i.id = o.ID + 1
    i.update(**d)

  def update(i, **d): i.__dict__.update(d); return i

  def __getitem__(i, k): return i.__dict__[k]

  def __hash__(i): return i.id

  def __repr__(i):
    keys = [k for k in sorted(i.__dict__.keys()) if k[0] is not "_"]
    show = [":%s %s" % (k, i.__dict__[k]) for k in keys]
    return '{' + ' '.join(show) + '}'


def read(src="./dataset"):
  """
  read data from arff files, return all data in a dictionary

  {'AEEEM':[{name ='./datasetcsv/SOFTLAB/ar6.csv'
             attributes=['ck_oo_numberOfPrivateMethods', 'LDHH_lcom', 'LDHH_fanIn'...]
             instances=[[.....],[.....]]},]
   'MORPH':....
   'NASA':....
   'Relink':....
   'SOFTLAB':....]
  }
  """
  data = {}
  folders = [i for i in listdir(src) if not isfile(i) and i != ".DS_Store"]
  for f in folders:
    path = join(src, f)
    for val in [join(path, i) for i in listdir(path) if i != ".DS_Store"]:
      arff = loadWekaData(val)
      attributes = [str(i).split(" ")[1] for i in arff.attributes()][:-1]  # exclude the label
      columns = [arff.values(i) for i in range(arff.class_index)]  # exclude the class label
      data[f] = data.get(f, []) + [o(name=val, attr=attributes, data=columns)]
  return data


def readsrc(src="./dataset"):
  """
  read all data files in src folder into dictionary,
  where subfolder src are keys, corresponding file srcs are values
  :param src: the root folder src
  :type src: str
  :return: src of all datasets
  :rtype: dictionary
  """
  data = {}
  subfolder = [join(src, i) for i in listdir(src) if not isfile(join(src, i))]
  for one in subfolder:
    data[one] = [join(one, i) for i in listdir(one) if isfile(join(one, i)) and i != ".DS_Store"]
  return data


def loadWekaData(src):
  if not jvm.started: jvm.start()
  loader = Loader(classname="weka.core.converters.ArffLoader")
  data = loader.load_file(src)
  data.class_is_last()
  return data


def wekaCALL(source_src, target_src, source_attr=[], test_attr=[], isHDP=False):
  """
  weka wrapper to train and test based on the datasets
  :param source_src: src of traininng data
  :type source_src: str
  :param target_src: src of testing data
  :type target_src: str
  :param source_attr: features selected for building a learner
  :type source_attr:list
  :param test_attr: features selected in target data to predict labels
  :type test_attr: list
  :param isHDP: flag
  :type isHDP:bool
  :return: AUC
  :rtype: float
  """

  def getIndex(data, used_attr):
    # pdb.set_trace()
    del_attr = []
    for k, attr in enumerate(data.attributes()):
      temp = str(attr).split(" ")
      if temp[1] not in used_attr:
        del_attr += [k]
    return del_attr

  def delAttr(data, index):
    order = sorted(index, reverse=True)
    for i in order[1:]:  # delete from big index, except for the class attribute
      data.delete_attribute(i)
    return data

  source_data = loadWekaData(source_src)
  target_data = loadWekaData(target_src)
  cls = Classifier(classname="weka.classifiers.functions.Logistic")
  if isHDP:
    # pdb.set_trace()
    source_del_attr = getIndex(source_data, source_attr)
    target_del_attr = getIndex(target_data, test_attr)
    source_data = delAttr(source_data, source_del_attr)
    target_data = delAttr(target_data, target_del_attr)
  cls.build_classifier(source_data)
  eval = Evaluation(source_data)
  eval.test_model(cls, target_data)
  # target_data.num_attributes
  # print(eval.percent_correct)
  # print(eval.summary())
  # print(eval.class_details())
  # print(eval.area_under_roc(1))
  return eval.area_under_roc(1)


def filter(data, toSave=False, file_name="test", filter_name="weka.filters.unsupervised.attribute.Remove",
           option=["-R", "first-3,last"]):
  # remove = Filter(classname="weka.filters.unsupervised.attribute.Remove", options = option)
  # option = ["-N","2","-F","2","-S","1"]
  remove = Filter(classname=filter_name, options=option)
  remove.inputformat(data)
  filtered = remove.filter(data)
  if toSave:
    saver = Saver(classname="weka.core.converters.ArffSaver")
    saver.save_file(filtered, "./exp/" + file_name + ".arff")
  # print(filtered)
  return filtered


def featureSelection(data, num_of_attributes):
  """
  feature selection
  :param data: data to do feature selection
  :type data : Instance
  :param num_of_attributes : # of attributes to be selected
  :type num_of_attributes : int
  :return: data with selected feature
  :rtype: Instance
  """
  search = ASSearch(classname="weka.attributeSelection.Ranker", options=["-N", str(num_of_attributes)])
  evaluator = ASEvaluation(classname="weka.attributeSelection.ReliefFAttributeEval",
                           options=["-M", "-1", "-D", "1", "-K", "10"])
  attsel = AttributeSelection()
  attsel.search(search)
  attsel.evaluator(evaluator)
  attsel.select_attributes(data)
  # print("# attributes: " + str(attsel.number_attributes_selected))
  # print("attributes: " + str(attsel.selected_attributes))
  # print("result string:\n" + attsel.results_string)
  for i in reversed(range(data.class_index)):  # delete feature
    if i not in attsel.selected_attributes:
      data.delete_attribute(i)
  # pdb.set_trace()
  return data


if __name__ == "__main__":
  read()
  # if not jvm.started: jvm.start()
  # loader = Loader(classname="weka.core.converters.ArffLoader")
  # data = loader.load_file("./dataset/AEEEM/EQ.arff")
  # data.class_is_last()
  # featureSelection(data, 9)
  # filter()
  # filter()