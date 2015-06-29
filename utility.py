# __author__ = 'WeiFu'
from __future__ import print_function, division
import jnius_config
jnius_config.add_options('-Xrs', '-Xmx4096')
jnius_config.set_classpath('.', '/Users/WeiFu/Github/HDP_Jython/jar/weka.jar','/Users/WeiFu/Github/HDP_Jython/jar/commons-math3-3.5/commons-math3-3.5.jar')
import pdb
import random
from os import listdir
from os.path import isfile, join
from jnius import autoclass

#
#
#
#
#
#
# import weka.core.Instances
# import java.io.BufferedReader
# import java.io.FileReader
# import weka.attributeSelection.Ranker as Ranker
# import weka.attributeSelection.ReliefFAttributeEval as ReliefFAttributeEval
# import weka.attributeSelection.AttributeSelection as attributeSelection
# import weka.classifiers.functions.Logistic as Logistic
# import weka.classifiers.Evaluation as Evaluation
# import weka.core.converters.ArffSaver as Saver
# import weka.filters.unsupervised.attribute.Remove as Remove
# import weka.filters.unsupervised.instance.Randomize as Randomize
# import weka.filters.unsupervised.instance.RemoveFolds as RemoveFolds

# import weka.core.jvm as jvm
# import weka.core.converters
# from weka.core.converters import Loader, Saver
# from weka.classifiers import Classifier, Evaluation
# from weka.experiments import SimpleCrossValidationExperiment
# from weka.filters import Filter
# from weka.attribute_selection import ASSearch, ASEvaluation, AttributeSelection


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


def enumerateToList(enum):
  result =[]
  while enum.hasMoreElements():
    result.append(enum.nextElement().toString())
  return result

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
      attributes = [str(i).split(" ")[1] for i in enumerateToList(arff.enumerateAttributes())]  # exclude the label
      columns = [arff.attributeToDoubleArray(i) for i in range(int(arff.classIndex()))]  # exclude the class label
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
  source = autoclass('weka.core.converters.ConverterUtils$DataSource')(src)
  data = source.getDataSet()
  data.setClassIndex(data.numAttributes()-1)
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
    for k, attr in enumerate(enumerateToList(data.enumerateAttributes())):
      temp = str(attr).split(" ")
      if temp[1] not in used_attr:
        del_attr += [k]
    return del_attr

  def delAttr(data, index):
    order = sorted(index, reverse=True)
    for i in order[1:]:  # delete from big index, except for the class attribute
      data.deleteAttributeAt(i)
    return data

  source_data = loadWekaData(source_src)
  target_data = loadWekaData(target_src)
  # cls = Classifier(classname="weka.classifiers.functions.Logistic")
  cls = autoclass('weka.classifiers.functions.Logistic')
  if isHDP:
    # pdb.set_trace()
    source_del_attr = getIndex(source_data, source_attr)
    target_del_attr = getIndex(target_data, test_attr)
    source_data = delAttr(source_data, source_del_attr)
    target_data = delAttr(target_data, target_del_attr)
  cls.buildClassifier(source_data)
  eval = autoclass('weka.classifiers.Evaluation')(source_data)
  eval.evaluateModel(cls, target_data)
  # target_data.num_attributes
  # print(eval.percent_correct)
  # print(eval.summary())
  # print(eval.class_details())
  # print(eval.area_under_roc(1))
  return eval.areaUnderROC(1)


def filter(data, toSave=False, file_name="test", filter_name="weka.filters.unsupervised.attribute.Remove",
           option=["-R", "first-3,last"]):
  # remove = Filter(classname="weka.filters.unsupervised.attribute.Remove", options = option)
  # option = ["-N","2","-F","2","-S","1"]
  remove = None
  if toSave: # removeFolds
    remove = autoclass('weka.filters.unsupervised.instance.RemoveFolds')()
  else:
    remove = autoclass('weka.filters.unsupervised.instance.Randomize')()
  remove.setOptions(option)
  remove.setInputFormat(data)
  remove.input(data)
  filtered = remove.useFilter(data)
  if toSave:
    saver = autoclass('weka.core.converters.ArffSaver')()
    saver.setInstances(filtered)
    saver.setFile("./exp/" + file_name + ".arff")
    saver.writeBatch()
    # saver.save_file(filtered, "./exp/" + file_name + ".arff")
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
  search = autoclass('weka.attributeSelection.Ranker')()
  evaluator = autoclass('weka.attributeSelection.ReliefFAttributeEval')()
  attsel = autoclass('weka.attributeSelection.AttributeSelection')()
  search.setOptions(['-N',str(num_of_attributes)])
  attsel.setSearch(search)
  attsel.setEvaluator(evaluator)
  attsel.SelectAttributes(data)
  features = attsel.selectedAttributes()[:num_of_attributes]
  index = [i-1 for i in features] # for some reason, weka return index form 1-based not zero-based
  return index

if __name__ == "__main__":
  read()
  # if not jvm.started: jvm.start()
  # loader = Loader(classname="weka.core.converters.ArffLoader")
  # data = loader.load_file("./dataset/AEEEM/EQ.arff")
  # data.class_is_last()
  # featureSelection(data, 9)
  # filter()
  # filter()