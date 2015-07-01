# __author__ = 'WeiFu'
from __future__ import division, print_function
import sys
from utility import *
from wpdp import *
from cpdp import *
from hdp import *


def readMatch(src="./result/source_target_match.txt"):
  def getStrip(lst):
    result = []
    for one in lst:
      result.append(one[one.index("'") + 1:one.rindex("'")])
    return result

  result = []
  f = open(src, "r")
  X = f.readlines()[0].split("}, {")
  for each in X:
    attr_source = getStrip(
      each[each.index("attr_source") + len("attr_source") + 2:each.index("attr_target") - 3].split(","))
    attr_target = getStrip(each[each.index("attr_target") + len("attr_target") + 2:each.index("group") - 3].split(","))
    group = each[each.index("group") + len("group") + 1:each.index("id") - 2]
    score = float(each[each.index("score") + len("score") + 2:each.index("source_src") - 2])
    source_src = (each[each.index("source_src") + len("source_src") + 1:each.index("target_src") - 2])
    target_src = (each[each.index("target_src") + len("target_src") + 1:])
    temp = o(score=score, attr_source=attr_source, attr_target=attr_target, source_src=source_src,
             target_src=target_src)
    result.append(temp)
  return result
  # pdb.set_trace()


def process(match, target_src, result):
  total = []
  for i in match:
    one_source_result = None
    if i.target_src == target_src:
      one_source_result = [j.result[0] for j in result if
                           j.source_src == i.source_src and j.result != []]  # put all the results from one source
      # together.
    if not one_source_result:
      continue
    ordered = sorted(one_source_result)
    one_median = ordered[int(len(ordered) * 0.5)]
    print(i.source_src, "===>", target_src, one_median)
    total += [one_median]
  if len(total) == 0:
    print("no results for ", target_src)
    return
  total_median = sorted(total)[int(len(total) * 0.5)]
  print("final ====>", target_src, total_median)
  return total_median


def run():
  datasrc = readsrc()
  # source_target_match = KSanalyzer()
  source_target_match = readMatch()
  for group, srclst in datasrc.iteritems():
    for one in srclst:
      random.seed(1)
      data = loadWekaData(one)
      out_wpdp, out_cpdp, out_hdp = [], [], []  # store results for three methods
      for _ in xrange(500):
        randomized = filter(data, False, "", "weka.filters.unsupervised.instance.Randomize", ["-S", str(_)])
        train = filter(randomized, True, "train", "weka.filters.unsupervised.instance.RemoveFolds",
                       ["-N", "2", "-F", "1", "-S", "1"])
        test = filter(randomized, True, "test", "weka.filters.unsupervised.instance.RemoveFolds",
                      ["-N", "2", "-F", "2", "-S", "1"])
        # out_wpdp += wpdp(tarin, test)
        # cpdp(group,one)
        temp = hdp(one, source_target_match)
        if len(temp) == 0:
          continue
        else:
          out_hdp += temp
      process(source_target_match, one, out_hdp)
      # re_sorted = sorted(out_hdp)
      # print(one, "===>", re_sorted[int(len(re_sorted) * 0.5)])
      # pdb.set_trace()
      # print("next=======>")


if __name__ == "__main__":
  # readMatch()
  run()