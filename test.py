# __author__ = 'WeiFu'

import jnius_config
jnius_config.add_options('-Xrs', '-Xmx4096')
jnius_config.set_classpath('.', '/Users/WeiFu/Github/HDP_Jython/jar/weka.jar','/Users/WeiFu/Github/HDP_Jython/jar/commons-math3-3.5/commons-math3-3.5.jar')
from jnius import autoclass
import pdb,sys
# import weka.core.Instances
# import java.io.BufferedReader
# import java.io.FileReader
# sys.path.append('/Users/WeiFu/Github/HDP_Jython/jar/weka.jar')
Stack = autoclass('java.util.Stack')
stack = Stack()
stack.push('hello')
stack.push('world')
# Instaces =('weka.core.Instances')


print stack.pop() # --> 'world'
print stack.pop() # --> 'hello'


src = "./dataset/AEEEM/EQ.arff"
filereader = autoclass('java.io.FileReader')(src)
Buffer =  autoclass('java.io.BufferedReader')(filereader)
weka = autoclass('weka.core.Instances')
test = autoclass('org.apache.commons.math3.stat.inference.KolmogorovSmirnovTest')

Reader = autoclass('weka.core.converters.ConverterUtils$DataSource')
pdb.set_trace()
data = weka(Buffer)



pdb.set_trace()
bufferreader = java.io.BufferedReader(filereader)
data = weka.core.Instances(bufferreader)
data.setClassIndex(data.numAttributes()-1)

