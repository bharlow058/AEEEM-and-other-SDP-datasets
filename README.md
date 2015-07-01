# HDP_Pyjnius
Since I need weka, apache.commons.math and other java libs, but also need python library like netowrkx to do max_weight_bipartite, then I decide to use cpython + pyjnius to combine them 
# Setup
  * install [pyjnius](https://pyjnius.readthedocs.org/en/latest/)
  * when import java library, refer to the following example:
```
System = autoclass('java.lang.System')
System.out.println('Hello World')
```
  
# Limitation
you need to take care of java heap size. it may cause a crash after 40 hours.
