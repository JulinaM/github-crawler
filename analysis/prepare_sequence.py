from readme_cleanup import readme_cleanup
import numpy as np
import pandas as pd
from datetime import datetime
import difflib
from collections import defaultdict
from numba import jit

@jit
def diff_calculator(str1, str2):
   s = difflib.SequenceMatcher(lambda x : x == '')
   s.set_seqs(str1, str2)
   i = 1
   # codes = []
   # delete = []
   # replace = {}
   insert = []
   for (opcode, before_start, before_end, after_start, after_end) in s.get_opcodes():
       if opcode == 'equal':
           continue
       # codes.append(opcode)
       # # print (i, ". %7s '%s :'  ----->  '%s'" % (opcode, test[0][before_start:before_end], test[1][after_start:after_end]))
       # if opcode == 'replace':
       #     replace[str1[before_start:before_end]]  = str2[after_start:after_end]
       # if opcode == 'delete':
       #     delete.append(str1[before_start:before_end])
       if opcode == 'insert':
           if str2[after_start:after_end]:
            insert.append(str2[after_start:after_end])
       i = i + 1
   # return replace, delete, insert
   return insert

@jit
def create_a_sequence(readmeList):
    result = []
    for i in range(0,len(readmeList)-1):
        first = readme_cleanup(readmeList[i])
        second = readme_cleanup(readmeList[i+1])
        insert = diff_calculator(first, second)
        result.append(','.join(insert))
    return result

@jit
def prepareSequenceForBERT(readmeList):
    diffList = create_a_sequence(readmeList)
    s = '[CLS]' + "[SEP]".join([str(i) for i in diffList])
    return s +'[SEP]'

