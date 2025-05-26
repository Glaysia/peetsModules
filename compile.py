#!/usr/bin/env python
# Copyright © 2024 ANSYS, Inc. Unauthorized use, distribution, or duplication is prohibited. 

from __future__ import print_function

import time
import datetime
import argparse
import glob
import sys
import os
import traceback
import platform
import tempfile
import py_compile
import zipapp
import shutil

print("Started at:", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
print("Current Working Directory:", os.getcwd())
print("Version:",sys.version)

def clean():
   toIgnore = []
   toDelete = []
   files = [f for f in glob.glob("**/*",recursive=True)]
   for f in files:
      if f.endswith('.pyc') or f.endswith('.pyz'):
         toDelete.append(f)
      elif "__pycache__" in f:
         toDelete.append(f)
      elif os.path.isfile(f):
         toIgnore.append(f)

   if False: print("toIgnore:\n  ",'\n  '.join(toIgnore),sep='')
   print("toDelete:\n  ",'\n  '.join(toDelete),sep='')
   
   for f in toDelete:
      if os.path.isfile(f):
         os.remove(f)
   for f in toDelete:
      if os.path.isdir(f):
         os.rmdir(f)

def compileSingles():
   py_compile.compile("PythonClient.py",cfile="PythonClient.pyc")
   py_compile.compile("pythonStructs.py",cfile="pythonStructs.pyc")
   py_compile.compile("PHXPythonGlobals.py",cfile="PHXPythonGlobals.pyc")


def getMainPyText():
   mainPyText="""#!/usr/bin/env python
def main():
   print('This is just a library')"""
   return mainPyText

def archive():
   with open("libs/main.py", "w") as pyFile:
      pyFile.write(getMainPyText())
  
   zipapp.create_archive('libs',main="main:main",target="gen.pyz")

   print("Archive Done")

   sys.path.insert(0,"gen.pyz")
   print("sys.path:\n  ", "\n  ".join(sys.path),sep='')

   print("Test Imports")
   from com.phoenix_int.aserver.util.scriptwrapper.interprocess.ThriftScriptWrapperService import Client
   from com.phoenix_int.aserver.util.scriptwrapper.interprocess.ttypes import GeneralError
   from thrift.protocol import TJSONProtocol
   from thrift.protocol import TBinaryProtocol
   from thrift.transport import TTransport
   from thrift.transport import TSocket
   from thrift import Thrift
   print("Imports Happy")
   
   import main
   main.main()


clean()

compileSingles()

archive()

print("End Of File")

