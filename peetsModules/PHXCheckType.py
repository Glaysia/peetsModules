# Copyright © 2024 ANSYS, Inc. Unauthorized use, distribution, or duplication is prohibited. 
import math
import sys

java_long_min = int(math.pow(-2, 63))     # Java long lower bound.
java_long_max = int(math.pow(2, 63) - 1)  # Java long upperbound

def checkIsInstance(var, expType):
   """
   Checks if var is an instance of expType.
   If expType checked for is a list, the leftmost variable will be the type assumed the list will contain.
   Ex.) checkIsInstance(var, (int, float, list)) 
      - This will check that var is a list made of only int values.
   @param var: A variable to be checked.
   @type var: object
   @param expType: The class to be checked for.
   @type expType: type or list of types
      Ex.) int OR (int, float)
   @return var # TODO: Perform PHX casting here
   """
   if not isinstance(var, expType):
      # TODO: Fix the weird output.
      raise TypeError("Expected type " + str(expType) + " while actual type was " + str(type(var)) + ".")

   # Check that values of int do not overflow in java.
   if isinstance(var, int):
      castInt = int(var)
      if castInt > java_long_max or castInt < java_long_min:
         raise OverflowError("Long out of bounds: " + str(var))

   # Check that values are uniform in a list.
   if isinstance(var, list):
      expElemType = expType[0]
      var = checkIsInstanceArray(var, expElemType)

   return var


def checkIsInstanceArray(arr, expType):
   """
   Checks if arr's values are all instances of expType
   @param arr: A variable to be checked.
   @type arr: object
   @param expType: The class to be checked for.
   @type expType: type
   @return arr # TODO: Perform PHX casting here
   """
   if not isinstance(arr, list):
      raise TypeError("Expected type list")

   # iterate through each value and check the type
   if isinstance(arr, list):
      for v in arr:
         if not isinstance(v, expType):
            # TODO: Fix the weird output.
            raise TypeError("Array does not contain " + str(expType) + ". Failed on value: " + str(v))

   return arr
