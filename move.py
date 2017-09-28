#coding=utf-8

import os
from pprint import pprint

# pprint(os.listdir("imas/01"))

file_list = os.listdir("imas/01")
f = open("character_list.txt")
idol_list = f.read().split()

print(idol_list)
