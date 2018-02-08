# -*- coding: utf-8 -*-
from random import randint
import pickle
import string
import numpy as np
import argparse
import threading
import word_corpus_data
import string
import glob
import re

################################################################
# Options
################################################################

parser = argparse.ArgumentParser(description='Data Generator')
parser.add_argument('--asterisk', type=bool, default=True,
                    help='Generate asterisked PTB corpus?')
parser.add_argument('--convert', type=bool, default=True,
                    help='Convert asterisked PTB corpus to one np array?')
parser.add_argument('--split', type=bool, default=True,
                    help='Split asterisked PTB corpus to train_data and val_data?')
parser.add_argument('--train', type=bool, default=True,
                    help='Convert train_data to np array?')
parser.add_argument('--val', type=bool, default=True,
                    help='Convert val_data to np array?')
args = parser.parse_args()

#################################################################
# Extract words from the xmls
#################################################################
txt = ""
with open("word_data/old_books.txt", "w", encoding='UTF-8', newline='') as write_file:
	with open("old_books.txt", "r", encoding='UTF-8', newline='') as read_file:
		for line in read_file:
			# line.rstrip("\n")
			txt += line.rstrip("\n")
	
	write_file.write(txt)

# with open("word_data/old_books.txt", "r", encoding='UTF-8', newline='') as a:
# 	for aa in a:
# 		print (aa)


		# first = True
		# for line in read_file:
		# 	if re.match(".*lemma=.*", line):
		# 		word = line.split("</w>")[0].split(">")[1]
		# 		if first == True:
		# 			write_file.write(word)
		# 			first = False
		# 		else:
		# 			write_file.write(" " + word)
		# 	elif re.match(".*</pc>", line):
		# 		word = line.split("</pc>")[0].split(">")[1]
		# 		write_file.write(word)

