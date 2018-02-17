# -*- coding: utf-8 -*-
import word_train_RL as train

for i in range(10):
	print ("===== Cut off " + str(i * 0.1) + " =====")
	train.iterate(i * 0.1)
