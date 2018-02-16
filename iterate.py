# -*- coding: utf-8 -*-

type = "active"
# type = "active_no_weight"
# type = "baseline"

if type == "active":
	import word_train_RL as train
elif type == "active_no_weight":
	import word_train_RL_no_weight as train
elif type == "baseline":
	import word_train_RL_baseline as train

for i in range(11):
	print ("===== Cut off " + str(i * 0.1) + " =====")
	train.iterate(i * 0.1)
