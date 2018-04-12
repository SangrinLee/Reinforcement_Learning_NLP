# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

# type = "active"
# type = "active_no_weight"
# type = "baseline"
type = "active"

if type == "active":
	import word_train_RL as train
elif type == "active_no_weight":
	import word_train_RL_no_weight as train
elif type == "baseline":
	import word_train_RL_baseline as train

elif type == "data_active":
	import rare_proportion as rare_prop
	path = 'dataset/active learning/'
elif type == "data_active_no_weight":
	import rare_proportion as rare_prop
	path = 'dataset/active learning without weight/'
elif type == "data_active_baseline":
	import rare_proportion as rare_prop
	path = 'dataset/baseline/'

prop_list = []
for i in range(11):
	print ("===== Cut off " + str(i * 0.1) + " =====")
	train.iterate(i * 0.1)
	# prop_list.append(rare_prop.iterate(path, i * 0.1))

# print (type)
# plt.plot(prop_list)
# plt.savefig("rare_proportion_" + type + '.png')