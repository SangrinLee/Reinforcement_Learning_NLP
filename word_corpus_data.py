import os
import torch
import numpy as np
import re
import threading
import pickle

class MyThread(threading.Thread):
    def __init__(self, id, data, corpus):
        threading.Thread.__init__(self)
        self.id = id
        self.data = data
        self.corpus = corpus
    def run(self):
        data_array = np.array([])
        for i in range(len(self.data) // 8 * self.id, min(len(self.data) // 8 * (self.id + 1), len(self.data))):
            pattern = re.compile(r'●')
            text_list = (pattern.findall(self.data[i]))

            if len(text_list) >= 1:
                data_array = np.append(data_array, self.corpus.dictionary.word2idx["<bd>"]) # added
            else:
                if self.data[i] in self.corpus.unique_word:
                    data_array = np.append(data_array, self.corpus.dictionary.word2idx[self.data[i]]) # added
                else:
                    data_array = np.append(data_array, self.corpus.dictionary.word2idx["<unk>"]) # added

            if i % (len(self.data) // 50) == 0:
                print("Thread {} at {:2.1f}%".format(self.id, 100 * (i - len(self.data) // 8 * self.id) /
                      (min(len(self.data) // 8 * (self.id + 1), len(self.data)) - len(self.data) // 8 * self.id)))

        with open('word_data/rare_data_array_{}'.format(self.id), 'wb') as handle:
            pickle.dump(data_array, handle, protocol=pickle.HIGHEST_PROTOCOL)


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()

        self.test_dictionary = []
        self.unique = {}
        self.unique_word = []

        self.train_data = self.generate_train_set(os.path.join(path, 'old_books.txt'))
        self.test_data = self.generate_test_set(os.path.join(path, 'old_books.txt'))
    
    # generate train_data
    def generate_train_set(self, path):
        assert os.path.exists(path)

        # get the dictionary of unique words
        with open(path, 'r', encoding='UTF-8') as f:
            for line in f:
                words = re.split("[,.:; ]", line)
                words = list(filter(None, words))

                for word in words:
                    pattern = re.compile(r'●')
                    text_list = (pattern.findall(word))

                    if word in self.unique:
                        self.unique[word] += 1
                    else:
                        self.unique[word] = 1

        # adjust the size of dictionary based on highest frequency(e.g. 50000 fixed unique ids)
        for k, v in self.unique.items():
            if v >= 2:
                self.unique_word.append(k)
        sorted_d = sorted(self.unique, key=self.unique.get, reverse=True)
        sorted_d = sorted_d[:50000]
        self.unique_word = sorted_d

        # generate the list of old_books applying the fixed number of unique ids
        with open(path, 'r', encoding='UTF-8') as f:
            for line in f:
                words = re.split("[,.:; ]", line)
                words = list(filter(None, words))

                for word in words:
                    pattern = re.compile(r'●')
                    text_list = (pattern.findall(word))
                    if len(text_list) >= 1:
                        self.dictionary.add_word("<bd>")
                    else:
                        if word in self.unique_word:
                            self.dictionary.add_word(word)
                        else:
                            self.dictionary.add_word("<unk>")

        # Tokenize file content
        with open(path, 'r', encoding='UTF-8') as f:
            ids = np.array([])
            for line in f:
                words = re.split("[,.:; ]", line)
                words = list(filter(None, words))
                                
                thread0 = MyThread(0, words, self)
                thread1 = MyThread(1, words, self)
                thread2 = MyThread(2, words, self)
                thread3 = MyThread(3, words, self)
                thread4 = MyThread(4, words, self)
                thread5 = MyThread(5, words, self)
                thread6 = MyThread(6, words, self)
                thread7 = MyThread(7, words, self)
                thread8 = MyThread(8, words, self)
                thread0.start()
                thread1.start()
                thread2.start()
                thread3.start()
                thread4.start()
                thread5.start()
                thread6.start()
                thread7.start()
                thread8.start()
                thread0.join()
                thread1.join()
                thread2.join()
                thread3.join()
                thread4.join()
                thread5.join()
                thread6.join()
                thread7.join()
                thread8.join()

        with open('word_data/rare_data_array_0', 'rb') as handle:
            data_array_0 = pickle.load(handle)
        with open('word_data/rare_data_array_1', 'rb') as handle:
            data_array_1 = pickle.load(handle)
        with open('word_data/rare_data_array_2', 'rb') as handle:
            data_array_2 = pickle.load(handle)
        with open('word_data/rare_data_array_3', 'rb') as handle:
            data_array_3 = pickle.load(handle)
        with open('word_data/rare_data_array_4', 'rb') as handle:
            data_array_4 = pickle.load(handle)
        with open('word_data/rare_data_array_5', 'rb') as handle:
            data_array_5 = pickle.load(handle)
        with open('word_data/rare_data_array_6', 'rb') as handle:
            data_array_6 = pickle.load(handle)
        with open('word_data/rare_data_array_7', 'rb') as handle:
            data_array_7 = pickle.load(handle)
        with open('word_data/rare_data_array_8', 'rb') as handle:
            data_array_8 = pickle.load(handle)

        data_array = np.append(data_array_0, [data_array_1, data_array_2, data_array_3, data_array_4, data_array_5, data_array_6, data_array_7])
        data_array = np.append(data_array, data_array_8)

        return data_array

    # generate test_data
    def generate_test_set(self, path):
        assert os.path.exists(path)
        
        index = 0
        num = 0
        ids = []
        test_data_list = [] # added for real testing
        with open(path, 'r', encoding='UTF-8') as f:
            for line in f:
                words = re.split("[,.:; ]", line)
                words = list(filter(None, words))

                for word in words:
                    pattern = re.compile(r'●')
                    text_list = (pattern.findall(word))
                    if len(text_list) >= 1:
                        test_data_list.append(word) # added for real testing
                        ids.append([index, str(num)])
                        self.test_dictionary.append([word, num])
                        num += 1
                    index += 1

        # added for real testing
        test_data2=open('word_data/actual_test_data', 'wb')
        pickle.dump(test_data_list, test_data2)
        test_data2.close()

        return ids
