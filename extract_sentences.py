import numpy as np
import chainer

# Overhead
# Extract data
train, val, test = chainer.datasets.get_ptb_words()
# Dictionary between id and word
ptb_dict = chainer.datasets.get_ptb_words_vocabulary()
# Number of words
words_num = len(ptb_dict)

# Reverse dictionary
ptb_word_id_dict = ptb_dict
ptb_id_word_dict = dict((v,k) for k,v in ptb_word_id_dict.items())

# Extract sentences sampled using cutoff_percent and then concatenated.
def extract_sentences_rare_words(data,cutoff_percent):
    # Number of words in data
    data_len = len(data)

    # id for <eos>
    eos_idx = ptb_dict['<eos>']

    # Indices in data corresponding to <eos>
    eos_list = np.where(data==eos_idx)[0]

    # Extract sentences
    sentence_list = []

    eos_idx_prev = 0
    for eos_idx_curr in eos_list:
        sentence = data[eos_idx_prev:eos_idx_curr]
        sentence_list.append(sentence)
        eos_idx_prev = eos_idx_curr+1
    sentence_num = len(sentence_list) # Number of sentences
        
    # Frequency list of words
    freq_list = np.zeros(words_num)
    for i in range(words_num):
        freq_list[i] = np.sum(data==i)
        
    prob_words = freq_list/data_len # Probability list

    # Sort
    idx_list_sorted = np.argsort(freq_list) # Low to high
    freq_list_sorted = freq_list[idx_list_sorted]

    # Remove top 50%
    idx_list_rare = idx_list_sorted[0:int(words_num*cutoff_percent)]    # Words corresponding to bottom 50% frequency
    freq_list_rare = freq_list_sorted[0:int(words_num*cutoff_percent)]

    # Sample sentences based on rare words proportion
    p_list = []                 # Rare word fraction (inverse weights)
    sentence_kept_list = []     # Sentences kept for labelling

    for i in range(sentence_num):   # Loop through every sentence
        sentence = sentence_list[i]
        num_total = len(sentence)   # Total number of words
        num_rare = 0                # Number of rare words
        for word in sentence:       # Loop through every word
            if word in idx_list_rare:   # If word is rare
                num_rare += 1
                
        p_i = (num_rare+1)/num_total    # Prob. of sampling
        
        # Cutoff
        if p_i > 1:
            p_i = 1
            
        r = np.random.random()
        if r < p_i:
            p_list.append(p_i)  
            sentence_kept_list.append(sentence)
            
    data_less = np.concatenate(sentence_kept_list)
    
    return sentence_kept_list, p_list
