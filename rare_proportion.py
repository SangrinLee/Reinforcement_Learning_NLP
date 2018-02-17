import pickle
import numpy as np
import extract_sentences as es

# Frequency list
freq_list = es.extract_frequencies(es.train)
# Sort
idx_list_sorted = np.argsort(freq_list) # Low to high

path = '\dataset\active learning\\'

def iterate(path,cutoff):
    print (path)
    with open(path+'sentence_kept_list_' + str(cutoff), 'rb') as handle:
        sentence_kept_list = pickle.load(handle)
        
    # Rare word list
    idx_list_rare = idx_list_sorted[0:int(es.words_num*cutoff)]    # Words corresponding to bottom cutoff_percent frequency
    
    # Data
    sentences_kept = np.concatenate(sentence_kept_list)
    
    num_words = len(sentences_kept)
    num_rare = 0
    for word in sentences_kept:
        if word in idx_list_rare:
            num_rare += 1
            
    rare_prop = num_rare/num_words
    
    return rare_prop