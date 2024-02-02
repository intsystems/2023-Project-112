import numpy as np
import pandas as pd
import os
import pickle


def save_object(obj, filename):
    # Overwrites any existing file.
    with open(os.path.join(os.path.dirname(os.getcwd()), "src", filename), 'wb') as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    with open(os.path.join(os.path.dirname(os.getcwd()), "src", filename), 'rb') as inp:
        obj = pickle.load(inp)
        return obj


def preprocess(v):
    return (v - v.min()) / (v.max() - v.min())


def MSE(A):
    m, n = A.shape
    return (np.linalg.norm(A, "fro") ** 2) / (m * n)

# ================

def extract_last_word_from_filename(filename):
    # Используем регулярное выражение для поиска последнего слова в имени файла
    match = re.search(r'([^_]+)\.tsv$', filename)
    if match:
        return match.group(1)
    else:
        return None
    
def get_occur():
    
    filenames = []
    objectnames = []    

    for root, dirs, files in os.walk(os.path.join(os.path.dirname(os.getcwd()), 'src', 'annotations', 'video')):
        for filename in files:
            filenames.append(os.path.join(root, filename))
            objectname = extract_last_word_from_filename(filename)
            objectnames.append(objectname)
            
    filenames.sort()
    objectnames.sort()
    
    occur = list()
    #objects = dict()
    #pairs = list()

    for filename, objectname in zip(filenames, objectnames):
        df = pd.read_csv(filename, sep='\t')
        df = (df * 25).astype(dtype=int)
        vector = np.zeros(9750, dtype=int)
        for ts in df.values:
            vector[ts[0]:ts[1]+1] = 1
        occur.append(vector)
        #objects[objectname] = {}
        #objects[objectname]['count'] = df.shape[0]
        #objects[objectname]['occurences'] = vector
        #pairs.append((objectname, df.shape[0]))
        
    occur = np.array(occur)
    #pairs.sort(key=lambda x: x[1], reverse=True) # пары (объек, число появлений в кадре), отсортированные по убыванию числа появлений
   
    return occur