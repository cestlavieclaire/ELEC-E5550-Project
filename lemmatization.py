'''
Created on 18 Apr 2018

@author: tamperm1
'''

from texture_detection import TextureDetection
from text_processing import TextProcessing
import numpy as np
import csv
    
def main():
    fname = "/u/32/tamperm1/unix/git/nlp-course-project/test_data/word2vec.csv"
    fname_train = "/u/32/tamperm1/unix/git/nlp-course-project/test_data/word2vec.csv"
    directory = "/u/32/tamperm1/unix/git/nlp-course-project/data/kko_abstract"
    search_pattern = "*.txt"
    documents = None
    
    tp = TextProcessing(directory, search_pattern, documents)
    tp.read_documents()
    tp.get_baseformed_data()
    
    write_data("document_abstract_data_lemmatized.csv",tp.get_documents())
    
    #data = read_data(fname)
    #train_dataset  = read_data(fname_train)
    
    #t = TextureDetection()

    #X, y = t.train(train_dataset)
    #t.pca_analysis(np.array(X),np.array(y))

    #texture = t.category_recognition(data)

def read_data(fname):
    data = dict()
    
    with open(fname, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in spamreader:
            data[row[0]]=row[1:len(row)]
            print(row[0], row[1:len(row)])
    return data

def write_data(fname, docs):
    
    with open(fname, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for doc in docs:
            row = [doc.get_filename()] + doc.get_lemmatized()
            spamwriter.writerow(row)

if __name__ == '__main__':
    main()
