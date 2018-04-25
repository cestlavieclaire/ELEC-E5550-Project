'''
Created on 25 Apr 2018

@author: tamperm1
'''
import csv
from collections import OrderedDict
import copy

def main():
    filename="/u/32/tamperm1/unix/git/nlp-course-project/data/kko_document_filtered_categories.csv"
    data = get_y_dataset(filename)
    write_results(data)
    
def get_y_dataset(dataset):
        dct_y = dict()
        vocab = OrderedDict()
        with open(dataset, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for row in spamreader:
                mapped, vocab = map_these(list(filter(len, row[1:])),vocab)
                doc = row[0].split('.')[0]
                if doc not in dct_y:
                    #if doc in dct_y and dct_y[doc]!=mapped:
                    print("write",doc,"with", mapped)
                    dct_y[doc]=mapped
                
        return dct_y
    
def map_these(data, vocab):
    mapped=copy.deepcopy(data)
    new_id = 0
    if data:
        for i in range(len(data)):
            category = data[i]
            if category:
                if category not in vocab:
                    if len(vocab) > 0:
                        last = list(vocab.keys())[-1]
                        new_id = vocab[last] + 1
                    vocab[category] =new_id
                #else:
                #    print(vocab)
                    
                mapped[i] = vocab[category]
            else:
                print('false',category)
        
        #print("From", data, "to", mapped)
    else:
        print("Empty data", data)
    return mapped, vocab
            
            
def write_results(data):
    print("Write results to file")
    with open('anonym_categories.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for doc, related in data.items():
            row = [doc]+related
            print(row)
            spamwriter.writerow(row)
if __name__ == '__main__':
    main()