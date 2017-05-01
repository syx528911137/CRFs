#ecoding:utf-8
from itertools import chain
import codecs
#import sklearn
#from sklearn.metrics import classification_report
#from sklearn.preprocessing import LabelBinarizer
import pycrfsuite

#str:context of line
#return [(word,tag),....]
def sent(str):
    tmp = str.strip().split(" ")
    tmpsent = []
    for tup in tmp:
        wordAndTag = tup.split(":")
        tmpsent.append((wordAndTag[0],wordAndTag[1]))
    return tmpsent

def train_sents(filename):
    trainSents = []
    trainFile = codecs.open(filename,"r","utf-8")
    contextOfLine = trainFile.readline()
    while len(contextOfLine) != 0:
        sen = sent(contextOfLine)
        trainSents.append(sen)
        contextOfLine = trainFile.readline()
    return trainSents

#sent:sentence,input of this parameter is train_sents[k],the k is lineNo(from 0),data format:[(word,tag)...]
#i: the ith word in sentence,i form 1 to len(sent) - 1
def wordFeatures(sent,i):
    word = sent[i][0]
    word_before = sent[i - 1][0]
    word_after = sent[i + 1][0]
    features = ['bias','word_before='+word_before,'word='+word,'word_after='+word_after]
    return  features

#sent:sentence,input of this parameter is train_sents[k],the k is lineNo(from 0),data format:[(word,tag)...]
def sentlabels(sent):
    features = []
    for i in range(1,len(sent) - 1):
       features.append(sent[i][1])
    return features

#sent:sentence,input of this parameter is train_sents[k],the k is lineNo(from 0),data format:[(word,tag)...]
def sentFeatures(sent):
    features = []
    # i from 1 to len(sent) - 1
    for i in range(1,len(sent) - 1):
        features.append(wordFeatures(sent,i))
    return features

#sent:sentence,input of this parameter is train_sents[k],the k is lineNo(from 0),data format:[(word,tag)...]
def sentToken(sent):
    token = []
    for i in range(1,len(sent)):
        token.append(sent[i][0])
    return token

#trainSet:train_sents
#num:num of trainSet
def trainModel(trainSet,begin,num):
    X_train_features = []
    Y_train_labels = []
    X_test_features = []
    Y_test_labels = []
    result = []
    for n in range(begin + num + 1,begin + len(trainSents) + 1):
        tmp3 = n%len(trainSents)
        X_test_features.append(sentFeatures(trainSents[tmp3]))
    for m in range(begin + num + 1,begin + len(trainSents) + 1):
        tmp4 = m%len(trainSents)
        Y_test_labels.append(sentlabels(trainSents[tmp4]))
    #i:line
    for i in range(begin,begin + num):
        tmp1 = i%len(trainSents)
        X_train_features.append(sentFeatures(trainSents[tmp1]))
    #j:line
    for j in range(begin,begin + num):
        tmp2 = j%len(trainSents)
        Y_train_labels.append(sentlabels(trainSents[j]))
    trainer = pycrfsuite.Trainer(verbose = False)
    nn = 0
    for xseq,yseq in zip(X_train_features,Y_train_labels):
        nn = nn + 1
        print nn
        trainer.append(xseq,yseq)
    trainer.set_params({'c1':1.0,'c2':1e-3,'max_iterations':50,'feature.possible_transitions': True})
    trainer.params()
    trainer.train('nlp4.crfsuite')
    result.append(X_train_features)
    result.append(Y_train_labels)
    result.append(X_test_features)
    result.append(Y_test_labels)
    return result

#test:X_test_features
def predictions(X_test):
    tagger = pycrfsuite.Tagger()
    tagger.open('nlp4.crfsuite')
    y_pred = [tagger.tag(xseq) for xseq in X_test]
    return y_pred

def classificationReport(y_ture,y_pred,str):
    result = []
    tureNum_recall = 0.0
    predNum_recall = 0.0
    for i in range(0,len(y_ture)):
        for j in range(0,len(y_ture[i])):
            tmp1=y_ture[i][j]
            if(str == tmp1):
                tureNum_recall = tureNum_recall + 1.0
                if(y_pred[i][j] == y_ture[i][j]):
                    predNum_recall = predNum_recall + 1.0
    recall = predNum_recall/tureNum_recall
    tureNum_precision = 0.0
    predNum_precision = 0.0
    for m in range(0,len(y_pred)):
        for n in range(0,len(y_pred[m])):
            if(y_pred[m][n] == str):
                predNum_precision = predNum_precision + 1.0
                if(y_pred[m][n] == y_ture[m][n]):
                    tureNum_precision = tureNum_precision + 1.0
    precision = tureNum_precision/predNum_precision
    f1_score = (2 * precision * recall)/(precision + recall)
    result.append(precision)
    result.append(recall)
    result.append(f1_score)
    result.append(predNum_precision)
    #result[0]:precision，result[1]:recall,result[2]:f1-score;result[3]:support
    return result





'''
def classificationReport(y_ture,y_pred):
    lb = LabelBinarizer()
    y_ture_combined = lb.fit_transform(list(chain.from_iterable(y_ture)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))
    tagset = set(lb.classes_)-{'O'}
    tagset = sorted(tagset,key=lambda  tag:tag.split('-',1)[::-1])
    class_indices={cls:idx for idx,cls in enumerate(lb.classes_)}
    return classification_report(y_ture_combined,y_pred_combined,labels=[class_indices[cls] for cls in tagset],tagset_name=tagset)
'''
if __name__ == '__main__':
    fileName = "labeledData.txt"
    trainSents = train_sents(fileName)
    result = trainModel(trainSents,0,1500);#X_train_features,Y_train_labels,X_test_features,Y_test_labels
    y_pred = predictions(result[2])
    print "pred"
    labels = ["time","location","person_name","org_name","company_name","product_name"]
    for str in labels:
        tmp_str1 = str + "_B"
        print tmp_str1 + "   " + "precision" + "   " +"recall"+"   " +"f1-score" + "   "+"support"
        print classificationReport(result[3],y_pred,tmp_str1)
        tmp_str2 = str + "_I"
        print tmp_str2 + "   " + "precision" + "   " +"recall"+"   " +"f1-score" + "   "+"support"
        print classificationReport(result[3],y_pred,tmp_str2)
