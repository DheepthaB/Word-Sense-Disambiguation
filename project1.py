# The code performs word sense disambiguation using word2vec and cosine similarity function.


from nltk.corpus import wordnet
from nltk.corpus import stopwords
import sys
from gensim.models import word2vec
from nltk.stem.wordnet import WordNetLemmatizer
from itertools import chain
from sklearn.decomposition import PCA
from matplotlib import pyplot
from matplotlib import pyplot
import re
import numpy

target_word=sys.argv[1]
sentence=sys.argv[2:]

max_similarity=0
best_sense=''
lmtzr = WordNetLemmatizer()
target_word=lmtzr.lemmatize(target_word)

context=''
for i in range(0,len(sentence)):
    context+="".join(lmtzr.lemmatize(sentence[i]))+" "

context=context.split()
stop_words=set(stopwords.words('english'))

context=[[w for w in context if w not in stop_words]]

senses=wordnet.synsets(target_word)


model_context=word2vec.Word2Vec(context,min_count=1,workers=4)
X = model_context[model_context.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model_context.wv.vocab)
for i,word in enumerate(words):
    pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
#pyplot.show()

for sense in senses:          #Build word2vec model for each sense and compute cosine similarity with input sentence
    #print(sense)
    signature=[]
    gloss=sense.definition()
    gloss=re.sub(r'[^A-Za-z]',' ',gloss)
    gloss=gloss.split()
    gloss=[w for w in gloss if w not in stop_words]
    example=sense.examples()
    example=(" ".join(example))
    example=re.sub(r'[^A-Za-z]',' ',example)
    example=example.split()
    example=[w for w in example if w not in stop_words]
    signature.append(gloss)
    signature.append(example)
    signature.append(sense.lemma_names()) 
    hyperhypo = (sense.hyponyms()+sense.hypernyms()+sense.instance_hyponyms()+sense.instance_hypernyms())
    s=set(chain(*[i.lemma_names() for i in hyperhypo]))
    l=(list(s))
    l=" ".join(i for i in l)
    l=re.sub(r'[^A-Za-z]',' ',l)
    l=l.split()
    l=[w for w in l if w not in stop_words]
    signature.append(l)
    related_senses = (sense.member_holonyms()+sense.part_holonyms()+sense.substance_holonyms()+sense.member_meronyms()+sense.part_meronyms()+sense.substance_meronyms()+sense.similar_tos())
    s1=set(chain(*[i.lemma_names() for i in related_senses]))
    l1=(list(s1))
    l1=" ".join(i for i in l1)
    l1=re.sub(r'[^A-Za-z]',' ',l1)
    l1=l1.split()
    l1=[w for w in l1 if w not in stop_words]
    signature.append(l1)
    model = word2vec.Word2Vec(signature,min_count=1,workers=4)
    model.init_sims(replace=True)    
    X = model[model.wv.vocab]
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    pyplot.scatter(result[:, 0], result[:, 1])
    words = list(model.wv.vocab)
    for i,word in enumerate(words):
        pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
    #pyplot.show()
    
    overlap=0
    similarity=0
    for i in range(0,len(list(model_context.wv.vocab))):
        sim=0
        for j in range(0,len(words)):
            cosine_similarity = numpy.dot(model[words[j]], model_context[list(model_context.wv.vocab)[i]])/(numpy.linalg.norm(model[words[j]])* numpy.linalg.norm(model_context[list(model_context.wv.vocab)[i]]))
            sim=max(sim,cosine_similarity) 
        similarity+=sim 

    #print(similarity)
    if(similarity>max_similarity):
        max_similarity=similarity
        best_sense=(sense)

print('best_sense: ',best_sense)
print("Definition:",best_sense.definition())

#####################################################################################################################################################################################################################