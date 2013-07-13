# -*- coding: utf8 -*-
import re, collections
import cPickle as pickle

def words(text):
    return re.findall('[a-z]+', text.lower()) 

def train(features):
    model = collections.defaultdict(int)
    for f in features:
        model[f] += 1
    return model

class Corrector:
    def __init__(self):
        Corrector.alphabet = 'abcdefghijklmnopqrstuvwxyz'
        Corrector.NWORDS = pickle.load(open('dict.pkl','rb'))

    def edits1(self,word):
       splits     = [(word[:i], word[i:]) for i in range(len(word) + 1)]
       deletes    = [a + b[1:] for a, b in splits if b]
       transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b)>1]
       replaces   = [a + c + b[1:] for a, b in splits for c in Corrector.alphabet if b]
       inserts    = [a + c + b     for a, b in splits for c in Corrector.alphabet]
       return set(deletes + transposes + replaces + inserts)

    def known_edits2(self,word):
        return set(e2 for e1 in self.edits1(word) for e2 in self.edits1(e1) if e2 in Corrector.NWORDS)

    def known(self,words):
        return set(w for w in words if w in Corrector.NWORDS)

    def correct(self,word):
        candidates = self.known([word]) or self.known(self.edits1(word)) or self.known_edits2(word) or [word]
        return max(candidates, key=Corrector.NWORDS.get)

if __name__ == '__main__':
    print 'begin training.'
    Dictionary = train(words(file('dict.txt').read()))
    pickle.dump(Dictionary,open('dict.pkl','wb'))
