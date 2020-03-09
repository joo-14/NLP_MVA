import nltk.treetransforms
import nltk
from nltk.tree import * 
import progressbar
from extraction_preparation import *
from PCYK import *
from PCFG_Gramar import *
from OOV import *
from shuffle import *


print('Creation of the PCFG...')
""" run this cells if you want to shuffle the data set
s = Shuffle() 
s.shuffle()
s.creation_test_file_shuffle()
"""

# run this cell if you don't want to shuffle the data set
preparation()


oov = OOV()
gramar = PCFG()
gramar.chomsky()
print('PCFG created')

#/Users/johnlevy/Desktop/TP2 NLP/test_shuffle_sentence if you are using shuffle data set

output = open("output","w")
with open("test_sentence","r") as file :
	for sentence in progressbar.progressbar(file) :
		s = PCYK(sentence, oov, gramar)
		t = nltk.tree.Tree.fromstring(s[0])
		t.un_chomsky_normal_form(unaryChar='_')
		output.write(parser_final(t) + '\n') 

output.close()


