import os 
import nltk
import numpy as np
import nltk.tree as Tree
from nltk import treetransforms, induce_pcfg,  Nonterminal

class preparation():

	def __init__(self):
		train_file = open("filename","w")
		with open("sequoia-corpus+fct.mrg_strict.txt","r") as file:
			data = file.readlines() #split into lines
	
			for line in data :
				#we remove first and last 2 character, we cannot do that in words because when we split we have words and no more characters
				words = line[1:-2].split() #words is the list of words for sentence line, we supprime the first and the last 2 character
				for w in words : # we take each words
					if w[0] == "(": #If the first character is a "(" we know that it is not terminal so we remove "-" when we find one
						if "-" in w:
							train_file.write(w[:w.index("-")]) #W.index("-") gives the place where "-" is.
						else :
							train_file.write(w)
					else : #it's terminal, so we write the word even if we have "-" because it could be a a date for example
						train_file.write(w)
					train_file.write(" ") # we need to add a spaxce between words
				train_file.write("\n") #we finish the sentence, we write in another line

		train_file.close()



		#WE SPLIT INTO TRAIN VALIDATION and TEST set
		file1 = open("train_file","w")
		file2 = open("validation_file","w")
		file3 = open("test_file","w")
		with open("filename","r") as f:
			data = f.readlines()
			longueur = len(data)
			for line in data[:int(longueur*0.8)]:
				words = line.split()
				for w in words : 
					file1.write(w)
					file1.write(" ")
				file1.write("\n")

			for line in data[int(longueur*0.8) : int(longueur*0.9)]:
				words = line.split()
				for w in words : 
					file2.write(w)
					file2.write(" ")
				file2.write("\n")

			for line in data[int(longueur*0.9):]:
				words = line.split()
				for w in words : 
					file3.write(w)
					file3.write(" ")
				file3.write("\n")

		file1.close()
		file2.close()
		file3.close()



	#We create the file with just word and not the grammar
		test_sentence = open("test_sentence","w")
		with open("test_file","r") as txt:
			for phrase in txt :
				test_tree = nltk.tree.Tree.fromstring(phrase)
				for word in test_tree.leaves():
					test_sentence.write(word)
					test_sentence.write(" ")
				test_sentence.write("\n")

	
		test_sentence.close()














