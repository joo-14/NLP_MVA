import numpy as np
import nltk
import nltk.tree as Tree
from nltk import induce_pcfg
from OOV import *



def parser_tree(back, words, low, high, tag, tags_dict):
    
    if back[low, high, tags_dict[tag]] is None:  
        
        s = '(' + str(tag) + ' ' + words[low] + ')'
        return s
    else:
        
        k, B, C = back[low, high, tags_dict[tag]]
        left = parser_tree(back, words, low, k, B, tags_dict)
        right = ''
        if C is not None:
            
            right = parser_tree(back, words, k, high, C, tags_dict)
        if right != '':
            
            return '(' + str(tag) + ' ' + left + ' ' + right + ')'
        return '(' + str(tag) + ' ' + left + ')' 
            


def PCYK(sentence,oov,gramar):

    words = sentence.strip().split(' ')

    lhs_voc_prob = gramar.lhs_terminal_proba
    lhs_rhs_prob = gramar.lhs_non_terminal_proba
    voc_count = gramar.lexique_occur

    tags_dict = {}
    count = 0
    for word in gramar.lhs_occur.keys():
      tags_dict[word] = count
      count += 1

    S = nltk.Nonterminal('SENT')
    

    table = np.zeros((len(words), len(words)+1, len(tags_dict.keys())))
    
    back = np.full((len(words),len(words)+ 1, len(tags_dict)),None)
    

    for j in range(1, len(words)+1): 

        word = words[j-1]

        print(word)

        if word not in voc_count.keys():

            print("original word: " + word)

            if j == 1:
                prev_word = None
            else:
                prev_word = words[j-2]
            if j == len(words):
                next_word = None
            else:
                next_word = words[j]

            word = oov.verication_OOV(word, prev_word, next_word)
            
        
        tags = []
        for cle in lhs_voc_prob.keys():

          if cle[1] == word:

            tags.append(cle[0])

        for tag in tags:
            table[j-1, j, tags_dict[tag]] = lhs_voc_prob[(tag, word)]
        
        for i in range(j-1, -1, -1):

            for lhs, rhs in lhs_rhs_prob.keys():

                    if type(rhs) is tuple and len(rhs)>1:

                        B = rhs[0]
                        C = rhs[1]
                        
                        for k in range(i+1, j):

                            if table[i, k, tags_dict[B]] > 0 and table[k, j, tags_dict[C]] > 0:

                                if table[i, j, tags_dict[lhs]] < lhs_rhs_prob[(lhs, rhs)] * table[i, k, tags_dict[B]] * table[k, j, tags_dict[C]]:

                                    table[i, j, tags_dict[lhs]] = lhs_rhs_prob[(lhs, rhs)] * table[i, k, tags_dict[B]] * table[k, j, tags_dict[C]]

                                    back[i, j, tags_dict[lhs]] = (k, B, C)
                    else:
                        if type(lhs) is tuple:

                            lhs = lhs[0]
                            
                        rhs = rhs[0]

                        if table[i, j, tags_dict[rhs]] > 0:
                            
                            if table[i, j, tags_dict[lhs]] < lhs_rhs_prob[(lhs, (rhs,))] * table[i, j, tags_dict[rhs]]:

                                table[i, j, tags_dict[lhs]] = lhs_rhs_prob[(lhs, (rhs,))] * table[i, j, tags_dict[rhs]]
                                
                                back[i, j, tags_dict[lhs]] = (j, rhs, None)
        
    return parser_tree(back, words, 0, len(words), S, tags_dict), table[0, len(words), tags_dict[S]]




def parser_final(t):
    tag = t.label()
    if len(t) == 1:
        if type(t[0]) == str:
            return '(' + tag + ' ' + str(t[0]) + ')'
        else:
            return '(' + tag + ' ' + parser_final(t[0]) + ')'
    else:
        s = []
        for i in range(len(t)):
            s.append(parser_final(t[i]))
        return '(' + tag + ' ' + ' '.join(s) + ')'

