# NLP_MVA
This work is an implementation of a probabilistic parser for French based on PCYK, CYK and handling Out-Of-Vacabulary words.

You can directly run the shell script 'run.sh' by tapping ***./run.sh*** on the terminal. \
This file will run the main.py file, which calls all the others files.\

Notes : 
All the needed files as the polyglo-fr.pkl file are already in the system folder.

Information about the .py files :

1) The extraction_preparation file enables to preprocess the data and split into a train, dev and test set. We also create a test_sentence file which is the the test file without all the grammar but just the sentence with its words.\

2) If you prefer to shuffle the data, in the main file you should put into commentary the "preparation()" and running the lines where it is written "shuffle"\

ATTENTION :\
If you directly run the run.sh shell script, the main.py will automaticcaly erase and recreate the evaluation_data.parser_output file and so erase the result that we obtain after 1h30 of running. 


------------------------------------------------------------------------------
File "evaluation_data.parser_output" was generated running "python3 eval.py"

------------------------------------------------------------------------------
