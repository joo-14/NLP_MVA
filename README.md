# NLP_MVA
This work is an implementation of a probabilistic parser for French based on PCYK, CYK and handling Out-Of-Vacabulary words.

You can directly run the shell script run.sh by tapping ./run.sh on the terminal. \
This file will run the main.py file, which calls all the other file.

Notes : 
All the needed file as the polyglo-fr.pkl file are already in the system folder.

Information about the .py files :

1) The extraction_preparation file enables to preprocess the data and split into a train, dev and test set. We also create a test_sentence file which is the the test file without all the grammar but just the sentence with its words.\

2) If you prefer to shuffle the data, in the main file you should put into commentary the "preparation()" and running the lines where it is written "shuffle"\



------------------------------------------------------------------------------
File "evaluation_data.parser_output" was generated running "python3 eval.py"

------------------------------------------------------------------------------
