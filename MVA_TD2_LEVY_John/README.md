# NLP_MVA
This work is an implementation of a probabilistic parser for French based on PCYK, CYK and handling Out-Of-Vacabulary words.

You can directly run the shell script **run.sh** by tapping ***./run.sh*** on the terminal. \
This file will run the main.py file, which calls all the others files, basically, it will parse all test data set. 

IMPORTANT NOTES : 
You  need to download the polyglot embeddings and put it in the system folder, here is the link: https://sites.google.com/site/rmyeid/projects/polyglot


Information about the .py files : 

1) The extraction_preparation file enables to preprocess the data and split into a train, dev and test set. We also create a test_sentence file which is the the test file without all the grammar but just the sentence with its words.


2) If you prefer to shuffle the data, in the main file you should put into commentary the "preparation()" and running the lines where it is written "shuffle"


-----------------------------------------------------------------------------------------
File "evaluation_data.parser_output" was generated running "./run.sh" or "python main.py"

-----------------------------------------------------------------------------------------

To evaluate our parser, one should use the open-package EVALB : \
-cd EvalB \
-./evalb -p sample/sample.prm ../system/test_file ../system/evaluation_data.parser_output

