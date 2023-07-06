## Directory tree

Exercise1:

+ main.py: the source code
+ requirements.txt: packages to install
+ report_ex1.pdf: a report on the model

Exercise2: 

+ report_ex2.pdf

Miscellaneous files:

+ keras_model.h5: model to use
+ labels.txt: lables to classify

## Requirements for the environment

I use Python 3.11 as the main intepreter. For the environments, I have put the required packages for this environment. 

## Source code's information for Exercise 1

The code is written on the Streamlit backbone so when running the file, the following command is:

`python -m streamlit run main.py`

After running the code, it should be running a Streamlit within your local machine. The program will ask you to upload an image to test. Clicking the "Classify" button will initialize the model and the answer will be returned.

If you don't upload any and click the "Classify" button, it will re-ask you to do until you upload one. 

I have also included a deployed Streamlit website in the mail. If you cannot find one, please refer to this [link](https://dog-cat-classifier.streamlit.app/)