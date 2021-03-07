![](https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?&style=for-the-badge&logo=PyTorch&logoColor=white)
![](https://img.shields.io/badge/pandas%20-%23150458.svg?&style=for-the-badge&logo=pandas&logoColor=white)
![](https://img.shields.io/badge/numpy%20-%23013243.svg?&style=for-the-badge&logo=numpy&logoColor=white)
![](https://img.shields.io/badge/python%20-%2314354C.svg?&style=for-the-badge&logo=python&logoColor=white)
![](https://img.shields.io/badge/Jupyter%20-%23F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white)

----

# Q&A model for SQUAD 2.0 using BERT

BERT-based model which returns “an answer”, given a user question and a passage which includes the answer of the question.


#### ‼️ If notebook doesn't open in GitHub follow the link [jupyter nbviewer](https://nbviewer.jupyter.org/github/Nikoletos-K/QA-with-BERT-for-SQuAD/blob/main/BERT_SQUAD.ipynb)


## Steps for building this Q&A model

In this task I made the following steps for creating the wanted model:

1. Downloaded SQuAD 2.0 from the official site and linked it with the notebook
2. Tokenized data with BertWordPieceTokenizer that is implemented with Rust and
hence it’s a little bit faster than simple BertTokenizer.
3. Pre-processed data as I read every "title" with QAs in the SQuAD 2.0 json
    1. Encoding based on the previous tokenizer for context and questions
    2. Stored start and end of each answer in given corpus
    3. Converted text to ids
    4. Added the mask
    5. Made the padding
    6. Stored every useful information for every QA in a dataframe    
4. Split all the transformed data to 2 sets (X,y) with X: data and y: true answers (one
pair for the training and one for the validation set )
5. Transformed all these data to tensors and splitted them to batches size 16 (CUDA
memory problems for bigger sizes)
6. Added the optimizer (Adam)
7. Trained and validated the model for multiple epochs
8. Tested the model in a small paragraph for Apollo program and Beyonce career
(context taken from Wikipedia)
9. Created a small chatbot for Beyonce

## Evaluation
Validation set gave a 74% accuracy, which is not bad.

## Fine-tunning
I did some fine tuning to the following parameters:
- ___Batch size:___ Almost every batch size bigger than 16 lead to memory crash (CUDA)
- ___Max sequence length:___ Almost every sequence length bigger 256 lead to memory
crash (CUDA)
- ___Adam learning rate:___ Tried values 0.001 and 0.0001 (lots of hours to be executed
only the two) but the results were worse than 0.00001 that had also on the repository that I followed for rhis implementation.
- ___Epochs:___ CUDA crashed either for memory either for time of execution, after 2
epochs as the model needed approximately 2 hours for each epoch.

---
