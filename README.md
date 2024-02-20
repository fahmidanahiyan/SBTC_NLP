# Scientific Bengali Text Classification using Transformer-based Techniques

- Developed a corpus consisting 6000 Bengali texts, categorized into 6 classes: **Physics, Chemistry, Biology, Information and Communications Technology (ICT), Mathematics and Others**
- Performed necessary preprocessing steps and conducted statistical analysis on the dataset
- Performed hyperparameter tuning and investigated the outcomes of 5 transformer-based models: **mBERT, Distil mBERT, Bangla BERT, Indic BERT, XLM-RoBERTa**
- For comparsion with ML models, 2 word embedding techniques were explored- **BoW, TF-IDF** and result of 4 ML models investigated- **NB, LR, SVM, RF**
- For comparsion with DL models, 3 word embedding techniques were explored- **Word2Vec, FastText, GloVe** and result of 3 ML models investigated- **CNN, BiLSTM, CNN+BiLSTM**

## Data Summary

A balanced dataset of 1000 data per class is developed-

![image](https://github.com/fahmidanahiyan/SBTC_NLP/assets/117027098/998480f5-ec41-405d-9f76-5d20117fcca4)

Information on total number of words, unique words and the most frequent words in each class are retrieved-

![phy info](https://github.com/fahmidanahiyan/SBTC_NLP/assets/117027098/dcceefa3-e438-4a30-a801-ad807264da84)
![chem info](https://github.com/fahmidanahiyan/SBTC_NLP/assets/117027098/3f16f22d-86c1-437e-8167-a978ee1a4bdd)
![bio info](https://github.com/fahmidanahiyan/SBTC_NLP/assets/117027098/b359b699-6eef-4077-9f8d-f36d5a5850d0)
![ict info](https://github.com/fahmidanahiyan/SBTC_NLP/assets/117027098/9d9ba16d-4abc-4e5f-ba38-192fde9b2664)
![math ifo](https://github.com/fahmidanahiyan/SBTC_NLP/assets/117027098/4dd03294-d2e1-480a-996b-39d3585daf5c)
![other info](https://github.com/fahmidanahiyan/SBTC_NLP/assets/117027098/da39a293-6149-4b86-bc6d-cf1f36989cab)
![image](https://github.com/fahmidanahiyan/SBTC_NLP/assets/117027098/f171e364-7b8a-437e-a644-ad128118e770)

The length distribution of the texts in the dataset is shown in a histogram-

![image](https://github.com/fahmidanahiyan/SBTC_NLP/assets/117027098/864f4a52-9186-4ce4-b104-b5ce22c5771e)

Unigrams, Bigrams, Trigrams for each class-

![uni phy](https://github.com/fahmidanahiyan/SBTC_NLP/assets/117027098/bbc753fb-bb61-49c3-904b-568727c759fc)
![bi phy](https://github.com/fahmidanahiyan/SBTC_NLP/assets/117027098/147dc7c1-97be-44e6-8f6a-007553f5163a)
![tri phy](https://github.com/fahmidanahiyan/SBTC_NLP/assets/117027098/49222ac5-a509-4abe-ae85-ec01e7309086)

![uni chem](https://github.com/fahmidanahiyan/SBTC_NLP/assets/117027098/6858ba6a-e7ed-4cf2-a210-8e159a303f13)
![bi chem](https://github.com/fahmidanahiyan/SBTC_NLP/assets/117027098/db87a1ac-248b-48d4-ac2e-608d2d1d3f4f)
![tri chem](https://github.com/fahmidanahiyan/SBTC_NLP/assets/117027098/ee0dec6e-1cef-4268-89c6-2e0f01e47468)

![uni bio](https://github.com/fahmidanahiyan/SBTC_NLP/assets/117027098/fc82ff24-53fc-41e4-a3e3-702660c48064)
![bi bio](https://github.com/fahmidanahiyan/SBTC_NLP/assets/117027098/0edca542-5556-4cfc-9c6e-a7f837b4fab5)
![tri bio](https://github.com/fahmidanahiyan/SBTC_NLP/assets/117027098/a9bc440c-94d3-458d-9e0e-f2d396ccc7f6)

![uni ict](https://github.com/fahmidanahiyan/SBTC_NLP/assets/117027098/ae06beae-01b5-4776-88a4-d6c71f428240)
![bi ict](https://github.com/fahmidanahiyan/SBTC_NLP/assets/117027098/07020113-ea29-4f2e-bb2e-93a3963f0ade)
![tri ict](https://github.com/fahmidanahiyan/SBTC_NLP/assets/117027098/5c659444-d628-4494-8fe1-5b6b5f7f9950)

![uni math](https://github.com/fahmidanahiyan/SBTC_NLP/assets/117027098/c901290a-cc54-4601-ac36-06d408f81704)
![bi math](https://github.com/fahmidanahiyan/SBTC_NLP/assets/117027098/3ced36e9-03c7-4374-a1cc-a859379f23a4)
![tri math](https://github.com/fahmidanahiyan/SBTC_NLP/assets/117027098/0ffcd207-3e81-47e2-be96-cd14c1eb2030)

![uni other](https://github.com/fahmidanahiyan/SBTC_NLP/assets/117027098/cbda58d5-14dd-4674-ad60-4153cf00cded)
![bi others](https://github.com/fahmidanahiyan/SBTC_NLP/assets/117027098/34917d17-8424-40fd-ab14-cfd658e4a1bc)
![tri other](https://github.com/fahmidanahiyan/SBTC_NLP/assets/117027098/6dfa93bc-9df6-4036-a4bf-e140341b257e)

Word Cloud for each class-

![image](https://github.com/fahmidanahiyan/SBTC_NLP/assets/117027098/ddcb2850-1119-45c0-9437-bc5f4646e57c)
![image](https://github.com/fahmidanahiyan/SBTC_NLP/assets/117027098/8c2d9124-a9f9-4139-9765-208d8e3c8636)
![image](https://github.com/fahmidanahiyan/SBTC_NLP/assets/117027098/292631cb-895c-40de-a940-10723db8a0c8)
![image](https://github.com/fahmidanahiyan/SBTC_NLP/assets/117027098/2717fe6c-f7ec-4a71-a04d-41d6db5b328e)
![image](https://github.com/fahmidanahiyan/SBTC_NLP/assets/117027098/b4a41690-1bbb-4ab7-89a9-4dc3343148f5)
![image](https://github.com/fahmidanahiyan/SBTC_NLP/assets/117027098/a9c61c15-a683-40a9-9053-6445f022e688)

### Abstract Process of SBTC

![process](https://github.com/fahmidanahiyan/SBTC_NLP/assets/117027098/e25efdab-68bf-4c73-97c6-40291fcf919c)

#### Hyperparameter Tuning

![TUNING](https://github.com/fahmidanahiyan/SBTC_NLP/assets/117027098/3f1bf67b-d518-49e9-ac6d-6862068a2b83)

##### Model Evaluation

Confusion matrix of the 5 transformer-based models-

![mbert](https://github.com/fahmidanahiyan/SBTC_NLP/assets/117027098/65f95123-0eaf-48ac-b230-249e86ae7c3e)
![distil](https://github.com/fahmidanahiyan/SBTC_NLP/assets/117027098/e35d0640-9fde-4f27-8cc4-b3d58c921e48)
![bangla](https://github.com/fahmidanahiyan/SBTC_NLP/assets/117027098/ccb8e6cb-7c39-4793-b095-c8f0a933e66e)
![indic](https://github.com/fahmidanahiyan/SBTC_NLP/assets/117027098/39b8efe5-5682-4a57-a07f-2ee1fd94741d)
![xlm](https://github.com/fahmidanahiyan/SBTC_NLP/assets/117027098/c548e06b-10b0-4b00-8fd6-cd23b91df4a2)

XLM-RoBERTa outperformed all the other models achieving 91.56% accuracy, 91.62% precision, 91.60% recall and 91.58% f1-score.




