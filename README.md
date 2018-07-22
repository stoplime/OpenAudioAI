# OpenAudioAI
Open Audiobooks
Work Logging:
https://docs.google.com/document/d/1fe0iCKSy-xPkjXN8e_PjmjxUOdGY7QqCyvKVCkFh2d8/edit?usp=sharing

To do:
Dataset Preparation:
  1) Potentially, extract string from html dataset already provided.
  2) Categorize string into utterances and labels.
  3) Divide up the utterences into sentences.(Or maybe divide per word)
  4) Set up hyperparameters for dynamically sizing sentences.
  5) Organizing every word from sentences into a hash table with their own keys.
  6) Convert the raw sentences into keys. 
  7) Create another hash table for word embeddings we create or downloaded from NLTK.
  8) Convert keys back to word embeddings.
  
  
  Current model Idea: 
  
  ![model1](https://user-images.githubusercontent.com/10410430/43050281-4167ac72-8dd4-11e8-85f2-f37859e15f30.jpg)
  
