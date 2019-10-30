###### Predicting Persuasiveness of Comments

To build model for Predicting Persuasiveness of Comments I have first extracted the features from the comment.
In this code I have used sentiment score, cosine similarity to get the similarity between Opinion and Comment. I used svm,
Random Forest Classifier to train the model and compared them. The accuracy is around 56%.

To install the required packages, run the following command
`pip3 install -r requirements.txt`

The repo contains data folder for data to train and test model. Alongside this the baseline.py file where model is written.

NOTE:: To RUN this code you will need to download nltk.