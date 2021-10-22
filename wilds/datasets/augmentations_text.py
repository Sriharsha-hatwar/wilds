'''
Making sure these are already installed and inplace. 

pip install torch>=1.6.0 transformers>=4.0.0 sentencepiece
pip install simpletransformers>=0.61.10
pip install nltk>=3.4.5
pip install numpy requests nlpaug

@author : Sriharsha-hatwar 

This file contains augmentation operators.

'''

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

model_dir = '/content/drive/MyDrive/OOD-data-robustness-in-NLP/data/'


#1
def simulateKeyBoardAug(given_text):
    # This simulates the augmentation that might happen due to keystrokes missclicks.
    aug = nac.KeyboardAug()
    augmented_text = aug.augment(given_text)
    return augmented_text
#2
def randomCharAug(given_text):
    aug = nac.RandomCharAug(action='insert')
    augmented_text = aug.augment(given_text)
    return augmented_text
#3
def substitutedWord2Vec(given_text):
    # Can do it later.
    aug = naw.WordEmbsAug(
        model_type='word2vec', model_path=model_dir+'GoogleNews-vectors-negative300.bin',
        action="insert")
    augmented_text = aug.augment(given_text)
    return augmented_text
#4
def tfIdfinsert(given_text):
    aug = naw.TfIdfAug(
        model_path=model_dir,
        action="insert")
    augmented_text = aug.augment(given_text)
    return augmented_text
"""
def contextualAugmenterInsert(give_text):
    aug = naw.ContextualWordEmbsAug(
        model_path='bert-base-uncased', action="insert")
    augmented_text = aug.augment(give_text)
    return augmented_text

def contextualAugmenterSubstitute(give_text):
    aug = naw.ContextualWordEmbsAug(
        model_path='bert-base-uncased', action="substitute")
    augmented_text = aug.augment(give_text)
    return augmented_text
"""
#5
def synonymAugmenter(given_text):
    aug = naw.SynonymAug(aug_src='wordnet')
    augmented_text = aug.augment(given_text)
    return augmented_text
#6
def randomWordAugmenter(given_text):
    aug = naw.RandomWordAug(action="swap")
    augmented_text = aug.augment(given_text)
    return augmented_text
#7
def delWordRandomAugmenter(given_text):
    aug = naw.RandomWordAug()
    augmented_text = aug.augment(given_text)
    return augmented_text

augmentations_all = [simulateKeyBoardAug, randomCharAug, substitutedWord2Vec, tfIdfinsert, delWordRandomAugmenter, randomWordAugmenter, synonymAugmenter]

if __name__ == '__main__':
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')