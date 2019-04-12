from model import *
from utils import *
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
from numpy import argmax
from pickle import load

# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
    # seed the generation process
    in_text = 'startseq'
    # iterate over the whole length of the sequence
    for i in range(max_length):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
        yhat = model.predict([photo, sequence], verbose=0)
        # convert probability to integer
        yhat = argmax(yhat)
        # map integer to word
        word = word_for_id(yhat, tokenizer)
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        in_text += ' ' + word
        # stop if we predict the end of the sequence
        if word == 'endseq':
            break
    return in_text

# evaluate the skill of the model
def evaluate_model(model, descriptions, photos, tokenizer, max_length):
    actual, predicted = list(), list()
    # step over the whole set
    for key, desc_list in descriptions.items():
        # generate description
        yhat = generate_desc(model, tokenizer, photos[key], max_length)
        # store actual and predicted
        references = [d.split() for d in desc_list]
        actual.append(references)
        predicted.append(yhat.split())
    # calculate BLEU score
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

def main():

    filename = 'Flickr_8k.trainImages.txt'
    train = load_set(filename)
    print('Dataset: %d' % len(train))

    # descriptions
    train_descriptions = load_clean_descriptions('descriptions.txt', train)
    print('Descriptions: train=%d' % len(train_descriptions))

    # prepare tokenizer
    tokenizer = create_tokenizer(train_descriptions)
    vocab_size = len(tokenizer.word_index) + 1
    print('Vocabulary Size: %d' % vocab_size)

    # determine the maximum sequence length
    max_len = max_length(train_descriptions)
    print('Description Length: %d' % max_len)

    # load test set
    filename = 'Flickr_8k.testImages.txt'
    test = load_set(filename)
    print('Dataset: %d' % len(test))

    # descriptions
    test_descriptions = load_clean_descriptions('descriptions.txt', test)
    print('Descriptions: test=%d' % len(test_descriptions))

    # photo features
    test_features = load_photo_features('features.pkl', test)
    print('Photos: test=%d' % len(test_features))

    # load the model
    filename = 'model_2.h5'
    model = load_model(filename)

    # evaluate the model
    evaluate_model(model, test_descriptions, test_features, tokenizer, max_len)

if __name__ == '__main__':

    main()
