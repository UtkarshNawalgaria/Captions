from pickle import load
import numpy as np
from os import system
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model, Model
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def extract_features(filename):
    # load the model
    model = VGG16('./vgg16_weights.h5')
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)

    # load the photo
    image = load_img(filename, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)

    # get the features
    feature = model.predict(image, verbose=0)
    return feature

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

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
        yhat = np.argmax(yhat)
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

def main():

    tokenizer = load(open('./pickle/tokenizer.pkl', 'rb'))
    max_length = 34

    # load the model
    model = load_model('model_4.h5')
    # load and prepare the image
    photo = extract_features('example.jpg')
    # generate description
    description = generate_desc(model, tokenizer, photo, max_length)
    desc = description.split()
    desc = ' '.join(desc[1:len(desc)-1])
    print(desc)

if __name__ == '__main__':

    main()
