from utils import *
from model import *
from pickle import dump

def train(train_desc, train_features, vocab_size, max_length, tokenizer):

    # create a model
    model = define_model(vocab_size, max_length)
    epochs = 5
    steps = len(train_desc)
    for i in range(epochs):
        # create the data generator
        generator = data_generator(train_desc, train_features, tokenizer, max_length)
        # fit the model for one epoch
        model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
        # save the model
        model.save('model_' + str(i) + '.h5')

def main():

    # loading the training dataset.
    filename = 'Flickr_8k.trainImages.txt'
    train = load_set(filename)
    length = len(train)
    print("Dataset: %d" % length)

    # load the descriptions
    train_desc = load_clean_descriptions('descriptions.txt', train)
    print("Descriptions: %d" % len(train_desc))

    # load all the stored features of images in the dataset.
    train_features = load_photo_features('features.pkl', train)
    print("Photos: %d" % len(train_features))

    # prepare the tokenizer
    tokenizer = create_tokenizer(train_desc)
    # save the tokenizer
    dump(tokenizer, open('tokenizer.pkl', 'wb'))
    vocab_size = len(tokenizer.word_index) + 1
    print("Vocabulary size: %d" % vocab_size)

    # determine the maximum sequence length
    max_len = max_length(train_desc)
    print("Description Length: %d" % max_len)

    train(train_desc, train_features, vocab_size, max_len, tokenizer)


if __name__ == '__main__':
    main()
