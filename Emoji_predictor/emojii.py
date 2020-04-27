import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation


def pred(glove_dictionary, sent):
    path = "Files/Emoji_Predictor/"
    emoji_to_image = {"0": 'Emoticons/heart.png ',
                      "1": "Emoticons/Baseball.png",
                      "2": "Emoticons/smile.png",
                      "3": 'Emoticons/sad.png',
                      "4": 'Emoticons/Fork.png', }

    # we wil be using only these emoji to calssify sentence
    # emoji_dictionary={"0": ":beating_heart:",
    #                  "1": ":baseball:",
    #                  "2": ":grinning_face_with_big_eyes:",
    #                  "3": ":disappointed_face:",
    #                  "4": ":fork_and_knife:",
    #                  }

    '''we need to convert the given words into vectors as machine only understand numbers and each word should have 
    different number representation so we can do this manually or we can use Glove developed by Stanford university 
    or word2vec of goggle for this model we will be using glove of 6B.50d '''

    # glove=open("Emoji predictor/Files/glove.6B.50d.txt",encoding='utf-8')
    # glove_dictionary={}
    # for line in glove:
    #     value=line.split()
    #     word=value[0]
    #     coefficient=np.asarray(value[1:],dtype=float)
    #     glove_dictionary[word]=coefficient
    #  glove.close()

    # import required Dataset for training and testing
    # training_data=pd.read_csv(path + "Dataset/train_emoji.csv",header=None)
    # testing_data=pd.read_csv(path + "Dataset/test_emoji.csv",header=None)
    # X_train=training_data[0]
    # Y_train=training_data[1]
    # X_test=testing_data[0]
    # Y_test=testing_data[1]

    # converting the sentence into vectors
    # def create_embedding_matrix(X):
    #     embedding_matrix=np.zeros((X.shape[0],10,50))
    #     for i in range(X.shape[0]):
    #         X[i]=X[i].split()
    #         for j in range(len(X[i])):
    #             try:
    #                 embedding_matrix[i][j]=glove_dictionary[X[i][j].lower()]
    #             except:
    #                 embedding_matrix[i][j]=np.zeros((50,))
    #     return embedding_matrix
    # X_train_vector=create_embedding_matrix(X_train)
    # X_test_vector=create_embedding_matrix(X_test)

    # now we have to convert Y part of train and test set to vector
    # Y_train_vector=to_categorical(Y_train)
    # Y_test_vector=to_categorical(Y_test)

    # here we will create a model to analyze the sentence and convert it to emoji

    # Creating basic structure of Model

    model = Sequential()
    model.add(LSTM(64, input_shape=(10, 50), return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(64, input_shape=(10, 50)))
    model.add(Dropout(0.4))
    model.add(Dense(5))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    # model.fit(X_train_vector,Y_train_vector,epochs=40,shuffle=True,batch_size=32)
    model.load_weights(path+"model.h5")

    def create_embedding_matrix(sentence):
        embedding_matrix = np.zeros((1, 10, 50))
        sentence = sentence.split()
        for j in range(len(sentence)):
            try:
                embedding_matrix[0][j] = glove_dictionary[sentence[j].lower()]
            except KeyError:
                embedding_matrix[0][j] = np.zeros((50,))
        return embedding_matrix

    X_input = create_embedding_matrix(sent)
    predict = model.predict_classes(X_input)

    # cv2.imread("Emoji predictor\Files\Emoticons\heart.png")
    # while(cv2.waitKey(1)!=ord("q")):
    #     cv2.imshow("PRESS Q TO EXIT",cv2.imread(emoji_to_image[str(pred[0])]))
    return path + emoji_to_image[str(predict[0])]
