import cv2
import numpy as np
import dlib
import pandas as pd
import os, glob
from imutils import face_utils
import argparse
import imutils
import tensorflow as tf
from keras.models import model_from_json


#--------------------------------------------------------#
#                       Image Mood                       #
#--------------------------------------------------------#
# -------------------------------------------------------
script_dir = os.path.dirname(__file__) 
rel_path = "emotiondetector100.json"
filename = os.path.join(script_dir, rel_path)
json_file = open(filename, "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

rel_path = "emotiondetector100.h5"
filename1 = os.path.join(script_dir, rel_path)
model.load_weights(filename1)

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

def extract_featuresIMG(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

def detect_emotionIMG(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(image, 1.3, 5)
    labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
    try: 
        for (p, q, r, s) in faces:
            face_img = gray[q:q+s, p:p+r]
            face_img = cv2.resize(face_img, (48, 48))
            img = extract_featuresIMG(face_img)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]
            return prediction_label
    except cv2.error:
        pass

#--------------------------------------------------------#
#                   Video Mood                           #
#--------------------------------------------------------#
# ------------------------------------------------------------
script_dir = os.path.dirname(__file__) 
rel_path = "emotiondetector60.json"
filename = os.path.join(script_dir, rel_path)

json_file = open(filename, "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

rel_path = "emotiondetector100.h5"
filename1 = os.path.join(script_dir, rel_path)
model.load_weights(filename1)

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

def detect_emotion():
    webcam = cv2.VideoCapture(0)
    labels = {0 : 'Stressed', 1 : 'Stressed', 2 : 'Stressed', 3 : 'Not Stressed', 4 : 'Stressed', 5 : 'Stressed', 6 : 'Stressed'}
    while True:
        success, im = webcam.read()
        if not success:
            break
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(im, 1.3, 5)
        try: 
            for (p, q, r, s) in faces:
                image = gray[q:q+s, p:p+r]
                cv2.rectangle(im, (p, q), (p+r, q+s), (255, 0, 0), 2)
                image = cv2.resize(image, (48, 48))
                img = extract_features(image)
                pred = model.predict(img)
                prediction_label = labels[pred.argmax()]
                cv2.putText(im, '% s' %(prediction_label), (p-10, q-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))
            ret, buffer = cv2.imencode('.jpg', im)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except cv2.error:
            pass



#--------------------------------------------------------#
#                       Audio Mood                       #
#--------------------------------------------------------#

# Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, mfcc, chroma, mel):
    import librosa
    import soundfile
    import os, glob, pickle
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score
    import pandas as pd

    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    return result

def audio_prediction(path):
    import os, glob, pickle
    feature = extract_feature(path, mfcc=True, chroma=True, mel=True)
    script_dir = os.path.dirname(__file__) 
    rel_path = "static/voice_dataset/voice_emotions_model.sav"
    filename = os.path.join(script_dir, rel_path)
    
    model = pickle.load(open(filename, 'rb'))
    
    result = model.predict(feature.reshape(1, -1))
    
    result_dict = {'audio':result[0]}
    
    return result_dict

# To be used as a reference only
# emotions={
#       '01':'neutral',
#       '02':'neutral',
#       '03':'happy',
#       '04':'sad',
#       '05':'angry',
#       '06':'fearful',
#       '07':'disgust',
#       '08':'surprised'
# }





#--------------------------------------------------------#
#                     Text Sentiment                     #
#--------------------------------------------------------#

def text_cleaning(data):
    import pandas as pd
    import numpy as np
    import nltk
    # nltk.download()
    from nltk import word_tokenize
    from nltk.corpus import stopwords
    import string
    import pickle
    
    # load the model from disk
    script_dir = os.path.dirname(__file__) 
    rel_pathTEXT1 = "static/text_dataset/text_emotions_model.sav"
    model_filename = os.path.join(script_dir, rel_pathTEXT1)
    model = pickle.load(open(model_filename, 'rb'))

    # load the vectorizer from disk
    rel_pathTEXT2 = "static/text_dataset/tfidf_vect.pk"
    vectorizer_filename = os.path.join(script_dir, rel_pathTEXT2)
    tfidf_vect = pickle.load(open(vectorizer_filename, 'rb'))

    text = data.lower()
    cleaned = [char for char in text if char not in string.punctuation]
    cleaned = "".join(cleaned)
    result = np.array([cleaned])
    
    result_prediction = text_features(result, tfidf_vect, model)
    
    emotion = {'text':result_prediction}
    
    return emotion


def text_features(text, tfidf_vect, model):
    import pandas as pd
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import MultinomialNB
    
    text_vect = tfidf_vect.transform(text).toarray()
    
    emotion = model.predict(text_vect.reshape(1, -1))[0]
    
    emotions = {0:'Negative',
                1:'Positive'}
    
    result = emotions[emotion]
    
    return result


