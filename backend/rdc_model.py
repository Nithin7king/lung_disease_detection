import os
import tensorflow
from tensorflow.keras.models import load_model
import numpy as np
import librosa as lb

# load trained model
model = load_model('model/model.h5')


def fix_feature_size(feature, size=259):
    if feature.shape[1] < size:
        pad_width = size - feature.shape[1]
        feature = np.pad(feature, ((0, 0), (0, pad_width)), mode='constant')
    else:
        feature = feature[:, :size]
    return feature


def getFeaturesForNeuralNetwork(path):

    soundArr, sample_rate = lb.load(path)

    mfcc = lb.feature.mfcc(y=soundArr, sr=sample_rate, n_mfcc=20)
    cstft = lb.feature.chroma_stft(y=soundArr, sr=sample_rate)
    mSpec = lb.feature.melspectrogram(y=soundArr, sr=sample_rate)

    mfcc = fix_feature_size(mfcc)
    cstft = fix_feature_size(cstft)
    mSpec = fix_feature_size(mSpec)

    mfcc = mfcc.reshape(1, mfcc.shape[0], mfcc.shape[1], 1)
    cstft = cstft.reshape(1, cstft.shape[0], cstft.shape[1], 1)
    mSpec = mSpec.reshape(1, mSpec.shape[0], mSpec.shape[1], 1)

    return mfcc, cstft, mSpec


def classificationResults(soundFilePath):

    res_list = []

    if os.path.exists(soundFilePath):

        mfcc_test, croma_test, mspec_test = getFeaturesForNeuralNetwork(soundFilePath)

        result = model.predict({
            "mfcc": mfcc_test,
            "croma": croma_test,
            "mspec": mspec_test
        })

        diseaseArray = [
            'Asthma', 'Bronchiectasis', 'Bronchiolitis', 'COPD',
            'Healthy', 'LRTI', 'Pneumonia', 'URTI'
        ]

        result = result.flatten()

        indexMax = np.argmax(result)

        indexSecMax = 0
        secMax = result[0]

        for i in range(len(result)):
            if result[i] > secMax and result[i] < result[indexMax]:
                indexSecMax = i
                secMax = result[i]

        res1 = "Respiratory disorder detected: " + diseaseArray[indexMax] + " with probability " + str(result[indexMax]*100) + "%"
        res2 = "Second prediction: " + diseaseArray[indexSecMax] + " with probability " + str(result[indexSecMax]*100) + "%"

        res_list.append(res1)
        res_list.append(res2)

        return res_list

    else:

        res_list.append("Sorry, No File Found")
        res_list.append("Please upload the file in .wav format")

        return res_list