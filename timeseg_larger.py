import glob
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.layers import Conv1D, Conv1DTranspose
from tensorflow.python.keras.layers.core import Dropout

import pandas as pd 
import numpy as np
from scipy import interpolate 

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from sklearn.metrics._classification import accuracy_score


class TimeSeg:
    def smooth(x,window_len=11,window='hanning'):
        """smooth the data using a window with requested size.

        This method is based on the convolution of a scaled window with the signal.
        The signal is prepared by introducing reflected copies of the signal 
        (with the window size) in both ends so that transient parts are minimized
        in the begining and end part of the output signal.

        input:
            x: the input signal 
            window_len: the dimension of the smoothing window; should be an odd integer
            window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
                flat window will produce a moving average smoothing.

        output:
            the smoothed signal
        
        example:

        t=linspace(-2,2,0.1)
        x=sin(t)+randn(len(t))*0.1
        y=smooth(x)

        see also: 

        numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
        scipy.signal.lfilter
    
        TODO: the window parameter could be the window itself if an array instead of a string
        NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
        """

        x = np.array(x)
        if x.ndim != 1:
            raise ValueError("smooth only accepts 1 dimension arrays.")

        if x.size < window_len:
            raise ValueError("Input vector needs to be bigger than window size.")

        if window_len<3:
            return x

        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError("Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

        #s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
        #print(len(s))
        if window == 'flat': #moving average
            w=np.ones(window_len,'d')
        else:
            w=eval('np.'+window+'(window_len)')

        N = window_len
        spad = np.pad(x, (N//2, N-1-N//2), mode='edge')
        ret = np.convolve(w/w.sum(),spad,mode='valid')
        #ret = y[int(window_len/2)-1:-int(window_len/2)-1] 

        assert len(ret) == len(x)
        return ret
    ### Artificial data generation function
    def gen_trace(noise_amp, amp = 0.9, n=500):
        # Generate fake top hat data
        rnd = sorted(np.random.random_sample((2,)))
        start,stop = rnd[0],rnd[1]
        t = np.linspace(0, 1, n)
        vals = np.zeros(n)
        vals[t>start]= amp
        vals[t>stop] = 0
        vals = TimeSeg.smooth(vals)
        #vals = savgol_filter(vals, 51, 3)
        noise = np.random.normal(0, noise_amp, (500))
        raw = vals + noise

        # label fake top hat data
        labels = np.zeros(n)
        labels[t>start]= 2
        labels[t>stop] = 0
        labels[abs(t-start)<.025] = 1
        labels[abs(t-stop)<.025] = 1

        encoded_labels = np.zeros((n,3))
        encoded_labels[:,0] = labels==0
        encoded_labels[:,1] = labels==1
        encoded_labels[:,2] = labels==2
        print(raw)
        print(encoded_labels)
        return  raw, encoded_labels
    
    def train_model():
        ### Generate and label artificial data for training
        x_train, y_train = [],[]
        for i in range(0,20000):
            if(i < 5000):
                raw,labels = TimeSeg.gen_trace(noise_amp=0.015)
            elif(i < 10000): 
                raw,labels = TimeSeg.gen_trace(noise_amp=0.025)
            elif(i < 15000):
                raw,labels = TimeSeg.gen_trace(noise_amp=0.035)
            else: 
                raw,labels = TimeSeg.gen_trace(noise_amp=0.045)
            x_train.append(raw)
            y_train.append(labels)
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_train = np.expand_dims(x_train, axis=2)

        plt.plot(raw)
        plt.plot(labels)
        plt.show()

        #Create 1D encoder CNN model using sequential API from Keras
        model = keras.models.Sequential()
        model.add(Conv1D(filters= 32, kernel_size=7, activation='relu', input_shape= (500,1), strides=2, padding='same'))
        model.add(Dropout(0.25))
        model.add(Conv1D(filters=16, kernel_size=7, activation='relu', strides=2, padding='same'))
        model.add(Dropout(0.25))
        model.add(Conv1DTranspose(filters=16, kernel_size=7, activation='relu', strides=2, padding='same'))
        model.add(Dropout(0.25))
        model.add(Conv1DTranspose(filters=32, kernel_size=7, activation='relu', strides=2, padding='same'))
        model.add(Dropout(0.25))
        model.add(Conv1DTranspose(filters=32, kernel_size=7, activation='relu', strides=1, padding='same'))
        model.add(Dropout(0.25))
        model.add(Conv1DTranspose(filters=16, kernel_size=7, activation='relu', strides=1, padding='same'))
        model.add(Dropout(0.25))
        model.add(Conv1DTranspose(filters=16, kernel_size=7, activation='relu', strides=1, padding='same'))
        model.add(Dropout(0.25))
        model.add(Conv1DTranspose(filters=3, kernel_size=3, activation='softmax',padding='same'))
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy', tf.keras.metrics.Precision()], loss=keras.losses.CategoricalCrossentropy())
        model.summary()



        ### Train model and plot validation vs training error
        history = model.fit(
            x_train,
            y_train,
            epochs=3,
            batch_size=128,
            validation_split=0.1,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")
            ],
        )
        #model.save("arcjetmodel.h5")
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.legend(['accuracy','validation accuracy'])
        plt.show()

        plt.plot(history.history["loss"], label="Training Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.xlabel("Time")
        plt.ylabel("Percent Error")
        plt.legend()
        plt.show()

        x_test = [] 
        y_test = []
        raw,labels = TimeSeg.gen_trace(noise_amp=0.015)
        x_test.append(raw)
        y_test.append(labels)
        y_test = np.array(y_test)
        x_test = np.array(x_test)
        x_test = np.expand_dims(x_test, axis=2)
        out = model.predict(x_test)
        plt.plot(x_test[0,:,0])
        plt.plot(out[0,:,:])
        plt.xlabel("Time(s)")
        plt.ylabel("Arc Current scaled 0-1")
        plt.show()
        evals = model.evaluate(x_test, y_test)
        print(evals)

TimeSeg.gen_trace(noise_amp=0.015)