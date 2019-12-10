# Music-Genre-Classificator

 * Classify music files based on genre from the GTZAN music corpus
 * GTZAN corpus is included for easy of use
 * CNN (to be implemented)
 * CRNN (to be implemented)
 * Implementated in Tensorflow

### Audio features extracted

 * [MFCC](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum) as 128x128 frequencies spectograms
 * (We might also try) [Spectral Centroid](https://en.wikipedia.org/wiki/Spectral_centroid)
 * (We might also try) [Chroma](http://labrosa.ee.columbia.edu/matlab/chroma-ansyn/)
 * (We might also try) [Spectral Contrast](http://ieeexplore.ieee.org/document/1035731/)

### Preprocesing dependencies
The jupyter notebook for the pre-processing uses the following dependencies:
 * numpy (for array operations)
 * librosa (for audio feature extraction)
 * os
 * glob (to read files)
 * matplotlib (to save spectrograms)


### Model dependencies
To be modelled

### Accuracy

 * Training (at Epoch N):
    - Training loss: X
    - Training accuracy: X

 * Testing:
    - Test loss:   X
    - Test accuracy:  X
