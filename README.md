# Music-Genre-Classificator

 * Classify music files based on genre from the GTZAN music corpus
 * GTZAN corpus not included for git memory problems (> 1GB) 
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

To run it properly, include in the /data folder a "/genres" folder, which can be downlaoded from here:

http://marsyas.info/downloads/datasets.html

Keep in the "genres" folder only the subfolder with the sound files (i.e. "data/genres/hip-hop", "data/genres/jazz", ...) and nothing else

### Model dependencies
To be modelled

### Accuracy

 * Training (at Epoch N):
    - Training loss: X
    - Training accuracy: X

 * Testing:
    - Test loss:   X
    - Test accuracy:  X
