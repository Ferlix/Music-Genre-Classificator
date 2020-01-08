# Music-Genre-Classificator

 * Classify music files based on genre from the GTZAN music corpus
 * CNN
 * RNN
 * CRNN 
 * SVM
 * Deep RNN with transfer learning

(Implementated with Tensorflow)

### Audio features extracted

 * [MFCC](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum) as frequencies spectograms
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


### Results

| Model | Trainig accuracy | Test accuracy |
| --- | --- | --- |
| Deep CNN | 0.99 | 0.7 |
| CNN | x | x |
| RNN | x | x |
| CRNN | x | x |
| SVM | x | x |

