# Music-Genre-Classificator

Different architectures to lassify music files based on genre from the GTZAN music corpus, namely:
 * Convolutional Neural Network (CNN)
 * Recurrant Neural Network (RNN)
 * Mixed CNN and RNN 
 * SVM
 * Deep RNN with transfer learning

(Implementated with Tensorflow)

### Dataset
In the  GTZAN music corpus there's 10 genres with 100 songs each (1000 in total): 80% of it was used during the training phase (800 images), and 20% for testing (200 images). After the split, each song of 30 seconds is split in chunks of 10 seconds (resulting in 2400 and 600 training and testing samples).

*Dataset can be downlaoded here:*
*http://marsyas.info/downloads/datasets.html*

### Audio augmentation
To increase further the amount of data, some augmentation were done on the audio files. For each song chunk, we applied:
* Add light random noise in the wave form 
* Add intense random noise in the wave form
* Increase randomly pitch (2% at most) 


### Audio features extracted
For exctracting the audio features, the library *librosa* was used. 

 * [Mel-frequency spectrogram](https://librosa.github.io/librosa/generated/librosa.feature.melspectrogram.html) as images of size 512x512 (in black and white)
 
 * Combination of [Mel-frequency spectogram](https://librosa.github.io/librosa/generated/librosa.feature.melspectrogram.html), [spectral centroid](https://librosa.github.io/librosa/generated/librosa.feature.spectral_centroid.html) and [spectral contrast](https://librosa.github.io/librosa/generated/librosa.feature.spectral_contrast.html) stacked as images of size 512x512

Example of Mel-frequency Spectrograms
<img src="https://github.com/Ferlix/Music-Genre-Classificator/blob/master/pre-processing/examples_preproc.png" width="500" align="middle">


### Results

| Model | Trainig accuracy | Test accuracy |
| --- | --- | --- |
| [MobileNet V2](https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4) (TL) | 99% | 70% |
| [Inception V3](https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4)(TL) | 99% | 83% |
| CNN | x | x |
| RNN | x | 0.6 |
| CRNN | x | x |
| SVM | x | x |

