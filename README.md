# Chord Recognition
## Description
The project uses Template-based matching and Hidden Markov Model to recognize chords from audio files.

## Principle
### HMM
For the Hidden Markov Model to work, the parameters (transition probability) need to be estimiated (Estimation Problem). This can be solved using the Baum-Welch algorithm that takes into account the
observation sequence and the possible outgoing transitions

After the parameters are estimated, the Viterbi algorithm can be used to infer the most likely sequence of states given the observation sequence.

In general, this is a supervised learning problem where the states are the chords and the observations are the audio features.

### Dataset used: 

Chorale Dataset

Isophonics: http://isophonics.net

Piano Triads: https://www.kaggle.com/datasets/davidbroberts/piano-triads-wavset

### References:

Template-Based Chord Recognition: https://www.audiolabs-erlangen.de/resources/MIR/FMP/C5/C5S2_ChordRec_Templates.html

HMM on Chord Recognition: https://www.audiolabs-erlangen.de/resources/MIR/FMP/C5/C5S3_ChordRec_HMM.html

Viterbi Algorithm: https://www.audiolabs-erlangen.de/resources/MIR/FMP/C5/C5S3_Viterbi.html

Experiment on "Let it be": https://www.audiolabs-erlangen.de/resources/MIR/FMP/C5/C5S3_ChordRec_Beatles.html



