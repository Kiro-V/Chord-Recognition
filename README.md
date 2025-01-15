# Chord Recognition
## Description
The project uses Template-based matching and Hidden Markov Model to recognize chords from audio files.

## Principle
### HMM
For the Hidden Markov Model to work, the parameters (transition probability) need to be estimiated (Estimation Problem). This can be solved using the Baum-Welch algorithm that takes into account the
observation sequence and the possible outgoing transitions

After the parameters are estimated, the Viterbi algorithm can be used to infer the most likely sequence of states given the observation sequence.

In general, this is a supervised learning problem where the states are the chords and the observations are the audio features.

Dataset used: Chorale Dataset.

### Template Matching

## Required dependencies

