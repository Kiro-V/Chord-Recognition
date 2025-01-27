import numpy as np
import librosa
import pandas as pd
import os
from sklearn.preprocessing import normalize

# pip install libfmp
import libfmp.b
import libfmp.c3
import libfmp.c4

##################################################
#    Hidden Markov Model for Chord Recognition   #
##################################################

### Helpers functions ###

def split_labels(label):
    # Split the chord labels into root and quality
    if ':' not in label:
        # Take first letter as root or 2 letters if it's C#
        if '#' in label or 'b' in label:
            root = label[:2]
        else: 
            root = label[0]
        quality = 'maj'
    else:
        x = label.split(':')
        root = x[0]
        quality = x[1][:3]
    return [root, quality]

def enharmonic_compensation(label):
    # Compensate for enharmonic equivalence Eb = D#, B# = C, A# = Bb
    label = label.replace('B#', 'C')
    label = label.replace('A#', 'Bb')
    label = label.replace('Ab', 'G#')
    label = label.replace('Db', 'C#')
    label = label.replace('Eb', 'D#')
    return label

def bigram_counting(array):
    # Create a dictionary to store the bigram counts
    bigram_counts = {}
    
    # Loop through the array to count the bigrams
    for i in range(len(array) - 1):
        bigram = (array[i], array[i+1])
        if bigram in bigram_counts:
            bigram_counts[bigram] += 1
        else:
            bigram_counts[bigram] = 1
            
    return bigram_counts

def simplify_chords(chord):
    # Simplify the chord labels to [<root> <maj/min>]
    # <sus>, <maj> -> <maj>
    # <dim>, <min> -> <min>
    # all others -> <maj>
    if 'maj' in chord[1] or 'sus' in chord[1]:
        return chord[0] + 'maj'
    elif 'dim' or 'min' in chord[1]:
        return chord[0] + 'min'
    else:
        return chord[0] + 'maj'

def bigram_counting(array):
    # Create a dictionary to store the bigram counts
    bigram_counts = {}
    
    # Loop through the array to count the bigrams
    for i in range(len(array) - 1):
        bigram = (array[i], array[i+1])
        if bigram in bigram_counts:
            bigram_counts[bigram] += 1
        else:
            bigram_counts[bigram] = 1
            
    return bigram_counts

# Load chord label function
def load_chord_labels(file_path, sep=','):
    """
    Reading dataset csv file, [start_meas, end_meas, chord]

    Args:
        file_path (str): Path to the dataset
        sep (str): Delimiter used in the dataset (Default value = ',')

    Returns:
        chord_labels (pd.DataFrame): DataFrame containing the chord labels
    """
    # Load the chord labels
    chord_labels = pd.read_csv(file_path, sep=sep)
    chord_labels.columns = ['start_meas', 'end_meas', 'chord']

    # Clean up the chord labels
    chord_labels['chord'] = chord_labels['chord'].apply(lambda x: x.strip())

    # Discard rows with missing chord labels ('N')
    chord_labels = chord_labels[chord_labels['chord'] != 'N']

    # Simplify the chord labels to <root>:<quality[:2]>
    chord_labels['chord'] = chord_labels['chord'].apply(lambda x: split_labels(x))
    
    return chord_labels

def uniform_transition_matrix(p=0.01, N=24):
    """Computes uniform transition matrix

    Notebook: C5/C5S3_ChordRec_HMM.ipynb

    Args:
        p (float): Self transition probability (Default value = 0.01)
        N (int): Column and row dimension (Default value = 24)

    Returns:
        A (np.ndarray): Output transition matrix
    """
    off_diag_entries = (1-p) / (N-1)     # rows should sum up to 1
    A = off_diag_entries * np.ones([N, N])
    np.fill_diagonal(A, p)
    return A

####################################################
#                Viterbi Algorithm                 #
####################################################

def viterbi_log_likelihood(A, B_O, C=None):
    """Viterbi algorithm (log variant) for solving the uncovering problem

    Notebook: C5/C5S3_Viterbi.ipynb

    Args:
        A (np.ndarray): State transition probability matrix of dimension I x I
        C (np.ndarray): Initial state distribution  of dimension I
        B_O (np.ndarray): Likelihood matrix of dimension I x N

    Returns:
        S_opt (np.ndarray): Optimal state sequence of length N
        S_mat (np.ndarray): Binary matrix representation of optimal state sequence
        D_log (np.ndarray): Accumulated log probability matrix
        E (np.ndarray): Backtracking matrix
    """
    if C is None:
        # Assume C is uniform
        C = 1 / 24 * np.ones((1, 24))

    I = A.shape[0]    # Number of states
    N = B_O.shape[1]  # Length of observation sequence
    tiny = np.finfo(0.).tiny
    A_log = np.log(A + tiny)
    C_log = np.log(C + tiny)
    B_O_log = np.log(B_O + tiny)

    # Initialize D and E matrices
    D_log = np.zeros((I, N))
    E = np.zeros((I, N-1)).astype(np.int32)
    D_log[:, 0] = C_log + B_O_log[:, 0]

    # Compute D and E in a nested loop
    for n in range(1, N):
        for i in range(I):
            temp_sum = A_log[:, i] + D_log[:, n-1]
            D_log[i, n] = np.max(temp_sum) + B_O_log[i, n]
            E[i, n-1] = np.argmax(temp_sum)

    # Backtracking
    S_opt = np.zeros(N).astype(np.int32)
    S_opt[-1] = np.argmax(D_log[:, -1])
    for n in range(N-2, -1, -1):
        S_opt[n] = E[int(S_opt[n+1]), n]

    # Matrix representation of result
    S_mat = np.zeros((I, N)).astype(np.int32)
    for n in range(N):
        S_mat[S_opt[n], n] = 1

    return S_mat, S_opt, D_log, E