import numpy as np
import librosa

# pip install libfmp
import libfmp.b
import libfmp.c3
import libfmp.c4

#################################################
#   Template-based Chord Recognition Matching   #
#################################################

def compute_chromagram_from_filename(fn_wav, Fs=22050, N=4096, H=2048, gamma=None, version='STFT', norm='2', start=0, dur = None):
    """Compute chromagram for WAV file specified by filename

    Notebook: C5/C5S2_ChordRec_Templates.ipynb

    Args:
        fn_wav (str): Filenname of WAV
        Fs (scalar): Sampling rate (Default value = 22050)
        N (int): Window size (Default value = 4096)
        H (int): Hop size (Default value = 2048)
        gamma (float): Constant for logarithmic compression (Default value = None)
        version (str): Technique used for front-end decomposition ('STFT', 'IIS', 'CQT') (Default value = 'STFT')
        norm (str): If not 'None', chroma vectors are normalized by norm as specified ('1', '2', 'max')
            (Default value = '2')

    Returns:
        X (np.ndarray): Chromagram
        Fs_X (scalar): Feature reate of chromagram
        x (np.ndarray): Audio signal
        Fs (scalar): Sampling rate of audio signal
        x_dur (float): Duration (seconds) of audio signal
    """
    x, Fs = librosa.load(fn_wav, sr=Fs)
    if dur is not None:
        x = x[start*Fs:(start+dur)*Fs]
    x_dur = x.shape[0] / Fs
    if version == 'STFT':
        # Compute chroma features with STFT
        X = librosa.stft(x, n_fft=N, hop_length=H, pad_mode='constant', center=True)
        if gamma is not None:
            X = np.log(1 + gamma * np.abs(X) ** 2)
        else:
            X = np.abs(X) ** 2
        X = librosa.feature.chroma_stft(S=X, sr=Fs, tuning=0, norm=None, hop_length=H, n_fft=N)
    if version == 'CQT':
        # Compute chroma features with CQT decomposition
        X = librosa.feature.chroma_cqt(y=x, sr=Fs, hop_length=H, norm=None)
    if version == 'IIR':
        # Compute chroma features with filter bank (using IIR elliptic filter)
        X = librosa.iirt(y=x, sr=Fs, win_length=N, hop_length=H, center=True, tuning=0.0)
        if gamma is not None:
            X = np.log(1.0 + gamma * X)
        X = librosa.feature.chroma_cqt(C=X, bins_per_octave=12, n_octaves=7,
                                       fmin=librosa.midi_to_hz(24), norm=None)
    if norm is not None:
        X = libfmp.c3.normalize_feature_sequence(X, norm=norm)
    Fs_X = Fs / H
    return X, Fs_X, x, Fs, x_dur

def plot_chromagram_annotation(ax, X, Fs_X, ann=None, color_ann=None, x_dur=None, cmap='gray_r', title=''):
    """Plot chromagram and annotation

    Notebook: C5/C5S2_ChordRec_Templates.ipynb

    Args:
        ax: Axes handle
        X: Feature representation
        Fs_X: Feature rate
        ann: Annotations (Default value = None)
        color_ann: Color for annotations (Default value = None)
        x_dur: Duration of feature representation (Default value = None)
        cmap: Color map for imshow (Default value = 'gray_r')
        title: Title for figure (Default value = '')
    """
    libfmp.b.plot_chromagram(X, Fs=Fs_X, ax=ax,
                             chroma_yticks=[0, 4, 7, 11], clim=[0, 1], cmap=cmap,
                             title=title, ylabel='Chroma', colorbar=True)
    if ann != None:
        libfmp.b.plot_segments_overlay(ann=ann, ax=ax[0], time_max=x_dur,
                                    print_labels=False, colors=color_ann, alpha=0.1)

def get_chord_labels(ext_minor='m', nonchord=False):
    """Generate chord labels for major and minor triads (and possibly nonchord label)

    Notebook: C5/C5S2_ChordRec_Templates.ipynb

    Args:
        ext_minor (str): Extension for minor chords (Default value = 'm')
        nonchord (bool): If "True" then add nonchord label (Default value = False)

    Returns:
        chord_labels (list): List of chord labels
    """
    chroma_labels = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    chord_labels_maj = chroma_labels
    chord_labels_min = [s + ext_minor for s in chroma_labels]
    chord_labels = chord_labels_maj + chord_labels_min
    if nonchord is True:
        chord_labels = chord_labels + ['N']
    return chord_labels

def generate_chord_templates(nonchord=False):
    """Generate chord templates of major and minor triads (and possibly nonchord)

    Notebook: C5/C5S2_ChordRec_Templates.ipynb

    Args:
        nonchord (bool): If "True" then add nonchord template (Default value = False)

    Returns:
        chord_templates (np.ndarray): Matrix containing chord_templates as columns
    """
    template_cmaj = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]).T
    template_cmin = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]).T
    num_chord = 24
    if nonchord:
        num_chord = 25
    chord_templates = np.ones((12, num_chord))
    for shift in range(12):
        chord_templates[:, shift] = np.roll(template_cmaj, shift)
        chord_templates[:, shift+12] = np.roll(template_cmin, shift)
    return chord_templates

def chord_recognition_template(X, norm_sim='1', nonchord=False, templates=None):
    """Conducts template-based chord recognition
    with major, minor chords

    Notebook: C5/C5S2_ChordRec_Templates.ipynb

    Args:
        X (np.ndarray): Chromagram
        norm_sim (str): Specifies norm used for normalizing chord similarity matrix (Default value = '1')
        nonchord (bool): If "True" then add nonchord template (Default value = False)

    Returns:
        chord_sim (np.ndarray): Chord similarity matrix
        chord_max (np.ndarray): Binarized chord similarity matrix only containing maximizing chord
    """
    if templates is not None:
        chord_templates = templates
    else:
        chord_templates = generate_chord_templates(nonchord=nonchord)
    X_norm = libfmp.c3.normalize_feature_sequence(X, norm='2')
    chord_templates_norm = libfmp.c3.normalize_feature_sequence(chord_templates, norm='2')
    chord_sim = np.matmul(chord_templates_norm.T, X_norm)
    if norm_sim is not None:
        chord_sim = libfmp.c3.normalize_feature_sequence(chord_sim, norm=norm_sim)
    # chord_max = (chord_sim == chord_sim.max(axis=0)).astype(int)
    chord_max_index = np.argmax(chord_sim, axis=0)
    chord_max = np.zeros(chord_sim.shape).astype(np.int32)
    for n in range(chord_sim.shape[1]):
        chord_max[chord_max_index[n], n] = 1

    return chord_sim, chord_max

def generate_averaged_templates():
    '''
    The following code uses the Piano Triads Wavset dataset to generate templates by averaging the chrome values for each chord (https://www.kaggle.com/datasets/davidbroberts/piano-triads-wavset).
    '''
    standard_notes = np.array(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])

    triad_audios = librosa.util.find_files('piano_triads/')
    triad_audios = np.asarray(triad_audios)

    notes = np.array(['C', 'Cs', 'D', 'Eb', 'E', 'F', 'Fs', 'G', 'Gs', 'A', 'Bb', 'B']) # notation used in the dataset

    templates = np.empty((0, 12))
    #fig, ax = plt.subplots(8, figsize=(10,30))

    # major chords:
    for note in notes:
        # major chords:
        matching_files = triad_audios[np.char.find(triad_audios, note + "_maj") != -1]
        template = np.zeros(12)
        for i, file in enumerate(matching_files[4:]):   # discard lower octaves because they look like noise
            x, Fs = librosa.load(file)
            S = np.abs(librosa.stft(x[:int(1.5*Fs)]))
            chroma = librosa.feature.chroma_stft(S=S, sr=Fs, norm=None)
            template += np.sum(chroma, axis=-1)
            #ax[i].set_title(file)
            #img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax[i])
        template /= np.max(template)
        templates = np.vstack((templates, template))

    # minor chords:
    for note in notes:
        matching_files = triad_audios[np.char.find(triad_audios, note + "_min") != -1]
        template = np.zeros(12)
        for i, file in enumerate(matching_files[4:]):   # discard lower octaves because they look like noise
            x, Fs = librosa.load(file)
            S = np.abs(librosa.stft(x[:int(1.5*Fs)]))
            chroma = librosa.feature.chroma_stft(S=S, sr=Fs, norm=None)
            template += np.sum(chroma, axis=-1)
            #ax[i].set_title(file)
            #img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax[i])
        template /= np.max(template)
        templates = np.vstack((templates, template))

    

    #with open('example_templates.csv', 'w', newline='') as csvfile:
    #    writer = csv.writer(csvfile, delimiter=',')
    #    writer.writerow(standard_notes)
    #    writer.writerows(templates)

    return templates