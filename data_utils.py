from music_utils import * 
from preprocess import * 
from tensorflow.keras.utils import to_categorical

from collections import defaultdict
from mido import MidiFile
from pydub import AudioSegment
from pydub.generators import Sine
import math

#chords, abstract_grammars = get_musical_data('data/original_metheny.mid')
#corpus, tones, tones_indices, indices_tones = get_corpus_data(abstract_grammars)
#N_tones = len(set(corpus))
n_a = 64
x_initializer = np.zeros((1, 1, 90))
a_initializer = np.zeros((1, n_a))
c_initializer = np.zeros((1, n_a))

def load_music_utils(file):
    chords, abstract_grammars = get_musical_data(file)
    corpus, tones, tones_indices, indices_tones = get_corpus_data(abstract_grammars)
    N_tones = len(set(corpus))
    X, Y, N_tones = data_processing(corpus, tones_indices, 60, 30)   
    return (X, Y, N_tones, indices_tones, chords)


def generate_music(inference_model, indices_tones, chords, diversity = 0.5):
    """
    Generates music using a model trained to learn musical patterns of a jazz soloist. Creates an audio stream
    to save the music and play it.
    
    Arguments:
    model -- Keras model Instance, output of djmodel()
    indices_tones -- a python dictionary mapping indices (0-77) into their corresponding unique tone (ex: A,0.250,< m2,P-4 >)
    temperature -- scalar value, defines how conservative/creative the model is when generating music
    
    Returns:
    predicted_tones -- python list containing predicted tones
    """
    
    # set up audio stream
    out_stream = stream.Stream()
    
    # Initialize chord variables
    curr_offset = 0.0                                     # variable used to write sounds to the Stream.
    num_chords = int(len(chords) / 3)                     # number of different set of chords
    
    print("Predicting new values for different set of chords.")
    # Loop over all 18 set of chords. At each iteration generate a sequence of tones
    # and use the current chords to convert it into actual sounds 
    for i in range(1, num_chords):
        
        # Retrieve current chord from stream
        curr_chords = stream.Voice()
        
        # Loop over the chords of the current set of chords
        for j in chords[i]:
            # Add chord to the current chords with the adequate offset, no need to understand this
            curr_chords.insert((j.offset % 4), j)
        
        # Generate a sequence of tones using the model
        _, indices = predict_and_sample(inference_model)
        indices = list(indices.squeeze())
        pred = [indices_tones[p] for p in indices]
        
        predicted_tones = 'C,0.25 '
        for k in range(len(pred) - 1):
            predicted_tones += pred[k] + ' ' 
        
        predicted_tones +=  pred[-1]
                
        #### POST PROCESSING OF THE PREDICTED TONES ####
        # We will consider "A" and "X" as "C" tones. It is a common choice.
        predicted_tones = predicted_tones.replace(' A',' C').replace(' X',' C')

        # Pruning #1: smoothing measure
        predicted_tones = prune_grammar(predicted_tones)
        
        # Use predicted tones and current chords to generate sounds
        sounds = unparse_grammar(predicted_tones, curr_chords)

        # Pruning #2: removing repeated and too close together sounds
        sounds = prune_notes(sounds)

        # Quality assurance: clean up sounds
        sounds = clean_up_notes(sounds)

        # Print number of tones/notes in sounds
        print('Generated %s sounds using the predicted values for the set of chords ("%s") and after pruning' % (len([k for k in sounds if isinstance(k, note.Note)]), i))
        
        # Insert sounds into the output stream
        for m in sounds:
            out_stream.insert(curr_offset + m.offset, m)
        for mc in curr_chords:
            out_stream.insert(curr_offset + mc.offset, mc)

        curr_offset += 4.0
        
    # Initialize tempo of the output stream with 130 bit per minute
    out_stream.insert(0.0, tempo.MetronomeMark(number=130))

    # Save audio stream to fine
    mf = midi.translate.streamToMidiFile(out_stream)
    mf.open("output/my_music.midi", 'wb')
    mf.write()
    print("Your generated music is saved in output/my_music.midi")
    mf.close()
    
    
    
    return out_stream


def predict_and_sample(inference_model, x_initializer = x_initializer, a_initializer = a_initializer, 
                       c_initializer = c_initializer):
    """
    Predicts the next value of values using the inference model.
    
    Arguments:
    inference_model -- Keras model instance for inference time
    x_initializer -- numpy array of shape (1, 1, 78), one-hot vector initializing the values generation
    a_initializer -- numpy array of shape (1, n_a), initializing the hidden state of the LSTM_cell
    c_initializer -- numpy array of shape (1, n_a), initializing the cell state of the LSTM_cel
    Ty -- length of the sequence you'd like to generate.
    
    Returns:
    results -- numpy-array of shape (Ty, 78), matrix of one-hot vectors representing the values generated
    indices -- numpy-array of shape (Ty, 1), matrix of indices representing the values generated
    """
    
    #Predicts the next value using the inference model
    pred = inference_model.predict([x_initializer, a_initializer, c_initializer])
    #Finds the index of the highest probability value along the last axis of the predictions
    indices = np.argmax(pred, axis = -1)
    #Converts the indices to one-hot vectors
    results = to_categorical(indices, num_classes=90)
    return results, indices


def note_to_freq(note, concert_A=440.0):
  '''
  This function implements the standard formula for converting MIDI note numbers
  to frequencies, following the MIDI Tuning Standard. The formula calculates the
    frequency based on the distance from A4 (MIDI note 69), which has a default
    frequency of 440 Hz.

    Args:
        note (int or float): The MIDI note number to convert. Standard MIDI notes
            range from 0 to 127, where 60 is middle C (C4). Fractional values can
            be used for notes that fall between semitones (e.g., for pitch bends).
        concert_A (float, optional): The frequency in Hz to use for A4 (MIDI note 69).
            Defaults to 440.0 Hz, which is the modern standard concert pitch.

    Returns:
        float: The frequency in Hz corresponding to the given MIDI note number.
  from wikipedia: http://en.wikipedia.org/wiki/MIDI_Tuning_Standard#Frequency_values
  '''
  return (2.0 ** ((note - 69) / 12.0)) * concert_A

def ticks_to_ms(ticks, tempo, mid):
    """
    Converts MIDI ticks to milliseconds based on tempo and time division.

    This function calculates the duration in milliseconds corresponding to a given
    number of MIDI ticks. The conversion depends on the current tempo (in beats per
    minute) and the ticks per beat value defined in the MIDI file.

    Args:
        ticks (int): The number of MIDI ticks to convert to milliseconds.
        tempo (float): The tempo in beats per minute (BPM).
        mid (MidiFile): A MIDI file object that contains the ticks_per_beat attribute,
            which defines the time division of the MIDI file.

    Returns:
        int: The duration in milliseconds, rounded up to the nearest integer.
    """
    tick_ms = math.ceil((60000.0 / tempo) / mid.ticks_per_beat)
    return ticks * tick_ms

def mid2wav(file):
    """
    Converts a MIDI file to a WAV audio file.

    This function reads a MIDI file, processes its tracks and notes, and renders
    them as sine waves to create an audio representation of the MIDI data. The
    resulting audio is exported as a WAV file.

    Args:
        file (str): Path to the input MIDI file.

    Returns:
        None: The function exports the rendered audio directly to a WAV file
        at "./output/rendered.wav" and doesn't return a value.

    """
    mid = MidiFile(file)
    output = AudioSegment.silent(mid.length * 1000.0)

    tempo = 130 # bpm

    for track in mid.tracks:
        # position of rendering in ms
        current_pos = 0.0
        current_notes = defaultdict(dict)

        for msg in track:
            current_pos += ticks_to_ms(msg.time, tempo, mid)
            if msg.type == 'note_on':
                if msg.note in current_notes[msg.channel]:
                    current_notes[msg.channel][msg.note].append((current_pos, msg))
                else:
                    current_notes[msg.channel][msg.note] = [(current_pos, msg)]


            if msg.type == 'note_off':
                start_pos, start_msg = current_notes[msg.channel][msg.note].pop()

                duration = math.ceil(current_pos - start_pos)
                signal_generator = Sine(note_to_freq(msg.note, 500))
                #print(duration)
                rendered = signal_generator.to_audio_segment(duration=duration-50, volume=-20).fade_out(100).fade_in(30)

                output = output.overlay(rendered, start_pos)

    output.export("./output/rendered.wav", format="wav")