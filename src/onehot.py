import tensorflow as tf

alphabet = ' ABCDEFGHIJKLMNOPQRSTUVWXYZ'
alphabet_set = set(alphabet)
alphabet_size = len(alphabet)
reverse_dict_alphabet = {c:i for i,c in enumerate(alphabet)}

def char_to_index(char):
    return reverse_dict_alphabet[char]

def index_to_char(index):
    return alphabet[index]

def convert_to_chars(transcripts):
    """Converts a transcript (a string) to a sequence of indices"""
    converted = []
    for transcript in transcripts:
        transcript = transcript.upper()
        # Note: we currently don't encode any punctuation
        transcript = [char_to_index(c) for c in transcript if c in alphabet]
        converted.append(transcript)
    return converted

def decode_transcript(transcript):
    return ''.join(index_to_char(i) for i in transcript)

def decode_transcripts(transcripts):
    """Decodes a sequence of one-hot vectors into characters."""
    decoded_strings = []
    for transcript in transcripts:
        decoded_str = decode_transcript(transcript)
        decoded_strings.append(decoded_str)

    return decoded_strings

def decode_prob_transcripts(transcripts):
    """Decodes a sequence of one-hot vectors into characters."""
    decoded_strings = []
    for transcript in transcripts:
        transcript = tf.random.categorical(transcript, 1, dtype=tf.dtypes.int32)
        transcript = tf.squeeze(transcript, axis=-1)
        transcript = decode_transcript(transcript)
        decoded_strings.append(transcript)

    return decoded_strings

