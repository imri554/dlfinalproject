from pathlib import Path

from audio import load_audio_file

def read_transcript(path):
    transcripts = []
    with open(path) as transcript_file:
        for line in transcript_file:
            space = line.find(' ')

            name = line[:space]
            text = line[space+1:]
            transcripts.append((name, text))
    return transcripts

def get_librispeech_data(data_path):
    """Returns labeled data"""
    data_path = Path(data_path)
    assert data_path.is_dir()

    all_names = []
    all_audio = []
    all_transcripts = []

    for book_path in data_path.iterdir():
        print(f'Loading book {book_path}')
        for chapter_path in book_path.iterdir():
            # Read transcript file
            transcript_file = next(chapter_path.glob('*.trans.txt'))
            transcripts = read_transcript(transcript_file)

            for name, text in transcripts:
                # Read sound clip
                sound_clip_path = chapter_path / f'{name}.flac'
                audio_data, sample_rate = load_audio_file(sound_clip_path)

                # Going to assume that all clips have the same sample rate
                assert sample_rate == 16000

                all_names.append(name)
                all_audio.append(audio_data)
                all_transcripts.append(text)

    return all_names, all_audio, all_transcripts
