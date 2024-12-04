import soundfile as sf
import librosa
import numpy as np

def load_audio(sample, min_len: float, max_len: float, sampling_rate: int, pad_to_min_length: bool = False):
    """ 
    Decodes audio from sample file into a waveform with the given
    `sampling rate` and a length in the intervall `[min_len, max_len]`.
    The exception to this is when the file of a sample is shorter
    than `min_len` and `pad_to_min_len` isn't set. In this case the
    returned audio will be the whole file.  
    If `pad_to_min_len` is set and a file is shorter than `min_len`
    then the returned audio will be symmetrically padded to `min_len`
    with its mean.

    Returns:
        Tuple: A tuple of audio waveform and sampling rate
    """

    path = sample["filepath"]

    file_info = sf.info(path)
    sr = file_info.samplerate
    total_duration = file_info.duration

    # convert all time information into sample index information for better accuracy
    original_min_len = min_len
    total_duration = int(total_duration * sr)
    min_len = int(min_len * sr)
    max_len = int(max_len * sr)


    if sample["detected_events"] is not None:
        start = int(sample["detected_events"][0] * sr)
        end = int(sample["detected_events"][1] * sr)
        event_duration = end - start

        if event_duration < min_len:
            extension = (min_len - event_duration) // 2
            
            # try to extend equally 
            new_start = max(0, start - extension)
            new_end = min(total_duration, end + extension)
            
            # check whether extending equally was successful
            if new_end - new_start < min_len:
                if new_end < total_duration:
                    # shift new_end further to a point where min_len is reached
                    new_end = min(total_duration, new_end + (min_len - (new_end - new_start)))
                if new_start > 0:
                    # shift new_start back to a point where min_len is reached
                    new_start = max(0, new_start - (min_len - (new_end - new_start)))

            start, end = new_start, new_end


        if end - start > max_len:
            # if longer than max_len
            end = min(start + max_len, total_duration)
            if end - start > max_len:
                end = start + max_len
    else:
        start = int(sample["start_time"] * sr)
        end = int(sample["end_time"] * sr)

        # cut audio if it's over max_len
        if (end - start) > max_len:
            to_cut = (end - start) - max_len
            equal_cut = to_cut // 2

            start += equal_cut
            end -= equal_cut

            # possible cutting needed due to integer division
            cut_left = (end - start) - max_len
            end -= cut_left


    audio, sr = sf.read(path, start=start, stop=end, dtype="float32")

    if audio.ndim != 1:
        audio = audio.swapaxes(1, 0)
        audio = librosa.to_mono(audio)
    if sr != sampling_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sampling_rate)
        sr = sampling_rate

    if (((end - start) < min_len) or total_duration < min_len) and pad_to_min_length:
        audio = pad_audio(audio, sr, original_min_len)

    return audio, sr


def pad_audio(audio, sample_rate: int, wanted_length: float):
    """
    Pads the given `audio` waveform to the `wanted_length` symmetrically
    with its mean.

    Returns:
        NDArray: padded audio waveform
    """
    len_to_pad = int(wanted_length * sample_rate - audio.size)

    if len_to_pad <= 0:
        return audio

    # divide padding into equal parts
    left_pad = len_to_pad // 2
    right_pad = len_to_pad // 2

    # make sure it is padded to the right size
    if left_pad + right_pad != len_to_pad:
        right_pad = right_pad + (len_to_pad - (left_pad + right_pad))
    return np.pad(audio, [(left_pad, right_pad)], mode=("mean" if audio.size != 0 else "empty"))