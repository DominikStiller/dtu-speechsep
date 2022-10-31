import os


def get_storage_dir():
    storage_dir = os.getenv("LIBRIMIX_STORAGE_DIR")
    if storage_dir is None:
        raise "Set LIBRIMIX_STORAGE_DIR before loading audio files"
    return storage_dir


def get_audio_path(example: str, dataset: str):
    storage_dir = get_storage_dir()
    return f"{storage_dir}/Libri2Mix/wav8k/min/{dataset}/mix_both/{example}.wav"
