def clone_and_rename_wavs(root_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    i = 0

    for wav_path in collect_wav_paths(root_dir):
        parent_name = Path(wav_path).parents[3].name
        new_name = f"{parent_name}_utterance_{i}.wav"
        dest = Path(output_dir) / new_name
        shutil.copy2(wav_path, dest)
        i += 1

#clone_and_rename_wavs("data/libricss", "data/all_utterances_libricss")


def batch_process(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    extractor = SpectrogramExtractor()

    for wav_path in Path(input_dir).glob("*.wav"):
        spec = extractor(str(wav_path))
        out_path = Path(output_dir) / (wav_path.stem + ".pt")
        torch.save(spec, out_path)

#batch_process("data/all_utterances", "data/spectrograms")
