import os
import random
import torch
import torchaudio
from sklearn.model_selection import train_test_split
from datasets import Dataset


def load_data_from_directory(root_dir):
    data = []
    for label, folder_name in enumerate(["fake", "real"]):
        folder_path = os.path.join(root_dir, folder_name)
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                data.append({"path": file_path, "label": label})
    random.shuffle(data)
    return Dataset.from_list(data)


def resample_audio(audio, sample_rate, target_sample_rate):
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        audio = resampler(audio)
    return audio


def load_batch_audios(batch, target_sample_rate):
    batch_audios = []
    for path in batch['path']:
        try:
            audio, sample_rate = torchaudio.load(path)
            audio = resample_audio(audio, sample_rate, target_sample_rate)
            batch_audios.append(audio)
        except Exception as e:
            print(f"Error loading audio file '{path}': {e}")
    return batch_audios


def preprocess(batch, processor, num_channels=1):
    target_sample_rate = processor.feature_extractor.sampling_rate
    batch_audios = load_batch_audios(batch, target_sample_rate)
    batch_size = len(batch_audios)
    sequence_length = batch_audios[0].shape[1]
    batch_audios = torch.stack(batch_audios).reshape(batch_size, num_channels, sequence_length).squeeze(1)
    inputs = processor(batch_audios, return_tensors="pt", padding=True, truncation=True,
                       sampling_rate=target_sample_rate, max_length=target_sample_rate)
    return {"input_values": inputs.input_values.squeeze(0), "labels": batch["label"]}


def analyze_dataset(dataset, dataset_name):
    total_samples = len(dataset)
    real_samples = sum([1 for sample in dataset if sample['labels'] == 1])
    fake_samples = total_samples - real_samples
    real_ratio = real_samples / total_samples
    fake_ratio = fake_samples / total_samples

    print(f"{dataset_name} dataset:")
    print(f"  Total samples: {total_samples}")
    print(f"  Real samples: {real_samples} ({real_ratio * 100:.2f}%)")
    print(f"  Fake samples: {fake_samples} ({fake_ratio * 100:.2f}%)\n")


def get_dataset(path_to_data, dataset_name, processor):
    root_dir = os.path.join(path_to_data, dataset_name)
    dataset = load_data_from_directory(root_dir)
    dataset = dataset.map(preprocess, fn_kwargs={'processor': processor}, batched=True, batch_size=8, remove_columns=["path", "label"])
    dataset.set_format(type="torch", columns=["input_values", "labels"])
    analyze_dataset(dataset, dataset_name)
    return dataset


def split_dataset(dataset, test_size=0.2, seed=42):
    train_indices, test_indices = train_test_split(range(len(dataset)), test_size=test_size, random_state=seed)
    return dataset.select(train_indices), dataset.select(test_indices)
