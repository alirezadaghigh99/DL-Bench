import torch
from torch.utils.data import Dataset, DataLoader
import random
import string

class PytorchTextClassificationDataset(Dataset):
    def __init__(self, texts, labels, vocab, multi_label, target_labels):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.multi_label = multi_label
        self.target_labels = target_labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

def random_text_classification_dataset(num_samples=1000, max_length=50, num_classes=5, 
                                       multi_label=False, vocab_size=1000, device='cpu', 
                                       target_labels=None, dtype=torch.long):
    # Generate a random vocabulary
    vocab = [''.join(random.choices(string.ascii_lowercase, k=5)) for _ in range(vocab_size)]
    
    # Ensure target_labels are set
    if target_labels is None:
        target_labels = [f'class_{i}' for i in range(num_classes)]
    
    # Generate random text samples
    texts = []
    for _ in range(num_samples):
        text_length = random.randint(1, max_length)
        text = random.choices(vocab, k=text_length)
        texts.append(text)
    
    # Generate random labels
    labels = []
    if multi_label:
        for _ in range(num_samples):
            label = torch.zeros(num_classes, dtype=dtype)
            num_labels = random.randint(1, num_classes)
            chosen_labels = random.sample(range(num_classes), num_labels)
            label[chosen_labels] = 1
            labels.append(label)
    else:
        for _ in range(num_samples):
            label = random.randint(0, num_classes - 1)
            labels.append(label)
    
    # Ensure all labels are present in the dataset
    if not multi_label:
        for i in range(num_classes):
            if i not in labels:
                labels[random.randint(0, num_samples - 1)] = i
    
    # Convert texts to indices
    word_to_index = {word: idx for idx, word in enumerate(vocab)}
    texts_indices = [[word_to_index[word] for word in text] for text in texts]
    
    # Convert to tensors
    texts_tensor = [torch.tensor(text, dtype=dtype, device=device) for text in texts_indices]
    labels_tensor = torch.tensor(labels, dtype=dtype, device=device)
    
    # Create dataset
    dataset = PytorchTextClassificationDataset(texts_tensor, labels_tensor, vocab, multi_label, target_labels)
    
    return dataset

