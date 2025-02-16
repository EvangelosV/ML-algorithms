import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_length=100):
        """
        Μετατρέπει κάθε κείμενο (string) σε ακολουθία από δείκτες λέξεων, 
        χρησιμοποιώντας το λεξιλόγιο (0 για λέξεις που δεν υπάρχουν).
        Τα sequences παρεμβάλλονται ή περικόπττονται σε σταθερό max_length.
        """
        self.sequences = [self.text_to_sequence(text, vocab, max_length) for text in texts]
        self.labels = labels

    def text_to_sequence(self, text, vocab, max_length):
        tokens = text.split()  # Απλή tokenization – μπορείς να χρησιμοποιήσεις πιο προηγμένη επεξεργασία
        seq = [vocab.get(token, 0) for token in tokens]
        if len(seq) < max_length:
            seq = seq + [0] * (max_length - len(seq))
        else:
            seq = seq[:max_length]
        return torch.tensor(seq, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]
