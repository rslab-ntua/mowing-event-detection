from torch.nn.utils.rnn import pad_sequence
import torch
import random
def collate_fn(batch):

    sequences, labels = zip(*batch)
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)

    return padded_sequences, torch.tensor(labels, dtype=torch.long)
    
    
    


class ReverseTimeSeries(object):
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, sample):
        if random.random() < self.probability:
            return torch.flip(sample, dims=[0,1])
        return sample




