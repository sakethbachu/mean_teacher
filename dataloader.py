import numpy as np
import pandas as pd

class CustomDataLoader:
    def __init__(self, X_labeled, y_labeled, X_unlabeled, batch_size=100):
        self.X_labeled = X_labeled
        self.y_labeled = y_labeled
        self.X_unlabeled = X_unlabeled
        self.batch_size = batch_size
        
        self.labeled_size = len(X_labeled)
        self.unlabeled_size = len(X_unlabeled)
        
        self.num_batches = max(np.ceil(self.labeled_size / batch_size), np.ceil(self.unlabeled_size / batch_size))
        
        self.labeled_indices = np.arange(self.labeled_size)
        self.unlabeled_indices = np.arange(self.unlabeled_size)
        
        np.random.shuffle(self.labeled_indices)
        np.random.shuffle(self.unlabeled_indices)
        
    def __iter__(self):
        self.labeled_pointer = 0
        self.unlabeled_pointer = 0
        return self
    
    def __next__(self):
        if self.labeled_pointer >= self.labeled_size and self.unlabeled_pointer >= self.unlabeled_size:
            np.random.shuffle(self.labeled_indices)
            np.random.shuffle(self.unlabeled_indices)
            self.labeled_pointer = 0
            self.unlabeled_pointer = 0
            raise StopIteration
        
        labeled_end = min(self.labeled_pointer + self.batch_size, self.labeled_size)
        unlabeled_end = min(self.unlabeled_pointer + self.batch_size, self.unlabeled_size)
        
        labeled_batch_indices = self.labeled_indices[self.labeled_pointer:labeled_end]
        unlabeled_batch_indices = self.unlabeled_indices[self.unlabeled_pointer:unlabeled_end]
        
        X_l_batch = self.X_labeled[labeled_batch_indices]
        y_l_batch = self.y_labeled[labeled_batch_indices]
        X_u_batch = self.X_unlabeled[unlabeled_batch_indices]
        
        if len(X_l_batch) < self.batch_size:
            additional_indices = np.random.choice(self.labeled_indices, self.batch_size - len(X_l_batch), replace=True)
            X_l_batch = np.vstack([X_l_batch, self.X_labeled[additional_indices]])
            y_l_batch = np.concatenate([y_l_batch, self.y_labeled[additional_indices]])
        
        if len(X_u_batch) < self.batch_size:
            additional_indices = np.random.choice(self.unlabeled_indices, self.batch_size - len(X_u_batch), replace=True)
            X_u_batch = np.vstack([X_u_batch, self.X_unlabeled[additional_indices]])
        
        self.labeled_pointer += self.batch_size
        self.unlabeled_pointer += self.batch_size
        
        return (X_l_batch, y_l_batch), X_u_batch

# Example usage
# Generate example labeled and unlabeled datasets
X_labeled = np.random.randn(150, 20)  # 150 labeled examples, 20 features
y_labeled = np.random.randint(0, 2, 150)  # 150 labeled examples
X_unlabeled = np.random.randn(120, 20)  # 120 unlabeled examples, 20 features

# Create the dataloader
dataloader = CustomDataLoader(X_labeled, y_labeled, X_unlabeled, batch_size=100)

# Iterate through the batches
for (X_l_batch, y_l_batch), X_u_batch in dataloader:
    print(f"Labeled batch shape: {X_l_batch.shape}, {y_l_batch.shape}")
    print(f"Unlabeled batch shape: {X_u_batch.shape}")
