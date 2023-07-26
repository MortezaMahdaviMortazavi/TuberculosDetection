import torch
import config
import torchvision.transforms as transforms

from PIL import Image
from preprocessing import make_dataframes
from torch.utils.data import Dataset,DataLoader

class TBDataset(Dataset):
    def __init__(self, data_df, transform=None):
        self.data_df = data_df
        self.transform = transform if transform is not None else self._default_transform()
        self.class_labels = {'Normal': 0, 'Tuberculosis': 1}

    def __repr__(self):
        return f"TBDataset: Number of samples: {len(self)}"

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        filepath = self.data_df.iloc[index, 0]
        label = self.data_df.iloc[index, 1]

        # Load image using PIL
        image = Image.open(filepath)

        if self.transform:
            image = self.transform(image)

        return image, label
    
    def _default_transform(self):
        return transforms.Compose([
            transforms.Lambda(lambda x: x.convert('RGB')),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    

def create_dataloader(train_transform,valid_transform,test_transform):
    train_df, test_df, valid_df, class_count, average_height, average_weight, aspect_ratio = make_dataframes(config.DATASET_DIR)
    train_dataset = TBDataset(train_df, transform=train_transform)
    valid_dataset = TBDataset(valid_df, transform=valid_transform)
    test_dataset = TBDataset(test_df, transform=test_transform)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )

    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )

    return train_dataloader,valid_dataloader,test_dataloader