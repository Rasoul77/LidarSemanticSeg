# This file is covered by the LICENSE file in the root of this project.

from data.lidar_dataset import SemanticKittiDataset

from torch.utils.data.dataloader import DataLoader


def get_dataloaders(config: dict):
    """
    Initializes and returns PyTorch DataLoader instances for training and evaluation datasets.

    The function creates two SemanticKittiDataset instances:
    - One for training with optional ground truth exclusion (`include_full_gt=False`)
    - One for evaluation with full ground truth included (`include_full_gt=True`)

    It then wraps each dataset in a DataLoader with options defined in the configuration dictionary.

    Args:
        config (dict): A configuration dictionary that must contain:
            - 'data.kitti.train_split': list of sequences for training
            - 'data.kitti.valid_split': list of sequences for validation
            - 'train.batch_size': batch size for training
            - 'train.num_workers': number of worker threads for training DataLoader
            - 'eval.batch_size': batch size for evaluation
            - 'eval.num_workers': number of worker threads for evaluation DataLoader

    Returns:
        Tuple[DataLoader, DataLoader]: A tuple containing:
            - dataloader_train: DataLoader for training data
            - dataloader_eval: DataLoader for evaluation data
    """
    
    train_ds = SemanticKittiDataset(
        config=config,
        sequences=config['data']['kitti']['train_split'],
        include_full_gt=False,        
    )
    dataloader_train = DataLoader(
        train_ds,
        shuffle=True,
        batch_size=config['train']['batch_size'],
        drop_last=True,
        num_workers=config['train']['num_workers'],
    )

    eval_ds = SemanticKittiDataset(
        config=config,
        sequences=config['data']['kitti']['valid_split'],
        include_full_gt=True,
    )
    dataloader_eval = DataLoader(
        eval_ds,
        shuffle=False,
        batch_size=config['eval']['batch_size'],
        drop_last=False,
        num_workers=config['eval']['num_workers'],
    )

    return dataloader_train, dataloader_eval


def get_infer_dataloader(config: dict):
    """
    Prepares Inference data loader.
    """
    infer_ds = SemanticKittiDataset(
        config=config,
        sequences=config['data']['kitti']['test_split'],
        include_full_gt=True,
    )
    dataloader_infer = DataLoader(
        infer_ds,
        shuffle=False,
        batch_size=config['infer']['batch_size'],
        drop_last=False,
        num_workers=config['infer']['num_workers'],
    )

    return dataloader_infer
