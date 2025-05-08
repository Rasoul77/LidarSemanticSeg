# This file is covered by the LICENSE file in the root of this project.

import datetime
import math
import shutil
import sys
import wandb

from collections import OrderedDict
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from augmentation import apply_augmentation
from data.data_loader import get_dataloaders
from eval import KNN
from eval.torch_ioueval import iouEval
from model.model_creator import get_segmentation_model
from utility.utils import *


class Trainer:
    """
    Handles the training, evaluation, and logging of a semantic segmentation model.

    Attributes:
        config (DotDict): Configuration for training, evaluation, and model parameters.
        model (torch.nn.Module): The segmentation model.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        loss_function (callable): Loss function or list of loss functions.
        dataloader_train (DataLoader): DataLoader for training data.
        dataloader_eval (DataLoader): DataLoader for evaluation data.
        scheduler (Scheduler, optional): Learning rate scheduler.
        scaler (GradScaler): Gradient scaler for mixed precision training.
        post_process (KNN, optional): Post-processing method using KNN.
        device (torch.device): CUDA or CPU.
    """

    def __init__(self, config: dict):
        """
        Initializes the Trainer class by setting up model, optimizer, dataloaders, etc.

        Args:
            config (dict): A dictionary containing configuration parameters.
        """
        
        self.config = DotDict(config)
        
        # Checkpoint Directory
        self.checkpoint_path = os.path.join(
            self.config.train.root_chkpt, datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        )
        os.makedirs(self.checkpoint_path)

        # Copy configurations to the checkpoint directory
        shutil.copy2(self.config.config_path, self.checkpoint_path)
        shutil.copy2(self.config.data_config_path, self.checkpoint_path)

        self.projected_predictions_dir = None
        if self.config.eval.save_projected_pred:
            self.projected_predictions_dir = os.path.join(self.checkpoint_path, "visualization")
            os.makedirs(self.projected_predictions_dir)

        # Logging
        if self.config.wandb.enabled:
            wandb.login()
            self.wandb_exp = wandb.init(
                project=f'Lidar Semantic Segmentation',
                name=self.config.wandb.project_name,
                resume='allow',
            )
            wandb_url = wandb.run.get_url()
            with open(os.path.join(self.checkpoint_path, "wandb_url.txt"), "w") as f:
                f.write(wandb_url)
                f.write("\n")
                f.write(self.config.wandb.project_name)

        # Seed and Reproducibility
        random_seed = get_seed(config)
        print(f"Random seed: {random_seed}")
        set_random_seed(random_seed, deterministic=self.config.train.deterministic)
        print(f"Deterministic: {self.config.train.deterministic}")
        self.np_rand_gen = np.random.default_rng()

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Segmentaiton Model
        self.model = get_segmentation_model(
            base_model=self.config.model.base,
            encoder_name=self.config.model.encoder,
            encoder_weights=self.config.model.weights,
            num_classes=self.config.model.num_classes,
            in_channels=self.config.model.in_channels,
            device=self.device,
        )

        if self.config.model.load:
            self.model.load_state_dict(torch.load(self.config.model.load))

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.train.learning_rate,
            weight_decay=self.config.train.weight_decay,
        )
        
        # Loss Function
        self.loss_function = get_loss_function(config)

        # Automatic Mixed Precision
        self.autocast = torch.cuda.amp.autocast(enabled=True, dtype=torch.float16)
        self.scaler = torch.cuda.amp.GradScaler(enabled=True)

        # Datasets
        self.dataloader_train, self.dataloader_eval = get_dataloaders(config)
        self.data_mean, self.data_std = get_mean_std_dev_train_set(config)

        # Rare-objects Database
        self.objects_db = get_objects_database(config)

        # Learning Rate Scheduler
        self.scheduler = None
        if self.config.train.scheduler.enabled:
            num_steps_per_epoch = len(self.dataloader_train) 
            warmup_steps = int(self.config.train.scheduler.num_warmup_epochs * num_steps_per_epoch)
            num_total_steps = self.config.train.num_epochs * num_steps_per_epoch            
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_total_steps,
                num_cycles=self.config.train.scheduler.num_cycles)

        # Post processing
        self.post_process = None
        if self.config.eval.post_process.enabled:
            self.post_process = KNN(
                nclasses=self.config.model.num_classes,
                knn=self.config.eval.knn.num_nearest,
                search=self.config.eval.knn.search,
                sigma=self.config.eval.knn.sigma,
                cutoff=self.config.eval.knn.cut_off,
            )
    
        # Color map
        learning_map_inv = config['data']['kitti']['learning_map_inv']
        learning_map = config['data']['kitti']['learning_map']
        self.color_map = create_semantic_color_map(
            self.config.data.kitti.color_map.to_dict(), learning_map, learning_map_inv
        )

        self.best_score = -1
        self.best_loss = float('inf')
        self.epoch_idx = 1        

    def train_one_epoch(self) -> float:
        """
        Trains the model for a single epoch.

        Returns:
            float: Running average of the training loss over the epoch.
        """
        
        self.model.train()
        epoch_loss = 0.0
        tqdm_desc = f"Train Epoch {self.epoch_idx:02}/{self.config.train.num_epochs}"
        with tqdm(self.dataloader_train, leave=False, desc=tqdm_desc) as train_bar:
            for batch_idx, data in enumerate(train_bar):
                images = data['image']
                masks = data['mask']

                images, masks = apply_augmentation(
                    images=images,
                    masks=masks,
                    objects_db=self.objects_db,
                    rng=self.np_rand_gen,
                    rare_classes=self.config.data.kitti.rare_classes.to_dict(),
                    num_objects_per_cat=self.config.train.aug.num_objects_per_cat,
                    aug_prob=self.config.train.aug.prob,
                    instance_injection_prob=self.config.train.aug.inst_prob,
                    cut_mix_prob=self.config.train.aug.cutmix_prob,
                )
                    
                images = (images - self.data_mean[None, None, None, :]) / self.data_std[None, None, None, :]

                images = images.permute(0, 3, 1, 2).to(self.device)
                masks = masks.long().to(self.device)

                self.optimizer.zero_grad()

                with self.autocast:
                    logits = self.model(images)
                    if isinstance(self.loss_function, list):
                        batch_loss = sum([lf(logits, masks) * weight for lf, weight in self.loss_function])
                    else:
                        batch_loss = self.loss_function(logits, masks)

                    if not math.isfinite(batch_loss):
                        print(f"\tBatch loss is {batch_loss}, stopping training")
                        sys.exit(1)                

                    epoch_loss += batch_loss.item()

                self.scaler.scale(batch_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1e9)

                self.scaler.step(self.optimizer)
                self.scaler.update()

                if self.scheduler is not None:
                    self.scheduler.step()
                
                running_loss = epoch_loss / (batch_idx + 1)
                running_lr = self.optimizer.param_groups[0]["lr"]

                train_bar.set_postfix(
                    OrderedDict(
                        running_loss=f'{running_loss:.4f}',
                        lr=f'{running_lr:.3e}'
                    )
                )

                if self.config.wandb.enabled:
                    self.wandb_exp.log({
                        'batch loss': batch_loss.item(),
                        'running loss': running_loss,
                        'epoch': self.epoch_idx,
                        'learning rate': running_lr,
                    })

        return running_loss

    def eval_one_epoch(self) -> Tuple[float, float, np.ndarray]:
        """
        Evaluates the model for a single epoch.

        Returns:
            Tuple[float, float, np.ndarray]:
                - Running evaluation loss
                - Mean IoU score
                - Per-class IoU scores
        """
        
        self.model.eval()
        epoch_loss = 0.0        
        evaluator = iouEval(self.config.model.num_classes, ignore=[0])
        tqdm_desc = f"Eval Epoch {self.epoch_idx:02}/{self.config.train.num_epochs}"
        with tqdm(self.dataloader_eval, leave=False, desc=tqdm_desc) as eval_bar:
            with torch.no_grad():
                for idx, data in enumerate(eval_bar):
                    batchsize = int(data['image'].shape[0])
                    images = data['image']
                    masks = data['mask']

                    if self.config.eval.post_process:
                        proj_range = images[..., 0].cpu()

                    images = (images - self.data_mean[None, None, None, :]) / self.data_std[None, None, None, :]

                    images = images.permute(0, 3, 1, 2).to(self.device)
                    masks = masks.long().to(self.device)
                    
                    with self.autocast:
                        logits = self.model(images)
                        if isinstance(self.loss_function, list):
                            batch_loss = sum([lf(logits, masks) * weight for lf, weight in self.loss_function])
                        else:
                            batch_loss = self.loss_function(logits, masks)
                        
                        epoch_loss += batch_loss.item()

                    running_loss = epoch_loss / (idx + 1)

                    eval_bar.set_postfix(
                        OrderedDict(
                            running_loss=f'{running_loss:.4f}',
                        )
                    )

                    pred_mask = torch.argmax(logits, dim=1).cpu()
                    
                    for i in range(batchsize):
                        num_points = int(data['num_points'][i].numpy())
                        per_point_label = data['flat_label'][i, 0:num_points].numpy()
                        if self.config.eval.post_process.enabled:
                            per_point_pred = self.post_process(
                                proj_range = proj_range[i, ...],
                                unproj_range = data['unproj_range'][i, :num_points],
                                proj_argmax=pred_mask[i, ...],
                                px=data['p_x'][i, :num_points],
                                py=data['p_y'][i, :num_points],
                            ).numpy()
                        else:
                            point_index_mask = data['proj_idx'][i, :].numpy()                    
                            per_point_pred = get_per_point_pred(
                                pred_mask=pred_mask[i, ...].numpy(),
                                point_index_mask=point_index_mask,
                                num_points=num_points,
                            )                        
                        evaluator.addBatch(per_point_pred, per_point_label)

                        if self.config.eval.save_projected_pred:
                            sequence = f"{int(data['sequence'][i]):02}"
                            frame_number = data['frame_number'][i]
                            save_proj_pred(
                                pred_mask[i, ...],
                                os.path.join(
                                    self.projected_predictions_dir, f"{sequence}_{frame_number}.png"
                                ),
                                semantic_color_map=self.color_map
                            )

        avg_score, per_class_score = evaluator.getIoU()
        return running_loss, avg_score.item(), per_class_score.cpu().numpy()

    def run(self) -> None:
        """
        Executes the full training and evaluation loop across all epochs.
        """
        
        total_loss = 0.0
        for self.epoch_idx in range(1, self.config.train.num_epochs + 1):

            start_time = time.time()
            epoch_loss = self.train_one_epoch()            
            train_epoch_time = get_formatted_time_execution(start_time)

            total_loss += epoch_loss

            start_time = time.time()
            eval_loss, eval_score, per_class_score = self.eval_one_epoch()
            eval_epoch_time = get_formatted_time_execution(start_time)

            if self.config.wandb.enabled:
                self.wandb_exp.log({
                    'eval loss': eval_loss,
                    'eval score': eval_score,
                })

            print(f"Epoch{self.epoch_idx:02}/{self.config.train.num_epochs}"
                f" | Train > Epoch Loss: {epoch_loss:.4f} Time: {train_epoch_time}"
                f" | Eval > Epoch Loss: {eval_loss:.4f} SCORE: {eval_score*100.0:.2f}% Time: {eval_epoch_time}"
            ) 

            self.save_best_model(
                eval_loss=eval_loss,
                eval_score=eval_score,
                per_class_score=per_class_score,
            )

    def save_best_model(self, eval_loss: float, eval_score: float, per_class_score: np.ndarray) -> None:
        """
        Saves the best model based on evaluation loss and score.

        Args:
            eval_loss (float): Current epoch evaluation loss.
            eval_score (float): Current epoch evaluation score.
            per_class_score (np.ndarray): Per-class IoU scores.
        """
        
        best_eval_loss = read_number_from_file("eval_loss", self.checkpoint_path, default=np.inf)
        best_eval_score = read_number_from_file("eval_score", self.checkpoint_path, default=0.0)        
        
        to_print = f"Epoch{self.epoch_idx:02}/{self.config.train.num_epochs} Best Eval"
        if eval_loss < best_eval_loss:
            to_print += f" | Loss : {best_eval_loss:.4f} -> {eval_loss:.4f}"
            save_model(self.model, "best_val_loss_model.pt", self.checkpoint_path)
            save_number_to_file("eval_loss", self.checkpoint_path, eval_loss)

        if best_eval_score < eval_score:
            to_print += f" | Score : {best_eval_score*100.0:.2f}% -> {eval_score*100.0:.2f}%"
            save_model(self.model, "best_val_score_model.pt", self.checkpoint_path)
            save_number_to_file("eval_score", self.checkpoint_path, eval_score)
            save_per_class_score(self.checkpoint_path, per_class_score, self.config)

        if eval_loss < best_eval_loss or best_eval_score < eval_score:
            print(to_print)
