import argparse
import numpy as np
import torch
import random
import os
import time
import clip
from torch.utils.data import DataLoader

from petfinder_dataset import PetDataset
from models.baseline import Baseline, Baseline2, Baseline3, Baseline4, Baseline5, Baseline6
from models.animal_aware_model import AnimalAwareModel
from utils import AverageMeter, count_parameters

# wandb
import wandb
project = 'PetFinder'
my_name = 'uyenhnp'


def adjust_learning_rate(optimizer, lr_decay_rate):
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["lr"] * lr_decay_rate


def get_clip_prediction(model_clip, img_clip, clip_animal_label_features, 
        clip_color_label_features, clip_age_label_features):

    img_clip_features = model_clip.encode_image(img_clip)
    # normalized features
    img_clip_features = img_clip_features / img_clip_features.norm(dim=-1, keepdim=True)
    clip_animal_label_features = clip_animal_label_features / clip_animal_label_features.norm(dim=-1, keepdim=True)
    clip_color_label_features = clip_color_label_features / clip_color_label_features.norm(dim=-1, keepdim=True)
    clip_age_label_features = clip_age_label_features / clip_age_label_features.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logit_scale = model_clip.logit_scale.exp()

    animal_logits_per_image = logit_scale * img_clip_features @ clip_animal_label_features.t()
    animal_probs = animal_logits_per_image.softmax(dim=-1)

    color_logits_per_image = logit_scale * img_clip_features @ clip_color_label_features.t()
    color_probs = color_logits_per_image.softmax(dim=-1)

    age_logits_per_image = logit_scale * img_clip_features @ clip_age_label_features.t()
    age_probs = age_logits_per_image.softmax(dim=-1)

    return animal_probs.detach(), color_probs.detach(), age_probs.detach() # (N, 2)


def validate(model, dataloader, device, loss, use_clip, model_clip, 
             clip_animal_label_features, clip_color_label_features,
             clip_age_label_features):
    model.eval()
    val_mse_meter = AverageMeter()
    n = 0
    all_preds = []
    with torch.no_grad():
        for iter, batch in enumerate(dataloader):
            for k in batch:
                batch[k] = batch[k].to(device)
            img = batch['img']
            score = batch['score']
            meta = batch['meta']
            if use_clip:
                img_clip = batch['img_clip']
                animal_probs, color_probs, age_probs = get_clip_prediction(
                    model_clip, img_clip, clip_animal_label_features,
                    clip_color_label_features, clip_age_label_features)
            # Get the predicted score.
            if use_clip:
                y_pred = model(batch, animal_probs, color_probs, age_probs)
            else:
                y_pred = model(batch)
            all_preds.append(y_pred.cpu().squeeze())
            # Calculate RMSE.
            if loss == 'MSE':
                mse = ((score - y_pred)**2).sum()
            elif loss == 'BCE':
                mse = ((score*100 - y_pred*100)**2).sum()
            val_mse_meter.update(mse.item())
            n += img.shape[0]
    val_rmse = (val_mse_meter.sum/n)**0.5
    all_preds = torch.cat(all_preds, dim=0).numpy()
    return val_rmse, all_preds


def main():
    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True
    np.random.seed(24)
    torch.manual_seed(24)
    random.seed(24)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_small', action='store_true')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--feat_extractor_lr', type=float, default=5e-6)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--model', type=str, default='baseline')
    parser.add_argument('--exp_name', type=str, default='example')
    parser.add_argument('--drop', type=float, default=0.5)
    parser.add_argument('--crop', type=float, default=0.2)
    parser.add_argument('--img_size', type=int, default=384)
    parser.add_argument('--train_full', action='store_true')
    parser.add_argument('--loss', type=str, default='BCE')
    parser.add_argument('--backbone', type=str, default='resnet50') # resnet50, swin_base_patch4_window12_384, swin_large_patch4_window12_384
    parser.add_argument('--feat_eng', action='store_true')
    ######### KHOI #########
    parser.add_argument('--pawscore_jitter', type=int, default=0)
    parser.add_argument('--grayscale', action='store_true')
    parser.add_argument('--fc_nonlinear', action='store_true')
    parser.add_argument('--epoch_decay', type=int, default=-1)
    parser.add_argument('--use_animal_aware_embedding', action='store_true')
    ########################
    parser.add_argument('--train_split', type=str, default='train1')
    parser.add_argument('--val_split', type=str, default='validation1')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--use_clip', action='store_true')
    args = parser.parse_args()

    if args.train_full:
        print('Train full.')
    print(f'Train {args.model} with backbone={args.backbone}'
        f' batch_size={args.batch_size}, epochs={args.epochs},' 
        f' lr={args.lr}, feat_extractor_lr={args.feat_extractor_lr},' 
        f' weight_decay={args.weight_decay}, loss={args.loss},'
        f' drop={args.drop}, crop={args.crop}, img_size={args.img_size},'
        f' feat_eng={args.feat_eng}, train_split={args.train_split},'
        f' val_split={args.val_split}, epoch_decay={args.epoch_decay}')
    print(f'Using {device}')

    if args.use_wandb:
        print('Log using wandb')
        config = {
            'batch_size': args.batch_size,
            'lr': args.lr,
            'epochs': args.epochs,
            'model': args.model,
            'drop': args.drop,
        }
        wandb.init(
            project=project,
            name=args.exp_name,
            config=config,
            entity=my_name
        )
        wandb.define_metric("train/epoch")
        wandb.define_metric("train/*", step_metric="train/epoch")
        wandb.define_metric("val/*", step_metric="train/epoch")

    # Create checkpoint path.
    ckpt_path = f'checkpoints/{args.model}/{args.exp_name}'
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    # 1. Create a model.
    if args.model == 'baseline':
        model = Baseline(backbone=args.backbone, drop=args.drop, loss=args.loss,
                         fc_nonlinear=args.fc_nonlinear)
    elif args.model == 'baseline2':
        meta_dim = 2 if args.feat_eng else 12
        model = Baseline2(backbone=args.backbone, drop=args.drop, 
                          loss=args.loss, meta_dim=meta_dim)
    elif args.model == 'baseline3':
        meta_dim = 2 if args.feat_eng else 12
        model = Baseline3(backbone=args.backbone, drop=args.drop, 
                          loss=args.loss, meta_dim=meta_dim)
    elif args.model == 'baseline4':
        model = Baseline4(backbone=args.backbone, drop=args.drop, loss=args.loss,
                          fc_nonlinear=args.fc_nonlinear)
    elif args.model == 'baseline5' or args.model == 'baseline6':
        if args.model == 'baseline5':
            model = Baseline5(backbone=args.backbone, drop=args.drop, loss=args.loss)
        else:
            model = Baseline6(backbone=args.backbone, drop=args.drop, loss=args.loss)
        model_clip, preprocess_clip = clip.load("ViT-B/32", device=device)
        model_clip.eval()
        clip_animal_label = clip.tokenize(["a baby dog", "a baby cat", "an adult dog", "an adult cat"]).to(device)
        clip_animal_label_features = model_clip.encode_text(clip_animal_label)
    elif args.model == 'aam':
        model = AnimalAwareModel(
            backbone=args.backbone,
            drop=args.drop,
            loss=args.loss,
            use_embedding=args.use_animal_aware_embedding)
        model_clip, preprocess_clip = clip.load("ViT-B/32", device=device)
        model_clip.eval()
        animal_label = clip.tokenize(['a dog', 'a cat']).to(device)
        color_label = clip.tokenize([
            'a black pet', 'a brown pet', 'a white pet']).to(device)
        age_label = clip.tokenize(['a baby pet', 'an adult pet']).to(device)
        clip_animal_label_features = model_clip.encode_text(animal_label)
        clip_color_label_features = model_clip.encode_text(color_label)
        clip_age_label_features = model_clip.encode_text(age_label)

    model.to(device)
    total_params = count_parameters(model)
    if args.use_wandb:
        wandb.run.summary["num_parameters"] = total_params

    if not args.use_clip:
        preprocess_clip = None
        model_clip = None
        clip_animal_label_features = None
    if args.model != 'aam':
        clip_color_label_features = None # Uyen add
        clip_age_label_features = None # Uyen add

    # 2. Prepare data.
    train_split = args.train_split
    if args.train_full:
        train_split = 'train'
    dataset = PetDataset(split=train_split, debug_small=args.train_small, 
                         crop=(args.crop, 1), img_size=args.img_size,
                         loss=args.loss, feat_engineer=args.feat_eng,
                         pawscore_jitter=args.pawscore_jitter,
                         grayscale=args.grayscale, use_clip=args.use_clip,
                         preprocess_clip=preprocess_clip)
    print(f'# Number of training examples: {len(dataset)}')
    train_dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4)
    print(f'# batch: {len(train_dataloader)}')

    val_dataset = PetDataset(split=args.val_split, img_size=args.img_size, 
                             loss=args.loss, feat_engineer=args.feat_eng,
                             use_clip=args.use_clip, preprocess_clip=preprocess_clip)
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=4
    )
    print(f'# of val examples: {len(val_dataset)}')

    # 3. Lost function.
    if args.loss == 'MSE':
        loss_fn = torch.nn.MSELoss(reduction='mean')
    elif args.loss == 'BCE':
        loss_fn = torch.nn.BCELoss(reduction='mean')

    # 4. Optimizer.
    feat_extractor_params = [
        param
        for name, param in model.named_parameters()
        if 'feat_extractor' in name and param.requires_grad
    ]
    other_params = [
        param
        for name, param in model.named_parameters()
        if 'feat_extractor' not in name and param.requires_grad
    ]

    optimizer = torch.optim.Adam([
        {'params': feat_extractor_params, 'lr': args.feat_extractor_lr, 'weight_decay': 5e-5},
        {'params': other_params}
        ], lr=args.lr, weight_decay=args.weight_decay)

    # 5. Train model
    epochs = args.epochs
    best_val_rmse = 1000000000
    for epoch in range(epochs):
        model.train()
        loss_meter = AverageMeter()
        train_rmse_meter = AverageMeter()

        # Time check.
        batch_time = AverageMeter()
        data_time = AverageMeter()
        end_time = time.time()

        for iter, batch in enumerate(train_dataloader):
            data_time.update(time.time() - end_time)
            for k in batch:
                batch[k] = batch[k].to(device)
            score = batch['score']
            img = batch['img']
            meta = batch['meta']
            if args.use_clip:
                img_clip = batch['img_clip']
                animal_probs, color_probs, age_probs = \
                    animal_probs, color_probs, age_probs = get_clip_prediction(
                        model_clip, img_clip, clip_animal_label_features,
                        clip_color_label_features, clip_age_label_features)

            # Forward pass: calculate y_pred.
            if args.use_clip:
                y_pred = model(batch, animal_probs, color_probs, age_probs)
            else:
                y_pred = model(batch)
            # Calculate the loss.
            loss = loss_fn(y_pred, score)
            loss_meter.update(loss.item())
            # Zero the gradients before running the backward pass.
            optimizer.zero_grad() 
            # Backward pass: compute gradient of the loss with respect to all parameters.
            loss.backward()
            # Update the weights using gradient descent.
            optimizer.step()
            # Calculate RMSE.
            if args.loss == 'MSE':
                mse = ((score - y_pred)**2).mean()
            elif args.loss == 'BCE':
                mse = ((score*100 - y_pred*100)**2).mean()
            rmse = torch.sqrt(mse)
            train_rmse_meter.update(rmse.item())

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            if (iter + 1) % 50 == 0:
                print(f'Iter {iter+1}/{len(train_dataloader)}: RMSE = {train_rmse_meter.avg:.4f}'
                      f', Batch_time = {batch_time.avg:.2f}, Data_time = {data_time.avg:.2f}', flush=True)
                batch_time.reset()
                data_time.reset()

        # Compute validation RMSE.
        # Save the checkpoint.
        file_path = f'{ckpt_path}/model.pt'
      
        if args.train_full:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                # 'optimizer_state_dict': optimizer.state_dict()
            }, file_path)
            print(f'Save checkpoint at epoch {epoch}.')
        else:
            val_rmse, all_preds = validate(
                model, val_dataloader, device, loss=args.loss, 
                use_clip=args.use_clip, model_clip=model_clip, 
                clip_animal_label_features=clip_animal_label_features,
                clip_color_label_features=clip_color_label_features,
                clip_age_label_features=clip_age_label_features)
            
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    # 'optimizer_state_dict': optimizer.state_dict()
                }, file_path)
                print(f'Save checkpoint at epoch {epoch}.')

                # Save prediction.
                # df = val_dataset.data.copy()
                # df["pred"] = all_preds
                # df.to_csv(f'{ckpt_path}/{args.val_split}_pred.csv', index=0)
                # with open(f'{ckpt_path}/{args.val_split}_rmse.txt', 'w') as f:
                #     f.write(f'epoch,val_rmse')
                #     f.write(f'{epoch + 1},{val_rmse:.4f}')

            print(f'epoch={epoch+1}, loss={loss_meter.avg:.4f}, train_rmse={train_rmse_meter.avg:.4f},'
                  f' val_rmse={val_rmse:.4f}')

        if args.epoch_decay != -1 and epoch+1 == args.epoch_decay:
            adjust_learning_rate(optimizer, lr_decay_rate=0.1)

        if args.use_wandb:
            # Plot on wandb.
            if args.train_full:
                val_rmse = 0
            wandb.log({
                'train/loss': loss_meter.avg,
                'train/rmse': train_rmse_meter.avg,
                'val/rmse': val_rmse,
                'train/epoch': epoch + 1
            }, commit=True)
    
    if args.use_wandb:
        wandb.finish()

if __name__ == '__main__':
    main()