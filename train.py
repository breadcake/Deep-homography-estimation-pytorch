import torch
from torch import nn, optim
from dataset import CocoDdataset
from model import HomographyNet
from torch.utils.data import DataLoader
import argparse
import time
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def train(args):
    MODEL_SAVE_DIR = 'checkpoints/'
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)

    model = HomographyNet()

    TrainingData = CocoDdataset(args.train_path)
    ValidationData = CocoDdataset(args.val_path)
    print('Found totally {} training files and {} validation files'.format(len(TrainingData), len(ValidationData)))
    train_loader = DataLoader(TrainingData, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(ValidationData, batch_size=args.batch_size, num_workers=4)

    if torch.cuda.is_available():
        model = model.cuda()

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    # decrease the learning rate after every 1/3 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(args.epochs / 3), gamma=0.1)

    print("start training")
    glob_iter = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        epoch_start = time.time()
        # Training
        model.train()
        train_loss = 0.0
        for i, batch_value in enumerate(train_loader):
            # save model
            if (glob_iter % 4000 == 0 and glob_iter != 0):
                filename = 'homographymodel' + '_iter_' + str(glob_iter) + '.pth'
                model_save_path = os.path.join(MODEL_SAVE_DIR, filename)
                state = {'epoch': args.epochs, 'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict()}
                torch.save(state, model_save_path)

            ori_images = batch_value[0].float()
            inputs = batch_value[1].float()
            pts1 = batch_value[2]
            target = batch_value[3].float()

            if torch.cuda.is_available():
                inputs = inputs.cuda()
                target = target.cuda()

            optimizer.zero_grad()  # 梯度清零
            outputs = model(inputs)
            loss = criterion(outputs, target.view(-1, 8))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if (i + 1) % 200 == 0 or (i+1) == len(train_loader):
                print("Training: Epoch[{:0>3}/{:0>3}] Iter[{:0>3}/{:0>3}] Mean Squared Error: {:.4f} lr={:.6f}".format(
                    epoch+1, args.epochs, i+1, len(train_loader), train_loss / 200, scheduler.get_lr()[0]))
                train_loss = 0.0

            glob_iter += 1
        scheduler.step()

        # Validation
        with torch.no_grad():
            model.eval()
            val_loss = 0.0
            for i, batch_value in enumerate(val_loader):
                ori_images = batch_value[0].float()
                inputs = batch_value[1].float()
                pts1 = batch_value[2]
                target = batch_value[3].float()
                if torch.cuda.is_available():
                    inputs, target = inputs.cuda(), target.cuda()
                outputs = model(inputs)
                loss = criterion(outputs, target.view(-1, 8))
                val_loss += loss.item()
            print("Validation: Epoch[{:0>3}/{:0>3}] Mean Squared Error:{:.4f}, epoch time: {:.1f}s".format(
                epoch + 1, args.epochs, val_loss / len(val_loader), time.time() - epoch_start))

    elapsed_time = time.time() - t0
    print("Finished Training in {:.0f}h {:.0f}m {:.0f}s.".format(
        elapsed_time // 3600, (elapsed_time % 3600) // 60, (elapsed_time % 3600) % 60))


if __name__ == "__main__":
    train_path = 'data/training/'
    val_path = 'data/validation/'

    total_iteration = 90000
    batch_size = 64
    num_samples = 500000
    steps_per_epoch = num_samples // batch_size
    epochs = int(total_iteration / steps_per_epoch)

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=batch_size, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=0.005, help="learning rate")
    parser.add_argument("--epochs", type=int, default=epochs, help="number of epochs")

    parser.add_argument("--train_path", type=str, default=train_path, help="path to training imgs")
    parser.add_argument("--val_path", type=str, default=val_path, help="path to validation imgs")
    args = parser.parse_args()
    train(args)
