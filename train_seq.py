from model import CVSeqLocation
from read_seqgeo import SeqGeoDataset, data_collect
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import os
import numpy as np
import torch
from loss import cross_entropy_loss, regression_loss


# config
otherlayers_lr = 1e-4
weight_decay = 1e-4
start_epoch = 0
end_epoch = 100
batch_size = 1
lambda_cross_entropy = 1
lambda_regression = 1
device = "cuda" if torch.cuda.is_available() else "cpu"
input_size = 256
grid_size = (32, 32)
stride = input_size / grid_size[0]
sequence_size = 6
resolution = 0.285   # 640 / 256 x 0.114


def save_checkpoint(state, filename="checkpoint.pth"):
    torch.save(state, filename)

def main():
    # setup train/val dataset
    train_dataset = SeqGeoDataset(mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collect, num_workers=4)

    val_dataset = SeqGeoDataset(mode='test')
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collect, num_workers=4)
    torch.cuda.empty_cache()
    model = CVSeqLocation(d_model=256).to(device)
    # load pretrained baseline weights
    checkpoint = torch.load('CVSeq_CVTrans_baseline.pth')
    model_dict = model.state_dict()
    model_dict.update(checkpoint['state_dict'])
    model.load_state_dict(model_dict)
    del checkpoint
    # Freeze extractor part parameters
    for p in model.sat_extractor.parameters():
        p.requires_grad = False
    for p in model.grd_extractor.parameters():
        p.requires_grad = False
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=otherlayers_lr, weight_decay=weight_decay)

    # learning rate scheduler
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50)
    best_mean_distance_error = 9999

    for epoch_idx in range(start_epoch, end_epoch):
        model.train()
        epoch_loss = []
        iteration = 0
        for iteration, (sat, grd, labels, _) in tqdm(enumerate(train_dataloader)):
            sat_img = sat.to(device)   # 1, 3, 256, 256
            grd_imgs = grd.to(device)  # 1, 7, 3, 260, 480
            targets = labels[0]           # labels: [[(index, ty, tx), (), (), (), (), (), ()]]

            grd_prev_feats = None
            seq_loss = []
            # print(targets.shape)
            optimizer.zero_grad()
            for t in range(sequence_size):
                pred_location, coordinate_reg, grd_curt = model(sat_img, grd_imgs[:, t, :, :, :], grd_prev_feats)
                location_loss = cross_entropy_loss(pred_location, [targets[t]])
                txy_loss = regression_loss(coordinate_reg, [targets[t]])
                frame_loss = lambda_cross_entropy * location_loss + lambda_regression * txy_loss
                seq_loss.append(frame_loss)
                grd_prev_feats = grd_curt
            loss = sum(seq_loss)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
        curr_training_loss = sum(epoch_loss) / (iteration + 1)
        train_file = 'training_loss.txt'
        with open(train_file, 'a') as file:
            file.write(f"Epoch {epoch_idx} Training Loss: {curr_training_loss}" + '\n')

        print(f"Epoch {epoch_idx} Training Loss: {curr_training_loss}")

        model.eval()
        val_epoch_loss = []
        distances = []
        with torch.set_grad_enabled(False):
            for i, (val_sat, val_grd, val_labels, val_ground_yx) in tqdm(enumerate(val_dataloader)):
                val_sat_img = val_sat.to(device)
                val_grd_imgs = val_grd.to(device)
                val_targets = val_labels[0]
                val_ground_yx = val_ground_yx[0]

                val_sequence_distance = []
                val_grd_prev_feats = None
                val_seq_loss = []

                for t in range(sequence_size):
                    val_pred_location, val_coordinate_reg, val_grd_curt = model(val_sat_img, val_grd_imgs[:, t, :, :, :], val_grd_prev_feats)
                    val_location_loss = cross_entropy_loss(val_pred_location, [val_targets[t]])
                    val_txy_loss = regression_loss(val_coordinate_reg, [val_targets[t]])
                    val_frame_loss = lambda_cross_entropy * val_location_loss + lambda_regression * val_txy_loss
                    val_seq_loss.append(val_frame_loss)
                    val_grd_prev_feats = val_grd_curt

                    ground_y, ground_x = val_ground_yx[t]
                    cur_pred_location = val_pred_location[0].cpu().detach().numpy()
                    cur_pred_index = cur_pred_location.argmax()
                    pred_grid_y, pred_grid_x = np.unravel_index(cur_pred_index, grid_size)
                    pred_ty, pred_tx = val_coordinate_reg[0].cpu().detach().numpy()[cur_pred_index]
                    pred_ground_y, pred_groundx = (pred_grid_y + pred_ty) * stride, (pred_grid_x + pred_tx) * stride
                    val_sequence_distance.append(np.sqrt((ground_y - pred_ground_y)**2 + (ground_x - pred_groundx)**2) * resolution)

                distances.append(sum(val_sequence_distance) / sequence_size)
                val_loss = sum(val_seq_loss)
                val_epoch_loss.append(val_loss.item())

            curr_val_loss = sum(val_epoch_loss) / (i+1)
            distance_mean_error = np.mean(distances)
            distance_median_error = np.median(distances)

            val_loss_file = 'val_loss.txt'
            val_distance_error = 'val_distance_error.txt'
            with open(val_loss_file, 'a') as file:
                file.write(f"Epoch {epoch_idx} val Loss: {curr_val_loss}" + '\n')
            print(f"Epoch {epoch_idx} validation Loss: {curr_val_loss}")
            with open(val_distance_error, 'a') as file:
                file.write(f"Epoch {epoch_idx} val distance error: {distance_mean_error}, {distance_median_error}" + '\n')
            print(f"Epoch {epoch_idx} validation distance mean error: {distance_mean_error}")
            print(f"Epoch {epoch_idx} validation distance median error: {distance_median_error}")

            if distance_mean_error < best_mean_distance_error:
                if not os.path.exists("checkpoint"):
                    os.mkdir("checkpoint")
                model_path = "checkpoint/CVSeqLocation.pth"
                save_checkpoint(
                    {"state_dict": model.state_dict(),
                     "optimizer": optimizer.state_dict()
                     },
                    model_path
                )
                best_mean_distance_error = distance_mean_error
                print(f"Model saved at distance error: {best_mean_distance_error}")

if __name__ == '__main__':
    main()

