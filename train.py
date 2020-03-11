import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import dispnet, dataloader, losses, utils
import os
import cv2 
import numpy as np
from datetime import datetime

# hyperparameters
batch_size = 6 
learning_rate = 0.0001
total_epoch = 10 
report_rate = 20 
save_rate = 4000 

# set training device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# dataset
dataset = dataloader.KITTIDataset()
data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=batch_size,
                                             shuffle=True)

# load model
model = dispnet.DispNet(h=375,w=1424).to(device)
model.train()

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
   
# load previous checkpoint
ckpt_path = 'logs/checkpoint.ckpt'
if os.path.isfile(ckpt_path):
    model.load_state_dict(torch.load(ckpt_path))
    print("check point loaded!")        


# start training
for epoch in range(total_epoch):
    loss_sum = 0
    step = 0
    writer = SummaryWriter()
    for i, (image, depth) in enumerate(data_loader):
        # send to device
        image = image.to(device)
        depth = depth.to(device)

        # model
        output, _ = model(image)
        output = torch.squeeze(output, dim=1) # squeeze extra channel
        # loss
        #d_loss = losses.BerHuLoss(output, depth, mask=(depth > 0.0).double())
        d_loss = losses.SILogLoss(output, depth)
        s_loss = losses.SmoothLoss(output.unsqueeze(dim=1), image)
        loss = d_loss + s_loss
        #loss = d_loss
        loss_sum += loss.item()
        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # save
        if step % save_rate == 0:
            torch.save(model.state_dict(), ckpt_path) # save lastest model
            now = datetime.now()
            timestring = now.strftime("%m%d%H%M")
            torch.save(model.state_dict(), os.path.join('logs', 'archive', '_'.join(['DispNet', str(epoch), str(step), timestring]) + '.ckpt')) 
        # report
        if step % report_rate == 0:
            print('Epoch [{}/{}], step [{}/{}], loss {}'.format(epoch+1, total_epoch, step, len(data_loader), loss_sum/report_rate))
            loss_sum = 0
            ground_depth = torch.unsqueeze(depth[0], 0).float().expand(3, -1, -1)
            pred_depth = utils.convert_to_colormap(1.0/output[0])
            pred_depth = pred_depth.to(device)
            img_grid = torchvision.utils.make_grid([image[0].float(), ground_depth, pred_depth], nrow=1)
            writer.add_image('visualize', img_grid, global_step=step)
            writer.add_scalar('BerHuLoss', d_loss.item(), step)
            writer.add_scalar('SmoothLoss', s_loss.item(), step)
            writer.add_scalar('TotalLoss', loss.item(), step)
            ErrorEval(writer, step, output, depth)
        step += 1 
    writer.close()


def ErrorEval(writer, step, pred, target):
    abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = Eval(pred, target)
    writer.add_scalar('AbsRel', abs_rel.item(), step)
    writer.add_scalar('SqRel', sq_rel.item(), step)
    writer.add_scalar('RMSE', rmse.item(), step)
    writer.add_scalar('RMSE_Log', rmse_log.item(), step)
    writer.add_scalar('delta_1.5', a1.item(), step)
    writer.add_scalar('delta_1.5^2', a2.item(), step)
    writer.add_scalar('delta_1.5^3', a3.item(), step)
