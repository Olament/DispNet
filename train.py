import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import dispnet, dataloader, losses, utils
import os
import cv2 
import numpy as np

# hyperparameters
batch_size = 2 
learning_rate = 0.001
total_epoch = 10 
report_rate = 10 

# set training device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# dataset
dataset = dataloader.KITTIDataset()
data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=batch_size,
                                             shuffle=False)

# load model
model = dispnet.DispNet(h=375,w=1424).to(device)
model.train()

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.9))
criterion = losses.RelLoss()
   
# load previous checkpoint
ckpt_path = 'logs/checkpoint.ckpt'
if os.path.isfile(ckpt_path):
    model.load_state_dict(torch.load(ckpt_path))
    print("check point loaded!")        

# summary writer for tensorboard
writer = SummaryWriter()

# start training
for epoch in range(total_epoch):
    loss_sum = 0
    step = 0
    for i, (image, depth) in enumerate(data_loader):
        # send to device
        image = image.to(device)
        depth = depth.to(device)

        # model
        output, _ = model(image)
        output = torch.squeeze(output, dim=1) # squeeze extra channel
        # loss
        loss = criterion(output, depth)
        loss_sum += loss.item()
        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # report
        if step % report_rate == 0:
            torch.save(model.state_dict(), ckpt_path)
            print('Epoch [{}/{}], step [{}/{}], loss {}'.format(epoch+1, total_epoch, step, len(data_loader), loss_sum/report_rate))
            loss_sum = 0
            #writer.add_image('image', image[0], global_step=step)
            #writer.add_image('ground_depth', torch.unsqueeze(depth[0], 0), global_step=step)
            #writer.add_image('pred_depth', torch.unsqueeze(output[0], 0), global_step=step)
            ground_depth = torch.unsqueeze(depth[0], 0).float().expand(3, -1, -1)
            pred_depth = torch.unsqueeze(output[0], 0).float().expand(3, -1, -1)
            img_grid = torchvision.utils.make_grid([image[0].float(), ground_depth, pred_depth], nrow=1)
            writer.add_image('comp', img_grid, global_step=step)
            writer.add_scalar('Loss', loss.item(), step)
            #utils.visualize_img(image[0], 'image')
            #utils.visualize_dep(depth[0], 'groundtruth')
            #utils.visualize_dep(output[0], 'pred')
            #cv2.destroyAllWindows()
        step += 1 
        
writer.close()
