############################################
#
#   Visualize results of trained model
#
############################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

from tqdm import tqdm

from pose_dataset import *
from pose_resnet import *
from pose_resnet_2d import *
from pose_hrnet import *
import arguments
from make_log import *
from evaluate import *
from loss import *

args = arguments.get_arguments()

# model name
model_name = '{}_nlayer{}_{}_lr{}_batch{}_momentum{}_schedule{}_nepoch{}_{}'.format(
        args.name,
        args.nlayer,
        args.optimizer,
        args.lr,
        args.batch_size,
        args.momentum,
        args.schedule,
        args.nepochs,
        args.arch
    )
logger = make_logger(log_file=model_name)
logger.info("saved model name "+model_name)        

arguments.print_arguments(args, logger)


multi_gpu = args.multi_gpu
set_gpu_num = args.gpu_num

if torch.cuda.is_available():
    print("gpu", torch.cuda.current_device(), torch.cuda.get_device_name())
else:
    print("cpu")
#----- model -----
if args.arch =='hrnet':
    model = get_pose_hrnet()
else:
    if args.flatten:
        model = get_2d_pose_net(num_layer=args.nlayer, input_depth=1)
    else:
        model = get_pose_net(num_layer=args.nlayer, input_depth=2048-args.cutoff)

if multi_gpu is True:
    model = torch.nn.DataParallel(model).cuda()
    logger.info("Let's use multi gpu\t# of gpu : {}".format(torch.cuda.device_count()))
else:
    torch.cuda.set_device(set_gpu_num)
    logger.info("Let's use single gpu\t now gpu : {}".format(set_gpu_num))
    #model.cuda()
    model = torch.nn.DataParallel(model, device_ids = [set_gpu_num]).cuda()
#model.cuda() # torch.cuda_set_device(device) 로 send
#model.to(device) # 직접 device 명시

#----- loss function -----
criterion = nn.MSELoss().cuda()
cr = JointsMSELoss().cuda()
#----- optimizer and scheduler -----
if args.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    logger.info('use adam optimizer')
else:
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4, nesterov=False)
    logger.info('use sgd optimizer')

lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=args.gammas)

#----- dataset -----
train_data = PoseDataset(mode='train', args=args)
train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True)

#----- training -----
max_acc = 0
max_acc_epoch = 0

#name = 'save_model/201011_resnet18_SGD_lr0.001_batch32_momentum0.9_schedule[90, 110]_nepoch140_epoch88.pt'
#state_dict = torch.load(name)
#model.module.load_state_dict(state_dict)
begin_time = datetime.now()
print(begin_time)
for epoch in range(args.nepochs):
    logger.info("Epoch {}\tcurrent lr : {} {}".format(epoch, optimizer.param_groups[0]['lr'], lr_scheduler.get_last_lr()))
    epoch_loss = []
    avg_acc = 0
    sum_acc = 0
    total_cnt = 0

    iterate = 0

    for rf, target_heatmap in tqdm(train_dataloader):
        #print(rf.shape, target_heatmap.shape)
        #print(rf.dtype, target_heatmap.dtype)
        rf, target_heatmap = rf.cuda(), target_heatmap.cuda()
        
        out = model(rf)
        #loss = 0.5 * criterion(out, target_heatmap)
        loss = cr(out, target_heatmap)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss)
        _, temp_avg_acc, cnt, pred = accuracy(out.detach().cpu().numpy(),
                target_heatmap.detach().cpu().numpy())
        sum_acc += temp_avg_acc * cnt
        total_cnt += cnt
        avg_acc = sum_acc / total_cnt if total_cnt != 0 else 0
        if iterate % 500 == 0:
            logger.info("iteration[%d] batch loss %.6f\tavg_acc %.4f\ttotal_count %d"%(iterate, loss.item(), avg_acc, total_cnt))
        iterate += 1

    logger.info("epoch loss : %.6f"%torch.tensor(epoch_loss).mean().item())
    logger.info("epoch acc on train data : %.4f"%(avg_acc))
    
    if avg_acc > max_acc:
        logger.info("epoch {} acc {} > max acc {} epoch {}".format(epoch, avg_acc, max_acc, max_acc_epoch))
        max_acc = avg_acc
        max_acc_epoch = epoch
        #if args.multi_gpu == 1:
        torch.save(model.module.state_dict(), "save_model/" + model_name + "_best.pt")
        #else:
            #torch.save(model.state_dict(), "save_model/" + model_name + "_best.pt")
    lr_scheduler.step()
    #if args.multi_gpu == 1:
    torch.save(model.module.state_dict(), "save_model/" + model_name + "_epoch{}.pt".format(epoch))
    #else:
    #    torch.save(model.state_dict(), "save_model/" + model_name + "_epoch{}.pt".format(epoch))

logger.info("training end | elapsed time = " + str(datetime.now() - begin_time))


