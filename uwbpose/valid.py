import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import sys

import arguments
from pose_dataset import *
from pose_resnet_2d import *
from pose_resnet import *
from pose_hrnet import *
from loss import *
from visualize import *
from inference import *
from make_log import *
from evaluate import *

def prediction(model, rf, target_heatmap, criterion):
    out = model(rf)

    loss = criterion(out, target_heatmap)

    _, temp_avg_acc, cnt, pred = accuracy(out.detach().cpu().numpy(),
            target_heatmap.detach().cpu().numpy())

    preds, maxvals = get_final_preds(out.clone().cpu().numpy())

    target_label, target_maxvals = get_final_preds(target_heatmap.clone().cpu().numpy())

    temp_true_det, temp_whole_cnt = pck(preds*4, target_label*4)

    return out, loss, temp_avg_acc, cnt, preds, target_label, temp_true_det, temp_whole_cnt

def validate(dataloader, model, logger, criterion, debug_img=False):
    model.eval()
    criterion = JointsMSELoss().cuda()
    vis = Visualize(show_debug_idx=False) 
    with torch.no_grad():
        epoch_loss = []
        avg_acc = 0
        sum_acc = 0
        total_cnt = 0

        iterate = 0

        true_detect = np.zeros((4, 13))
        whole_count = np.zeros((4, 13))
        
        if debug_img == True:
            for rf, target_heatmap, img in tqdm(dataloader):
                rf, target_heatmap, img = rf.cuda(), target_heatmap.cuda(), img.cuda()
                out, loss, temp_avg_acc, cnt, preds, target_label, temp_true_det, temp_whole_cnt = prediction(model, rf, target_heatmap, criterion)
                
                epoch_loss.append(loss)
                sum_acc += temp_avg_acc * cnt
                total_cnt += cnt
                avg_acc = sum_acc / total_cnt if total_cnt != 0 else 0
                true_detect += temp_true_det
                whole_count += temp_whole_cnt

                #save_debug_images(img, target_label*4, target_heatmap, preds*4, out, './vis/batch_{}'.format(iterate))
                #vis.detect_and_draw_person(img.clone().cpu().numpy(), preds*4, iterate, 'pred')
                #vis.detect_and_draw_person(img.clone().cpu().numpy(), target_label*4, iterate, 'gt')
                vis.compare_visualize(img.clone().cpu().numpy(), preds*4, target_label*4, iterate)

                #if iterate % 100 == 0:
                #    logger.info("iteration[%d] batch loss %.6f\tavg_acc %.4f\ttotal_count %d"%(iterate, loss.item(), avg_acc, total_cnt))
                iterate += 1

        else:
            for rf, target_heatmap in tqdm(dataloader):
                rf, target_heatmap = rf.cuda(), target_heatmap.cuda()
                out, loss, temp_avg_acc, cnt, preds, target_label, temp_true_det, temp_whole_cnt = prediction(model, rf, target_heatmap, criterion)
                
                epoch_loss.append(loss)
                sum_acc += temp_avg_acc * cnt
                total_cnt += cnt
                avg_acc = sum_acc / total_cnt if total_cnt != 0 else 0
                true_detect += temp_true_det
                whole_count += temp_whole_cnt

                #if iterate % 100 == 0:
                #    logger.info("iteration[%d] batch loss %.6f\tavg_acc %.4f\ttotal_count %d"%(iterate, loss.item(), avg_acc, total_cnt))
                iterate += 1
        
        logger.info("epoch loss : %.6f"%torch.tensor(epoch_loss).mean().item())
        logger.info("epoch acc on test data : %.4f"%(avg_acc))
        pck_res = true_detect / whole_count * 100
        thr = [0.1, 0.2, 0.3, 0.5]
        for t in range(4):
            logger.info("PCK {} average {} - {}".format(thr[t], np.average(pck_res[t]), pck_res[t]))


if __name__ == '__main__':

    args = arguments.get_arguments()
   
    model_name = args.model_name
    #model_name = "210109_newdata_normalize_nlayer18_adam_lr0.001_batch32_momentum0.9_schedule[10, 20]_nepoch30"
    #model_name = "210112_mixup_nlayer18_adam_lr0.001_batch32_momentum0.9_schedule[10, 20]_nepoch30"
    #model_name = '210113_intensity_nlayer18_adam_lr0.001_batch32_momentum0.9_schedule[10, 20]_nepoch30'
    model_name = '210119_hrnet_nlayer18_adam_lr0.001_batch16_momentum0.9_schedule[10, 20]_nepoch30_hrnet'
    if len(model_name) == 0:
        print("You must enter the model name for testing")
        sys.exit()
    
    #if model_name[-3:] != '.pt':
    #    print("You must enter the full name of model")
    #    sys.exit()

    #model_name = model_name.split('_epoch')[0]
    print("vaildate mode = ", model_name)
    log_name = model_name.split('/')[-1]
    print("log_name = ", log_name)
    logger = make_logger(log_file='valid_'+log_name)
    logger.info("saved valid log file "+'valid_'+log_name)        

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
            model = get_pose_net(num_layer=args.nlayer, input_depth=2048 - args.cutoff)

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
    #criterion = nn.MSELoss().cuda()
    criterion = JointsMSELoss().cuda()

    #----- dataset -----
    test_data = PoseDataset(mode='test', args=args)
    dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True)

    model_name = model_name + '_epoch{}.pt'
    # 원하는 모델 구간 지정해야함.
    #for i in range(20, 30):
    for i in range(5, 10):
        logger.info("epoch %d"%i)
        logger.info('./save_model/' + model_name.format(i))
        model.module.load_state_dict(torch.load('./save_model/'+model_name.format(i)))

        #model.module.load_state_dict(torch.load(model_name.format(i)))
        #model.module.load_state_dict(torch.load(model_name))
        validate(dataloader, model, logger, criterion, debug_img=args.vis)
