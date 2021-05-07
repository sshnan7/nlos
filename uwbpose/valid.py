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
from human_idnet import *

#def prediction(model, rf, target_heatmap, criterion, target_h_label):
def prediction(model, generator, rf, target_heatmap, criterion, target_h_label):
    #out, out_human = model(rf) #for pose + id
    rf = rf.view(-1, 1792)# 3은 안테나 수 아닐땐 1 #1792 default
    #print(rf.shape)
    rf = rf.float()
    gen_rf = generator(rf.unsqueeze(1))
    gen_rf = gen_rf
    gen_rf = gen_rf.cuda()
    out_human = model(gen_rf) # for only id
    #out_human = model(rf.unsqueeze(1))
    

    #loss = criterion(out, target_heatmap)
    loss = (float)(0)

    #_, temp_avg_acc, cnt, pred = accuracy(out.detach().cpu().numpy(),
    #        target_heatmap.detach().cpu().numpy()) ###for pose + id
            
    h_temp_avg_acc, h_cnt = human_accuracy(out_human, target_h_label)

    #preds, maxvals = get_final_preds(out.clone().cpu().numpy())

    #target_label, target_maxvals = get_final_preds(target_heatmap.clone().cpu().numpy())

    #temp_true_det, temp_whole_cnt = pck(preds*4, target_label*4)

    #return out, out_human, loss, temp_avg_acc, cnt, preds, target_label, temp_true_det, temp_whole_cnt, h_temp_avg_acc, h_cnt #for pose + id
    return out_human, h_temp_avg_acc, h_cnt #for only id

#def validate(dataloader, model, logger, criterion, debug_img=False):
def validate(dataloader, model, generator, logger, criterion, debug_img=False):
    criterion = JointsMSELoss().cuda()
    human_cr = nn.CrossEntropyLoss().cuda()
    vis = Visualize(show_debug_idx=False) 
    label_num_list = []
    label_correct_list = []
    for i in range(10):
        label_num_list.append(0)
        label_correct_list.append(0)
        
    with torch.no_grad():
        epoch_loss = []
        avg_acc = 0
        sum_acc = 0
        total_cnt = 0
        
        h_avg_acc = 0
        h_sum_acc = 0
        h_total_cnt = 0

        iterate = 0

        true_detect = np.zeros((4, 13))
        whole_count = np.zeros((4, 13))
        
        if debug_img == True:
            for rf, target_heatmap, img in tqdm(dataloader):
            
                rf, target_heatmap, img = rf.cuda(), target_heatmap.cuda(), img.cuda()
                out, loss, temp_avg_acc, cnt, preds, target_label, temp_true_det, temp_whole_cnt = prediction(model, rf, target_heatmap, criterion) # for pose + id
                
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
            for rf, target_heatmap, target_h_label, antena_label, domain in tqdm(dataloader):
                #print("rf", rf.shape)
                for i in target_h_label :    #int 형인지 확인하기
                    label_num_list[i] += 1
                rf, target_heatmap, target_h_label, antena_label = rf.cuda(), target_heatmap.cuda(), target_h_label.cuda(), antena_label.cuda()
                #out, h_out, loss, temp_avg_acc, cnt, preds, target_label, temp_true_det, temp_whole_cnt, h_temp_avg_acc, h_cnt = prediction(model, rf, target_heatmap, criterion, target_h_label) #for pose + id
                
                h_out,  h_temp_avg_acc, h_cnt = prediction(model, generator, rf, target_heatmap, criterion, target_h_label) #for only id
                
                #epoch_loss.append(loss) #for pose + id
                #sum_acc += temp_avg_acc * cnt #for pose + id
                h_sum_acc += h_temp_avg_acc * h_cnt #only id
                #total_cnt += cnt #only id
                h_total_cnt += h_cnt
                id_loss = human_cr(h_out, target_h_label)
                #avg_acc = sum_acc / total_cnt if total_cnt != 0 else 0 #for pose + id
                h_avg_acc = h_sum_acc / h_total_cnt if h_total_cnt != 0 else 0
                #true_detect += temp_true_det # for pose + id
                #whole_count += temp_whole_cnt # for pose + id
                _, h_predict = torch.max(h_out, 1)
                for i in range(len(h_predict)) :
                    if h_predict[i] == target_h_label[i]:
                        label_correct_list[h_predict[i]] += 1
                    
                
                #if iterate % 100 == 0:
                #    logger.info("iteration[%d] batch loss %.6f\tavg_acc %.4f\ttotal_count %d"%(iterate, loss.item(), avg_acc, total_cnt))
                iterate += 1
                
                #logger.info("human predict  : {},\t human_label : {},\t human_loss : {}".format(h_predict, target_h_label, id_loss))
                #if iterate % 100 == 0:
                #    logger.info("human predict : {} \t human_label : {}".format(h_predict,target_h_label))
        
        #logger.info("epoch loss : %.6f"%torch.tensor(epoch_loss).mean().item())
        #logger.info("epoch acc on test data : %.4f"%(avg_acc))
        logger.info("epoch h_acc on test data : %.4f"%(h_avg_acc))
        logger.info("epoch h_loss on test data : %.4f"%(id_loss))
        
        #####for pose + id
        '''
        #pck_res = true_detect / whole_count * 100 
        #thr = [0.1, 0.2, 0.3, 0.5]
        for t in range(4):
            logger.info("PCK {} average {} - {}".format(thr[t], np.average(pck_res[t]), pck_res[t]))
        '''
        #######
        for i in range(len(label_num_list)):
            print("{} acc {}".format(i, label_correct_list[i]/label_num_list[i]))
            
            
        

if __name__ == '__main__':

    args = arguments.get_arguments()
   
    model_name = args.model_name
    #model_name = "210109_newdata_normalize_nlayer18_adam_lr0.001_batch32_momentum0.9_schedule[10, 20]_nepoch30"
    #model_name = "210112_mixup_nlayer18_adam_lr0.001_batch32_momentum0.9_schedule[10, 20]_nepoch30"
    #model_name = '210113_intensity_nlayer18_adam_lr0.001_batch32_momentum0.9_schedule[10, 20]_nepoch30'
    #model_name = '210119_hrnet_nlayer18_adam_lr0.001_batch16_momentum0.9_schedule[10, 20]_nepoch30_hrnet'
    model_name = '210507_model_final007_nlayer18_adam_lr0.001_batch64_momentum0.9_schedule[10, 20]_nepoch60_resnet'
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
            #model = get_2d_pose_net(num_layer=args.nlayer, input_depth=1) # for pose
            model = get_human_id_net(num_layer=args.nlayer)
            generator = get_generator(num_layer=args.nlayer)
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
        #generator = torch.nn.DataParallel(generator, device_ids = [set_gpu_num]).cuda()
        generator = torch.nn.DataParallel(generator, device_ids = [set_gpu_num]).cuda()

    #model.cuda() # torch.cuda_set_device(device) 로 send
    #model.to(device) # 직접 device 명시
    

    #----- loss function -----
    #criterion = nn.MSELoss().cuda()
    criterion = JointsMSELoss().cuda()

    #----- dataset -----
    #test_data = PoseDataset(mode='test', args=args)
    test_data = PoseDataset(mode='test', args=args)
    dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True)

    model_name = model_name + '_epoch{}.pt'
    gen_name = 'generator_final_model007' + '_epoch{}.pt'
    # 원하는 모델 구간 지정해야함.
    #for i in range(20, 30):
    for i in range(0, 60):
        logger.info("epoch %d"%i)
        logger.info('./save_model/' + model_name.format(i))
        model.module.load_state_dict(torch.load('./save_model/'+model_name.format(i)))
        model.eval()
        generator.module.load_state_dict(torch.load('./save_model/' + gen_name.format(i)))
        generator.eval()
        

        #model.module.load_state_dict(torch.load(model_name.format(i)))
        #model.module.load_state_dict(torch.load(model_name))
        #print(dataloader, model, logger, criterion)
        #validate(dataloader, model, logger, criterion, debug_img=args.vis)
        validate(dataloader, model, generator, logger, criterion, debug_img=args.vis)
