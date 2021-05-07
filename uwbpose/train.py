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
from human_idnet import *
import numpy as np
from random import *

args = arguments.get_arguments()

# model name
dis_train = False
gen_train = True
model_train = True

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
gen_name = 'generator_final_model007'
dis_name = 'pre_discriminator_final_model007'
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
        #model = get_2d_pose_net(num_layer=args.nlayer, input_depth=1) #for pose estimate
        if model_train == True :
            #generator = get_generator()
            model = get_human_id_net(num_layer=args.nlayer)
        if gen_train == True :
            discriminator = get_discriminator(num_layer=args.nlayer)
            generator = get_generator(num_layer=args.nlayer)
            discriminator2 = get_discriminator2(num_layer=args.nlayer)
        if dis_train == True:
            discriminator = get_discriminator(num_layer=args.nlayer)
            discriminator2 = get_discriminator2(num_layer=args.nlayer)
    else:
        model = get_pose_net(num_layer=args.nlayer, input_depth=2048-args.cutoff)

if multi_gpu is True:
    model = torch.nn.DataParallel(model).cuda()
    logger.info("Let's use multi gpu\t# of gpu : {}".format(torch.cuda.device_count()))
else:
    torch.cuda.set_device(set_gpu_num)
    logger.info("Let's use single gpu\t now gpu : {}".format(set_gpu_num))
    #model.cuda()
    if model_train == True:
        model = torch.nn.DataParallel(model, device_ids = [set_gpu_num]).cuda()
        #model.module.load_state_dict(torch.load('./save_model/210422_GAN_model1_nlayer18_adam_lr0.001_batch64_momentum0.9_schedule[10, 20]_nepoch60_resnet_epoch1.pt'))
       # generator = torch.nn.DataParallel(generator, device_ids = [set_gpu_num]).cuda()
    if gen_train == True :
        generator = torch.nn.DataParallel(generator, device_ids = [set_gpu_num]).cuda()
        #generator.module.load_state_dict(torch.load('./save_model/generator_epoch1.pt'))
        discriminator = torch.nn.DataParallel(discriminator, device_ids = [set_gpu_num]).cuda()
        discriminator2 = torch.nn.DataParallel(discriminator2, device_ids = [set_gpu_num]).cuda()
        #discriminator.module.load_state_dict(torch.load('./save_model/pre_discriminator_duel_model_epoch0.pt'))
    if dis_train == True :
        discriminator = torch.nn.DataParallel(discriminator, device_ids = [set_gpu_num]).cuda()
    
#model.cuda() # torch.cuda_set_device(device) 로 send
#model.to(device) # 직접 device 명시

#----- loss function -----
#criterion = nn.MSELoss().cuda()
signal_cr = nn.BCELoss().cuda()
human_cr = nn.CrossEntropyLoss().cuda()
cr = JointsMSELoss().cuda()
#----- optimizer and scheduler -----
args.lr = 1e-4
if args.optimizer == 'adam':
    if model_train == True :
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if gen_train == True :
        gen_optimizer = torch.optim.Adam(generator.parameters(), lr=3e-3) #1e-4 ~ 1e-2
        dis_optimizer = torch.optim.Adam(discriminator.parameters(), lr=7e-4)
        dis2_optimizer = torch.optim.Adam(discriminator2.parameters(), lr=1e-6)
    #if dis_train == True :
    
    logger.info('use adam optimizer')
else:
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4, nesterov=False)
    logger.info('use sgd optimizer')

#lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=args.gammas)
if model_train == True :
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma=args.gammas)
if gen_train == True :
    lr_scheduler_gen = torch.optim.lr_scheduler.StepLR(gen_optimizer, step_size = 3, gamma=2)
    lr_scheduler_dis = torch.optim.lr_scheduler.StepLR(dis_optimizer, step_size = 3, gamma=2)
    lr_scheduler_dis2 = torch.optim.lr_scheduler.StepLR(dis2_optimizer, step_size = 3, gamma=2)

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
#if model_train == True :
    #generator.eval()
#    model.train()
#if gen_train == True :
#    discriminator.eval()
#    generator.train()
#if dis_train == True :
#    discriminator.train()
for epoch in range(args.nepochs):
    
    if gen_train == True :
        if epoch % 2 == 1 :
            dis_train = False
            logger.info("generator_train")
            model.train()
            generator.train()
            discriminator.eval()
            discriminator2.eval()
        else :
            dis_train = True
            logger.info("discriminator_train")
            #model.eval()
            generator.eval()
            discriminator.train()
            discriminator2.train()
    if model_train == True and gen_train == False :
        model.train()
            
    if model_train == True :
        logger.info("Epoch {}\tcurrent lr : {} {}".format(epoch, optimizer.param_groups[0]['lr'], lr_scheduler.get_last_lr()))
    if gen_train == True :
        logger.info("Epoch {}\tcurrent lr : {} {}".format(epoch, gen_optimizer.param_groups[0]['lr'], lr_scheduler_gen.get_last_lr()))
    if dis_train == True :
        logger.info("Epoch {}\tcurrent lr : {} {}".format(epoch, dis_optimizer.param_groups[0]['lr'], lr_scheduler_dis.get_last_lr()))
    epoch_loss = []
    avg_acc = 0
    sum_acc = 0
    h_avg_acc = 0
    h_sum_acc = 0
    dis_avg_acc = 0
    dis_sum_acc = 0
    dis2_avg_acc = 0
    dis2_sum_acc = 0
    total_cnt = 0
    h_total_cnt = 0
    dis_total_cnt = 0
    dis2_total_cnt = 0
    iterate = 0
    total_use_loss = 0
    bool_dis2_train = False

    for rf, target_heatmap, target_label, antena_label, domain in tqdm(train_dataloader):
        #print(rf.shape, target_heatmap.shape)
        #print(rf.dtype, target_heatmap.dtype)
        #rf, target_heatmap, target_label, antena_label, domain = rf.cuda(), target_heatmap.cuda(), target_label.cuda(), antena_label.cuda(), domain.cuda()#rf 는 input raw signal
        rf, target_label, antena_label = rf.cuda(), target_label.cuda(), antena_label.cuda()
        #out = model(rf)
        
        
        
        loss = 0
        
        ###############discriminator pre train##############
        
        if dis_train == True and epoch == 0:
            discriminator.train()
            dis_label = torch.zeros(rf.shape[0]).cuda()
            new_rf = torch.zeros((rf.shape[0]),1792*2).cuda() #안테나 1개는 *3->*1로
            new_rf = new_rf.unsqueeze(0)
            new_rf = new_rf.view([-1, 2, 1792]) #안테나 1개는 *3->*1로
            for i in range(rf.shape[0]):
                if i < (int(rf.shape[0]/2)):
                    for k in range(int(rf.shape[0]/2)):
                        target = randint(0, rf.shape[0]-1)
                        if antena_label[i] == antena_label[target] and i != target:
                            break
                target_rf = rf[target]
                new_rf[i] = torch.cat([rf[i].unsqueeze(0), target_rf.unsqueeze(0)], dim = 0)
                if antena_label[i] == antena_label[target] :
                    dis_label[i] = float(1.0)
                else :
                    dis_label[i] = float(0.0)
            dis_label = dis_label.unsqueeze(1)
            dis_out = discriminator(new_rf)
            dis_out = dis_out.cuda()
                #print("dis_out", dis_out)
                #print("dis_label", dis_label)
            loss = signal_cr(dis_out, dis_label)
            #loss = my_BCELoss(dis_out, dis_label)
                #print(loss)
            dis_optimizer.zero_grad()
            if loss != 0 :
                loss.backward()
            dis_optimizer.step()
            #_, h_predict = torch.max(dis_out, 1)
            '''
        ########################discriminator2 train########################
        ###############step 1 : discriminator 2 #####################
        
        if epoch == 0 :#% 2 == 1:
            bool_dis2_train = True
            #continue
            rf = rf.float()
            generator.requires_grad = False
            discriminator2.requires_grad = True
            gen_rf = generator(rf.unsqueeze(1))
            gen_rf = gen_rf.cuda()
            
            dis_label = torch.ones(rf.shape[0]).cuda()
            dis2_label = torch.zeros(rf.shape[0]).cuda()
            dis_label = dis_label.unsqueeze(1)
            dis2_label = dis2_label.unsqueeze(1)

            
            dis_out = discriminator2(rf.unsqueeze(1))
            dis_out = dis_out.cuda()
            dis2_out = discriminator2(gen_rf)
            dis2_out = dis2_out.cuda()
            loss = signal_cr(dis2_out, dis2_label) # 진짜인지 아닌지 loss
            loss += signal_cr(dis_out, dis_label)
            #loss = my_BCELoss(dis_out, dis_label)
            if loss != 0 :
                dis2_optimizer.zero_grad()
                loss.backward()
                dis2_optimizer.step()
                torch.cuda.empty_cache()
                #_, h_predict = torch.max(dis_out, 1)
        if iterate % 1000 == 0 :
            print(loss)
            
        #########################for 2d################################
        #out, out_human = model(rf) # 32개의 13*120*120 out
        
        ################################################################
        
        #loss = 0.5 * criterion(out, target_heatmap)
        
        #by 승환
        ###################for d loss################################
        #d_loss, use_loss = dual_loss(out_human, target_label)
        #total_use_loss += use_loss
        ############################################################
        #loss = 10*cr(out, target_heatmap) + 0.005*id_loss + 30*d_loss # 0.03 기존
        #loss = 10*cr(out, target_heatmap) + 0.003*id_loss + 50*d_loss # 0.03 기존
        #loss = 0.005*id_loss + 30*d_loss 
        '''
        '''
        ###############discriminator pre train##############
        if epoch %2 == 1:
            generator.eval()
            discriminator.train()
            dis_label = torch.zeros(rf.shape[0]).cuda()
            new_rf = torch.zeros((rf.shape[0]),1792*2).cuda() #안테나 1개는 *3->*1로
            new_rf = new_rf.unsqueeze(0)
            new_rf = new_rf.view([-1, 2, 1792]) #안테나 1개는 *3->*1로
            
            rf = rf.float()
            gen_rf = generator(rf.unsqueeze(1))
            
            for i in range(rf.shape[0]):
                if i < (int(rf.shape[0]/2)):
                    for k in range(int(rf.shape[0]/2)):
                        target = randint(0, rf.shape[0]-1)
                        if antena_label[i] == antena_label[target] and i != target:
                            break
                target_rf = gen_rf[target]
                new_rf[i] = torch.cat([gen_rf[i], target_rf], dim = 0)
                if antena_label[i] == antena_label[target] :
                    dis_label[i] = float(1.0)
                else :
                    dis_label[i] = float(0.0)
            dis_label = dis_label.unsqueeze(1)
            dis_out = discriminator(new_rf)
            dis_out = dis_out.cuda()
                #print("dis_out", dis_out)
                #print("dis_label", dis_label)
            loss = signal_cr(dis_out, dis_label)
            #loss = my_BCELoss(dis_out, dis_label)
                #print(loss)
            dis_optimizer.zero_grad()
            if loss != 0 :
                loss.backward()
            dis_optimizer.step()
            #_, h_predict = torch.max(dis_out, 1)
        '''
        #############step 2 : generator######################
        if gen_train == True and epoch%2 == 1 :# > 0 :#% 2 == 1:
            generator.train()
            #discriminator2.eval()
            discriminator.eval()
            loss = 0
            generator.requires_grad = True
            discriminator.requires_grad = False
            discriminator2.requires_grad = False
            dis_label = torch.ones(rf.shape[0]).cuda() #무조건 1이 나오도록 학습 하기 위해 label 모두 1
            dis_label = dis_label.unsqueeze(1)
            #dis2_label = torch.ones(rf.shape[0]).cuda()
            #dis2_label = dis2_label.unsqueeze(1)
            
            
            
            new_rf = torch.zeros((rf.shape[0], 1792*2)).cuda() #1 안테나 1792*2 3안테나 1792*2*3 # 1792 default unet 936
            new_rf = new_rf.unsqueeze(1)
            new_rf = new_rf.view([-1, 2, 1792]) #1 안테나 1792, 3안테나 1792*3
            
            rf = rf.float()
            gen_rf = generator(rf.unsqueeze(1))
            gen_rf = gen_rf.cuda()
            #print(gen_rf.shape)
            #print((gen_rf).shape)
            #new_rf2 = gen_rf
            for i in range(rf.shape[0]):
                target = randint(0, rf.shape[0]-1)
                target_rf = gen_rf[target]
                new_rf[i] = torch.cat([gen_rf[i], target_rf], dim = 0)
                
            
            #dis2_out = discriminator2(gen_rf)
            #dis2_out = dis2_out.cuda()
            #real_loss = signal_cr(dis2_out, dis2_label) # 진짜인지 아닌지 loss
            
            #else :
            #continue
            dis_out = discriminator(new_rf)
            dis_out = dis_out.cuda()
            #print("dis_out", dis_out)
            #print("dis_label", dis_label)
            #signal_loss = my_BCELoss(dis_out, dis_label)
            signal_loss = signal_cr(dis_out, dis_label)
            '''
            if iterate % 3000 == 0:
                for k in range(gen_rf.shape[0]):
                    plt.plot(gen_rf[k].cpu())
                    plt.savefig('./train_sig/signal_plot_normal_{}.png'.format(k))
                    plt.clf()
            '''
            #loss += real_loss
            loss += 0.01*signal_loss
            #print(loss)
            
            #gen_optimizer.zero_grad()
            #loss.backward()
            #gen_optimizer.step()
            
            #loss.backward()
            #gen_optimizer.step()
            #_, h_predict = torch.max(dis_out, 1)
            
        ####################step 3 : model train########################
        if gen_train == True and epoch > 0 :#% 2 == 0 and epoch != 0 :
            model.train()
            #generator.eval()
            #generator.requires_grad = False
            if epoch % 2 == 0 :
                generator.eval()
                generator.requires_grad = False
                loss = 0
            
            gen_rf = generator(rf.unsqueeze(1))
            gen_rf = gen_rf.cuda()
            
            #rf = rf.view(-1, 1792)
            #print(rf.shape)
            #rf = rf.float()
            #gen_rf = generator(rf)
            #gen_rf = gen_rf.cuda()
            #print((target_rf).shape)
            out = model(gen_rf)
            out = out.cuda()
            #print(out.shape, target_label.shape)
            id_loss = human_cr(out, target_label)
            
            loss += id_loss
            #gen_optimizer.zero_grad()
            optimizer.zero_grad()
            if epoch % 2 == 1 :
                gen_optimizer.zero_grad()
            loss.backward()
            #gen_optimizer.step()
            if epoch %2 == 1 :
                gen_optimizer.step()
            optimizer.step()
            _, h_predict = torch.max(out, 1)
                
        ##############3only model##############
        if model_train == True and gen_train == False :
            rf = rf.float()
            out = model(rf.unsqueeze(1))
            out = out.cuda()
            id_loss = human_cr(out, target_label)
            loss += id_loss
            optimizer.zero_grad()
            loss.backward()
            #gen_optimizer.step()
            optimizer.step()
            _, h_predict = torch.max(out, 1)
            
    
        
        #_, h_predict = torch.max(out_human, 1) #for pose + id

        #epoch_loss.append(loss)
        #_, temp_avg_acc, cnt, pred = accuracy(out.detach().cpu().numpy(), target_heatmap.detach().cpu().numpy())
        if model_train == True :
            if epoch > 0 :#% 2 == 0 and epoch != 0:
                h_temp_avg_acc, h_cnt = human_accuracy(out, target_label)
                h_sum_acc += h_temp_avg_acc * h_cnt
                h_total_cnt += h_cnt
                h_avg_acc = h_sum_acc / h_total_cnt if h_total_cnt != 0 else 0
        #h_temp_avg_acc, h_cnt = human_accuracy(out_human, target_label) #for pose + id
        if dis_train == True :
            #print("target", dis_label)
            #print("output", dis_out)
            dis_temp_avg_acc, dis_cnt = dis_accuracy(dis_out, dis_label)
            dis_sum_acc += dis_temp_avg_acc*dis_cnt
            dis_total_cnt += dis_cnt
            dis_avg_acc = dis_sum_acc / dis_total_cnt if dis_total_cnt != 0 else 0
            
        
        if gen_train == True :
            if epoch >0 :#% 2 == 1:
                dis_temp_avg_acc, dis_cnt = dis_accuracy(dis_out, dis_label)
                dis_sum_acc += dis_temp_avg_acc*dis_cnt
                dis_total_cnt += dis_cnt
                dis_avg_acc = dis_sum_acc / dis_total_cnt if dis_total_cnt != 0 else 0
            if bool_dis2_train :
                dis2_temp_avg_acc, dis2_cnt = dis_accuracy(dis2_out, dis2_label)
                dis2_sum_acc += dis2_temp_avg_acc*dis2_cnt
                dis2_total_cnt += dis2_cnt
                dis2_avg_acc = dis2_sum_acc / dis2_total_cnt if dis2_total_cnt != 0 else 0
            
        #sum_acc += temp_avg_acc * cnt 
        sum_acc = 0 # pose 안할 때
        #total_cnt += cnt
        total_cnt = 0 # pose 안할 때
        avg_acc = sum_acc / total_cnt if total_cnt != 0 else 0        
        avg_acc = 0 # pose 안할 때
        
        if iterate % 500 == 0:
        #    for i in range(rf_data.shape[0]):
        #        plt.plot(gen_rf[i])
        #        plt.savefig('./train_sig/signal_plot_normal_{}.png'.format(i))
        #        plt.clf()
        
            #logger.info("iteration[%d] batch loss %.6f\tavg_acc\t%.6f\th_avg_acc %.4f\ttotal_count %d\tpeople_loss %.6f"%(iterate, loss.item(), avg_acc, h_avg_acc, total_cnt, id_loss))
            if model_train == True and epoch >0 :#%2 == 0 and epoch != 0:
                logger.info("iteration[%d] batch loss %.6f\th_avg_acc %.4f\ttotal_count %d\tpeople_loss "%(iterate, id_loss.item(), h_avg_acc, total_cnt))
            if model_train == True and gen_train == False :
                logger.info("iteration[%d] batch loss %.6f\th_avg_acc %.4f\ttotal_count %d\tpeople_loss "%(iterate, id_loss.item(), h_avg_acc, total_cnt))
            if dis_train == True :
                logger.info("iteration[%d] batch loss %.6f\tdis_avg_acc %.4f\ttotal_count %d"%(iterate, loss, dis_avg_acc, total_cnt))
                if bool_dis2_train :
                    logger.info("iteration[%d] batch loss %.6f\tdis_avg_acc %.4f\ttotal_count %d"%(iterate, loss, dis2_avg_acc, total_cnt))
            if gen_train == True and epoch>0:#epoch % 2 == 1:
                #logger.info("iteration[%d] batch loss %.6f\tdiscriminate_avg_acc %.4f\ttotal_count %d\tdis2_avg_acc %.4f "%(iterate, signal_loss.item(), dis_avg_acc, total_cnt,  dis2_avg_acc))
                logger.info("iteration[%d] batch loss %.6f\tdiscriminate_avg_acc %.4f "%(iterate, signal_loss.item(), dis_avg_acc))
        iterate += 1
        
    #logger.info("human predict : {} \t human_label : {}".format(out, target_label))
    #logger.info("human predict : {} \t human_label : {}".format(out_human, target_label)) #for pose + id
    #logger.info("epoch loss : %.6f"%torch.tensor(epoch_loss).mean().item()) #for pose 
    #logger.info("epoch acc on train data : %.4f"%(avg_acc)) #for pose
    if model_train == True and gen_train == True:
        if epoch > 0 :#% 2 == 0 and epoch != 0 :
            logger.info("epoch h_acc on train data : %.4f"%(h_avg_acc))
            lr_scheduler.step()
            torch.save(model.module.state_dict(), "save_model/" + model_name + "_epoch{}.pt".format(int(epoch)-1))#(int(epoch/2)-1))
    if model_train == True and gen_train== False :
        logger.info("epoch h_acc on train data : %.4f"%(h_avg_acc))
        lr_scheduler.step()
        torch.save(model.module.state_dict(), "save_model/" + model_name + "_epoch{}.pt".format(epoch))
    if dis_train == True :
        logger.info("epoch dis_acc on train data : %.4f\t%.4f"%(dis_avg_acc, dis2_avg_acc))
        
        #if epoch == 0 :
        lr_scheduler_dis.step()
        torch.save(discriminator.module.state_dict(), "save_model/" + dis_name + "_epoch{}.pt".format(epoch))
    if gen_train == True :
        if epoch >0 :#% 2 == 0 and epoch != 0 :
        #logger.info("epoch gen_acc on train data : %.4f"%(h_avg_acc))
            #logger.info("generator makes discriminator : {}\nlabel : {}".format(dis_out, dis_label))
            #logger.info("discriminator : {}\nlabel : {}".format(dis2_out, dis2_label))
            lr_scheduler_gen.step()
            torch.save(generator.module.state_dict(), "save_model/" + gen_name + "_epoch{}.pt".format(int(epoch)-1))#(int(epoch/2)-1))
    #logger.info("use d_loss : %d"%(total_use_loss)) #dloss 사용 횟수
    
    if avg_acc > max_acc:
        logger.info("epoch {} acc {} > max acc {} epoch {}".format(epoch, avg_acc, max_acc, max_acc_epoch))
        max_acc = avg_acc
        max_acc_epoch = epoch
        #if args.multi_gpu == 1:
        torch.save(model.module.state_dict(), "save_model/" + model_name + "_best.pt")
        #else:
            #torch.save(model.state_dict(), "save_model/" + model_name + "_best.pt")
    
    #if args.multi_gpu == 1:
    

    #else:
    #    torch.save(model.state_dict(), "save_model/" + model_name + "_epoch{}.pt".format(epoch))

logger.info("training end | elapsed time = " + str(datetime.now() - begin_time))


