import torch
import numpy as np

from torch.utils.data import DataLoader
import time
from torch.utils.tensorboard import SummaryWriter

from dataset import MyDataset
import tools
from collections import OrderedDict
from Dmodel import DRFKNet
import visualizer
from GraFormer import adj_mx_from_edges, GraFormer
from semgcn.Semgcn import SemGCN
from GCT import GCT
def main():
    # Network Arguments
    args = {}
    args['use_cuda'] = True
    args['isTrain'] = True
    args['isGNN'] = True
    args['isSemgcn'] = False
    args['OutputDM'] = '9D'
    print(args['isSemgcn'])

    src_mask = torch.tensor([[[True, True, True, True, True, True, True, True, True, True, True,
                               True]]]).cuda()
    writer = SummaryWriter("logs")
    # Initialize network
    # net = DRFKNet().double()
    # k-hop matrix
    gan_edges = (np.array([[1, 2], [1, 6], [2, 3], [3, 4], [4, 5], [5, 6], [7, 8],
                          [7, 12], [8, 9], [9, 10], [10, 11], [11, 12], [1, 7],
                          [2, 8], [3, 9], [4, 10], [5, 11], [6, 12]])-1)

    gan_edges1 = (np.array([[1, 8], [1, 12], [1, 3], [1, 5], [2, 9], [2, 7], [2, 4],
                          [2, 6], [3, 10], [3, 8], [3, 5], [3, 1], [4, 11],
                          [4, 9], [4, 6], [4, 2], [5, 12], [5, 10], [5, 1], [5, 3], [6, 7], [6, 11], [6, 2], [6, 4]])-1)

    gan_edges2 = (np.array([[1, 9], [1, 11], [1, 4],
                          [2, 10], [2, 12],  [2, 5],
                          [3, 11], [3, 7], [3, 6],
                          [4, 12], [4, 8], [4, 1],
                          [5, 7], [5, 9], [5, 2],
                          [6, 8], [6, 10], [6, 3]])-1)
    adj = adj_mx_from_edges(num_pts=12, edges=gan_edges, sparse= False).cuda()
    adj1 = adj_mx_from_edges(num_pts=12, edges=gan_edges1, sparse=False).cuda()
    adj2 = adj_mx_from_edges(num_pts=12, edges=gan_edges2, sparse=False).cuda()
    net = GCT(adj, adj1, adj2).double()

    device = torch.device('cuda:0')
    #pre train model
    # path = "Semgcn_499_9D.pth"
    # checkpoint = torch.load(path, map_location=device)
    # net = checkpoint
    # dict_trained = torch.load('best_modeldict_9D_325.pth', map_location=torch.device('cpu'))
    # dict_new = net.state_dict()
    # net_dict = net.state_dict()
    # 1. filter out unnecessary keys
    # dict_trained = {k: v for k, v in dict_trained.items() if k in dict_new}
    # 2. overwrite entries in the existing state dict
    # net_dict.update(dict_trained)
    # net.load_state_dict(dict_new)
    net.to(device)
    ## Initialize optimizer
    ## pretrainEpochs = 5
    trainEpochs = 800
    mean_criterion1 = torch.nn.MSELoss(reduction='mean')
    # mean_criterion2 = torch.nn.MSELoss(reduction='mean')
    lr = 0.0001

    batch_size = 4000
    beta = 200
    gamma = 0.5
    total_steps = 0
    print_freq = 10*10000
    save_epoch_freq = 300
    ## Initialize data loaders
    data = MyDataset()
    trSet = data[0: 40*10000, :]
    valSet = data[40*10000:50*10000, :]
    train_num = 0
    test_num = 0
    iteration = 0
    log_frequency = 25
    val_frequency = 25

    trDataloader = DataLoader(trSet, batch_size=batch_size, shuffle=False, num_workers=0)
    dataset_size = len(trDataloader)
    print(dataset_size)

    valDataloader = DataLoader(valSet, batch_size=batch_size, shuffle=False, num_workers=0)
    testset_size = len(valDataloader)
    print(testset_size)
    for epoch in range(trainEpochs):
        epoch_start_time = time.time()
        epoch_iter = 0
        loss = 0
        loss_val = 0
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        for i, data in enumerate(trDataloader):
            iter_start_time = time.time()
            total_steps += batch_size
            epoch_iter += batch_size
            input_dis = data[:, 8: 14].cuda()
            output_pose = data[:, 1:8]
            gt_motion = output_pose[:, 0:3].cuda()
            gt_rotation = output_pose[:, 3:7].cuda()
            input_data = data[:, 14:158]
            data1 = input_data.view(input_data.shape[0], 12, 12)
            nonZeroRows = torch.abs(data1) > 0
            input_feature = data1[nonZeroRows].view(input_data.shape[0], 12, 3).cuda()

            train_num += 1


            # Forward pass

            pre_value = net(input_feature, src_mask)

            # Loss computation
            pre_motion = pre_value[:, 0:3].cuda()
            pre_rotation = pre_value[:, 3:].cuda()


            loss_weights = [0.3, 0.3, 1]
            # rm1 = tools.compute_rotation_matrix_from_quaternion(pre_rotation).cuda()
            rm2 = tools.compute_rotation_matrix_from_quaternion(gt_rotation).cuda()
            orm1 = tools.symmetric_orthogonalization(pre_rotation).cuda()


            mse_pos = mean_criterion1(pre_motion, gt_motion)
            mse_ori = mean_criterion1(rm2, orm1)

            # mse_dis = mean_criterion1(Ikp_dis, l_dis)

            loss_G = (mse_pos + mse_ori * beta)
            loss += loss_G
            # Backprop and update weights
            optimizer.zero_grad()
            loss_G.backward()
            optimizer.step()

            #  lr = adjust_learning_rate_by_epoch(optimizer, epoch, trainEpochs)
            if iteration % log_frequency == 0:
                err_pos = torch.dist(pre_motion,gt_motion)
                err_deg = torch.rad2deg(tools.compute_geodesic_distance_from_two_matrices(rm2,orm1))
                writer.add_scalar('train/pos_mean', err_pos.mean().item(), iteration)
                writer.add_scalar('train/ori_mean', err_deg.mean().item(), iteration)



            if total_steps % print_freq == 0:
                errors = get_current_errors(pre_motion, gt_motion, rm2, orm1, args['isTrain'])
                t = (time.time() - iter_start_time) / batch_size
                visualizer.Visualizer.print_current_errors(None,epoch, epoch_iter, errors, t)
            iteration += 1
                # visualizer.plot_current_errors(epoch, float(epoch_iter) / dataset_size, errors)
        print('loss value is : %f \t' % (loss/dataset_size))
        print('End of epoch %d \t Time Taken: %d sec' %
              (epoch, time.time() - epoch_start_time))
        writer.add_scalar("trian_loss", (loss/dataset_size), epoch+1)
        writer.add_scalar('train/lr', lr, epoch+1)


        # val set
        with torch.no_grad():
            test_err_deg = np.array([])
            test_err_pos = np.array([])
            for i, data in enumerate(valDataloader):
                test_num += 1
                iter_start_time = time.time()
                total_steps += batch_size
                epoch_iter += batch_size
                input_dis = data[:, 8: 14].cuda()
                output_pose = data[:, 1:8]
                gt_motion = output_pose[:, 0:3].cuda()
                gt_rotation = output_pose[:, 3:7].cuda()
                input_data = data[:, 14:158]
                data1 = input_data.view(input_data.shape[0], 12, 12)
                nonZeroRows = torch.abs(data1) > 0
                input_feature = data1[nonZeroRows].view(input_data.shape[0], 12, 3).cuda()


                # Forward pass

                pre_value = net(input_feature, src_mask)


                # Loss computation
                pre_motion = pre_value[:, 0:3].cuda()
                pre_rotation = pre_value[:, 3:].cuda()

                rm2 = tools.compute_rotation_matrix_from_quaternion(gt_rotation).cuda()
                orm1 = tools.symmetric_orthogonalization(pre_rotation).cuda()
                mse_pos = mean_criterion1(pre_motion, gt_motion)
                mse_ori = mean_criterion1(rm2, orm1)
                loss_v = (mse_pos + mse_ori * beta)
                loss_val += loss_v

                if iteration % val_frequency == 0:
                    # err_pos = torch.dist(pre_motion, gt_motion)
                    err_pos1 = torch.abs((pre_motion-gt_motion))
                    err_deg = torch.rad2deg(tools.compute_geodesic_distance_from_two_matrices(rm2, orm1))
                    writer.add_scalar('test/pos_mean', err_pos.mean().item(), iteration)

                    #ori_error
                    test_err_deg = np.append(test_err_deg, err_deg.detach().cpu().numpy())
                    writer.add_scalar('test/ori_err_median', np.median(test_err_deg), iteration)
                    writer.add_scalar('test/ori_err_mean', np.mean(test_err_deg), iteration)
                    writer.add_scalar('test/ori_err_max', np.max(test_err_deg), iteration)
                    writer.add_scalar('test/acc_0.1deg', (test_err_deg < 0.1).sum() / len(test_err_deg),
                                      iteration)
                    writer.add_scalar('test/acc_0.2deg', (test_err_deg < 0.2).sum() / len(test_err_deg), iteration)
                    writer.add_scalar('test/acc_0.3deg', (test_err_deg < 0.3).sum() / len(test_err_deg), iteration)
                    writer.add_scalar('test/acc_0.4deg', (test_err_deg < 0.4).sum() / len(test_err_deg), iteration)
                    writer.add_scalar('test/acc_0.5deg', (test_err_deg < 0.5).sum() / len(test_err_deg), iteration)
                    writer.add_scalar('test/acc_0.6deg', (test_err_deg < 0.6).sum() / len(test_err_deg), iteration)
                    writer.add_scalar('test/acc_0.7deg', (test_err_deg < 0.7).sum() / len(test_err_deg),
                                      iteration)
                    writer.add_scalar('test/acc_0.8deg', (test_err_deg < 0.8).sum() / len(test_err_deg), iteration)
                    writer.add_scalar('test/acc_0.9deg', (test_err_deg < 0.9).sum() / len(test_err_deg), iteration)
                    writer.add_scalar('test/acc_1.0deg', (test_err_deg < 1.0).sum() / len(test_err_deg), iteration)
                    writer.add_scalar('test/acc_1.1deg', (test_err_deg < 1.1).sum() / len(test_err_deg), iteration)
                    writer.add_scalar('test/acc_1.2deg', (test_err_deg < 1.2).sum() / len(test_err_deg), iteration)
                    writer.add_scalar('test/acc_1.3deg', (test_err_deg < 1.3).sum() / len(test_err_deg), iteration)
                    writer.add_scalar('test/acc_1.4deg', (test_err_deg < 1.4).sum() / len(test_err_deg), iteration)
                    writer.add_scalar('test/acc_1.5deg', (test_err_deg < 1.5).sum() / len(test_err_deg), iteration)
                    writer.add_scalar('test/acc_2.0deg', (test_err_deg < 2.0).sum() / len(test_err_deg),
                                      iteration)
                    writer.add_scalar('test/acc_3.0deg', (test_err_deg < 3.0).sum() / len(test_err_deg),
                                      iteration)
                    # pos_error
                    test_err_pos = np.append(test_err_pos, err_pos1.detach().cpu().numpy())
                    writer.add_scalar('test/pos_err_median', np.median(test_err_pos), iteration)
                    writer.add_scalar('test/pos_err_mean', np.mean(test_err_pos), iteration)
                    writer.add_scalar('test/pos_err_max', np.max(test_err_pos), iteration)
                    writer.add_scalar('test/acc_0.1mm', (test_err_pos < 0.01).sum() / len(test_err_pos), iteration)
                    writer.add_scalar('test/acc_0.2mm', (test_err_pos < 0.02).sum() / len(test_err_pos), iteration)
                    writer.add_scalar('test/acc_0.3mm', (test_err_pos < 0.03).sum() / len(test_err_pos), iteration)
                    writer.add_scalar('test/acc_0.4mm', (test_err_pos < 0.04).sum() / len(test_err_pos), iteration)
                    writer.add_scalar('test/acc_0.5mm', (test_err_pos < 0.05).sum() / len(test_err_pos), iteration)
                    writer.add_scalar('test/acc_0.6mm', (test_err_pos < 0.06).sum() / len(test_err_pos), iteration)
                    writer.add_scalar('test/acc_0.7mm', (test_err_pos < 0.07).sum() / len(test_err_pos), iteration)
                    writer.add_scalar('test/acc_0.8mm', (test_err_pos < 0.08).sum() / len(test_err_pos), iteration)
                    writer.add_scalar('test/acc_0.9mm', (test_err_pos < 0.09).sum() / len(test_err_pos), iteration)
                    writer.add_scalar('test/acc_1.0mm', (test_err_pos < 0.1).sum() / len(test_err_pos), iteration)
                    writer.add_scalar('test/acc_1.1mm', (test_err_pos < 0.11).sum() / len(test_err_pos), iteration)
                    writer.add_scalar('test/acc_1.2mm', (test_err_pos < 0.12).sum() / len(test_err_pos), iteration)
                    writer.add_scalar('test/acc_1.3mm', (test_err_pos < 0.13).sum() / len(test_err_pos), iteration)
                    writer.add_scalar('test/acc_1.4mm', (test_err_pos < 0.14).sum() / len(test_err_pos), iteration)
                    writer.add_scalar('test/acc_1.5mm', (test_err_pos < 0.15).sum() / len(test_err_pos), iteration)
                    writer.add_scalar('test/acc_1.6mm', (test_err_pos < 0.16).sum() / len(test_err_pos),
                                      iteration)
                    writer.add_scalar('test/acc_1.7mm', (test_err_pos < 0.17).sum() / len(test_err_pos),
                                      iteration)
                    writer.add_scalar('test/acc_1.8mm', (test_err_pos < 0.18).sum() / len(test_err_pos),
                                      iteration)
                    writer.add_scalar('test/acc_1.9mm', (test_err_pos < 0.19).sum() / len(test_err_pos),
                                      iteration)
                    writer.add_scalar('test/acc_2.0mm', (test_err_pos < 0.2).sum() / len(test_err_pos),iteration)
                    writer.add_scalar('test/acc_2.5mm', (test_err_pos < 0.25).sum() / len(test_err_pos),iteration)
                    writer.add_scalar('test/acc_3.0mm', (test_err_pos < 0.3).sum() / len(test_err_pos), iteration)
                    writer.add_scalar('test/acc_3.5mm', (test_err_pos < 0.35).sum() / len(test_err_pos), iteration)
                    writer.add_scalar('test/acc_4.0mm', (test_err_pos < 0.4).sum() / len(test_err_pos), iteration)
                    writer.add_scalar('test/acc_4.5mm', (test_err_pos < 0.45).sum() / len(test_err_pos), iteration)
                    writer.add_scalar('test/acc_5.0mm', (test_err_pos < 0.5).sum() / len(test_err_pos), iteration)


            if (loss / dataset_size) < 0.025:
                torch.save(net, "best_model_9D_{}.pth".format(epoch))
                torch.save(net.state_dict(), "best_modeldict_9D_{}.pth".format(epoch))

            print('val loss is : %f \t'%(loss_val/testset_size) )
            writer.add_scalar("test_loss", (loss_val / testset_size), epoch)
        print('-----------------------------------------------------------------------')



    if args['isGNN']:
        if args['OutputDM']=='4D':
            torch.save(net, "model_{}_4D.pth".format(epoch))
            torch.save(net.state_dict(), "model_dict_4D_{}.pth".format(epoch))
        elif args['OutputDM']=='6D':
            torch.save(net, "model_{}_6D.pth".format(epoch))
            torch.save(net.state_dict(), "model_dict_6D_{}.pth".format(epoch))
        elif args['OutputDM'] == '9D':
            torch.save(net, "model_{}_9D_rad.pth".format(epoch))
            torch.save(net.state_dict(), "model_dict_9D_rad_{}.pth".format(epoch))
        elif args['OutputDM'] == '10D':
            torch.save(net, "model_{}_10D.pth".format(epoch))
            torch.save(net.state_dict(), "model_dict_10D_{}.pth".format(epoch))
        elif args['OutputDM'] == 'Euler':
            torch.save(net, "model_{}_Euler.pth".format(epoch))
            torch.save(net.state_dict(), "model_dict_Euler_{}.pth".format(epoch))
    elif args['isSemgcn']:
        torch.save(net, "Semgcn_{}_9D.pth".format(epoch))
        torch.save(net.state_dict(), "Semgcn_dict_9D_{}.pth".format(epoch))
    else: torch.save(net, "DNN_model_{}.pth".format(epoch))



def get_current_errors(mpred, minput, rpred, rinput, isTrain):
    pos_err = torch.dist(mpred, minput)
    ori_err = tools.compute_geodesic_distance_from_two_matrices(rpred, rinput)
    ori_err = ori_err.mean() * 180 / np.pi
    if isTrain:
        return OrderedDict([('pos_err', pos_err),
                                ('ori_err', ori_err),
                                ])
    else:
        return [pos_err.item(), ori_err.item()]








if __name__ == '__main__':
    main()



