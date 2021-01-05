import argparse
import time
import torch
import numpy as np
import torch.optim as optim

# custom modules

from loss import MonodepthLoss
from utils import get_model, to_device, prepare_dataloader
import scipy.io
# plot params
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (15, 10)
import cv2
from tqdm import tqdm
def return_arguments():
    parser = argparse.ArgumentParser(description='PyTorch Monodepth')

    parser.add_argument('--data_dir', default="Kaist_data",
                        help='path to the dataset folder. \
                        It should contain subfolders with following structure:\
                        "image_02/data" for left images and \
                        "image_03/data" for right images'
                        )
    parser.add_argument('--val_data_dir', default="Kaist_data",
                        help='path to the validation dataset folder. \
                            It should contain subfolders with following structure:\
                            "image_02/data" for left images and \
                            "image_03/data" for right images'
                        )
    parser.add_argument('--model_path', default="models",help='path to the trained model')
    parser.add_argument('--output_directory', default="models",
                        help='where save dispairities\
                        for tested images'
                        )
    parser.add_argument('--input_height', type=int, help='input height',
                        default=448)
    parser.add_argument('--input_width', type=int, help='input width',
                        default=512)
    parser.add_argument('--model', default='resnet18_md',
                        help='encoder architecture: ' +
                        'resnet18_md or resnet50_md ' + '(default: resnet18)'
                        + 'or torchvision version of any resnet model'
                        )
    parser.add_argument('--pretrained', default=False,
                        help='Use weights of pretrained model'
                        )
    parser.add_argument('--mode', default='train',
                        help='mode: train or test (default: train)')
    parser.add_argument('--epochs', default=100,
                        help='number of total epochs to run')
    parser.add_argument('--learning_rate', default=5e-4,
                        help='initial learning rate (default: 1e-4)')
    parser.add_argument('--batch_size', default=8,
                        help='mini-batch size (default: 256)')
    parser.add_argument('--adjust_lr', default=True,
                        help='apply learning rate decay or not\
                        (default: True)'
                        )
    parser.add_argument('--device',
                        default='cuda:0',
                        help='choose cpu or cuda:0 device"'
                        )
    parser.add_argument('--do_augmentation', default=True,
                        help='do augmentation of images or not')
    parser.add_argument('--augment_parameters', default=[
        0.8,
        1.2,
        0.5,
        2.0,
        0.8,
        1.2,
        ],
            help='lowest and highest values for gamma,\
                        brightness and color respectively'
            )
    parser.add_argument('--print_images', default=False,
                        help='print disparity and image\
                        generated from disparity on every iteration'
                        )
    parser.add_argument('--print_weights', default=False,
                        help='print weights of every layer')
     parser.add_argument('--RGB', default=False,
                        help='Use RGB for input')
    parser.add_argument('--input_channels', default=3,
                        help='Number of channels in input tensor')
    parser.add_argument('--num_workers', default=4,type=int, 
                        help='Number of workers in dataloader')
    parser.add_argument('--use_multiple_gpu', default=True)
    args = parser.parse_args()
    return args


def adjust_learning_rate(optimizer, epoch, learning_rate):
    """Sets the learning rate to the initial LR\
        decayed by 2 every 10 epochs after 30 epoches"""

    if epoch >= 60 and epoch < 75:
        lr = learning_rate / 2
    elif epoch >= 40:
        lr = learning_rate / 4
    else:
        lr = learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def post_process_disparity(disp):
    (_, h, w) = disp.shape
    l_disp = disp[0, :, :]
    r_disp = np.fliplr(disp[1, :, :])
    m_disp = 0.5 * (l_disp + r_disp)
    (l, _) = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


class Model:

    def __init__(self, args):
        self.args = args

        # Set up model
        self.device = args.device
        self.model = get_model(args.model, input_channels=args.input_channels, pretrained=args.pretrained)
        self.model = self.model.to(self.device)
        if args.use_multiple_gpu:
            self.model = torch.nn.DataParallel(self.model)

        if args.mode == 'train':
            self.loss_function = MonodepthLoss(
                n=4,
                SSIM_w=0.85,
                disp_gradient_w=0.1, lr_w=1).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=args.learning_rate)
            self.val_n_img, self.val_loader = prepare_dataloader(args.val_data_dir, "val",
                                                                 args.augment_parameters,
                                                                 False, args.batch_size,
                                                                 (args.input_height, args.input_width),
                                                                 args.num_workers)
        else:
            self.model.load_state_dict(torch.load(args.model_path))
            args.augment_parameters = None
            args.do_augmentation = False
            args.batch_size = 1

        # Load data
        self.output_directory = args.output_directory
        self.input_height = args.input_height
        self.input_width = args.input_width

        self.n_img, self.loader = prepare_dataloader(args.data_dir, args.mode, args.augment_parameters,
                                                     args.do_augmentation, args.batch_size,
                                                     (args.input_height, args.input_width),
                                                     args.num_workers,RGB=args.RGB)


        if 'cuda' in self.device:
            torch.cuda.synchronize()


    def train(self):
        losses = []
        val_losses = []
        best_loss = float('Inf')
        best_val_loss = float('Inf')

        running_val_loss = 0.0
        self.model.eval()
        for data in self.val_loader:
            data = to_device(data, self.device)
            left = data['left_image']
            right = data['right_image']
            thermal = data['thermal_image']
            if self.args.RGB:
                disps = self.model(left)
            else:
                disps = self.model(thermal)
            loss = self.loss_function(disps, [left, right])
            val_losses.append(loss.item())
            running_val_loss += loss.item()

        running_val_loss /= self.val_n_img / self.args.batch_size
        print('Val_loss:', running_val_loss)
        
        for epoch in range(self.args.epochs):
            if self.args.adjust_lr:
                adjust_learning_rate(self.optimizer, epoch,
                                     self.args.learning_rate)
            c_time = time.time()
            running_loss = 0.0
            self.model.train()
            for data in tqdm(self.loader):
                # Load data
                data = to_device(data, self.device)
                left = data['left_image']
                right = data['right_image']
                thermal = data['thermal_image']
            
                # One optimization iteration
                self.optimizer.zero_grad()
                if self.args.RGB:
                    disps = self.model(left)
                else:
                    disps = self.model(thermal)
                    
                loss = self.loss_function(disps, [left, right])
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())

                # Print statistics
                if self.args.print_weights:
                    j = 1
                    for (name, parameter) in self.model.named_parameters():
                        if name.split(sep='.')[-1] == 'weight':
                            plt.subplot(5, 9, j)
                            plt.hist(parameter.data.view(-1))
                            plt.xlim([-1, 1])
                            plt.title(name.split(sep='.')[0])
                            j += 1
                    plt.show()

                if self.args.print_images:
                    print('disp_left_est[0]')
                    plt.imshow(np.squeeze(
                        np.transpose(self.loss_function.disp_left_est[0][0,
                                     :, :, :].cpu().detach().numpy(),
                                     (1, 2, 0))))
                    plt.show()
                    print('left_est[0]')
                    plt.imshow(np.transpose(self.loss_function\
                        .left_est[0][0, :, :, :].cpu().detach().numpy(),
                        (1, 2, 0)))
                    plt.show()
                    print('disp_right_est[0]')
                    plt.imshow(np.squeeze(
                        np.transpose(self.loss_function.disp_right_est[0][0,
                                     :, :, :].cpu().detach().numpy(),
                                     (1, 2, 0))))
                    plt.show()
                    print('right_est[0]')
                    plt.imshow(np.transpose(self.loss_function.right_est[0][0,
                               :, :, :].cpu().detach().numpy(), (1, 2,
                               0)))
                    plt.show()
                running_loss += loss.item()

            running_val_loss = 0.0
            self.model.eval()
            for data in self.val_loader:
                data = to_device(data, self.device)
                left = data['left_image']
                right = data['right_image']
                thermal = data['thermal_image']
            
                if self.args.RGB:
                    disps = self.model(left)
                else:
                    disps = self.model(thermal)
                loss = self.loss_function(disps, [left, right])
                val_losses.append(loss.item())
                running_val_loss += loss.item()

            # Estimate loss per image
            running_loss /= self.n_img / self.args.batch_size
            running_val_loss /= self.val_n_img / self.args.batch_size
            print (
                'Epoch:',
                epoch + 1,
                'train_loss:',
                running_loss,
                'val_loss:',
                running_val_loss,
                'time:',
                round(time.time() - c_time, 3),
                's',
                )
            self.save(self.args.model_path + '_last.pth')
            if running_val_loss < best_val_loss:
                self.save(self.args.model_path + '_cpt.pth')
                best_val_loss = running_val_loss
                print('Model_saved')

        print ('Finished Training. Best loss:', best_val_loss)
        self.save(self.args.model_path)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def test(self):
        self.model.eval()
        disparities = np.zeros((self.n_img,
                               self.input_height, self.input_width),
                               dtype=np.float32)
        disparities_pp = np.zeros((self.n_img,
                                  self.input_height, self.input_width),
                                  dtype=np.float32)
        with torch.no_grad():
            for (i, data) in enumerate(self.loader):
                # Get the inputs
                data = to_device(data, self.device)
                left = data.squeeze()
                # Do a forward pass
                disps = self.model(left)
                disp = disps[0][:, 0, :, :].unsqueeze(1)
                disparities[i] = disp[0].squeeze().cpu().numpy()
                disparities_pp[i] = \
                    post_process_disparity(disps[0][:, 0, :, :]\
                                           .cpu().numpy())

        np.save(self.output_directory + '/disparities.npy', disparities)
        np.save(self.output_directory + '/disparities_pp.npy',
                disparities_pp)
        print('Finished Testing')
    def viz(self):
        self.model.eval()

        import cv2

        cmap = plt.get_cmap('magma');
        cval = [[cmap(x)[0],cmap(x)[1],cmap(x)[2]] for x in range(0,255)];
        cval.append( cval[-1] );
        cval = np.array( cval );

        max_depth = 80;
        min_depth = 1;
        min_disp  = 1;
        def compute_depth_errors(gt, pred):
            """Computation of error metrics between predicted and ground truth depths
            """
            # thresh = torch.max((gt / pred), (pred / gt))
            # a1 = (thresh < 1.25     ).float().mean()
            # a2 = (thresh < 1.25 ** 2).float().mean()
            # a3 = (thresh < 1.25 ** 3).float().mean()

            rmse = (gt - pred) ** 2
            rmse = np.sqrt(rmse.mean())

            rmse_log = (np.log(gt) - np.log(pred)) ** 2
            rmse_log = np.sqrt(rmse_log.mean())

            abs_rel = np.mean(np.abs(gt - pred) / gt)

            sq_rel = np.mean((gt - pred) ** 2 / gt)

            return abs_rel, sq_rel, rmse, rmse_log#, a1, a2, a3

        def disp2depth( disp, max_disp ):
            disp[ disp < min_disp ] = min_disp;
            depth = (3233.93339530 * 0.245) / disp;
            depth[depth < min_depth] = min_depth;
            depth[depth > max_depth] = max_depth;

            return depth

        def disp2disp( disp, max_disp ):
            disp[ disp < min_disp ] = min_disp;
            disp = ( disp / max_disp ) * 255.;
            disp[ disp < min_disp ] = min_disp;
            disp = cval[(disp).astype(np.uint8)];
            disp = (disp*255.).astype(np.uint8);
            disp = disp[:,:,[2,1,0]];
            return disp
        def colormap(depth):
            depth = depth * (255./max_depth);
            depth = cval[(depth).astype(np.uint8)];
            depth = (depth*255.).astype(np.uint8);
            depth = depth[:,:,[2,1,0]];
            return depth

        if not os.path.exists(self.output_directory):
           os.makedirs( self.output_directory );
        txt=txt_depth=open("MTN_data/txt/test_depth.txt","r")
        depthdata=[]
        for line in txt:
            depthdata.append(os.path.join("MTN_data",line[:-1]))
        abs_rels, sq_rels, rmses, rmse_logs =[],[],[],[]
        with torch.no_grad():
            for (i, data) in enumerate(tqdm(self.loader)):
                # Get the inputs
                ###
                
                Depth=scipy.io.loadmat(depthdata[i])["depth"]
                ###
                data = to_device(data, self.device)
                left = data.squeeze()
                # Do a forward pass
                disps    = self.model(left)
                disp     = disps[0][:, 0, :, :].unsqueeze(1)
                disp     = disp[0].squeeze().cpu().numpy()
                disp_pp  = post_process_disparity(disps[0][:, 0, :, :].cpu().numpy())
                #disp     = disp *1280#* disp_pp.shape[1];
                disp_pp  = disp_pp *1280# disp_pp.shape[1];
                #depth    = disp2depth( disp, 0.1*disp_pp.shape[1] );
                depth_pp = disp2depth( disp_pp, 0.1*disp_pp.shape[1] );
                left     = left[0].permute((1,2,0)).cpu().numpy() * 255.;
                abs_rel, sq_rel, rmse, rmse_log=compute_depth_errors(Depth,depth_pp)
 
                abs_rels.append(abs_rel)
                sq_rels.append(sq_rel)
                rmses.append(rmse)
                rmse_logs.append(rmse_log)
                
                depth_pp=colormap(depth_pp)
                Depth=colormap(Depth)
                viz      = np.concatenate( (left, Depth, depth_pp), 1 );
                cv2.imwrite(os.path.join( self.output_directory, '%06d.jpg' %(i)), viz);
                
        print("abs_rel: ",np.array(abs_rels).mean()," sq_rels: ",np.array(sq_rels).mean()," rmses: ",np.array(rmses).mean()," rmse_logs: ",np.array(rmse_logs).mean())
        save_folder=self.output_directory
        pathIn=self.output_directory
        pathOut = save_folder+'.mp4'
        fps = 10
        frame_array = []
        for idx , path in enumerate(os.listdir(pathIn)) :
            
            if path[0]==".":
                continue
            if "npy" in path:
                #import pdb;pdb.set_trace()
                continue
            img = cv2.imread(os.path.join(pathIn,path))
            try:
                height, width, layers = img.shape
            except:
                import pdb;pdb.set_trace()
            size = (width,height)
            frame_array.append(img)

        out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'MP4V'), fps, size)
        for i in range(len(frame_array)):
            # writing to a image array
            out.write(frame_array[i])
        out.release()
        print("Done")
        print('Finished Testing')

def main():
    args = return_arguments()
    if args.mode == 'train':
        model = Model(args)
        model.train()
    elif args.mode == 'test':
        model_test = Model(args)
#         model_test.test()
        model_test.viz()

if __name__ == '__main__':
    main()

