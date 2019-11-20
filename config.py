import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='test', help='train or test')
parser.add_argument('--step_num', type=int, default=151469, help='model number to load')
parser.add_argument('--model', type=str, default='original_capsule', help='original_capsule, matrix_capsule or vector_capsule')

# Training logs
parser.add_argument('--max_step', type=int, default=100000, help='# of step for training (only for mnist)')
parser.add_argument('--max_epoch', type=int, default=20, help='# of step for training (only for nodule data)')
parser.add_argument('--epoch_based', type=bool, default=True, help='Running the training in epochs')
parser.add_argument('--SAVE_FREQ', type=int, default=1000, help='Number of steps to save model')
parser.add_argument('--SUMMARY_FREQ', type=int, default=100, help='Number of step to save summary')
parser.add_argument('--VAL_FREQ', type=int, default=500, help='Number of step to evaluate the network on Validation data')

# Hyper-parameters
parser.add_argument('--loss_type', type=str, default='margin', help='spread or margin')
parser.add_argument('--add_recon_loss', type=bool, default=False, help='To add reconstruction loss')

# For margin loss
parser.add_argument('--m_plus', type=float, default=0.9, help='m+ parameter')
parser.add_argument('--m_minus', type=float, default=0.1, help='m- parameter')
parser.add_argument('--lambda_val', type=float, default=0.5, help='Down-weighting parameter for the absent class')
# For reconstruction loss
parser.add_argument('--alpha', type=float, default=0.0005, help='Regularization coefficient to scale down the reconstruction loss')
# For training
parser.add_argument('--batch_size', type=int, default=20, help='training batch size')
parser.add_argument('--init_lr', type=float, default=1e-4, help='Initial learning rate')
parser.add_argument('--lr_min', type=float, default=1e-5, help='Minimum learning rate')

# data
parser.add_argument('--N', type=int, default=19807, help='Total number of training samples')
parser.add_argument('--dim', type=int, default=2, help='2D or 3D for nodule data')
parser.add_argument('--one_hot', type=bool, default=False, help='one-hot-encodes the labels')
parser.add_argument('--data_augment', type=bool, default=False, help='Adds augmentation to data')
parser.add_argument('--max_angle', type=int, default=180, help='Maximum rotation angle along each axis; when applying augmentation')
parser.add_argument('--height', type=int, default=50, help='Network input height size')
parser.add_argument('--width', type=int, default=50, help='Network input width size')
parser.add_argument('--depth', type=int, default=32, help='Network input depth size (in the case of 3D input images)')
parser.add_argument('--channel', type=int, default=7, help='Network input channel size')
parser.add_argument('--num_cls', type=int, default=5, help='Number of output classes')

# Training directories
parser.add_argument('--run_name', type=str, default='run02', help='Run name')
parser.add_argument('--logdir', type=str, default='./Results/log_dir/', help='Logs directory')
parser.add_argument('--modeldir', type=str, default='./Results/model_dir/', help='Saved models directory')
parser.add_argument('--reload_step', type=int, default=0, help='Reload step to continue training')
parser.add_argument('--model_name', type=str, default='model', help='Model file name')

# network architecture
parser.add_argument('--prim_caps_dim', type=int, default=8, help='Dimension of the PrimaryCaps in the Original_CapsNet')
parser.add_argument('--digit_caps_dim', type=int, default=16, help='Dimension of the DigitCaps in the Original_CapsNet')
parser.add_argument('--h1', type=int, default=512, help='Number of hidden units of the first FC layer of the reconstruction network')
parser.add_argument('--h2', type=int, default=1024, help='Number of hidden units of the second FC layer of the reconstruction network')

# Matrix Capsule architecture
parser.add_argument('--use_bias', type=bool, default=True, help='Adds bias to init capsules')
parser.add_argument('--use_BN', type=bool, default=True, help='Adds BN before conv1 layer')
parser.add_argument('--add_coords', type=bool, default=True, help='Adds capsule coordinations')
parser.add_argument('--grad_clip', type=bool, default=False, help='Adds gradient clipping to get rid of exploding gradient')
parser.add_argument('--L2_reg', type=bool, default=False, help='Adds L2-regularization to all the network weights')
parser.add_argument('--lmbda', type=float, default=5e-04, help='L2-regularization coefficient')
parser.add_argument('--add_decoder', type=bool, default=False, help='Adds a fully connected decoder and reconstruction loss')
parser.add_argument('--iter', type=int, default=1, help='Number of EM-routing iterations')
parser.add_argument('--A', type=int, default=32, help='A in Figure 1 of the paper')
parser.add_argument('--B', type=int, default=32, help='B in Figure 1 of the paper')
parser.add_argument('--C', type=int, default=32, help='C in Figure 1 of the paper')
parser.add_argument('--D', type=int, default=32, help='D in Figure 1 of the paper')

# test save parameters
parser.add_argument('--data_path', type=str, default=r'/project/hnguyen/xiaoyang/exps/Data/50_plex/jj_final/multiclass/capsuleNet/[retest]-autosegPseed-mrcnntest-imagenet/data.h5', help='path to the h5 data file')
parser.add_argument('--OUTPUT_DIR', type=str, default='Results/test', help='Saved models directory')


args, _ = parser.parse_known_args()
