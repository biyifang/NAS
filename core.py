import numpy as np
import csv
import argparse
import tensorflow as tf
from keras import backend as K
from keras.datasets import cifar10
from keras.utils import to_categorical
import torchvision.models as models
from control import Controller, StateSpace
from environment import CNNEnv, model_fn

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--data', metavar='DIR', default='/Users/biyifang/Desktop/research/AllState/experiment', type=str,
                    help='path to dataset')
parser.add_argument('--model_save', metavar='MS', default='', type=str,
                    help='path to student model')
parser.add_argument('--teacher_model_save', metavar='MS', default='', type=str,
                    help='path to teacher model')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--num_class', default=10, type=int, metavar='NoC',
                    help='number of class')
parser.add_argument('--num_boost_iter', default=50, type=int, metavar='N',
                    help='number of boosting iterations')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N',
                    help='mini-batch size (default: 1), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-gradient_acc', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')#default:0.1
parser.add_argument('--lr_dis', '--learning-rate-dis', default=0.001, type=float,
                    metavar='LRdis', help='learning rate for distillation', dest='lr_dis')
parser.add_argument('--lr_boost', '--learning-rate-boost', default=0.00001, type=float,
                    metavar='LRboost', help='learning rate for distillation', dest='lr_boost')
parser.add_argument('--temperature', '--temperature', default=3.0, type=float,
                    metavar='temperature', help='temperature for softmax', dest='temperature')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--boost_shrink', default=0.9, type=float, metavar='S',
                    help='boosting shrinkage parameter')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100000, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')


args = parser.parse_args()

# create a shared session between Keras and Tensorflow
policy_sess = tf.compat.v1.Session()
K.set_session(policy_sess)

NUM_LAYERS = 1  # number of layers of the state space
MAX_TRIALS = 2  # maximum number of models generated
MAX_EPOCHS = 10  # maximum number of epochs to train
CHILD_BATCHSIZE = 128  # batchsize of the child models
EXPLORATION = 0.8  # high exploration for the first 1000 steps
REGULARIZATION = 1e-3  # regularization strength
CONTROLLER_CELLS = 32  # number of cells in RNN controller
EMBEDDING_DIM = 20  # dimension of the embeddings for each state
ACCURACY_BETA = 0.8  # beta value for the moving average of the accuracy
CLIP_REWARDS = 0.0  # clip rewards in the [-0.05, 0.05] range
RESTORE_CONTROLLER = True  #

# construct a state space
state_space = StateSpace()

# add states
#state_space.add_state(name='kernel', values=[1,3])
#state_space.add_state(name='filters', values=[16, 32, 64])
state_space.add_state(name='partial_frac', values=[0.9, 0.75, 0.5])
state_space.add_state(name='input_size', values=[60, 100, 150, 224])
state_space.add_state(name='CNN_one', values=[2,3,4,5])
state_space.add_state(name='CNN_two', values=[2,3,4,5])
state_space.add_state(name='CNN_three', values=[2,3,4,5])
state_space.print_state_space()

'''
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

dataset = [x_train, y_train, x_test, y_test]
'''

previous_acc = 0.0
total_reward = 0.0

with policy_sess.as_default():
    # create the Controller and build the internal policy network
    controller = Controller(policy_sess, NUM_LAYERS, state_space,
                            reg_param=REGULARIZATION,
                            exploration=EXPLORATION,
                            controller_cells=CONTROLLER_CELLS,
                            embedding_dim=EMBEDDING_DIM,
                            restore_controller=RESTORE_CONTROLLER)

# create the Network Manager
manager = CNNEnv(args, epochs=MAX_EPOCHS, child_batchsize=CHILD_BATCHSIZE, clip_rewards=CLIP_REWARDS, acc_beta=ACCURACY_BETA)

# get an initial random state space if controller needs to predict an
# action from the initial state
state = state_space.get_random_state_space(NUM_LAYERS)
print("Initial Random State : ", state_space.parse_state_space_list(state))
print()

# clear the previous files
controller.remove_files()

# train for number of trails
for trial in range(MAX_TRIALS):
    with policy_sess.as_default():
        K.set_session(policy_sess)
        actions = controller.get_action(state)  # get an action for the previous state

    # print the action probabilities
    state_space.print_actions(actions)
    print("Predicted actions : ", state_space.parse_state_space_list(actions))

    # build a model, train and get reward and accuracy from the network manager
    reward, previous_acc = manager.get_rewards(model_fn, state_space.parse_state_space_list(actions))
    print("Rewards : ", reward, "Accuracy : ", previous_acc)

    with policy_sess.as_default():
        K.set_session(policy_sess)

        total_reward += reward
        print("Total reward : ", total_reward)

        # actions and states are equivalent, save the state and reward
        state = actions
        controller.store_rollout(state, reward)

        # train the controller on the saved state and the discounted rewards
        loss = controller.train_step()
        print("Trial %d: Controller loss : %0.6f" % (trial + 1, loss))

        # write the results of this trial into a file
        with open('train_history.csv', mode='a+') as f:
            data = [previous_acc, reward]
            data.extend(state_space.parse_state_space_list(state))
            writer = csv.writer(f)
            writer.writerow(data)
    print()

print("Total Reward : ", total_reward)
