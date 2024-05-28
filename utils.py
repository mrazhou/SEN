import os
import numpy as np
import h5py
import json
import torch
# from scipy.misc import imread, imresize
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample

from eval_func.bleu.bleu import Bleu
from eval_func.rouge.rouge import Rouge
from eval_func.cider.cider import Cider
from eval_func.meteor.meteor import Meteor
import numpy as np

import os
import numpy as np
import h5py
import json
import torch
from termcolor import cprint

from imageio import imread
from cv2 import resize as imresize

from tqdm import tqdm
from collections import Counter, OrderedDict
from random import seed, choice, sample
import random
import torchvision
import torch.nn as nn
import os
import subprocess
from torch.utils.data import Subset


def check_weight_path(weight_path):
    """Check if weight_path is a valid path
        if weight_path have 'resnet18' but no '.pth' in it, return False
            indicates that model trained from scratch with 'resnet18' architecture
        if weight_path have 'resnet18' and '.pth' in it, return True
            indicates that model trained from pretrained weight with 'resnet18' architecture
    """
    if weight_path and ('.pth' in weight_path or '.ckpt' in weight_path):
        return True
    else:
        return False

def get_net_type(weight_path, model_stage, proj_channel):
    proj = nn.Identity()
    out_channel = [64, 128, 256, 512][model_stage - 1]
    if 'resnet18' in weight_path or 'rn18' in weight_path:
        net_type = 'resnet18'
    elif 'resnet34' in weight_path or 'rn34' in weight_path:
        net_type = 'resnet34'
    elif 'resnet50' in weight_path or 'rn50' in weight_path:
        net_type = 'resnet50'
        out_channel = [256, 512, 1024, 2048][model_stage - 1]
    elif 'resnet101' in weight_path or 'rn101' in weight_path:
        net_type = 'resnet101'
        out_channel = [256, 512, 1024, 2048][model_stage - 1]
    else:
        raise NotImplementedError
    
    if proj_channel != out_channel:
        proj = nn.Conv2d(out_channel, proj_channel, kernel_size=1, stride=1, padding=0)
        out_channel = proj_channel
    return net_type, out_channel, proj

def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()
    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message

def remove_prefix(state_dict, flag_key="conv1.weight", max_depth=6):
    "recursive function to remove prefix. just for resnet"
    state_dict = {'.'.join(k.split('.')[1:]):v for k, v in state_dict.items()}
    try:
        state_dict[flag_key]
        return state_dict
    except KeyError:
        if max_depth <= 0:
            raise KeyError(f"Can't find {flag_key} in state_dict")
        return remove_prefix(state_dict, flag_key=flag_key, max_depth=max_depth-1)

def get_state_dict(weight_path, in_channel=6):
    if check_weight_path(weight_path):
        flag_key = 'conv1.weight'
        if 'seco' in weight_path:
            # for seco
            state_dict = torch.load(weight_path, map_location='cuda')
        else:
            # for SSL4EO moco
            state_dict = torch.load(weight_path, map_location='cuda')['state_dict']
            state_dict = remove_prefix(state_dict, flag_key=flag_key)
        
        if 'B13' in weight_path:
            state_dict[flag_key] = state_dict[flag_key][:, [3,2,1], :, :]
            if in_channel == 6:
                state_dict[flag_key] = state_dict[flag_key].repeat(1,2,1,1)
        elif 'B3' in weight_path and in_channel == 6:
            state_dict[flag_key] = state_dict[flag_key].repeat(1,2,1,1)
        elif 'B6' in weight_path and in_channel == 3:
            state_dict[flag_key] = state_dict[flag_key][:, :3, :, :]
    else:
        cprint('[Warning] weight_path is not existing. And will train model from scratch', 'red')
        state_dict = {}
    return state_dict

def load_weight_resnet(model_name, weight_path, in_channel=6, model_stage=4):
    weights = True if 'imagenet' in weight_path else False
    model = getattr(torchvision.models, model_name)(weights=weights)
    state_dict = get_state_dict(weight_path, in_channel=in_channel)
    if in_channel == 6:
        conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if 'B6' in weight_path and weights: # for B6_imagenet
            conv1.weight = nn.Parameter(model.conv1.weight.repeat(1,2,1,1))
        model.conv1 = conv1
    msg=model.load_state_dict(state_dict, strict=False)
    # print(msg)
    with open(os.environ.get('log_path'), 'a') as f:
        if weights:
            print(f'[Info] Load weight from {weight_path}', file=f)
        else:
            print(f'[Info] Load weight from {weight_path}', file=f)
            print(f'[Info] With message: {msg}',file=f)
            print(f'[Info] With message: {msg}')
    
    layers = OrderedDict([
        ("conv1", model.conv1),
        ("bn1", model.bn1),
        ("relu", model.relu),
        ("maxpool", model.maxpool),
    ])

    for i in range(model_stage):
        name = f'layer{i+1}'
        layers.update({name: getattr(model, name)})
    net = nn.Sequential(layers)
    return net

def fine_tune_net(net, layer=5, in_channel=6, weight_path=None):
    for param in net.parameters():
        param.requires_grad = False
    # If fine-tuning, only fine-tune convolutional blocks 2 through 4 for resnet50
    # for resnet101, fine-tune convolutional blocks 2 through 5
    children = list(net.children())
    head_layer = 2
    # for first and second layer due to the different input channel
    # if in_channel == 6 and 'B6' not in weight_path:
    #     for c in children[:head_layer]:
    #         for p in c.parameters():
    #             p.requires_grad = True
    for c in children[layer:]:  # layer >= 8 equal to fine-tune = False
        for p in c.parameters():
            p.requires_grad = True
    # print('Gradient of Model:',[(n,p.requires_grad) for n,p in net.named_parameters()])

def load_weight(model_name, weight_path, model_stage, ft_layer, in_channel=6):
    model = load_weight_resnet(model_name, weight_path, in_channel, model_stage)
    if check_weight_path(weight_path) or 'imagenet' in weight_path:
        fine_tune_net(model, layer=ft_layer, in_channel=in_channel, weight_path=weight_path)
    return model

def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' # https://blog.csdn.net/BeiErGeLaiDe/article/details/129306023
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.log = open(filename, "a")

    def write(self, *message, show=True):
        print(*message) if show else None
        self.log.write(' '.join(map(str, message))+'\n')
    
    def close(self):
        self.log.close()

    def flush(self):
        self.log.flush()

def create_input_files(dataset, karpathy_json_path, image_folder, captions_per_image, min_word_freq, output_folder,
                       max_len=100):
    """
    Creates input files for training, validation, and test data.

    :param dataset: name of dataset, one of 'coco', 'flickr8k', 'flickr30k'
    :param karpathy_json_path: path of Karpathy JSON file with splits and captions
    :param image_folder: folder with downloaded images
    :param captions_per_image: number of captions to sample per image
    :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>s
    :param output_folder: folder to save files
    :param max_len: don't sample captions longer than this length
    """

    assert dataset in {'RSICD','LEVIR_CC'}

    # Read Karpathy JSON
    with open(karpathy_json_path, 'r') as j:
        data = json.load(j)

    # Read image paths and captions for each image
    train_image_paths = []
    train_image_captions = []

    val_image_paths = []
    val_image_captions = []

    test_image_paths = []
    test_image_captions = []

    word_freq = Counter()  # 创建一个空的Counter类(计数
    for img in data['images']:
        captions = []
        for c in img['sentences']:
            # Update word frequency
            word_freq.update(c['tokens'])   # 其中c['tokens']是一个很多单词组成的句子‘列表’
            if len(c['tokens']) <= max_len:
                captions.append(c['tokens'])

        if len(captions) == 0:
            continue

        if dataset == 'coco':
            path = os.path.join(image_folder, img['filepath'], img['filename'])
        elif dataset == 'LEVIR_CC':
            # FIXME:need to change for levir_CC
            path1 = os.path.join(image_folder, img['split'], 'A', img['filename'])
            path2 = os.path.join(image_folder, img['split'], 'B', img['filename'])
            path = [path1,path2]
        else:
            path = os.path.join(image_folder, img['filename'])

        if img['split'] in {'train', 'restval'}:
            train_image_paths.append(path)
            train_image_captions.append(captions)
        elif img['split'] in {'val'}:
            val_image_paths.append(path)
            val_image_captions.append(captions)
        elif img['split'] in {'test'}:
            test_image_paths.append(path)
            test_image_captions.append(captions)

    # Sanity check
    assert len(train_image_paths) == len(train_image_captions)
    assert len(val_image_paths) == len(val_image_captions)
    assert len(test_image_paths) == len(test_image_captions)

    # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    # Create a base/root name for all output files
    base_filename = dataset + '_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'

    # Save word map to a JSON
    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)

    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    seed(123)
    for impaths, imcaps, split in [(train_image_paths, train_image_captions, 'TRAIN'),
                                   (val_image_paths, val_image_captions, 'VAL'),
                                   (test_image_paths, test_image_captions, 'TEST')]:

        with h5py.File(os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.hdf5'), 'a') as h:
            # Make a note of the number of captions we are sampling per image
            h.attrs['captions_per_image'] = captions_per_image

            # Create dataset inside HDF5 file to store images
            if dataset == 'LEVIR_CC':
                images = h.create_dataset('images', (len(impaths), 2, 3, 256, 256), dtype='uint8')
            else:
                images = h.create_dataset('images', (len(impaths), 3, 256, 256), dtype='uint8')

            print("\nReading %s images and captions, storing to file...\n" % split)

            enc_captions = []
            caplens = []

            for i, path in enumerate(tqdm(impaths)):

                # Sample captions
                if len(imcaps[i]) < captions_per_image:
                    captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
                else:
                    captions = sample(imcaps[i], k=captions_per_image)

                # Sanity check
                assert len(captions) == captions_per_image

                # Read images
                if dataset =='LEVIR_CC':
                    img_A = imread(impaths[i][0])
                    img_B = imread(impaths[i][1])
                    if len(img_A.shape) == 2:
                        img_A = img_A[:, :, np.newaxis]
                        img_A = np.concatenate([img_A, img_A, img_A], axis=2)
                    if len(img_B.shape) == 2:
                        img_B = img_B[:, :, np.newaxis]
                        img_B = np.concatenate([img_B, img_B, img_B], axis=2)
                    img_A = imresize(img_A, (256, 256))
                    img_A = img_A.transpose(2, 0, 1)
                    img_B = imresize(img_B, (256, 256))
                    img_B = img_B.transpose(2, 0, 1)
                    assert img_A.shape == (3, 256, 256)
                    assert img_B.shape == (3, 256, 256)
                    assert np.max(img_A) <= 255
                    assert np.max(img_B) <= 255

                    # Save image to HDF5 file
                    # images[i][0] = img_A
                    # images[i][1] = img_B
                    images[i] = [img_A,img_B]

                else:
                    img = imread(impaths[i])
                    if len(img.shape) == 2:
                        img = img[:, :, np.newaxis]
                        img = np.concatenate([img, img, img], axis=2)
                    img = imresize(img, (256, 256))
                    img = img.transpose(2, 0, 1)
                    assert img.shape == (3, 256, 256)
                    assert np.max(img) <= 255
                    # Save image to HDF5 file
                    images[i] = img

                for j, c in enumerate(captions):
                    # Encode captions
                    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))

                    # Find caption lengths
                    c_len = len(c) + 2

                    enc_captions.append(enc_c)
                    caplens.append(c_len)


            # Sanity check
            assert images.shape[0] * captions_per_image == len(enc_captions) == len(caplens)

            # Save encoded captions and their lengths to JSON files
            with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_captions, j)

            with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
                json.dump(caplens, j)


def init_embedding(embeddings):
    """
    Fills embedding tensor with values from the uniform distribution.

    :param embeddings: embedding tensor
    """
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)


def load_embeddings(emb_file, word_map):
    """
    Creates an embedding tensor for the specified word map, for loading into the model.

    :param emb_file: file containing embeddings (stored in GloVe format)
    :param word_map: word map
    :return: embeddings in the same order as the words in the word map, dimension of embeddings
    """

    # Find embedding dimension
    with open(emb_file, 'r') as f:
        emb_dim = len(f.readline().split(' ')) - 1

    vocab = set(word_map.keys())

    # Create tensor to hold embeddings, initialize
    embeddings = torch.FloatTensor(len(vocab), emb_dim)
    init_embedding(embeddings)

    # Read embedding file
    print("\nLoading embeddings...")
    for line in open(emb_file, 'r'):
        line = line.split(' ')

        emb_word = line[0]
        embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))

        # Ignore word if not in train_vocab
        if emb_word not in vocab:
            continue

        embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)

    return embeddings, emb_dim


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(args,data_name, epoch, epochs_since_improvement, encoder_image,encoder_feat, decoder,
                    encoder_image_optimizer,encoder_feat_optimizer, decoder_optimizer,bleu4, is_best):
    """
    Saves model checkpoint.

    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'bleu-4': bleu4,
             'encoder_image': encoder_image,
             'encoder_feat': encoder_feat,
             'decoder': decoder,
             'encoder_image_optimizer': encoder_image_optimizer,
             'encoder_feat_optimizer': encoder_feat_optimizer,
             'decoder_optimizer': decoder_optimizer,
             }
    filename = 'checkpoint_' + data_name + '.pth.tar'
    path = args.savepath #'./models_checkpoint/mymodel/3-times/'
    if os.path.exists(path)==False:
        os.makedirs(path)
        # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, os.path.join(path, 'BEST_' + filename))

    # torch.save(state, os.path.join(path, 'checkpoint_' + data_name +'_epoch_'+str(epoch) + '.pth.tar'))


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))

#
# def accuracy_our(scores, targets, k):
#     """
#     Computes top-k accuracy, from predicted and true labels.
#
#     :param scores: scores from the model
#     :param targets: true labels
#     :param k: k in top-k accuracy
#     :return: top-k accuracy
#     """
#     batch_size = targets.size(0)
#     _, ind = scores.topk(k, 1, True, True)
#     correct = ind.eq(targets.view(-1, 1).expand_as(ind))
#     correct = correct.view(-1).float()
#     mask = (targets.view(-1, 1).expand_as(ind)).eq(0 * targets.view(-1, 1).expand_as(ind))
#     mask = 1-mask.view(-1).float()
#     correct_total = (correct*mask).sum()  # 0D tensor
#     return correct_total.item() * (100.0 / mask.sum().item())


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def get_eval_score(references, hypotheses):
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]

    hypo = [[' '.join(hypo)] for hypo in [[str(x) for x in hypo] for hypo in hypotheses]]
    ref = [[' '.join(reft) for reft in reftmp] for reftmp in
           [[[str(x) for x in reft] for reft in reftmp] for reftmp in references]]

    score = []
    method = []
    for scorer, method_i in scorers:
        score_i, scores_i = scorer.compute_score(ref, hypo)
        score.extend(score_i) if isinstance(score_i, list) else score.append(score_i)
        method.extend(method_i) if isinstance(method_i, list) else method.append(method_i)
        print("{} {}".format(method_i, score_i))
    score_dict = dict(zip(method, score))

    return score_dict


def convert2words(sequences, rev_word_map):
    for l1 in sequences:
        caption = ""
        for l2 in l1:
            caption += rev_word_map[l2]
            caption += " "
        print(caption)