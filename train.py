from utils import *
import sys
import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
import argparse
from torch.optim.lr_scheduler import StepLR
from datasets import *
from eval import evaluate_transformer
import eval
import tqdm

from modules.RSICCformer import *
from modules.SEN import SEN
from modules.MCCFormer import MCCFormers_D, MCCFormers_S

import shutil
from torch.utils.tensorboard import SummaryWriter


def train(args, train_loader, encoder_image,encoder_feat, decoder, criterion, encoder_image_optimizer,encoder_image_lr_scheduler,encoder_feat_optimizer,encoder_feat_lr_scheduler, decoder_optimizer, decoder_lr_scheduler, epoch, logger):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """
    encoder_image.train()
    encoder_feat.train()
    decoder.train()  # train mode (dropout and batchnorm is used)


    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs_our = AverageMeter()
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    # Batches
    best_bleu4 = 0.  # BLEU-4 score right now
    for i, (img_pairs, caps, caplens) in enumerate(train_loader):
        # if i == 2:
        #    break
        data_time.update(time.time() - start)

        # Move to GPU, if available
        img_pairs = img_pairs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Forward prop.
        imgs_A = img_pairs[:, 0, :, :, :]
        imgs_B = img_pairs[:, 1, :, :, :]
        if args.model == 'RSICCformer':
            imgs_A = encoder_image(imgs_A)
            imgs_B = encoder_image(imgs_B)
            fused_feat = encoder_feat(imgs_A,imgs_B) # fused_feat: (S, batch, feature_dim)
        elif args.model == 'SEN':
            img_cat = torch.cat((imgs_A, imgs_B), dim=1)
            feature = encoder_image(img_cat) # feature: [batch_size, 1024, 14, 14]
            fused_feat = encoder_feat(img_cat, feature) # fused_feat: (S, batch, feature_dim)
        scores, caps_sorted, decode_lengths, sort_ind = decoder(fused_feat, caps, caplens)


        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data


        # Calculate loss
        loss = criterion(scores, targets)

        # Back prop.
        decoder_optimizer.zero_grad()
        encoder_feat_optimizer.zero_grad()
        if encoder_image_optimizer is not None:
            encoder_image_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if args.grad_clip is not None:
            clip_gradient(decoder_optimizer, args.grad_clip)
            if encoder_image_optimizer is not None:
                clip_gradient(encoder_image_optimizer, args.grad_clip)
            if args.encoder_feat_clip:
                clip_gradient(encoder_feat_optimizer, args.grad_clip)

        # Update weights
        decoder_optimizer.step()
        decoder_lr_scheduler.step()
        encoder_feat_optimizer.step()
        encoder_feat_lr_scheduler.step()
        if encoder_image_optimizer is not None:
            encoder_image_optimizer.step()
            encoder_image_lr_scheduler.step()

        # Keep track of metrics
        top5 = accuracy(scores, targets, 1)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()
        if i % args.print_freq == 0:
            # logger.write('TIME: ', time.strftime("%m-%d  %H : %M : %S", time.localtime(time.time())))
            msg = "Epoch: {}/{} step: {}/{} Loss: {:.4f} AVG_Loss: {:.4f} Top-5 Accuracy: {:.4f} Batch_time: {:.4f}s".format(epoch+0, args.epochs, i+0, len(train_loader), losses.val, losses.avg, top5accs.val, batch_time.val)
            logger.write(msg)
            logger.flush()
    return losses.avg, top5accs.avg


def main(args, logger):

    logger.write(time.strftime("%m-%d  %H : %M : %S", time.localtime(time.time())))

    start_epoch = 0
    best_bleu4 = 0.  # BLEU-4 score right now
    epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
    cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

    # Read word map
    word_map_file = os.path.join(args.data_folder, 'WORDMAP_' + args.data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    # Initialize
    # Encoder
    # encoder_image = ResTransformer(in_channels=6)
    if args.model == 'SEN':
        args.vocab_size = len(word_map)
        sen = SEN(args)
        encoder_image = sen.extractor
        encoder_feat = sen.encoder
        decoder = sen.decoder
    elif args.model == 'RSICCformer':

        encoder_image = CNN_Encoder(args.encoder_image)
        feature_dim = 1024 # resnet101
        if args.encoder_feat == 'MCCFormers-D':
            encoder_feat = MCCFormers_D(feature_dim = 1024,dropout=0.5,h=14,w=14,d_model=512,n_head=args.n_heads,n_layers=args.n_layers).to(device)
        elif args.encoder_feat == 'MCCFormer-S':
            encoder_feat = MCCFormers_S(feature_dim = 1024,h=14,w=14, n_head=args.n_heads,n_layers=args.n_layers).to(device)
        else:
            encoder_feat = MCCFormers_diff_as_Q(feature_dim=feature_dim, dropout=0.5, h=14, w=14, d_model=512, n_head=args.n_heads,
                               n_layers=args.n_layers)
        decoder = TransDecoder(feature_dim=512*2,
                                vocab_size=len(word_map),
                                n_head=args.n_heads,
                                n_layers=args.decoder_n_layers,
                                dropout=args.dropout)
        
    # RSICCformer not fine-tune the encoder_image
    encoder_image_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder_image.parameters()),
                                               lr=args.encoder_lr) if args.model == "SEN" else None
    encoder_image_lr_scheduler = StepLR(encoder_image_optimizer, step_size=900, gamma=1) if args.model == "SEN" else None

    encoder_feat_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder_feat.parameters()),
                                         lr=args.encoder_lr)
    encoder_feat_lr_scheduler = StepLR(encoder_feat_optimizer, step_size=900, gamma=1)

    decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                         lr=args.decoder_lr)
    decoder_lr_scheduler = StepLR(decoder_optimizer,step_size=900,gamma=1)

    # compile with pytorch2.0 to speed up
    try:
        encoder_image = torch.compile(encoder_image)
        encoder_feat = torch.compile(encoder_feat)
        decoder = torch.compile(decoder)
    except Exception:
        cprint("[Warning] torch.compile to speed up is not available", "yellow")

    # Move to GPU, if available
    encoder_image = encoder_image.to(device)
    encoder_feat = encoder_feat.to(device)
    decoder = decoder.to(device)
    
    get_trainable_params = lambda x: sum(p.numel() for p in x.parameters() if p.requires_grad)
    get_params = lambda x: sum(p.numel() for p in x.parameters())
    encoder_image_params = get_trainable_params(encoder_image)
    decoder_params = get_trainable_params(decoder)
    encoder_feat_params = get_trainable_params(encoder_feat)
    with open(os.path.join(args.savepath, 'model_summary.txt'), 'a') as f:
        print('encoder_iamge:', encoder_image, file=f)
        print('encoder_feat:', encoder_feat, file=f)
        print('decoder:', decoder, file=f)
    logger.write("--------Params--------")
    logger.write("encoder_image params:{:,}".format(encoder_image_params))
    logger.write("encoder_feat params:{:,}".format(encoder_feat_params))
    logger.write("decoder params:{:,}".format(decoder_params))
    logger.write("Total trainable params:{:,}".format(encoder_image_params + encoder_feat_params + decoder_params))
    logger.write("Total params:{:,}".format(get_params(encoder_image) + encoder_feat_params + decoder_params))
    logger.write("--------Params--------")
    # write gradient of encoder_image
    with open(os.path.join(args.savepath, 'gradient.txt'), 'a') as f:
        for n, p in encoder_image.named_parameters():
            f.write("【"+ n + '】 ' + str(p.requires_grad) + '\n')
        
    logger.write("\033[32m[Info]\033[0m Checkpoint_savepath:{}".format(args.savepath))
    logger.write("Encoder_image_mode:{}   Encoder_feat_mode:{}   Decoder_mode:{}".format(args.encoder_image,args.encoder_feat,args.decoder))
    logger.write("encoder_layers {} decoder_layers {} n_heads {} dropout {} encoder_lr {} "
          "decoder_lr {}".format(args.n_layers, args.decoder_n_layers, args.n_heads, args.dropout,
                                         args.encoder_lr, args.decoder_lr))

    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)

    # Custom dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    # pin_memory: If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
    # If your data elements are a custom type, or your collate_fn returns a batch that is a custom type.
    dataset = CaptionDataset(args.data_folder, args.data_name, 'TRAIN', transform=transforms.Compose([normalize]))
    train_loader = torch.utils.data.DataLoader(dataset,batch_size=args.batch_size, shuffle=True, 
                                               num_workers=args.workers, pin_memory=True, worker_init_fn=seed_worker)

    # Epochs
    pbar = tqdm.tqdm(range(start_epoch, args.epochs))
    tb_writer = SummaryWriter(log_dir=args.savepath)
    for epoch in pbar:
        # Decay learning rate if there is no improvement for x consecutive epochs, and terminate training after x
        if epochs_since_improvement == args.stop_criteria:
            logger.write("[Finished] the model has not improved in the last {} epochs".format(args.stop_criteria))
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % args.decay_interval == 0:
            adjust_learning_rate(decoder_optimizer, args.decay_rate)
            logger.write("[Decay LR] The new learning rate is %f\n" % (decoder_optimizer.param_groups[0]['lr'],))
            if encoder_image_optimizer is not None and args.decay_lr:
                adjust_learning_rate(encoder_image_optimizer, args.decay_rate)
                logger.write("[Decay encoder image LR] The new learning rate is %f\n" % (encoder_image_optimizer.param_groups[0]['lr'],))
            if args.encoder_feat_decay:
                adjust_learning_rate(encoder_feat_optimizer, args.decay_rate)
                logger.write("[Decay encoder feat LR] The new learning rate is %f\n" % (encoder_feat_optimizer.param_groups[0]['lr'],))
                

        # One epoch's training
        logger.write(time.strftime("%m-%d  %H : %M : %S", time.localtime(time.time())))
        
        loss, top5acc = train(args,
                            train_loader=train_loader,
                            encoder_image=encoder_image,
                            encoder_feat=encoder_feat,
                            decoder=decoder,
                            criterion=criterion,
                            encoder_image_optimizer=encoder_image_optimizer,
                            encoder_image_lr_scheduler=encoder_image_lr_scheduler,
                            encoder_feat_optimizer=encoder_feat_optimizer,
                            encoder_feat_lr_scheduler=encoder_feat_lr_scheduler,
                            decoder_optimizer=decoder_optimizer,
                            decoder_lr_scheduler=decoder_lr_scheduler,
                            epoch=epoch,
                            logger=logger)
        tb_writer.add_scalar('train/loss', loss, epoch)
        tb_writer.add_scalar('train/top5acc', top5acc, epoch)

        # One epoch's validation
        metrics, nochange_metric, change_metric = evaluate_transformer(args,
                            encoder_image=encoder_image,
                            encoder_feat=encoder_feat,
                           decoder=decoder,
                           logger=logger,
                           type=args.model)

        recent_bleu4 = metrics["Bleu_4"]
        tb_writer.add_scalar('Bleu_4', recent_bleu4, epoch)
        if nochange_metric is not None and change_metric is not None:
            tb_writer.add_scalar('metric/nochange_Bleu_4', nochange_metric["Bleu_4"], epoch)
            tb_writer.add_scalar('metric/change_Bleu_4', change_metric["Bleu_4"], epoch)
        # Check if there was an improvement
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            logger.write("\nEpochs since last improvement: %d" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0
        logger.write("\nThe best BLEU-4 score: {:.4f}\n".format(best_bleu4))

        # Save checkpoint
        checkpoint_name = args.encoder_image + '_'+args.encoder_feat + '_' + args.decoder #_tengxun_aggregation
        save_checkpoint(args, checkpoint_name, epoch, epochs_since_improvement, encoder_image,encoder_feat, decoder,
                        encoder_image_optimizer,encoder_feat_optimizer,decoder_optimizer, metrics, is_best)
    logger.close()
    tb_writer.close()

def create_folder_and_backup(save_path):
    def check_floder(path,i=1):
        if os.path.exists(path):
            path = os.path.join(save_path,str(i))
            return check_floder(path,i+1)
        else:
            return path
    init_path = os.path.join(save_path, '1')
    save_path = check_floder(init_path)
    os.makedirs(save_path)

    # backup important files
    shutil.copy('./modules/SEN.py', os.path.join(save_path, 'SEN.py'))
    return save_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image_Change_Captioning')

    # Data parameters
    parser.add_argument('--data_folder', default="./data",help='folder with data files saved by create_input_files.py.')
    parser.add_argument('--data_name', default="LEVIR_CC_5_cap_per_img_5_min_word_freq",help='base name shared by data files.')

    # Model parameters
    parser.add_argument('--model', type=str, default='SEN', choices=['SEN', 'RSICCformer'], help='model')
    parser.add_argument('--encoder_image', default="SEN_Extractor", help='which model does encoder use?')
    parser.add_argument('--encoder_feat', default='SEN_DiffEncoder') #
    parser.add_argument('--decoder', default='SEN_TransDecoder')
    parser.add_argument('--n_heads', type=int, default=8, help='Multi-head attention in Transformer.')
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--decoder_n_layers', type=int, default=1)
    parser.add_argument('--feature_dim_de', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
    # SEN
    parser.add_argument('--weight_path', type=str, help='pre-train weight path of resnet')
    parser.add_argument('--proj_channel', type=int, default=512, help='the channel of projection layer')
    parser.add_argument('--ft_layer', type=int, default=4, help='the number of layers to be fine-tuned')
    parser.add_argument('--model_stage', type=int, default=4, help='the stage of resnet')
    parser.add_argument('--encoder_n_layers', type=int, default=2, help='the number of layers of encoder')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=40, help='number of epochs to train for (if early stopping is not triggered).')
    parser.add_argument('--stop_criteria', type=int, default=40, help='training stop if epochs_since_improvement == stop_criteria')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--print_freq', type=int, default=50, help='logger.write training/validation stats every __ batches.')
    parser.add_argument('--workers', type=int, default=8, help='for data-loading; right now, only 0 works with h5pys in windows.')
    parser.add_argument('--encoder_lr', type=float, default=2e-4, help='learning rate for encoder if fine-tuning.')
    parser.add_argument('--decoder_lr', type=float, default=2e-4, help='learning rate for decoder.')
    parser.add_argument('--grad_clip', type=float, default=5., help='clip gradients at an absolute value of.')
    parser.add_argument('--decay_lr', action='store_true', default=False, help='decay learning rate if there is no improvement for x consecutive epochs, and terminate training after x.')
    parser.add_argument('--checkpoint', default=None, help='path to checkpoint, None if none.')
    
    parser.add_argument('--decay_rate', type=float, default=0.7, help='the decay rate of decoder_lr')
    parser.add_argument('--decay_interval', type=int, default=3)
    parser.add_argument('--encoder_feat_decay', action='store_true', default=False)
    parser.add_argument('--encoder_feat_clip', action='store_true', default=False)
    parser.add_argument('--more_reproducibility', action='store_true', default=False, help='more reproducibility but less speed. set to False will speed up but less reproducibility.')
    parser.add_argument('--seed', type=int, default=42)

    # Validation
    parser.add_argument('--Split', default="VAL", help='which')
    parser.add_argument('--beam_size', type=int, default=1, help='beam_size.')
    parser.add_argument('--savepath', default="./models_checkpoint/")
    args = parser.parse_args()
    
    # copy this modules.vision.py to the checkpoint directory
    # args.savepath = create_folder_and_backup(args.savepath)
    
    logger = Logger(os.path.join(args.savepath, 'train_log.txt'))
    os.environ['log_path'] = os.path.join(args.savepath, 'train_log.txt')
    logger.write("git:\n\t"+get_sha())
    cmd = sys.argv[1:]
    cmd = [arg.replace('--', '\\\n\t--') for arg in cmd]
    cmd = f'[CMD] python {sys.argv[0]} ' + ' '.join(cmd) + '\n'
    logger.write(cmd)
    logger.write('[args]\n', args)
    
    set_random_seed(args.seed, args.more_reproducibility)
    
    main(args, logger)
    
    args.Split = "TEST"
    args.path = args.savepath
    eval.main(args)
