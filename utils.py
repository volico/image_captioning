import os
import numpy as np
import h5py
import json
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample
from imageio import imread
from PIL import Image
import PIL
import torch
from torch import nn
import model



def load_models(checkpoint_name=None, encoded_image_size=None,
                word_embeddings_dim=None, attention_dim=None,
                decoder_hidden_size=None, vocab_size=None, device=None):
    '''
    :param checkpoint_name: name of checkpoint file
    :param encoded_image_size: params to initialize model if there is no checkpoint name
    :param word_embeddings_dim: params to initialize model if there is no checkpoint name
    :param attention_dim: params to initialize model if there is no checkpoint name
    :param decoder_hidden_size: params to initialize model if there is no checkpoint name
    :param vocab_size: params to initialize model if there is no checkpoint name
    :param device: on this device to store model
    :return: start_epoch, end_epoch, loss_fn, enc, dec, optimizer_encoder, optimizer_decoder
    '''
    loss_fn = nn.CrossEntropyLoss().to(device)
    end_epoch = 10_000
    if checkpoint_name == None:
        start_epoch = 0
        enc = model.Encoder(encoded_image_size=encoded_image_size).to(device)
        dec = model.Decoder(vocab_size=vocab_size,
                            word_embeddings_dim=word_embeddings_dim,
                            attention_dim=attention_dim,
                            decoder_hidden_size=decoder_hidden_size,
                            encoded_image_size=encoded_image_size).to(device)

        optimizer_decoder = torch.optim.Adam(enc.parameters(), lr=4e-4)
        optimizer_encoder = torch.optim.Adam(dec.parameters(), lr=1e-4)
    else:
        checkpoint = torch.load(checkpoint_name)
        start_epoch = checkpoint['epoch']
        dec = checkpoint['decoder'].to(device)
        optimizer_decoder = checkpoint['decoder_optimizer']
        enc = checkpoint['encoder'].to(device)
        optimizer_encoder = checkpoint['encoder_optimizer']

    return start_epoch, end_epoch, loss_fn, enc, dec, optimizer_encoder, optimizer_decoder


def save_checkpoint(epoch, batch_n, encoder, decoder, encoder_optimizer, decoder_optimizer):
    ''' Saving checkpoints of pytorch objects
    :param epoch: current epoch
    :param batch_n: current batch
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: encoder optimizer
    :param decoder_optimizer: decoder optimizer
    :return: None
    '''

    state = {'epoch': epoch,
             'batch_n': batch_n,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}
    filename = 'models/' + 'checkpoint_' + str(epoch) + '_' + str(batch_n) + '.pth.tar'
    torch.save(state, filename)


def create_input_files(dataset, karpathy_json_path, image_folder, output_folder, captions_per_image):
    '''
    :param dataset:
    :param karpathy_json_path:
    :param image_folder:
    :param output_folder:
    :param captions_per_image:
    :return:
    '''

    assert dataset in {'coco', 'flickr8k', 'flickr30k'}

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
    word_freq = Counter()
    captions_lenghts = []

    for img in data['images']:
        captions = []
        for c in img['sentences']:
            word_freq.update(c['tokens'])
            captions.append(c['tokens'])
            captions_lenghts.append(len(c['tokens']))

        if len(captions) == 0:
            continue
        if dataset == 'coco':
            path = os.path.join(image_folder, img['filepath'], img['filename'])
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
    words = [w for w in word_freq.keys() if word_freq[w] > 5]
    word_map = {k: v + 4 for v, k in enumerate(words)}
    word_map['<start>'] = 0
    word_map['<end>'] = 1
    word_map['<unk>'] = 2
    word_map['<pad>'] = 3

    # Create a base/root name for all output files
    base_filename = dataset

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

            images = h.create_dataset('images', ((len(impaths)), 3,  256, 256), dtype='uint8')

            print("\nReading %s images and captions, storing to file...\n" % split)

            enc_captions = []
            caplens = []

            for i, path in enumerate(tqdm(impaths)):
                seed(1567)
                if len(imcaps[i]) < captions_per_image:
                    captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
                else:
                    captions = sample(imcaps[i], k = captions_per_image)

                # Read images

                img = imread(impaths[i])
                if len(img.shape) == 2:
                    img = img[:, :, np.newaxis]
                    img = np.concatenate([img, img, img], axis=2)
                img = Image.fromarray(img)
                img = np.array(img.resize((256, 256), PIL.Image.BICUBIC))
                img = img.transpose(2, 0, 1)
                assert img.shape == (3, 256, 256)
                assert np.max(img) <= 255

                # Save image to HDF5 file
                images[i] = img
                for c in captions:
                    # Encode captions
                    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + \
                            [word_map['<end>']] + [word_map['<pad>']] * (max(captions_lenghts) - len(c))

                    # Find caption lengths
                    c_len = len(c) + 2

                    enc_captions.append(enc_c)
                    caplens.append(c_len)



            # Sanity check
            print(images.shape[0], captions_per_image,  len(enc_captions), len(caplens))
            assert images.shape[0] * captions_per_image == len(enc_captions) == len(caplens)
            # Save encoded captions and their lengths to JSON files
            with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_captions, j)

            with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
                json.dump(caplens, j)