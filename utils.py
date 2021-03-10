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



def save_checkpoint(epoch, batch_n, encoder, decoder, encoder_optimizer, decoder_optimizer):
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
             'batch_n': batch_n,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}
    filename = 'checkpoint_' + str(epoch) + '_' + str(batch_n) + '.pth.tar'
    torch.save(state, filename)


def create_input_files(dataset, karpathy_json_path, image_folder, output_folder, captions_per_image):
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

        path = os.path.join(image_folder, img['filepath'], img['filename'])

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