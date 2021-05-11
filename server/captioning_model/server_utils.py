import torch
import json
import numpy as np
import torchvision.transforms as transforms
import PIL
import cv2


def load_models(checkpoint_name):
    ''' Loading encoder and decoder models
    :param checkpoint_name:
    :return: encoder, decoder
    '''

    checkpoint = torch.load(checkpoint_name, map_location = torch.device('cpu'))
    dec = checkpoint['decoder']
    enc = checkpoint['encoder']
    del checkpoint

    return enc, dec


def load_wordmap(data_set):
    '''Loading dictionary mapping from word to word index
    :return: dictionary mapping from word to word index and dictionary from word index to word
    '''

    with open('WORDMAP_{}.json'.format(data_set), 'rb') as f:
        wordmap = json.load(f)
    res = dict((v, k) for k, v in wordmap.items())

    return wordmap, res



def image_preprocessing(image):
    ''' Resizing image
    :param image: image as numpy matrix
    :return: resized image as numpy matrix
    '''

    if len(image.shape) == 2:
        image= image[:, :, np.newaxis]
        image = np.concatenate([image, image, image], axis=2)
    image = PIL.Image.fromarray(image)
    image = np.array(image.resize((256, 256), PIL.Image.BICUBIC))
    image = image.transpose(2, 0, 1)
    return image


def image_normalisation(image, device):
    ''' Normalize image as imagenet images
    :param image: image of appropriate size
    :param device: device on which to store image
    :return: normalized image
    '''

    image = torch.FloatTensor(image / 255.).unsqueeze(0)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    image = transform(image).to(device)

    return image

def captioning(enc, dec, img, wordmap, device, res):
    '''Captioning image
    :param enc: encoder part of model
    :param dec: decoder part of model
    :param img: image
    :param wordmap: dictionary mapping from word to word index
    :param device: on which device to make computations
    :param res: dictionary from word index to word
    :return: predicted captions, encoded image and embedded words of captions
    '''

    with torch.no_grad():

        enc = enc.eval()
        dec = dec.eval()
        predicted_sentence = []
        embedded_words = []
        encoder_out = enc(img)
        encoder_out = encoder_out.view(1, -1, 2048)
        h = dec.h_init(encoder_out.mean(dim=1))
        c = dec.c_init(encoder_out.mean(dim=1))
        last_word = torch.tensor(wordmap['<start>']).unsqueeze(0).unsqueeze(0).to(device)

        while last_word != wordmap['<end>']:

            # Append last word
            predicted_sentence.append(res[int(last_word)])
            # Obtain embedding of previous word
            decoder_out = dec.embedding(last_word)[:, 0]
            # Append embedded word
            embedded_words.append(decoder_out[0].tolist())
            # Attention mechanism
            encoder_weighted = dec.Attention(encoder_out=encoder_out,
                                             decoder_out=decoder_out)
            gate = dec.sigmoid(dec.f_beta(h))
            encoder_weighted = gate * encoder_weighted

            # Concatenating attention and previous word
            decoder_in = torch.cat((encoder_weighted, decoder_out), 1)

            # Obtaining probabilities (not exectaly, because no softmax on this step) of word appearing on this step
            h, c = dec.LSTMCell(decoder_in, (h, c))
            predictions_word = dec.linear(h)
            # Get word with maximum probability
            maxword = torch.argmax(predictions_word)
            last_word = maxword.unsqueeze(0).unsqueeze(0).to(device)

    return predicted_sentence, encoder_out, embedded_words


def video_to_screenshots(video, path_to_the_saved_frames, period, video_hash):
    ''' Extracting screenshots from video
    :param video: path to video
    :param path_to_the_saved_frames: path where to save screenshots from video
    :param period: save screenshot every period seconds
    :return: None
    '''

    cam = cv2.VideoCapture(video)
    frame_per_second = int(cam.get(cv2.CAP_PROP_FPS))
    currentframe = 0
    names = []

    while (True):

        # Reading frames
        ret, frame = cam.read()

        # If there is still video continue
        if ret:

            # Checking if the frame is the one we need
            if currentframe % (period * frame_per_second) == 0:

                # Saving screenshot
                name = path_to_the_saved_frames + '/{}_'.format(str(video_hash)) + 'frame' + str(currentframe) + '.png'
                cv2.imwrite(name, frame)
                names.append(name)
            currentframe += 1

        else:
            break
    return names