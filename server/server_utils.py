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

    checkpoint = torch.load(checkpoint_name)
    dec = checkpoint['decoder']
    enc = checkpoint['encoder']
    del checkpoint

    return enc, dec


def load_wordmap():
    '''Loading dictionary mapping from word to word index
    :return: dictionary mapping from word to word index and dictionary from word index to word
    '''

    with open('WORDMAP_COCO.json', 'rb') as f:
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

    with torch.no_grad():

        enc = enc.eval()
        dec = dec.eval()
        predicted_sentence = []
        encoder_out = enc(img)
        encoder_out = encoder_out.view(1, -1, 2048)
        h = dec.h_init(encoder_out.mean(dim=1))  # (batch_size, decoder_hidden_size)
        c = dec.c_init(encoder_out.mean(dim=1))  # (batch_size, decoder_hidden_size)
        last_word = torch.tensor(wordmap['<start>']).unsqueeze(0).unsqueeze(0).to(device)

        while last_word != wordmap['<end>']:

            print(res[int(last_word)])
            predicted_sentence.append(res[int(last_word)])
            # Выбираем эмбеддинг слова, стоящего на позиции last_word
            decoder_out = dec.embedding(last_word)[:, 0]

            # Механизм внимания
            encoder_weighted = dec.Attention(batch_size=1,
                                             encoder_out=encoder_out,
                                             decoder_out=decoder_out)

            # Конкатенируем информцию из механизма внимания и информацию о предыдущем слове
            decoder_in = torch.cat((encoder_weighted, decoder_out), 1)

            # Предсказываем вероятности появления слов на текущей позиции
            h, c = dec.LSTMCell(decoder_in, (h, c))  # (batch_size, decoder_hidden_size)
            predictions_word = dec.linear(h)  # (batch_size, decoder_hidden_size)
            maxword = torch.argmax(predictions_word)
            last_word = maxword.unsqueeze(0).unsqueeze(0).to(device)

    return predicted_sentence, encoder_out


def video_to_screenshots(video, path_to_the_saved_frames, period):
    '''
    :param video: path to video
    :param path_to_the_saved_frames: path where to save screenshots from video
    :param period: save screenshot every period seconds
    :return: None
    '''

    cam = cv2.VideoCapture(video)
    frame_per_second = int(cam.get(cv2.CAP_PROP_FPS))
    currentframe = 0

    while (True):

        # Reading frames
        ret, frame = cam.read()

        # If there is still video continue
        if ret:

            # Checking if the frame is the one we need
            if currentframe % (period * frame_per_second) == 0:

                # Saving screenshot
                name = path_to_the_saved_frames + '/frame' + str(currentframe) + '.png'
                cv2.imwrite(name, frame)
            currentframe += 1

        else:
            break