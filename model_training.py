import torch
from torch.nn.utils.rnn import pack_padded_sequence
from nltk.translate.bleu_score import corpus_bleu
from utils import save_checkpoint
import neptune

def train(enc, dec, device, loss_fn, train_loader, optimizer_decoder, optimizer_encoder, epoch):
    ''' Train model
    :param enc: encoder part of model
    :param dec: decoder part of model
    :param device: on which device to train model
    :param loss_fn: loss function
    :param train_loader: pytorch loader of images
    :param optimizer_decoder: pytorch optimizer for decoder part of model
    :param optimizer_encoder: pytorch optimizer for encoder part of model
    :param epoch: current epoch of training
    :return: None
    '''

    dec.train()
    enc.train()

    dec = dec.to(device)
    enc = enc.to(device)

    # iterate through batches of train loader
    for batch_n, (imgs, caps, caplens) in enumerate(train_loader):

        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Encode images
        enc_output = enc(imgs)

        # Decode encodings and get captions
        dec_out, captions, captions_lengths, sort_ind = dec(captions=caps,
                                                            encoder_out=enc_output,
                                                            captions_lengths=caplens)

        # Remove words which we did not decode at (e.g. max length of sentence in batch is 15 words,
        # so for sentence of 10 words we did not decode 5 words, and we have to skip them during loss computing)
        dec_out = pack_padded_sequence(dec_out, captions_lengths.cpu(), batch_first=True).data.to(device)
        captions = pack_padded_sequence(captions, captions_lengths.cpu(), batch_first=True).data.to(device)

        loss = loss_fn(dec_out, captions)
        optimizer_decoder.zero_grad()
        optimizer_encoder.zero_grad()

        loss.backward()

        optimizer_decoder.step()
        optimizer_encoder.step()

        if batch_n % 3000 == 0:
            save_checkpoint(epoch, batch_n, enc, dec, optimizer_encoder, optimizer_decoder)
            print('Current loss', loss.item())

    # Log metric to neptune
    neptune.log_metric('loss', loss.item())


def validate(enc, dec, device, val_loader, wordmap, epoch):
    ''' Calculate validation metric
    :param val_loader: pytorch loader of images
    :param wordmap: dictionary mapping from word to word index
    :param epoch: current epoch of training
    :return: None
    '''

    enc.eval()
    dec.eval()

    dec = dec.to(device)
    enc = enc.to(device)

    references = list()  # True captions
    hypotheses = list()  # Predicted captions

    with torch.no_grad():

        for batch_n, (imgs, caps, caplens, allcaps) in enumerate(val_loader):

            print(batch_n)

            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            enc_output = enc(imgs)
            dec_out, captions, captions_lengths, sort_ind = dec(captions=caps,
                                                                encoder_out=enc_output,
                                                                captions_lengths=caplens)
            scores_copy = dec_out.clone()


            allcaps = allcaps[sort_ind]  # Resort because captions were sorted in decoder

            for j in range(allcaps.shape[0]):

                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {wordmap['<start>'], wordmap['<pad>']}],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)

            # Take predicted captions for each image
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:captions_lengths[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)

        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(references, hypotheses)

        # Log score to neptune and print metric
        neptune.log_metric('bleu4', bleu4)
        print('Epoch {}, BLEU4'.format(epoch), bleu4)