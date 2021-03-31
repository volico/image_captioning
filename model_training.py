import torch
from torch.nn.utils.rnn import pack_padded_sequence
from nltk.translate.bleu_score import corpus_bleu
from utils import save_checkpoint
import neptune
import time

def validate(enc, dec, device, loss_fn, val_loader, wordmap, epoch):
    enc.eval()
    dec.eval()
    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)
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
            dec_out = pack_padded_sequence(dec_out.cpu(), captions_lengths.cpu(), batch_first=True).data.to(device)
            captions = pack_padded_sequence(captions.cpu(), captions_lengths.cpu(), batch_first=True).data.to(device)

            loss = loss_fn(dec_out, captions)

            allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {wordmap['<start>'], wordmap['<pad>']}],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:captions_lengths[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)
        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(references, hypotheses)
        neptune.log_metric('bleu4', bleu4)
        print('Epoch {}, BLEU4'.format(epoch), bleu4)


def train(enc, dec, device, loss_fn, train_loader, optimizer_decoder, optimizer_encoder, epoch):

    dec.train()  # train mode (dropout and batchnorm is used)
    enc.train()

    for batch_n, (imgs, caps, caplens) in enumerate(train_loader):
        start = time.time()
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)
        enc_output = enc(imgs)
        dec_out, captions, captions_lengths, sort_ind = dec(captions=caps,
                                                            encoder_out=enc_output,
                                                            captions_lengths=caplens)
        #        if batch_n % 20 == 0:
        #            aaaa = [res.get(int(key)) for key in torch.argmax(dec_out[0], dim = 1)]
        #            print('epoch:', epoch, 'batch', batch_n, aaaa)
        #            img = Image.fromarray((unorm(imgs[0].cpu()).numpy()*255).astype('uint8').transpose(1, 2, 0))
        #            img.save('{}-{}.png'.format(epoch, batch_n))
        #            with open("captions.txt", "a") as f:
        #                # Append 'hello' at the end of file
        #                f.write("\n")
        #                f.write(str(epoch) + '_' + str(batch_n) + '_' + str(aaaa))
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

    neptune.log_metric('loss', loss.item())