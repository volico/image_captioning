from flask import Flask, jsonify, request
import numpy as np
from imageio import imread
import torch
from os import listdir
from os.path import isfile, join
from server_utils import *
import os
from waitress import serve

data_set = 'coco'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
enc, dec = load_models(checkpoint_name = 'model.pth.tar')
wordmap, res = load_wordmap(data_set)
app = Flask(__name__)


@app.route('/')
def base():
    return jsonify('Server works')

@app.route('/get_image_caption', methods=['POST'])
def get_image_caption():
    '''
    Getting captions for one particular image
    '''
    no_image = False
    try:
        # Try to load image from request and save it under unique name
        image = request.files['image']
        image_hash = hash(image)
        image.save('images/{}.png'.format(image_hash))
        image = imread('images/{}.png'.format(image_hash))

    except:
        no_image = True
        return jsonify('No image file on post request or it is in the wrong format')

    if no_image == False:
        # Get captions for image if there is one
        image = image_preprocessing(image)
        if (image.shape == (3, 256, 256)) & (np.max(image) <= 256):

            image = image_normalisation(image, device)
            predicted_captions, encoders_out, embedded_words = captioning(enc, dec, image, wordmap, device, res)
            # Return captions, encoded image and embedded words

            return jsonify({'captions': ' '.join(predicted_captions[1:]),
                            'encoders_out': encoders_out.tolist(),
                            'embedded_words': embedded_words})
        else:
            return 'Image is not in png with 3 channels'


@app.route('/get_video_captions', methods=['POST'])
def get_video_captions():
    '''
    Getting captions for every 10 second of video
    :return:
    '''
    no_video = False
    try:
        # Try to load video from request and save it under unique name
        video = request.files['video']
        video_hash = hash(video)
        video.save('videos/{}.mp4'.format(video_hash))

    except:
        no_video = True
        return jsonify('No video file in post requests')

    if no_video == False:
        # Every N second of video will be saved as screenshot
        every_n_second = request.args.get('every_n_second', default=10, type=int)
        # Distance metric to calculate
        metric = request.args.get('metric', default = 'euclidean', type = str)
        # Rolling window size to calculate previous mean of embeddings
        rolling_window_size = request.args.get('rolling_window_size', default = 3, type = int)
        rolling_window_size = rolling_window_size + 1

        # Saving screenshots from video
        names = video_to_screenshots('videos/{}.mp4'.format(video_hash),
                                     'saved_screenshots', every_n_second,  video_hash)
        all_captions = []
        all_encoders_out = []
        all_embedded_words = []

        for file in names:
            # Getting captions for each saved screenshot
            image = imread(file)
            os.remove(file)
            image = image_preprocessing(image)
            if (image.shape == (3, 256, 256)) & (np.max(image) <= 256):
                image = image_normalisation(image, device)
                predicted_captions, encoder_out, embedded_words = captioning(enc, dec, image, wordmap, device, res)
                all_captions.append(' '.join(predicted_captions[1:]))
                all_encoders_out.append(encoder_out.tolist())
                all_embedded_words.append(embedded_words)

        distance_from_previous = compute_distance(all_embedded_words, rolling_window_size, metric)
        # Return captions, encoded image and embedded words
        return jsonify({
                        'captions': all_captions,
#                        'encoders_out': all_encoders_out,
#                        'embedded_words': all_embedded_words,
                        'distance_from_previous': list(distance_from_previous)})

@app.route('/get_captions', methods=['POST', 'GET'])
def get_captions():
    '''
    Get captions (page with simple interface)
    '''

    if request.method == 'POST':
        # If there is already image in request, make captions
        no_image = False
        try:
            # Try to load image from request and save it under unique name
            image = request.files['image']
            image_hash = hash(image)
            image.save('images/{}.png'.format(image_hash))
            image = imread('images/{}.png'.format(image_hash))
        except:
            no_image = True
            return jsonify('No image file on post request or it is in the wrong format')
        if no_image == False:
            # Get captions for image if there is one
            image = image_preprocessing(image)
            if (image.shape == (3, 256, 256)) & (np.max(image) <= 256):
                image = image_normalisation(image, device)
                predicted_captions, _, _ = captioning(enc, dec, image, wordmap, device, res)
                # Return predicted captions
                return ' '.join(predicted_captions[1:])
            else:
                return 'Image is not in png'
    else:
        # If there is no image in request, upload it
        return '''
        <!doctype html>
        <title>Upload new File</title>
        <h1>Upload new File</h1>
        <form action="" method=post enctype=multipart/form-data>
          <p><input type=file name=image>
             <input type=submit value=Upload>
        </form>
        '''

if __name__ == '__main__':
    serve(app, host = "0.0.0.0", port = "6666")