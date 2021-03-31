from flask import Flask, jsonify, request
import numpy as np
from imageio import imread
import torch
from os import listdir
from os.path import isfile, join
from server_utils import load_models, load_wordmap, image_preprocessing, image_normalisation, captioning, video_to_screenshots
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
enc, dec = load_models(checkpoint_name = 'checkpoint_7_0.pth.tar')
wordmap, res = load_wordmap()
app = Flask(__name__)


@app.route('/')
def base():
    return jsonify('Everything works fine')

@app.route('/get_image_caption', methods=['POST'])
def get_image_caption():
    no_image = False
    try:
        image = request.files['image']
        print(type(image))
        image_hash = hash(image)
        image.save('images/{}.png'.format(image_hash))
        image = imread('images/{}.png'.format(image_hash))
    except:
        no_image = True
        return jsonify('No image file on post request or it is in the wrong format')
    if no_image == False:
        image = image_preprocessing(image)
        if (image.shape == (3, 256, 256)) & (np.max(image) <= 256):
            image = image_normalisation(image, device)
            predicted_captions, _ = captioning(enc, dec, image, wordmap, device, res)
            return ' '.join(predicted_captions[1:])
        else:
            return 'Image is not in png'


@app.route('/get_video_captions', methods=['POST'])
def get_video_captions():
    no_video = False
    try:
        video = request.files['video']
        print(type(video))
        video_hash = hash(video)
        video.save('videos/{}.mp4'.format(video_hash))
    except:
        no_video = True
        return jsonify('No video file in post requests')
    if no_video == False:
        video_to_screenshots('videos/{}.mp4'.format(video_hash), 'saved_screenshots', 200)
        list_of_files = [f for f in listdir('saved_screenshots') if isfile(join('saved_screenshots', f))]
        all_captions = []
        all_encoders_out = []
        for file in list_of_files:
            image = imread('saved_screenshots/' + file)
            os.remove('saved_screenshots/' + file)
            image = image_preprocessing(image)
            if (image.shape == (3, 256, 256)) & (np.max(image) <= 256):
                image = image_normalisation(image, device)
                predicted_captions, encoder_out = captioning(enc, dec, image, wordmap, device, res)
                all_captions.append(predicted_captions)
                all_encoders_out.append(encoder_out)

        return jsonify({'captions': all_captions,
                        'encoders_out': all_encoders_out})






@app.route('/get_captions', methods=['POST', 'GET'])
def get_captions():

    if request.method == 'POST':
        no_image = False
        try:
            image = request.files['image']
            print(type(image))
            image_hash = hash(image)
            image.save('images/{}.png'.format(image_hash))
            image = imread('images/{}.png'.format(image_hash))
        except:
            no_image = True
            return jsonify('No image file on post request or it is in the wrong format')
        if no_image == False:
            image = image_preprocessing(image)
            if (image.shape == (3, 256, 256)) & (np.max(image) <= 256):
                image = image_normalisation(image, device)
                predicted_captions, _ = captioning(enc, dec, image, wordmap, device, res)
                return ' '.join(predicted_captions[1:])
            else:
                return 'Image is not in png'
    else:
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
    app.run(host = "0.0.0.0", port = "8000", debug = True)