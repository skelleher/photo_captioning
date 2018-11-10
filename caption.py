#!/usr/bin/env python
import os
import sys
import argparse
import requests
import io
import numpy as np
import torch
import time
from PIL import Image
from torchvision import transforms
from pycocotools.coco import COCO
from vocabulary import Vocabulary
from model import EncoderCNN, DecoderRNN
from flask import Flask, jsonify, request, _app_ctx_stack
from flask_restful import Resource, Api

# Flask web server
_app   = None
_api   = None
_index = None
_args  = None

# REST resources

# Given an input image (.png or .jpeg) generate a caption
class ImageCaptionResource(Resource):
    def post(self):
        if request.headers["Content-Type"] != "application/octet-stream":
            return "Unsupported Media Type", 415

        start = time.time()
        image_bytes = request.data
        image = Image.open( io.BytesIO(image_bytes) )
        caption = _get_prediction_from_image( image )
        stop = time.time()
        msecs = (stop - start) * 1000
        print("%d ms: %s bytes -> %d" % (msecs, request.headers["Content-Length"], len(caption)))
    
        return jsonify(caption)
 


# Models pre-trained on MS-COCO:
#   encoder (Resnet + embedding layers)
#   decoder (LSTM)  
_encoder_file = "encoder.pkl"
_decoder_file = "decoder.pkl"

# Hack: should recover these from the saved model
_embed_size = 256
#_embed_size = 512
_hidden_size = 512
_num_layers = 3

_device  = None
_encoder = None
_decoder = None
_vocab   = None

    
def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="(optional) path to photograph, for which a caption will be generated", nargs = "?")
    parser.add_argument("--host", help="(optional) host to start a webserver on. Default: 0.0.0.0", nargs = "?", default = "0.0.0.0")
    parser.add_argument("--port", help="(optional) port to start a webserver on. http://hostname:port/query", nargs = "?", type = int, default = 1985)
    parser.add_argument("--verbose", "-v", help="print verbose query information", action="store_true")
   
    global _args
    _args = parser.parse_args()

    if not _args.filename and not _args.port:
        parser.print_help()
        sys.exit(-1)

    global _device
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("PyTorch device = ", _device)

    # Load the vocabulary dictionary
    vocab_threshold = None,
    vocab_file = "./vocab.pkl"
    start_word = "<start>"
    end_word   = "<end>"
    unk_word   = "<unk>"
    load_existing_vocab = True
    #annotations_file = "/opt/cocoapi/annotations/captions_train2014.json"
    annotations_file = None

    print("Loading vocabulary...")
    global _vocab
    _vocab = Vocabulary(vocab_threshold, vocab_file, start_word, end_word, unk_word, annotations_file, load_existing_vocab)
    vocab_size = len (_vocab)
    print("Vocabulary contains %d words" % vocab_size)

    # Load pre-trained models: 
    # encoder (Resnet + embedding layers)
    # decoder (LSTM)
    global _encoder
    global _decoder
    encoder_path = os.path.join("./models/", _encoder_file)
    decoder_path = os.path.join("./models/", _decoder_file)
    print("Loading ", encoder_path)
    _encoder = EncoderCNN(_embed_size)
    _encoder.load_state_dict(torch.load(encoder_path))
    _encoder.eval()
    _encoder.to(_device)

    print("Loading ", decoder_path)
    _decoder = DecoderRNN(_embed_size, _hidden_size, vocab_size, _num_layers)
    _decoder.load_state_dict(torch.load(decoder_path))
    _decoder.eval()
    _decoder.to(_device)

    # Caption the photo, or start a web server if no photo specified
    if _args.filename:
        _get_prediction_from_file(_args.filename)
    else:
        global _app
        global _api

        _app = Flask(__name__)
        _api = Api(_app)

        _api.add_resource(ImageCaptionResource,
                "/v1/caption",
                "/v1/caption/")
        _app.run(host = _args.host, port = _args.port)

 
# Convert model output from tokens back into English, with white space
def _clean_sentence(output):
    sentence = ""
    for token in output:
        word = _vocab.idx2word[token]
        space = True
        
        if word == _vocab.start_word:
            continue
        if word == _vocab.end_word:
            break
        if "." not in word:
            word = " " + word
            
        sentence += word

    return sentence.strip()


# Pre-process images for feeding to the encoder
_image_transform = transforms.Compose([ 
                transforms.Resize(256),                          # smaller edge of image resized to 256
                transforms.CenterCrop(224),                      # get 224x224 crop from center
                transforms.ToTensor(),                           # convert the PIL Image to a tensor
                transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                                     (0.229, 0.224, 0.225))])


def _get_prediction_from_file(filename):
    image = Image.open(filename)
    
    return _get_prediction_from_image(image)


def _get_prediction_from_image(image):
    image = _image_transform(image).unsqueeze(0)
    image = image.to(_device)
    
    features = _encoder(image).unsqueeze(1)
    output = _decoder.sample(features)    
    sentence = _clean_sentence(output)
    print(sentence)

    return sentence


if __name__ == "__main__":
    _main()

