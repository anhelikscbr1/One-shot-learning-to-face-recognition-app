
# Compute the 128D vector that describes the face in img identified by
# shape.  In general, if two face descriptor vectors have a Euclidean
# distance between them less than 0.6 then they are from the same
# person, otherwise they are from different people. 

#ruta: /content/drive/MyDrive/servicio_social/face_embedding.py

from __future__ import division
from __future__ import print_function
import weaviate
import tensorflow as tf
import numpy as np
#import argparse
import facenet
import os
#import sys
#import math
#import tqdm
#import pandas as pd
#from sklearn import metrics
#from scipy.optimize import brentq
#from scipy import interpolate
#from scipy.spatial import distance
import numpy as np
import pickle
import cv2
from align_mtcnn import AlignMTCNN
import tensorflow.compat.v1 as tf
#from google.colab.patches import cv2_imshow
from datetime import datetime

tf.disable_v2_behavior()

from operator import itemgetter
import matplotlib.pyplot as plt


class FaceEmbedding:
    
    def __init__(self, data_path='./people2', model_dir='model/20170511-185253'):
    #def __init__(self, data_path='people', model_dir='model/20180408-102900'):
        self.data_path = data_path
        self.model_dir = model_dir
        self.alignMTCNN = AlignMTCNN()
        self.images_placeholder = None
        self.embeddings = None
        self.phase_train_placeholder=None
        self.embedding_size=None
        self.image_size = 160
        self.threshold=[0.8, 0.8, 0.8]
        self.factor=0.7
        self.minsize = 20
        self.margin=44
        self.detect_multiple_faces = False

    def convert_to_embedding(self, single=False, img_path=None):
        extracted = []
        client = weaviate.Client(
            url="https://oneshot-learning-ugto-n5yo5ft6.weaviate.network",  # Replace with your endpoint
        )
        with tf.Graph().as_default():
                with tf.Session() as sess:
                    self.sess = sess
                    # Load the model
                    facenet.load_model(self.model_dir)
                    
                    # Get input and output tensors
                    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                    print(dir(images_placeholder))
                    print(tf.shape(images_placeholder))
                    self.images_placeholder = tf.image.resize_images(images_placeholder,(self.image_size, self.image_size))
                    self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                    self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                    self.embedding_size = self.embeddings.get_shape()[1]
                    if not single:
                        for filename in os.listdir(self.data_path):
                            print(filename)
                            img = cv2.imread(self.data_path+"/"+filename, 1)
                            #cv2_imshow(img)
                            bounding_boxes, points = self.alignMTCNN.get_bounding_boxes(image=img)
                            faces = self.get_faces(img, bounding_boxes, points, filename)
                            self.weaviate(faces, client)
                            if(len(faces)==0):
                                continue
                            extracted.append(faces)
                        with open('extracted_embeddings3.pickle','wb') as f:
                            pickle.dump(extracted,f)
                        return faces
                    else:
                        img = cv2.imread(img_path, 1)
                        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        bounding_boxes, points = self.alignMTCNN.get_bounding_boxes(image=img)
                        faces = self.get_faces(img, bounding_boxes, points, img_path)
                        return faces

    def weaviate(self, faces, client):
        aux = 0
        # file name with extension
        file_name = os.path.basename(faces[0].get("name"))

        # file name without extension
        file_name=os.path.splitext(file_name)[0]
        print('file= '+file_name)
#        where_filter = {
            #"path": ["name"],
            #"operator": "Equal",
            #"valueText": file_name,
        #}
        #result = (
            #client.query
            #.get("Img", ["name"])
            #.with_where(where_filter)
            #.do()
        #)
        #for i in result:
            #aux = len(result.get('data').get('Get').get('Img'))

        #if aux > 0:
            #print('Image Name alrady exists in DB')
            #return faces

        data_obj = {
            #"name": faces[0].get("name")
            "name": file_name 
        }

        data_uuid = client.data_object.create(
          data_obj,
          "Img",
          #uid="36ddd591-2dee-4e7e-a3cc-eb86d30a0923", # optional, if not provided, one is going to be generated
          vector = faces[0].get("embedding"), # supported types are `list`, `numpy.ndarray`, `torch.Tensor` and `tf.Tensor`.
        )
        a = client.query.aggregate("Img").with_meta_count().do()

    def get_faces(self, img, bounding_boxes, points, filename):
        faces = []
        nrof_faces = bounding_boxes.shape[0]
        print("No. of faces detected: {}".format(nrof_faces))

        if nrof_faces>0:
            det = bounding_boxes[:,0:4]
            det_arr = []
            img_size = np.asarray(img.shape)[0:2]
            if nrof_faces>1:
                if self.detect_multiple_faces:
                    for i in range(nrof_faces):
                        det_arr.append(np.squeeze(det[i]))
                else:
                    bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                    img_center = img_size / 2
                    offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                    offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                    index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
                    det_arr.append(det[index,:])
            else:
                det_arr.append(np.squeeze(det))

            for i, det in enumerate(det_arr):
                det = np.squeeze(det)
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0]-self.margin/2, 0)
                bb[1] = np.maximum(det[1]-self.margin/2, 0)
                bb[2] = np.minimum(det[2]+self.margin/2, img_size[1])
                bb[3] = np.minimum(det[3]+self.margin/2, img_size[0])
                cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                resized = cv2.resize(cropped, (self.image_size, self.image_size),interpolation=cv2.INTER_CUBIC)
                prewhitened = facenet.prewhiten(resized)
                faces.append({'name': filename,'rect':[bb[0],bb[1],bb[2],bb[3]],'embedding':self.get_embedding(prewhitened)})

        return faces

    def get_embedding(self, processed_img):
        reshaped = processed_img.reshape(-1, self.image_size, self.image_size, 3)
        feed_dict = {self.images_placeholder:reshaped, self.phase_train_placeholder:False }
        feature_vector = self.sess.run(self.embeddings, feed_dict=feed_dict)
        return feature_vector
    
    def create_embedding(self, face_embedding):
        #Creating the embedding
        embedding = face_embedding.convert_to_embedding()

if __name__ == '__main__':
    print('Inicializando')
    ##############Initializing the class ##############
    face_embedding = FaceEmbedding()
    ##############Create the embedding##############
    a = face_embedding.create_embedding(face_embedding)   