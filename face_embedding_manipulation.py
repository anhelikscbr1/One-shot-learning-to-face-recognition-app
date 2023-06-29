
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
import sys
#import math
#import tqdm
import pandas as pd
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
from scipy.spatial import distance
import numpy as np
import pickle
import cv2
from align_mtcnn import AlignMTCNN
import tensorflow.compat.v1 as tf

import shutil #To copy a new file to DB folder
#from google.colab.patches import cv2_imshow
from datetime import datetime

tf.disable_v2_behavior()

from operator import itemgetter
import matplotlib.pyplot as plt


class FaceEmbedding:
    def load_pickle(self):
        embeddings = pickle.load( open( "extracted_embeddings.pickle", "rb" ) )
        return embeddings 
    
    def __init__(self, data_path='people', model_dir='model/20170511-185253'):
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
                        with open('extracted_embeddings.pickle','wb') as f:
                            pickle.dump(extracted,f)
                        return faces
                    else:
                        img = cv2.imread(img_path, 1)
                        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        bounding_boxes, points = self.alignMTCNN.get_bounding_boxes(image=img)
                        faces = self.get_faces(img, bounding_boxes, points, img_path)
                        #self.weaviate(faces, client, single = True)
                        return faces
    def weaviate(self, faces, client, single=False):
        aux = 0
        # file name with extension
        file_name = os.path.basename(faces[0].get("name"))

        # file name without extension
        file_name=os.path.splitext(file_name)[0]
        print('file= '+file_name)
        where_filter = {
            "path": ["name"],
            "operator": "Equal",
            "valueText": file_name,
        }
        result = (
            client.query
            .get("Img", ["name"])
            .with_where(where_filter)
            .do()
        )
        for i in result:
            aux = len(result.get('data').get('Get').get('Img'))

        if aux > 0:
            print('Image Name alrady exists in DB')
            return faces
        else:
            if(single):
                print("Copying to DB")
                shutil.copy(faces[0].get("name"), "./people2/")
                #copiar imagen nueva a la ruta de imagenes de la db si el checkbox esta activo

        data_obj = {
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

    def boxplot_graph(self, matrix):
        matrix = np.unique(matrix)
        matrix = np.delete(matrix, 0)
        #print(len(matrix))
        #print(matrix)
        fig = plt.figure(figsize =(10, 7))

        # Creating plot
        plt.boxplot(matrix)
        # show plot
        plt.show()

    def create_matrix (self, face_embedding, distance_metric):
        emb_list = face_embedding.load_pickle()
        no_images = len(emb_list)
        print(no_images)
        matrix = np.zeros((no_images , no_images))

        for img in range(no_images): 
            for j in range(no_images):
              #Calculating the euclidean distance
              array1 = emb_list[img][0].get("embedding")
              array2 = emb_list[j][0].get("embedding")

              dist = distance.cdist(array1, array2, distance_metric)
              matrix[img, j] = dist
              
        self.boxplot_graph(matrix)

        try:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            today = now.strftime("%b-%d-%Y")   
            #DF = pd.DataFrame(matrix)
            #DF.to_csv("csv/" + str(no_images) + "|20170511-185253" + distance_metric + str(today) +"|"+ current_time +".csv")  
            print(distance_metric + " matrix successfully created")
            print('std deviation: ')
            print(matrix.std())
        except:
            print("Error")

    def nw_image_check(self, image_path):
        img = cv2.imread(image_path, 1)
        cv2.imshow('image',img)
        cv2.waitKey(0)
        print("Looking for best coincidences")
        embedding = face_embedding.convert_to_embedding(single=True, img_path = image_path)
        emb_list = face_embedding.load_pickle()
        no_images = len(emb_list)

        for img in range(no_images):
            array1 = embedding[0].get("embedding")
            array2 = emb_list[img][0].get("embedding")

            euclidean_dist = distance.cdist(array1, array2, 'euclidean')
            manhattan_dist = distance.cdist(array1, array2, 'cityblock')
            cosine_dist = distance.cdist(array1, array2, 'cosine')

            coincidence_factor = 0.9
            if(euclidean_dist < coincidence_factor or manhattan_dist/10 < coincidence_factor or cosine_dist < 0.4):
                print(emb_list[img][0].get("name"), euclidean_dist, manhattan_dist, cosine_dist)
                img = cv2.imread("people/"+ emb_list[img][0].get("name"), 1)
                cv2.imshow('image',img)
                cv2.waitKey(0)
        print("End")
        cv2.waitKey(0)
      
    def nw_images_range(self, image_path, no_coincedences):
        img = cv2.imread(image_path, 1)
        #cv2_imshow(img)
        print("Looking for best coincidences")
        embedding = face_embedding.convert_to_embedding(single=True, img_path = image_path)
        emb_list = face_embedding.load_pickle()
        #ord_emb_list = []
        no_images = len(emb_list)

        emb_list_mod = []
        for img in emb_list:
            aux_dict = img[0] 
            aux_dict['euclidean_distance'] = 0 
            aux_dict['cosine_distance'] = 0 
            aux_dict['manhattan_distance'] = 0 
            emb_list_mod.append(aux_dict)

        for img in range(no_images):
            array1 = embedding[0].get("embedding")
            array2 = emb_list[img][0].get("embedding")

            euclidean_dist = distance.cdist(array1, array2, 'euclidean')
            manhattan_dist = distance.cdist(array1, array2, 'cityblock')
            cosine_dist = distance.cdist(array1, array2, 'cosine')

            emb_list_mod[img]['euclidean_distance'] = euclidean_dist
            emb_list_mod[img]['cosine_distance'] = cosine_dist
            emb_list_mod[img]['manhattan_distance'] = manhattan_dist

        euclidean_array = sorted(emb_list_mod, key=itemgetter('euclidean_distance'))
        manhattan_array = sorted(emb_list_mod, key=itemgetter('manhattan_distance'))
        cosine_array = sorted(emb_list_mod, key=itemgetter('cosine_distance'))
        
        print ("euclidean")
        for i in range(no_coincedences):
          print(euclidean_array[i]['euclidean_distance'], euclidean_array[i]['name'])
          img = cv2.imread("people/"+ euclidean_array[i]['name'], 1)
          #cv2_imshow(img)
        
        print ("manhattan")
        for i in range(no_coincedences):
          print(manhattan_array[i]['manhattan_distance'], manhattan_array[i]['name'])
          img = cv2.imread("people/"+ manhattan_array[i]['name'], 1)
          #cv2_imshow(img)

        print ("cosine")
        for i in range(no_coincedences):
          print(cosine_array[i]['cosine_distance'], cosine_array[i]['name'])
          img = cv2.imread("people/"+ cosine_array[i]['name'], 1)
          #cv2_imshow(img)

        print("End")

    def nw_image_weaviate(self,  face_embedding, image_path, limit, flag):
        client = weaviate.Client(
            url="https://oneshot-learning-ugto-n5yo5ft6.weaviate.network",  # Replace with your endpoint
        )
        embedding = face_embedding.convert_to_embedding(single=True, img_path = image_path )
        results = client.query.get("Img", ["name"]).with_near_vector({"vector": embedding[0].get("embedding")}).with_additional(["distance"]).with_limit(limit).do()
        #print(results)
        results = results.get('data').get('Get').get('Img')
        if flag==True:
            self.weaviate(embedding, client, single = True)
        for i in results:
            print(i.get('_additional').get('distance'), i.get('name'))
            return i.get('name')
if __name__ == '__main__':

    face_embedding = FaceEmbedding()

    ########################Calculate the euclidean distance matrix########################
    #To create a matrix, insert a distance matric in the create_matrix() method:

    #The distance metric to use. If a string, the distance function can be ‘braycurtis’, 
    #‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’, ‘cosine’, ‘dice’, ‘euclidean’, 
    #‘hamming’, ‘jaccard’, ‘jensenshannon’, ‘kulczynski1’, ‘mahalanobis’, ‘matching’, ‘minkowski’,
    #‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’.

    #face_embedding.create_matrix(face_embedding, "cosine")

    ##############Looking for coincidences##############
    #face_embedding.nw_image_check('test6.jpg')

    ##############Looking for first n-coincidences##############  
    #face_embedding.nw_images_range('test4.jpg', 1)
    #face_embedding.nw_image_weaviate(face_embedding, 'test6.jpg', 1)