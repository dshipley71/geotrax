# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 17:09:12 2022

@author: dship
"""

import os
import shutil
import glob
import cv2
import warnings

from sklearn.cluster import DBSCAN
from mtcnn import MTCNN
#from dface import MTCNN
from dface import FaceNet

# supress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class Cluster(object):
    def __init__(self, image_path='output', models='models', device='cpu', minimum_samples=5, eps=0.32, metric='cosine'):

        self.device           = device                      # 'cpu' or 'cuda'
        self.model_path       = os.path.abspath(models)     # model path for use with dface library (mtcnn, facenet)
        self.image_path       = os.path.abspath(image_path) # model path for use with dface library (mtcnn, facenet)
        self.minimum_samples  = minimum_samples             # minimum samples
        self.maximum_distance = eps                         # EPS
        self.distance_metric  = 'cosine'                    # distance directory

        print(self.device)
        print(self.model_path)
        print(self.image_path)
        print(self.minimum_samples)
        print(self.maximum_distance)
        print(self.distance_metric)

        # create cluster directory
        self.cluster_path = self.image_path + "/clustered/"
        if os.path.exists(self.cluster_path):
            shutil.rmtree(self.cluster_path)
            os.mkdir(self.cluster_path)
        else:
            os.mkdir(self.cluster_path)
            
        # This mtcnn detector is different from the implementation used by dface
        # and does not require the mtcnn.pt model. It is strictly CPU based. The
        # face detection weight model is built into the code.
        self.detector = MTCNN()
        
        # This uses FaceNet's MTCNN implementation and requires the mtcnn.pt model.
        # This can be used with both CPU and GPU.
        #
        # Note: To use this, reference the old batman code. The method to obtain
        # facial embeddings (facial feature vector) is slightly different. The
        # above method is used to eliminate the need to use an external model.
        #self.detector = MTCNN(self.device, model=self.model_path + '/mtcnn.pt')
        
        # This is FaceNet's face recognition model. Requires facenet.pt model. This
        # can be used with both CPU and GPU.
        self.facenet = FaceNet(self.device, model=self.model_path + '/facenet.pt')

    def __get_images(self):
        faces = []
        names = []

        filenames = glob.glob(self.image_path + '/*')

        for filename in filenames:
            #print(f'==> {filename}')

            # Load the image
            image = cv2.imread(filename)

            # Detect the face in the image
            try:
                results = self.detector.detect_faces(image)
            except:
                print(f'Error: {filename} is an invalid image')
                continue

            if len(results) > 0:
                # Crop the face from the image
                x, y, w, h = results[0]['box']
                face = image[y:y+h, x:x+w]

                # Resize the face to 224x224
                face = cv2.resize(face, (224, 224))
                faces.append(face)
                names.append(filename)
                
        return faces, names
        
    def cluster_faces(self):
        """
        Function to form clusters using the DBSCAN clustering algorithm
        """
        dbscan = DBSCAN(eps=self.maximum_distance, min_samples=self.minimum_samples, metric=self.distance_metric, n_jobs=-1).fit(self.embeddings)
        labels = dbscan.labels_
        print(labels)
        
        # create cluster subfolders
        for cluster_id in range(min(labels), max(labels) + 1):
            os.mkdir(self.cluster_path + str(cluster_id)) 

        for i in range(len(labels)):
            src = self.names[i]
            dst = self.cluster_path + str(labels[i])
            #print(src)
            #print(dst)
            shutil.copy(src, dst)

    def run(self):
        """
        """        
        self.faces, self.names = self.__get_images()
        #print('images: ', len(self.faces))
        #print('names : ', len(self.names))
        self.embeddings = self.facenet.embedding(self.faces)
        #print('embed : ', len(self.embeddings))
        self.cluster_faces()
        pass

if __name__ == '__main__':
    c = Cluster(image_path='output/cropped_faces')
    c.run()
