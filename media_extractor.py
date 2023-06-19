# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 09:34:55 2022

@author: dship
"""
import os
import shutil
import glob
import fitz
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import magic
import cv2
#import pyexiv2 as pex
import warnings
import subprocess
import platform
import boto3

from loguru import logger
from streamlit import runtime
from streamlit.runtime.scriptrunner import get_script_run_ctx

from stqdm import stqdm
from st_aggrid import AgGrid
from zipfile import ZipFile
from PIL import Image as Img
from botocore.exceptions import ClientError

from sklearn.cluster import DBSCAN
from mtcnn import MTCNN
#from dface import MTCNN
from dface import FaceNet
from stqdm import stqdm

# supress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# suppress tensorflow logging output at launch of application
#   0 = all messages are logged (default behavior)
#   1 = INFO messages are not printed
#   2 = INFO and WARNING messages are not printed
#   3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Configure logging
logger.add(sink="media_extractor_{time:YYYYMMDD}.log",
           mode="a",
           rotation="00:00",      # new log created daily
           retention="30 days",   # keep logs 30 days
           level="INFO",          # record log level rom INFO and above
           backtrace=True,        # logging exceptions
           diagnose=True)

def get_remote_ip() -> str:
    """
    Retrieve the remote IP address of a client making a request.

    This function attempts to obtain the script run context and retrieves the
    session information associated with the session ID from the runtime. If
    successful, it returns the remote IP address from the session information.
    In case of any errors or if the required information is not available, it
    returns None.
    """

    try:
        ctx = get_script_run_ctx()
        if ctx is None:
            return None

        session_info = runtime.get_instance().get_client(ctx.session_id)
        if session_info is None:
            return None
    except Exception as e:
        return None

    return session_info.request.remote_ip

def find_process_ids(port):
    """
    Find the process IDs associated with a given port number.

    This function searches for process IDs that are listening on the specified
    port. It checks the system platform and performs different operations based
    on the platform. If the system is Windows, it uses the 'netstat' command to
    gather information about TCP connections and filters for the port and
    'LISTENING' status. On Linux systems, it utilizes the 'lsof' command to
    obtain information about TCP connections and extracts the process ID
    associated with the port. The function returns a list of process IDs found.

    Note: This function requires the 'subprocess' module and appropriate
          permissions to execute the system commands.
    """
    process_ids = []
    system = platform.system()
    if system == 'Windows':
        try:
            output = subprocess.check_output(['netstat', '-ano', '-p', 'tcp'])
            lines = output.decode().split('\n')
            for line in lines:
                if 'LISTENING' in line and f':{port}' in line:
                    process_id = int(line.split()[-1])
                    process_ids.append(process_id)
        except subprocess.CalledProcessError:
            logger.warning("Failed to find file server process id. If this is the initial launch of Media Extractor, then file server does not exist yet.")
            
    elif system == 'Linux':
        try:
            output = subprocess.check_output(['lsof', '-i', f'tcp:{port}'])
            lines = output.decode().split('\n')
            for line in lines[1:]:
                if line:
                    process_id = int(line.split()[1])
                    process_ids.append(process_id)
        except subprocess.CalledProcessError:
            logger.warning("Failed to find file server process id. If this is the initial launch of Media Extractor, then file server does not exist yet.")

    return process_ids

def kill_processes(process_ids):
    """
    Kill the specified process IDs.

    This function iterates over a list of process IDs and attempts to terminate
    each process. It checks the system platform and performs different operations
    based on the platform. On Windows systems, it uses the 'taskkill' command
    with the '/F' flag to forcefully terminate the process using its ID. On Linux
    systems, it uses the 'kill' command with the '-9' signal to forcefully
    terminate the process with the given ID. If the termination is successful,
    the function continues to the next process ID. If an error occurs while
    attempting to terminate a process, an error message is printed.

    Note: This function requires the 'subprocess' module and appropriate
          permissions to execute the system commands.
    """
    system = platform.system()
    for process_id in process_ids:
        try:
            if system == 'Windows':
                subprocess.check_output(['taskkill', '/F', '/PID', str(process_id)])
            elif system == 'Linux':
                subprocess.check_output(['kill', '-9', str(process_id)])
            #print(f"Process {process_id} killed successfully!")
        except subprocess.CalledProcessError:
            logger.warning(f"Failed to kill process id {process_id}")
            #print(f"Failed to kill process {process_id}.")

class MediaExtractor(object):
    """
    If using an S3 bucket to stored processed images, make sure to specify the
    bucket name. Otherwise processed images will only be stored locally.
    """
    def __init__(self, confidence=0.90, skip_frames=0, crop_margin=1.10, image_path='output',
                 models='models', device='cpu', minimum_samples=5, eps=0.32, metric='cosine',
                 bucket_name='batmanplus', bucket_folder='s3_download'):

        # supported file types
        self.supported_filetypes = [
            'docx', 'docm', 'dotx', 'dotm', 'xlsx', 'xlsm', 'xltx', 'xltm',
            'pptx', 'pptm', 'potm', 'potx', 'ppsx', 'ppsm', 'odt',  'ott',
            'ods',  'ots',  'odp',  'otp',  'odg',  'doc',  'dot',  'ppt',
            'pot',  'xls',  'xlt',  'pdf', 'zip', 'mp4', 'avi', 'webm', 'wmv',
            'jpeg', 'jpg', 'png', 'gif', 'bmp', 'tiff'
        ]

        # document table
        self.extract_df = pd.DataFrame(columns=['File', 'Type', 'Size', 'Count'])

        # media table
        self.media_df = pd.DataFrame(columns=['Media', 'EXIF', 'Size', 'Height', 'Width', 'Format', 'Mode', 'Hash'])        
        
        # image table
        self.image_df = pd.DataFrame(columns=['Image', 'BoxXY', 'Height', 'Width', 'Left Eye', 'Right Eye', 'Nose', 'Mouth Left', 'Mouth Right', 'IPD', 'Confidence', 'Media', 'Hash'])

        # determine file type
        self.mime = magic.Magic(mime=True)
        
        # this is the name of the top level of the output folder
        self.results_folder = './output/'
        
        self.subfolders = []
        
        # lower and upper bounds check
        if confidence < 0.25:
            self.confidence = 0.25
        elif confidence > 1.00:
            self.confidence = 1.00
        else:
            self.confidence = confidence
        
        # lower-bounds check
        if skip_frames < 1:
            self.skip_frames = 1
        else:
            self.skip_frames = skip_frames
        
        # lower bounds check
        if crop_margin < 1:
            self.crop_margin = 1        
        else:
            self.crop_margin = crop_margin
        
        self.server = None
        
        # To utilize upload/download data to an S3 bucket, a credentials file
        # needs to be created and stored in the ~/.aws folder. Alternatively,
        # environment variables can be used to store the required AWS keys.
        # See BOTO3 documentation for setup of a credentials file and/or
        # setting up of environment variables.
        # Reference:
        #   https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html

        # Mounting an S3 bucket inside a docker container
        # Reference: https://github.com/skypeter1/docker-s3-bucket
        
        # setup for upload/download to an S3 bucket
        self.bucket_name = bucket_name
        if self.bucket_name is not None:
            logger.info("S3 bucket enabled")
            #print(">>>>> S3 BUCKET ENABLED <<<<<")
            self.s3_download = bucket_folder
            self.s3_resource = boto3.resource('s3')
            self.s3_bucket = self.s3_resource.Bucket(self.bucket_name)
        else:
            logger.info("S3 Bucket disabled")
            #print("<<<<< S3 BUCKET DISABLED >>>>>")

        self.device           = device                      # 'cpu' or 'cuda'
        self.model_path       = os.path.abspath(models)     # model path for use with dface library (mtcnn, facenet)
        self.image_path       = os.path.abspath(image_path) # model path for use with dface library (mtcnn, facenet)
        self.minimum_samples  = minimum_samples             # minimum samples
        self.maximum_distance = eps                         # EPS
        self.distance_metric  = 'cosine'                    # distance directory
            
        # This mtcnn detector is different from the implementation used by dface
        # and does not require the mtcnn.pt model. It is strictly CPU based. The
        # face detection weight model is built into the code.
        # Reference: https://github.com/ipazc/mtcnn
        self.detector = MTCNN()
        
        # This uses FaceNet's MTCNN implementation and requires the mtcnn.pt model.
        # This can be used with both CPU and GPU.
        # Note: To use this, reference the old batman code. The method to obtain
        # facial embeddings (facial feature vector) is slightly different. The
        # above method is used to eliminate the need to use an external model.
        # Reference: https://github.com/deepware/dface
        #self.detector = MTCNN(self.device, model=self.model_path + '/mtcnn.pt')
        
        # This is FaceNet's face recognition model. Requires facenet.pt model. This
        # can be used with both CPU and GPU.
        # Reference: https://github.com/deepware/dface
        self.facenet = FaceNet(self.device, model=self.model_path + '/facenet.pt')
        
    def s3_upload_directory(self, path):
        """
        Upload a directory and its contents to an S3 bucket.

        This function recursively walks through the specified directory path
        and uploads each file to the configured S3 bucket. It utilizes the
        `os.walk()` function to traverse the directory tree and obtain the list
        of files. For each file found, it constructs the file path and the
        corresponding S3 bucket path relative to the specified directory.
        Finally, it uploads the file to the S3 bucket using the `upload_file()`
        method of the `s3_bucket` object.

        Note: This function requires the `os` module and a configured S3 bucket
              connection (`s3_bucket`) to perform the upload.
        """
        for root, dirs, files in os.walk(path):
            for filename in files:
                file_path = os.path.join(root, filename)
                bucket_path = os.path.relpath(file_path, path)
                self.s3_bucket.upload_file(file_path, bucket_path)

    def s3_download_directory(self, remote_directory_name, download_directory):
        """
        Download a directory and its contents from an S3 bucket.

        This function downloads the contents of a remote directory from the
        configured S3 bucket to a local download directory. It utilizes the
        `s3_bucket.objects.filter()` method to retrieve objects within the
        specified remote directory. For each object found, it creates the
        corresponding local directory (if it doesn't exist) and downloads the
        file to the local path using the `download_file()` method of the
        `s3_bucket` object.

        Note: This function requires the `os` module and a configured S3 bucket
              connection (`s3_bucket`) to perform the download.
        """
        for obj in self.s3_bucket.objects.filter(Prefix=remote_directory_name):
            # create directory if it does not exist
            local_directory = os.path.join(download_directory, os.path.dirname(obj.key)[len(remote_directory_name):].lstrip('/'))
            if not os.path.exists(local_directory):
                os.makedirs(local_directory)

            # download file
            local_path = os.path.join(local_directory, os.path.basename(obj.key))
            self.s3_bucket.download_file(obj.key, local_path)

    def not_extract(self, file):
        """
        Handle unsupported file types.

        This function is called when an unsupported file type is encountered. It
        extracts relevant information about the file, such as the file name, file
        type (determined using the `mime` module), and file size. It appends this
        metadata to the `extract_df` DataFrame for later reference.

        The function also generates informative messages based on the file extension
        and displays them using the `st.info()` function. Specific messages are
        provided for certain older file formats, such as .doc, .dot, .ppt, .pot, .xls,
        and .xlt, instructing the user to convert them to a supported format (docx,
        pdf, pptx, xlsx) using the respective Microsoft Office applications. For other
        unsupported file types, a generic message is displayed indicating that the
        file is not supported.

        Note: This function requires the `os` and `mime` modules, as well as a
              DataFrame (`extract_df`) and the ability to display messages (`st.info()`)
              using a suitable user interface framework.
        """
        file_name, file_ext = os.path.splitext(file.name)
        file_type = self.mime.from_buffer(file.read())
        file_size = file.seek(0, os.SEEK_END)
        file.seek(0,0)

        metadata = {
            'File': file_name,
            'Type': file_type,
            'Size': file_size,
            'Count': 0
        }
        self.extract_df = self.extract_df.append(metadata, ignore_index=True)

        messages = {
            '.doc': 'is an older file format (Word 1997-2003). Please convert to docx or pdf from Microsoft Word.',
            '.dot': 'is an older file format (Word 1997-2003). Please convert to docx or pdf from Microsoft Word.',
            '.ppt': 'is an older file format (PowerPoint 1997-2003). Please convert to pptx or pdf from Microsoft PowerPoint.',
            '.pot': 'is an older file format (PowerPoint 1997-2003). Please convert to pptx or pdf from Microsoft PowerPoint.',
            '.xls': 'is an older file format (Excel 1997-2003). Please convert to xlsx or pdf from Microsoft Excel.',
            '.xlt': 'is an older file format (Excel 1997-2003). Please convert to xlsx or pdf from Microsoft Excel.'
        }

        message = messages.get(file_ext, 'is not a supported file type.')
        st.info(f"{file.name} {message}")
                
    def mso_extract(self, file, location):
        """
        Extract Microsoft Office documents from 2004 to the present.

        This function is designed to extract various files and media content from
        Microsoft Office documents of different formats. It first reads enough
        data from the file to determine its MIME type using the `mime` module.
        It also retrieves the file size.

        The function then proceeds to extract the contents of the Microsoft
        Office document, which is expected to be in the form of a zip file with
        various subfolders. It uses the `ZipFile` class to extract the files to
        the specified `location`.

        Depending on the file type (doc, ppt, xl), the function handles specific
        subfolders (word, ppt, xl) containing media files (images) and renames
        and moves them to a destination subfolder. It removes the original
        subfolders (word, ppt, xl) and performs cleanup by removing certain
        directories and XML files.

        Metadata about the extracted files, such as the file name, type, size,
        and the count of extracted files, is added to the `extract_df` DataFrame
        for later reference.

        The function returns the destination subfolder path where the extracted
        files are located.

        Note: This function requires the `os`, `shutil`, `zipfile`, and `glob`
              modules, as well as a DataFrame (`extract_df`) for storing metadata.
        """
        # read enough of the data to determine the mime typ of the file
        file_type = self.mime.from_buffer(file.read())
        file.seek(0,0)
        
        # determine the size of the file read
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0,0)

        file_name = file.name
        base_name = os.path.splitext(file_name)[0]

        with ZipFile(file) as thezip:
            #st.write(thezip.infolist())
            thezip.extractall(path=location)

        # extract images
        if 'doc' in file.name:
            src = location + '/word/media/'
            root = os.path.splitext(file.name)[0]
            subfolder = os.path.splitext(file.name)[0] + self.extract_folder_name
            dest = os.path.abspath(location) + '/' + subfolder
            if os.path.exists(dest):
                shutil.rmtree(dest)
                os.makedirs(dest)
            else:
                os.makedirs(dest)               
            files = os.listdir(src)
            for f in files:
                root, ext = os.path.splitext(f)
                f_new = "{}-{}{}".format(base_name, root, ext)
                os.rename(src + f, src + f_new)
                shutil.move(src + f_new, dest)
            shutil.rmtree(location + '/word')
            metadata = {'File': file_name,
                        'Type': file_type,
                        'Size': file_size,
                        'Count': len(files)}
            self.extract_df = self.extract_df.append(metadata, ignore_index=True)

        elif 'ppt' in file.name:
            src = location + '/ppt/media/'
            root = os.path.splitext(file.name)[0]
            subfolder = os.path.splitext(file.name)[0] + self.extract_folder_name
            dest = os.path.abspath(location) + '/' + subfolder
            if os.path.exists(dest):
                shutil.rmtree(dest)
                os.makedirs(dest)
            else:
                os.makedirs(dest)
            files = os.listdir(src)
            for f in files:
                root, ext = os.path.splitext(f)
                f_new = "{}-{}{}".format(base_name, root, ext)
                os.rename(src + f, src + f_new)
                shutil.move(src + f_new, dest)
            shutil.rmtree(location + '/ppt')
            metadata = {'File': file_name,
                        'Type': file_type,
                        'Size': file_size,
                        'Count': len(files)}
            self.extract_df = self.extract_df.append(metadata, ignore_index=True)

        elif 'xl' in file.name:
            src = location + '/xl/media/'
            root = os.path.splitext(file.name)[0]
            subfolder = os.path.splitext(file.name)[0] + self.extract_folder_name
            dest = os.path.abspath(location) + '/' + subfolder
            if os.path.exists(dest):
                shutil.rmtree(dest)
                os.makedirs(dest)
            else:
                os.makedirs(dest)
            files = os.listdir(src)
            for f in files:
                root, ext = os.path.splitext(f)
                f_new = "{}-{}{}".format(base_name, root, ext)
                os.rename(src + f, src + f_new)
                shutil.move(src + f_new, dest)
            shutil.rmtree(location + '/xl')
            metadata = {'File': file_name,
                        'Type': file_type,
                        'Size': file_size,
                        'Count': len(files)}
            self.extract_df = self.extract_df.append(metadata, ignore_index=True)

        shutil.rmtree(location + '/_rels')
        shutil.rmtree(location + '/docProps')
        [os.remove(f) for f in glob.glob(location + '/*.xml')]

        return dest

    def zip_extract(self, file, location):
        """
        Extract files from a ZIP archive.

        This function extracts files from a ZIP archive. It accepts a file
        object and a location where the extracted files should be stored. The
        function begins by gathering information about the ZIP archive, such as
        the file name and size.

        Using the `ZipFile` class, the function iterates through each item in
        the ZIP archive. It determines the file type based on the file extension.
        For certain older file formats (doc, dot, ppt, pot, xls, xlt), it
        displays an informative message instructing the user to convert them to a
        supported format using the respective Microsoft Office applications.

        For specific file types (pdf, docx, docm, dotm, dotx, xlsx, xlsb, xlsm,
        xltm, xltx, potx, ppsm, ppsx, pptm, pptx, potm), the function calls
        corresponding extraction methods (`pdf_extract()`, `mso_extract()`, etc.)
        to handle the extraction and processing of the file.

        For file types related to media content (mp4, webm, avi, wmv, jpeg, jpg,
        png, gif, bmp, tiff), it calls appropriate methods (`vid_extract()`,
        `img_extract()`, etc.) to handle the extraction and processing of the media
        content.

        The function updates the `extract_df` DataFrame with metadata about the
        extracted files, such as the file name, type, size, and the count of extracted
        files.

        Note: This function requires the `os` and `zipfile` modules, as well as other
        extraction methods (`pdf_extract()`, `mso_extract()`, etc.) and a DataFrame
        (`extract_df`) for storing metadata.

        Reference: https://docs.python.org/3/library/zipfile.html#module-zipfile
        """
        # get file stats
        file_name = os.path.basename(file.name)
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0,0)
        
        with ZipFile(file) as thezip:

            for i in stqdm(range(len(thezip.infolist())),
                           leave=True,
                           desc='ZIP Extraction: ',
                           gui=True):

                zipinfo = thezip.filelist[i]

                file_type = os.path.splitext(zipinfo.filename)[1][1:]

                if file_type in ['doc', 'dot']:
                    st.info(f"{zipinfo.filename} is an older file format (Word 1997-2003). Please convert to docx or pdf from Microsoft Word. Open document then click save as.")
    
                elif file_type in ['ppt', 'pot']:
                    st.info(f"{zipinfo.filename} is an older file format (PowerPoint 1997-2003). Please convert to pptx or pdf from Microsoft PowerPoint. Open document then click save as.")
    
                elif file_type in ['xls', 'xlt']:
                    st.info(f"{zipinfo.filename} is an older file format (Excel 1997-2003). Please convert to xlsx or pdf from Microsoft Excel. Open document then click save as.")

                elif file_type in ['pdf']:
                    with thezip.open(zipinfo) as thefile:
                        imgpath = self.pdf_extract(thefile, self.output_folder)
                        self.__get_media(imgpath)
                        self.__get_images(imgpath)

                elif file_type in ['docx', 'docm', 'dotm', 'dotx', 'xlsx',
                                   'xlsb', 'xlsm', 'xltm', 'xltx', 'potx',
                                   'ppsm', 'ppsx', 'pptm', 'pptx', 'potm']:
                    with thezip.open(zipinfo) as thefile:
                        imgpath = self.mso_extract(thefile, self.output_folder)
                        self.__get_media(imgpath)
                        self.__get_images(imgpath)

                elif file_type in ['mp4', 'webm', 'avi', 'wmv']:
                    with thezip.open(zipinfo) as thefile:
                        imgpath = self.vid_extract(thefile, self.output_folder)
                        self.__get_media(imgpath)
                        self.__get_images(imgpath)

                    
                elif file_type in ['jpeg', 'jpg', 'png', 'gif', 'bmp', 'tiff']:
                    with thezip.open(zipinfo) as thefile:
                        imgpath = self.img_extract(thefile, self.output_folder)
                        self.__get_media(imgpath)
                        self.__get_images(imgpath)

                else:
                    logger.warning(f"File type `{file_type}` not supported")
                    #pass

        metadata = {'File': file_name, 'Type': 'application/zip', 'Size': file_size, 'Count': len(thezip.infolist())}
        self.extract_df = self.extract_df.append(metadata, ignore_index=True)

    def pdf_extract(self, file, location):
        """
        Extract images from a PDF file.

        This function extracts images from a PDF file using the PyMuPDF library.
        It accepts a file object and a location where the extracted images
        should be stored. The function first determines the MIME type of the
        file and retrieves its size.

        The function then reads the entire file into memory and opens it as a
        PDF file using the PyMuPDF library. It creates a subfolder path based
        on the file name and the specified `extract_folder_name`.

        For each page in the PDF, the function iterates through each image on
        that page. It saves the images as PNG files in the designated subfolder.
        If an image is in the CMYK color space, it converts it to the RGB color
        space before saving.

        The function updates the `extract_df` DataFrame with metadata about the
        extracted images, such as the file name, type, size, and the count of
        extracted images.

        The function returns the path to the subfolder where the extracted
        images are stored.

        Note: This function requires the `os` and `fitz` modules from the
              PyMuPDF library, as well as a DataFrame (`extract_df`) for
              storing metadata.

        Reference: https://pymupdf.readthedocs.io/en/latest/index.html
        """
        # determine the MIME type of the file
        file_type = self.mime.from_buffer(file.read())
        file.seek(0,0)
        
        # determine the size of the file read
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0,0)

        # read the whole file into memory
        file_data = file.read()

        # open pdf file
        pdf_file = fitz.open('pdf', file_data)

        root, ext = os.path.splitext(file.name)
        subfolder = os.path.splitext(file.name)[0] + self.extract_folder_name
        document_path = os.path.abspath(location) + '/' + subfolder
        os.makedirs(document_path, exist_ok=True)

        # image counter
        nimags = 0

        # iterating through each page in the pdf
        for current_page_index in range(pdf_file.page_count):

            #iterating through each image in every page of PDF
            for img_index, img in enumerate(pdf_file.get_page_images(current_page_index)):
                  xref = img[0]
                  image = fitz.Pixmap(pdf_file, xref)
                  
                  #if it is a is GRAY or RGB image
                  if image.n < 5:        
                      image.save("{}/{}-image{}.png".format(document_path, root, nimags))

                  #if it is CMYK: convert to RGB first
                  else:                
                      new_image = fitz.Pixmap(fitz.csRGB, image)
                      new_image.writePNG("{}/{}-image{}-{}.png".format(document_path, root, nimags, img_index))
                      
                  nimags = nimags + 1

        metadata = {'File': file.name, 'Type': file_type, 'Size': file_size, 'Count': nimags}
        self.extract_df = self.extract_df.append(metadata, ignore_index=True)

        return document_path

    def vid_extract(self, file, location):
        """
        Extract frames from a video file.

        This function extracts frames from a video file using the OpenCV
        library. It accepts a file object and a location where the extracted
        frames should be stored. The function first creates a subfolder path
        based on the file name and the specified `extract_folder_name`.

        The function copies the buffer content to the output folder and
        retrieves the file name, type, and size. It initializes a video frame
        capture using the OpenCV `VideoCapture` class and determines the total
        number of frames in the video.

        For each frame in the video, the function reads the frame using the
        `read()` method of `VideoCapture` and writes the image to the output
        path as a PNG file. The number of frames extracted depends on the
        `skip_frames` parameter, which determines the frame skipping interval.

        The function updates the `extract_df` DataFrame with metadata about the
        extracted frames, such as the file name, type, size, and the count of
        extracted frames.

        Finally, the function releases the video capture, removes the temporary
        video file, and returns the path to the subfolder where the extracted
        frames are stored.

        Note: This function requires the `os`, `shutil`, `cv2`, and `tqdm`
              modules, as well as a DataFrame (`extract_df`) for storing
              metadata.
        """
        # create extraction folder
        subfolder = os.path.splitext(file.name)[0] + self.extract_folder_name
        video_path = os.path.abspath(location) + '/' + subfolder
        if os.path.exists(video_path):
            shutil.rmtree(video_path)
            os.makedirs(video_path)
        else:
            os.makedirs(video_path)

        # copy buffer to output folder
        video_file = os.path.abspath(self.output_folder) + '/' + file.name        
        with open(video_file, "wb") as f:
            f.write(file.read())

        # get file stats
        file_name = os.path.basename(video_file)
        file_type = self.mime.from_file(video_file)
        file_size = os.path.getsize(video_file)

        # initialize video frame capture        
        vidcap = cv2.VideoCapture(video_file)

        # get total number of frames
        max_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

        # extract frames from video as png
        for i in stqdm(range(max_frames),
                       #st_container=st.sidebar,
                       leave=True,
                       desc='Media Extraction: ',
                       gui=True):

            # get frame
            success, image = vidcap.read()

            # break from loop is frame extraction fails
            if not success:
                break

            # write image to output path
            if i % self.skip_frames == 0:
                cv2.imwrite(video_path + os.path.splitext(file_name)[0] + f"_image{i+1}" + ".png", image)      

        # write file and media stats to dataframe
        metadata = {'File': file_name, 'Type': file_type, 'Size': file_size, 'Count': max_frames}
        self.extract_df = self.extract_df.append(metadata, ignore_index=True)

        # release video capture
        vidcap.release()
        
        # remove temporary video
        os.remove(video_file)

        return video_path
    
    def img_extract(self, file, location):
        """
        Extract an image file.

        This function extracts an image file. It accepts a file object and a
        location where the extracted image should be stored. The function retrieves
        the file name, type, and size. It creates a subfolder path based on the
        file name and the specified `extract_folder_name`.

        The function updates the `extract_df` DataFrame with metadata about the
        extracted image, such as the file name, type, size, and the count of
        extracted images.

        Finally, the function writes the image file to the output path and returns
        the path to the subfolder where the extracted image is stored.

        Note: This function requires the `os`, `shutil`, and `mime` modules, as
              well as a DataFrame (`extract_df`) for storing metadata.
        """
        file_name, ext = os.path.splitext(os.path.basename(file.name))
        file_type = self.mime.from_buffer(file.read())
        file.seek(0,0)
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0,0)

        subfolder = os.path.splitext(file_name)[0] + self.extract_folder_name
        imgpath = os.path.abspath(location) + '/' + subfolder
        if os.path.exists(imgpath):
            shutil.rmtree(imgpath)
            os.makedirs(imgpath)
        else:
            os.makedirs(imgpath) 

        metadata = {
            'File': file_name,
            'Type': file_type,
            'Size': file_size,
            'Count': 1
        }
        self.extract_df = self.extract_df.append(metadata, ignore_index=True)

        with open(imgpath + file_name + '-image1' + ext, "wb") as f:
            f.write(file.read())
            
        return imgpath

    def crop_face(self, img, box, margin=1):
        """Crop a facial image based on a bounding box.

        This function crops a facial image based on a given bounding box. The
        `img` parameter represents the input image, and the `box` parameter
        contains the coordinates of the bounding box in the format (x1, y1,
        x2, y2). The `margin` parameter controls the size of the resulting
        cropped image.

        A margin greater than 1 increases the size of the bounding box,
        resulting in a larger cropped image. A margin less than 1 decreases
        the bounding box size, resulting in a smaller cropped image. A margin
        equal to 1 keeps the bounding box size unchanged.

        The function calculates the size of the bounding box based on the
        maximum difference between the x and y coordinates of the box,
        multiplied by the margin. It then determines the center coordinates
        of the box and adjusts the bounding box coordinates accordingly.

        Finally, the function crops the input image using the adjusted bounding
        box coordinates and returns the resulting cropped image as a NumPy array.

        Note: This function requires the `PIL` (Python Imaging Library) module
              and the `numpy` module for image manipulation.
        """
        x1, y1, x2, y2 = box
        size = int(max(x2-x1, y2-y1) * margin)
        center_x, center_y = (x1 + x2)//2, (y1 + y2)//2

        if margin <= 1:
            x1, x2 = center_x-size//2, center_x+size//2
            y1, y2 = center_y-size//2, center_y+size//2
        else:
            x1, x2 = center_x-size//1.8, center_x+size//1.8
            y1, y2 = center_y-size//1.4, center_y+size//1.4

        face = Img.fromarray(img).crop([x1, y1, x2, y2])
        return np.asarray(face)
        
    def rotate_upside_down_face(self, image, bbox, keypoints):
        """
        Determine if a face is upside down and rotate it if necessary.

        This function analyzes the facial landmarks and determines if the face
        is upside down. It accepts an image, bounding box coordinates (`bbox`),
        and facial keypoints (`keypoints`) as input. The function calculates the
        angle of the face based on the landmarks.

        The nose, chin, and mouth points are extracted from the facial keypoints.
        The function calculates the lengths of the sides of the triangle formed
        by these landmarks. It then applies the law of cosines to calculate the
        angle between two sides of the triangle.

        If the angle is greater than 90 degrees or less than -90 degrees, it
        indicates that the face is upside down. In such cases, the function
        rotates the image by 180 degrees using the OpenCV library.

        The function returns the rotated image if the face is upside down, or the
        original image if it is not.

        Note: This function requires the `math` and `cv2` modules for calculations
              and image manipulation.
              
        Addendum: THIS FUNCTION STILL NEEDS TO BE FULLY TESTED.
        """
        import math
        # get landmarks of the face
        nose_point = keypoints['nose']
        chin_point = (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3])
        mouth_point = ((keypoints['mouth_left'][0] + keypoints['mouth_right'][0]) // 2, (keypoints['mouth_left'][1] + keypoints['mouth_right'][1]) // 2)
        #angle = imutils.face_utils.get_face_angle(nose_point, chin_point, mouth_point)
        #angle = np.degrees(np.arccos((nose_point**2 + chin_point**2 - mouth_point**2) / (2 * nose_point * chin_point)))

        st.write(nose_point, chin_point, mouth_point)

        # Calculate the lengths of the sides of the triangle formed by the landmarks
        a = nose_point[1] - chin_point[1]
        b = mouth_point[0] - nose_point[0]
        c = chin_point[0] - nose_point[0]


        st.write(a**2, b**b, c**c)
        st.write((a**2 + b**2 - c**2) / (2 * a * b))

        # Check for valid triangle
        if a <= 0 or b <= 0 or c <= 0:
            return image
        
        # Calculate the angle using the law of cosines
        cos_angle = math.acos((a**2 + b**2 - c**2) / (2 * a * b))
        st.write(cos_angle)

        # Check for valid input to acos
        if cos_angle < -1 or cos_angle > 1:
            return image
            
        angle = math.degrees(cos_angle)        
        st.write(f'Face Angle: {angle}')

        # check angle of face
        if angle > 90 or angle < -90:
            # The face is upside down, so rotate it by 180 degrees
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, 180, 1.0)
            image = cv2.warpAffine(image, M, (w, h))
        
        return image

    def face_align(self, image, keypoints):
        """
        Align a face so that the eyes are on a horizontal plane.

        This function aligns a face image based on the positions of the left and
        right eye keypoints. It accepts an image and facial keypoints as input.
        The function calculates the angle of rotation needed to align the eyes on
        a horizontal plane.

        The vertical distance (`dY`) between the y-coordinates of the right and
        left eye keypoints is calculated, as well as the horizontal distance
        (`dX`) between their x-coordinates. The `arctan2` function is then used
        to determine the angle in radians between the horizontal axis and the
        line connecting the eyes.

        The image is rotated using the OpenCV `getRotationMatrix2D` function with
        the calculated angle and no scaling. The center of the image is used as
        the rotation center. The rotated image is obtained by applying the
        rotation matrix to the image using `warpAffine`.

        The function returns the aligned face image.

        Note: This function requires the `numpy` module (`np`) for mathematical
              calculations and the OpenCV library (`cv2`) for image manipulation.
        """
        dY = keypoints['right_eye'][1] - keypoints['left_eye'][1]
        dX = keypoints['right_eye'][0] - keypoints['left_eye'][0]
        angle = np.degrees(np.arctan2(dY, dX)) #- 180

        # get the center of the image
        image_center = np.array((image.shape[1] // 2, image.shape[0] // 2), dtype=np.float32)

        # get the rotation matrix for rotating the face with no scaling
        M = cv2.getRotationMatrix2D(image_center, angle, scale=1)

        # get image dimensions
        width = image.shape[1]
        height = image.shape[0]

        image = cv2.warpAffine(image, M, (width, height), flags=cv2.INTER_CUBIC)
        
        return image

    def __get_media(self, output_folder):
        """
        Retrieve media information from a folder.

        This function retrieves information about media files (images) in a
        specified folder. It accepts an output folder path as input. The
        function iterates through the files in the folder and extracts
        relevant metadata.

        For each image file, the function attempts to open the file using the
        `PIL` (Python Imaging Library) module. It retrieves metadata such as
        the file size, height, width, format, and mode of the image. Additionally,
        it computes the average hash value of the image using the OpenCV
        `img_hash.averageHash` function.

        The extracted metadata is stored in the `media_df` DataFrame, which keeps
        track of the media information.

        Note: This function requires the `os`, `Img` (PIL), `cv2`, and `pandas`
              modules for file operations, image handling, image hashing, and
              data storage, respectively. Additionally, the function includes
              commented out code related to EXIF data extraction using the
              `pexif` library.
        
        Addendum: The `pexif` code is functional but was commented out for
                  90-day customer testing on a CentOS7 linux OS EC2 instance
                  for cloud support. This is fully functional using Ubuntu
                  linux.
        """
        try:
            files = os.listdir(output_folder)

            for f in files:
                imgfile = output_folder + '/' + f
                try:
                    im = Img.open(imgfile)
                except Exception as e:
                    logger.exception(e)
                    st.error(e)
                    break

#                # check for EXIV data
#                try:
#                    pimg = pex.Image(imgfile)
#                    data = pimg.read_exif()
#                    pimg.close()
#
#                    if data:
#                        exif_data = "yes"
#                    else:
#                        exif_data = "no"
#
#                except Exception as e:
#                    st.error(e)
#                    break

                cropped_hash = cv2.img_hash.averageHash(cv2.imread(imgfile))[0]
                cropped_hash = ''.join(hex(i)[2:] for i in cropped_hash)

                metadata = {'Media': f,
                            'EXIF': '', #exif_data,
                            'Size': os.path.getsize(output_folder + '/' + f),
                            'Height': im.height,
                            'Width': im.width,
                            'Format': im.format,
                            'Mode': im.mode,
                            'Hash': cropped_hash}
                self.media_df = self.media_df.append(metadata, ignore_index=True)
        except:
            logger.exception("Get Media Error")
            pass
        
    def __get_images(self, output_folder):
        """
        Retrieve and process facial images from a specified folder.

        This function processes facial images in a specified folder using a
        face detection model. It accepts an output folder path as input. The
        function performs the following steps:

        1.  Retrieve a list of media files (images) in the output folder.
        2.  Initialize counters and create a destination folder for the cropped
            face images.
        3.  Iterate through the media files and perform face detection using a
            face detection model.
        4.  Filter the detected faces based on a confidence threshold.
        5.  For each detected face, extract the bounding box coordinates, facial
            keypoints, and inter-pupillary distance (IPD).
        6.  Draw bounding boxes and keypoints on the image for visualization
            purposes.
        7.  Crop the face image by adjusting the bounding box size based on a
            specified margin.
        8.  Align the cropped face image based on the positions of the eyes.
        9.  Export the cropped face image to a destination folder and calculate
            an image hash.
        10. Store metadata about the detected face, including the image name,
            bounding box coordinates, dimensions, keypoints, IPD, confidence
            score, media file name, and image hash.
        11. Handle exceptions and display errors, if any.

        The extracted metadata is stored in the `image_df` DataFrame for further
        analysis.

        Note: This function requires the `glob`, `cv2`, `os`, `shutil`, `pandas`,
              and `tqdm` modules, as well as a face detection model, image
              manipulation functions (`crop_face` and `face_align`), and the
              `image_df` DataFrame for storing metadata.
        """
        media_files = glob.glob(output_folder + '*.*')

        face_count = 0        
        max_files = len(media_files)

        # set image path
        output_folder = output_folder.split(self.extract_folder_name)[0]
        image_path = output_folder + self.cropped_folder_name
        
        self.subfolders.append(output_folder)

        if os.path.exists(image_path):
            shutil.rmtree(image_path)
            os.mkdir(image_path)
        else:
            os.mkdir(image_path)

        for i in stqdm(range(max_files),
                        leave=True,
                        desc='Face Detection: ',
                        gui=True):

            f = media_files[i]
            
            try:
                image = cv2.imread(f)
                detection_image = image.copy()
                bgr = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                detections = self.detector.detect_faces(bgr)
                #st.write(detections)
                
                # filtering detections with confidence greater than confidence threshold
                for idx, det in enumerate(detections):
                    if det['confidence'] >= self.confidence:
                        face_count = face_count + 1
                        x, y, width, height = det['box']
                        keypoints = det['keypoints']
                        
                        # rotate image if upside down
#                        detection_image = self.rotate_upside_down_face(detection_image, det['box'], keypoints)

                        # calculate ipd by taking the delta between the eyes (pixel distance)
                        ipd = keypoints['right_eye'][0] - keypoints['left_eye'][0]
                        
                        # draw bounding box for face; and points for eyes, nose and mouth
                        cv2.rectangle(detection_image, (x,y), (x+width,y+height), (0,155,255), 1)

                        # crop the image by increasing the detection bounding box (i.e. margin)
                        #  a margin of 1 is the detection bounding box
                        box = x, y, x+width, y+height
                        cropped_image = self.crop_face(image, box, margin=self.crop_margin)

                        # face alignment
                        cropped_image = self.face_align(cropped_image, keypoints)

                        # export image to disk as a PNG file
                        cropped_image_name = image_path + os.path.splitext(os.path.basename(f))[0] + '-' + f'face{idx+1}' + '.png'

                        cv2.imwrite(cropped_image_name, cropped_image)

                        # convert cropped image into an image hash using cv2
                        cropped_hash = cv2.img_hash.averageHash(cropped_image)[0]
                        cropped_hash = ''.join(hex(i)[2:] for i in cropped_hash)

                        metadata = {
                            'Image': os.path.basename(cropped_image_name),
                            'BoxXY': (x, y),
                            'Height': height,
                            'Width': width,
                            'Left Eye': keypoints['left_eye'],
                            'Right Eye': keypoints['right_eye'],
                            'Nose': keypoints['nose'],
                            'Mouth Left': keypoints['mouth_left'],
                            'Mouth Right': keypoints['mouth_right'],
                            'IPD': ipd,
                            'Confidence': det['confidence'],
                            'Media': os.path.basename(f),
                            'Hash': cropped_hash
                        }
                        
                        self.image_df = self.image_df.append(metadata, ignore_index=True)

            except Exception as e:
                logger.exception(e)
                st.error(e)

    def process_images(self, filepath):
        """
        Process images for clustering.

        This function processes images in a specified file path for clustering
        purposes. It accepts a file path as input. The function performs the
        following steps:

        1. Retrieve a list of image file names in the specified file path using
           the `glob` module.
        2. Iterate through the image file names and perform the following operations
           for each image:
           - Load the image using OpenCV (`cv2`).
           - Detect the face in the image using a face detection model (`detector`).
           - If a face is detected, crop the face from the image based on the bounding
             box coordinates.
           - Resize the cropped face to a fixed size of 224x224 pixels.
           - Store the cropped face in a list of faces (`faces`) and the corresponding
             image file name in a list of names (`names`).
        3. Return the lists of faces and names.

        Note: This function requires the `glob` and `cv2` modules, as well as a face
              detection model (`detector`). Additionally, the function assumes that
              the face detection model has been properly initialized and is accessible
              within the class.
        """
        faces = []
        names = []

        filenames = glob.glob(filepath + '/*')

        #for filename in filenames:
        for i in stqdm(range(len(filenames)),
                       leave=True,
                       desc='Processing Images for Clustering: ',
                       gui=True):

            filename = filenames[i]
            #print(f'==> {filename}')

            # Load the image
            image = cv2.imread(filename)

            # Detect the face in the image
            try:
                results = self.detector.detect_faces(image)
            except:
                logger.warning(f"File {filename} is an invalid facial image")
                #print(f'Error: {filename} is an invalid image')
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
        
    def cluster_faces(self, filepath):
        """
        Cluster faces using the DBSCAN clustering algorithm.

        This function clusters faces using the Density-Based Spatial Clustering
        of Applications with Noise (DBSCAN) algorithm. It accepts a file path
        as input. The function performs the following steps:

        1. Create a directory to store the clustered identities based on the
           specified file path.
        2. Extract the facial feature vectors (512 points) using a facial
           recognition model (`facenet`).
        3. Cluster the identities using the DBSCAN algorithm with specified
           parameters (eps, min_samples, metric, n_jobs).
        4. Assign labels to the faces based on the clustering results.
        5. Create subfolders in the cluster directory for each cluster ID.
        6. Move the corresponding face images to the respective cluster subfolders.
        7. Rename the cluster folder for unclustered identities, if applicable.
        8. Return the number of clusters (i.e., the maximum label + 1).

        Note: This function requires the `os`, `shutil`, `DBSCAN`, and `tqdm`
              modules, as well as a facial recognition model (`facenet`).
              Additionally, the function assumes that the facial recognition
              model has been properly initialized and is accessible within the
              class.
        """

        # create cluster directory
        self.cluster_path = filepath + "/clustered_identities/"
        if os.path.exists(self.cluster_path):
            shutil.rmtree(self.cluster_path)
            os.mkdir(self.cluster_path)
        else:
            os.mkdir(self.cluster_path)

        # extract facial feature vector (512 points)
        self.embeddings = self.facenet.embedding(self.faces)
        
        # cluster identities
        dbscan = DBSCAN(eps=self.maximum_distance, min_samples=self.minimum_samples, metric=self.distance_metric, n_jobs=-1).fit(self.embeddings)
        labels = dbscan.labels_
        #print(labels)
        
        # create cluster subfolders
        for cluster_id in range(min(labels), max(labels) + 1):
            os.mkdir(self.cluster_path + str(cluster_id)) 

        #for i in range(len(labels)):
        #    src = self.names[i]
        #    dst = self.cluster_path + str(labels[i])
        #    shutil.copy(src, dst)

        for i in stqdm(range(len(labels)),
                       leave=True,
                       desc='Clustering Results: ',
                       gui=True):
        
            src = self.names[i]
            dst = self.cluster_path + str(labels[i])
            shutil.copy(src, dst)

        try:
            os.rename(self.cluster_path + '/-1', self.cluster_path + '/unclustered_identities')
        except:
            pass
            
        return max(labels) + 1
        
    def run(self):
        """
        Run the Media Extractor application.

        This function is responsible for running the Media Extractor application.
        It sets the configuration for the Streamlit page, handles user inputs,
        processes the media files, displays the metadata tables, and performs
        additional operations based on user selections.

        The function performs the following steps:

        1.  Set the Streamlit page configuration, including the layout, sidebar
            state, page title, and page icon.
        2.  Display the Media Input section, allowing the user to upload media files.
        3.  Display the Media Output section, allowing the user to enter a folder
            name to store the extracted images.
        4.  Process the uploaded files and extract media content based on their types
            (e.g., images, videos, documents).
        5.  Display the metadata tables for documents, media, and images.
        6.  Handle additional options such as removing subfolders, clustering
            identities, and adjusting the bounding box size.
        7.  Copy processed images to the top-level of the output folder and perform
            further operations if applicable, such as clustering identities.
        8.  Launch the file server to serve the output folder contents and provide a
            link to access the output folder.
        9.  If an S3 bucket name is provided, upload the extracted and cropped images
            to the bucket.
        10. Provide a button to clear all results and refresh the page.

        Note: This function assumes the availability of various modules and dependencies,
              such as Streamlit, OpenCV, NumPy, Pandas, and scikit-learn. Additionally,
              it references other class methods and assumes the existence of specific
              variables and file paths within the class.
        """
        # set streamlit page defaults
        st.set_page_config(
            layout = 'wide', # centered, wide, dashboard
            initial_sidebar_state = 'auto', # auto, expanded, collapsed
            page_title = 'Media Extractor',
            page_icon = ':eyes:' # https://emojipedia.org/shortcodes/
        )

        st.session_state.launch = False

        if not st.session_state.launch:
            logger.debug(f'Media Extractor application launched by {get_remote_ip()}')
            st.session_state.launch = True
            
        with st.form("my-form", clear_on_submit=False):
            # set title and format
            st.markdown(""" <style> .font {font-size:60px; font-family: 'Sans-serif'; text-align:center; color: blue;} </style> """, unsafe_allow_html=True)
            st.markdown('<p class="font">Media Extractor</p>', unsafe_allow_html=True)
            
            # get remote ip address - an ip address of "ip: ::1" means it is a local ip
            st.markdown(f'Connection from remote ip: {get_remote_ip()}')
            
            st.subheader('Media Input')
            self.uploaded_files = st.file_uploader("Choose a media file (image, video, or document):", type=self.supported_filetypes, accept_multiple_files=True)

            st.subheader('Media Output')
            project_folder = st.text_input('Enter folder name to store images:', value="", key="subfolder")
            
            self.output_folder = os.path.abspath(self.results_folder + project_folder) #st.session_state.subfolder)
            self.extract_folder_name = '/extracted_images_unedited/'
            self.detection_folder_name = '/detection/'
            self.cropped_folder_name = '/cropped_faces/'

            self.submitted = st.form_submit_button("PROCESS", type='primary')
            
            self.remove_subfolders = st.checkbox('Remove Subfolders', value=True, help='Remove subfolders from output folder')
            st.session_state.cluster = st.checkbox('Cluster Identities', value=False, help='Cluster cropped images by identity')
            #st.session_state.pose_sort = st.checkbox('Sort by Pose', value=False, help='Sort clustered identies by head pose')
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                bbox_option = st.selectbox('Bounding Box Size', options=('Tight', 'Narrow', 'Medium', 'Wide', 'Extra-Wide'), index=1, help='Adjusts the size of cropping. Tight will crop only the face region; narrow and higher performs cropping of the head by increasing the size of the bounding box. Narrow is the default setting.')
            with col2:
                st.markdown('')
            with col3:
                st.markdown('')
            with col4:
                st.markdown('')
            with col5:
                st.markdown('')
                
            if bbox_option == 'Tight':
                self.crop_margin = 0.80
            elif bbox_option == 'Narrow':
                self.crop_margin = 1.10
            elif bbox_option == 'Medium':
                self.crop_margin = 1.30
            elif bbox_option == 'Wide':
                self.crop_margin = 1.50
            elif bbox_option == 'Extra-Wide':
                self.crop_margin = 1.70

            if self.submitted and self.uploaded_files != [] and project_folder != "":
                max_files = len(self.uploaded_files)

                for i in stqdm(range(max_files),
                                leave=True,
                                desc='Media Extraction: ',
                                gui=True):

                    uploaded_file = self.uploaded_files[i] 

                    # split filename to get extension and remove the '.'
                    file_type = os.path.splitext(uploaded_file.name)[1][1:].lower()

                    if file_type in ['doc', 'dot']:
                        self.not_extract(uploaded_file)

                    elif file_type in ['ppt', 'pot']:
                        self.not_extract(uploaded_file)
                        
                    elif file_type in ['xls', 'xlt']:
                        self.not_extract(uploaded_file)

                    elif file_type in ['pdf']:
                        imgpath = self.pdf_extract(uploaded_file, self.output_folder)
                        self.__get_media(imgpath)
                        self.__get_images(imgpath)

                    elif file_type in ['zip']:
                        self.zip_extract(uploaded_file, self.output_folder)
                        
                    elif file_type in ['docx', 'docm', 'dotm', 'dotx', 'xlsx',
                                       'xlsb', 'xlsm', 'xltm', 'xltx', 'potx',
                                       'ppsm', 'ppsx', 'pptm', 'pptx', 'potm']:
                        imgpath = self.mso_extract(uploaded_file, self.output_folder)
                        self.__get_media(imgpath)
                        self.__get_images(imgpath)

                    elif file_type in ['mp4', 'avi', 'webm', 'wmv']:                
                        imgpath = self.vid_extract(uploaded_file, self.output_folder)
                        self.__get_media(imgpath)
                        self.__get_images(imgpath)
                        
                    elif file_type in file_type in ['jpeg', 'jpg', 'png', 'gif', 'bmp', 'tiff']:
                        imgpath = self.img_extract(uploaded_file, self.output_folder)
                        self.__get_media(imgpath)
                        self.__get_images(imgpath)

                    else:
                        self.not_extract(uploaded_file)

                    # logging processed data
                    try:
                        logger.info(f"IP Address: {get_remote_ip()} | Input File: {self.uploaded_files[i]} | Output Folder: {project_folder}' | Extracted Files Count: {self.extract_df['Count'].sum()} | Cropped Files Count: {len(self.image_df)}")
                    except:
                        logger.exception("Failed to log data")
                        #pass
                        
                st.success('Process completed')
                        
            else:
                st.info('Please select files to be processed.')

            # create metadata table
            if not self.extract_df.empty:
                st.subheader("Documents")
                AgGrid(self.extract_df, fit_columns_on_grid_load=True)
                st.info(f"* Total of {max_files} files processed, {self.extract_df['Count'].sum()} media files extracted")

            if not self.media_df.empty:
                st.subheader("Media")
                AgGrid(self.media_df, fit_columns_on_grid_load=True)

            if not self.image_df.empty:
                st.subheader("Images")
                AgGrid(self.image_df, fit_columns_on_grid_load=True)
                st.info(f"* Found a total of {len(self.image_df)} face(s) in media files")
                
        if project_folder != "":
            # copy processed images to top-level of output folder
            if os.path.exists(self.output_folder) and self.submitted==True:        
                cropped_list = glob.glob(self.output_folder + '/*/cropped_faces/*.png', recursive=True)
                extract_list = glob.glob(self.output_folder + '/*/extracted_images_unedited/*', recursive=True)

                extract_folder = os.path.abspath(self.output_folder + self.extract_folder_name)
                cropped_folder = os.path.abspath(self.output_folder + self.cropped_folder_name)
                            
                if os.path.exists(cropped_folder):
                    shutil.rmtree(cropped_folder)
                    shutil.rmtree(extract_folder)
                    os.mkdir(extract_folder)
                    os.mkdir(cropped_folder)
                else:
                    os.mkdir(extract_folder)
                    os.mkdir(cropped_folder)

                for file in cropped_list:
                    shutil.copy(file, cropped_folder)
                    
                for file in extract_list:
                    shutil.copy(file, extract_folder)
                    
                # remove subfolders from output directory
                if self.remove_subfolders:
                    for subfolder in self.subfolders:
                        try:
                            shutil.rmtree(subfolder)
                        except:
                            print('WARNING: Could not remove subfolder {subfolder}')
                
                # cluster identities or remove cluster folder if it exists from previous run
                if st.session_state.cluster:
                    self.faces, self.names = self.process_images(cropped_folder)
                    nclusters = self.cluster_faces(self.output_folder)
                    st.info(f"* Completed clustering identites. Found {nclusters} identities.")
                    logger.info(f"Clustered identites - found {nclusters} identities.")
                else:
                    self.cluster_path = self.output_folder + "/clustered_identities/"
                    if os.path.exists(self.cluster_path):
                        shutil.rmtree(self.cluster_path)

                # launch file server
                process_ids = find_process_ids("8506")
                if process_ids:
                    kill_processes(process_ids)
                
                if self.bucket_name is None:
                    cmd = ["python", "gallery.py", "--directory", os.path.abspath(self.output_folder), "--port", "8506"]
                else:
                    cmd = ["python", "s3gallery.py", "--directory", os.path.abspath(self.output_folder), "--port", "8506", "--bucket", self.bucket_name]
                subprocess.Popen(cmd)
                
                if self.bucket_name is not None:
                    # upload extracted and cropped images to an S3 bucket
                    self.s3_upload_directory(os.path.abspath(self.results_folder))

                    # download processed data from S3 bucket
                    #self.s3_download_directory(st.session_state.subfolder, self.s3_download + "/" + st.session_state.subfolder)

                    # remove local results folder
                    try:
                        shutil.rmtree(self.results_folder)
                    except:
                        logger.warning(f'Could not remove output folder {os.path.abspath(self.results_folder)}')

                # provide link to file server
                url = "http://localhost:8506"
                message = f"[Click here to open output folder.]({url})"

                def refresh():
                    components.html("<meta http-equiv='refresh' content='0'>", height=0)

                st.markdown("""
                <style>
                div.stButton > button:first-child {
                    background-color: #ff0000;
                    color:#ffffff;
                }
                </style>""", unsafe_allow_html=True)

                col1, col2 = st.columns([11,1])

                with col1:
                    st.markdown(message)
                    #st.markdown(f"Output Directory Location: **:blue[{os.path.abspath(self.output_folder)}]**")
        
                with col2:
                    if st.button("Clear All"):
                        refresh()
            
if __name__ == '__main__':
    # run profiler - profile results will appear at the bottome of the media extractor page
    #from streamlit_profiler import Profiler
    #with Profiler():
    #    mx = MediaExtractor()    
    #    mx.run()

    # run without profiling
    mx = MediaExtractor()    
    mx.run()

