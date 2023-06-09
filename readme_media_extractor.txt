Media Extractor

I. Summary:

The Media Extractor application is a tool that allows users to extract media content (images, videos, documents) from various file types and perform operations such as cropping faces, aligning faces, clustering identities, and organizing the extracted media files. The application is built using the Streamlit framework and incorporates functionality from modules such as OpenCV, NumPy, Pandas, and scikit-learn.

The application follows a modular approach with different functions dedicated to specific tasks:

1.	The `zip_extract` function extracts media files from a ZIP archive, categorizing them based on their file types. It supports various formats such as PDF, Microsoft Office documents (Word, PowerPoint, Excel), images, and videos.
2.	The `pdf_extract` function processes PDF files, extracting images and media content. It utilizes the PyMuPDF library to read PDF files, extract images from each page, and store them in the specified location. Metadata such as file type, size, and image count are recorded.
3.	The `vid_extract` function extracts frames from video files, saving them as individual images. It utilizes OpenCV to read the video file, capture frames, and store them in the specified location. Metadata such as file type, size, and frame count are recorded.
4.	The `img_extract` function saves images to the specified location. It extracts metadata such as file type, size, and image dimensions.
5.	The `crop_face` function crops facial images based on the provided bounding box coordinates and a margin parameter. It adjusts the size of the bounding box to crop only the face region or include additional head area.
6.	The `rotate_upside_down_face` function determines if a face is upside down based on facial landmarks and rotates it by 180 degrees if necessary.
7.	The `face_align` function aligns a face by rotating it to ensure that the eyes are on a horizontal plane.
8.	The `__get_media` function retrieves metadata of extracted media files, such as image size, format, mode, and hash values.
9.	The `__get_images` function performs face detection on the extracted images, crops the faces, aligns them, and saves them in a separate folder. It utilizes a face detection model and facial landmarks to locate and crop the faces. Metadata such as image dimensions, bounding box coordinates, facial landmarks, and image hash values are recorded.
10.	The `process_images` function processes a directory of images, detecting faces and preparing them for clustering by resizing them to a standardized size.
11.	The `cluster_faces` function clusters the cropped face images using the DBSCAN clustering algorithm. It generates clusters based on the similarity of facial feature vectors obtained from a pre-trained face recognition model. The clustered identities are saved in separate subfolders within the cluster directory.
12.	The `run` function is the main entry point of the application. It sets up the Streamlit page, handles user inputs, processes media files, displays metadata tables, performs additional operations based on user selections (such as clustering and removing subfolders), copies processed images to the output folder, launches a file server to serve the output folder contents, and provides a link to access the output folder.

The application works by allowing users to upload media files and select options such as the bounding box size for cropping faces, removing subfolders, and clustering identities. The uploaded files are processed based on their file types, and the corresponding functions extract and process the media content. Metadata tables are displayed to provide an overview of the extracted media and the detected faces. After processing, the application copies the processed images to the output folder and performs additional operations such as clustering identities if selected. The application then launches a file server to serve the output folder contents and provides a link for the user to access the output folder and view the results.

The Media Extractor application offers a user-friendly interface for efficiently extracting and organizing media content. Users can upload various types of media files, including images, videos, and documents. The application utilizes different modules and functions to process and extract media content based on the file type.

The application supports the extraction of media from ZIP archives, PDF files, and various Microsoft Office documents. For PDF files, images from each page are extracted and saved separately. Videos are processed by extracting frames and saving them as individual images. Images and other supported file types are saved to the specified output folder.

The application includes features for face cropping and alignment. Faces can be cropped based on bounding box coordinates, with options to adjust the size of the bounding box for tighter or wider crops. Additionally, faces can be aligned to ensure that the eyes are on a horizontal plane.

The Media Extractor application also provides the capability to cluster identities within the extracted face images. The DBSCAN clustering algorithm is used to group similar faces together based on facial feature vectors. Clustered identities are organized into separate subfolders within the output directory.

Metadata tables are displayed to provide detailed information about the processed media and detected faces. The tables include data such as file names, types, sizes, dimensions, formats, modes, and hash values. Users can easily review and analyze the extracted media and identify clusters of similar faces.

The application offers customization options such as adjusting the bounding box size, removing subfolders from the output directory, and choosing whether to perform clustering. Users can also choose to serve the output folder contents through a file server, allowing easy access to the organized media.

The Media Extractor application is designed to streamline the process of media extraction, face cropping, alignment, and identity clustering. It can be used in various domains where efficient media organization and analysis are required, such as image and video processing, facial recognition, identity verification, and data mining.

II. Installation and Running

Installation is straight forward and simple. Installation and testing performed using WSL Ubuntu Linux 22.0.4. The Media Extractor application will run on Windows 10/11 and other Linux distributions (e.g. CentOS7). There may need to be slight modifications to the installation procedure when installing the python packages on systems that are not Ubuntu. Regardless, the overall installation process should be easy to perform.
Installation procedure for WSL Ubuntu 22.0.4:
1.	Install Anaconda from Anaconda | The World’s Most Popular Data Science Platform
2.	Run the yaml file to create a python environment and install packages:
a.	Run `conda env create –file media_extractor.yml`
3.	Enter the new environment:
a.	`conda activate media_extractor`
4.	Launch application:
a.	`streamlit run media_extractor.py`


III. AWS S3 Bucket storage

1. Create a credentials file in .aws directory containing the following
   information:

	[default]
	aws_access_key_id=[your key]
	aws_secret_access_key=[your secret key]
	region=us-east-1

2. Sign up for an AWS account and obtain required keys.

3. Save file as ~/.aws/credentials (no file extension). lThis will allow media extractor application to uploiad and download processed file results.