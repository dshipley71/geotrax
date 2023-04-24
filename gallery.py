import os
import zipfile
import argparse
from datetime import datetime
from flask import Flask, render_template, send_from_directory, send_file, after_this_request, make_response

app = Flask(__name__, static_folder='assets')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/extracted_images_unedited/')
def extracted_images_unedited():
    folder = os.path.join(args.directory, 'extracted_images_unedited')
    image_names = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    return render_template('thumbnails.html', subdirectory_name='extracted_images_unedited', image_names=image_names)

@app.route('/cropped_faces/')
def cropped_faces():
    folder = os.path.join(args.directory, 'cropped_faces')
    image_names = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    return render_template('thumbnails.html', subdirectory_name='cropped_faces', image_names=image_names)

@app.route('/clustered_identities/')
def clustered_identities():
    folder = os.path.join(args.directory, 'clustered_identities')
    if not os.path.exists(folder):
        return "<b>WARNING: Clustered identities not found. Make sure to select the clustered identities checkbox on the Media Extractor page.</b>"
    return render_template('clustered_thumbnails.html', subdirectory_name='clustered_identities', folder=folder, os=os)

@app.route('/<path:subdirectory>/<path:image_name>')
def send_image(subdirectory, image_name):
    directory = os.path.join(args.directory, subdirectory)
    return send_from_directory(directory, image_name)

# @app.route('/download/')
# def download():
    # # Get the absolute path of the output directory
    # output_dir = os.path.abspath(args.directory)

    # # create a zip file containing all images in the directory
    # timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    # zip_filename = f'media_extractor_images_{timestamp}.zip'

    # # Iterate over all files in the output directory and add them to the zip file
    # with zipfile.ZipFile(zip_filename, 'w') as zip_file:
        # for root, dirs, files in os.walk(output_dir):
            # for file in files:
                # file_path = os.path.join(root, file)
                # zip_file.write(file_path, os.path.relpath(file_path, output_dir))

    # # send the zip file to the user for download
    # return send_file(zip_filename, as_attachment=True)

@app.route('/download/')
def download():
    # create a zip file containing all images in the directory
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    zip_filename = f'media_extractor_images_{timestamp}.zip'

    def generate():
        # Get the absolute path of the output directory
        output_dir = os.path.abspath(args.directory)

        # create a zip file containing all images in the directory
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        zip_filename = f'media_extractor_images_{timestamp}.zip'

        # Iterate over all files in the output directory and add them to the zip file
        with zipfile.ZipFile(zip_filename, 'w') as zip_file:
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    zip_file.write(file_path, os.path.relpath(file_path, output_dir))

        # stream the zip file to the user for download
        with open(zip_filename, 'rb') as f:
            while True:
                data = f.read(1024*1024)
                if not data:
                    break
                yield data

        # delete the zip file after sending
        os.remove(zip_filename)

    # set the response headers for streaming
    response = make_response(generate())
    response.headers.set('Content-Disposition', 'attachment', filename=zip_filename)
    response.headers.set('Content-Type', 'application/zip')
    response.headers.set('Cache-Control', 'no-cache, no-store, must-revalidate')
    response.headers.set('Pragma', 'no-cache')
    response.headers.set('Expires', '0')
    return response
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', type=int, default=8506, help='Port number')
    parser.add_argument('-d', '--directory', type=str, default='.', help='Parent directory location')
    args = parser.parse_args()

    app.run(host="0.0.0.0", debug=True, port=args.port)
