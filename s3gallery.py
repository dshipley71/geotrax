import os
import zipfile
import argparse
from datetime import datetime
from flask import Flask, render_template, send_from_directory, send_file, after_this_request, make_response
import boto3

app = Flask(__name__, static_folder='assets')

# Retrieve S3 bucket contents
def get_s3_bucket_contents(bucket_name):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)
    contents = [obj.key for obj in bucket.objects.all()]
    print('===> contents:', contents)
    return contents
    
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/extracted_images_unedited/')
def extracted_images_unedited():
    bucket_name = args.bucket
    contents = get_s3_bucket_contents(bucket_name)
    filtered_contents = list(filter(lambda x: 'extracted_images_unedited' in x, contents))
    print('===> filtered: ', filtered_contents)
    parent_folder = args.directory.split('/')[-1]
    filtered_contents = list(filter(lambda x: parent_folder in x, filtered_contents))
    print('===> filtered: ', filtered_contents)

    return render_template('thumbnails.html', image_names=filtered_contents)
    
@app.route('/cropped_faces/')
def cropped_faces():
    bucket_name = args.bucket
    contents = get_s3_bucket_contents(bucket_name)
    filtered_contents = list(filter(lambda x: 'cropped_faces' in x, contents))
    print('===> filtered: ', filtered_contents)
    parent_folder = args.directory.split('/')[-1]
    filtered_contents = list(filter(lambda x: parent_folder in x, filtered_contents))
    print('===> filtered: ', filtered_contents)
    return render_template('thumbnails.html', image_names=filtered_contents)
    
@app.route('/clustered_identities/')
def clustered_identities():
    bucket_name = args.bucket
    contents = get_s3_bucket_contents(bucket_name)
    filtered_contents = list(filter(lambda x: 'clustered_identities' in x, contents))
    print('===> filtered: ', filtered_contents)
    parent_folder = args.directory.split('/')[-1]
    filtered_contents = list(filter(lambda x: parent_folder in x, filtered_contents))
    print('===> filtered: ', filtered_contents)
    if filtered_contents == []:
        return "<b>WARNING: Clustered identities not found. Make sure to select the clustered identities checkbox on the Media Extractor page.</b>"

    subfolders = sorted(set([os.path.dirname(file) for file in filtered_contents]))
    print('===> subfolders: ', subfolders)

    subfolder_contents = {}
    for subfolder in subfolders:
        subfolder_files = [file for file in filtered_contents if file.startswith(subfolder)]
        subfolder_contents[subfolder] = subfolder_files

    return render_template('s3_clustered_thumbnails.html', subfolder_contents=subfolder_contents)

@app.route('/<path:subdirectory>/<path:image_name>')
def send_image(subdirectory, image_name):
    bucket_name = args.bucket
    key = f'{subdirectory}/{image_name}'
    print(f'===> KEY: {key}')
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket_name, Key=key)
    image_data = response['Body'].read()
    response = make_response(image_data)
    response.headers.set('Content-Type', 'image/jpeg')
    response.headers.set('Content-Disposition', 'inline', filename=image_name)
    return response

@app.route('/download/')
def download():
    bucket_name = args.bucket
    s3 = boto3.client('s3')
    objects = s3.list_objects_v2(Bucket=bucket_name)['Contents']

    parent_folder = args.directory.split('/')[-1]
    
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    zip_filename = f'media_extractor_images_{timestamp}.zip'
    
    with zipfile.ZipFile(zip_filename, 'w') as zip_file:
        for obj in objects:
            key = obj['Key']
            if parent_folder not in key:
                continue
            print(f'===> obj : {key}')
            response = s3.get_object(Bucket=bucket_name, Key=key)
            image_data = response['Body'].read()
            zip_file.writestr(key, image_data)

    @after_this_request
    def remove_zip_file(response):
        os.remove(zip_filename)
        return response
    
    return send_file(zip_filename, as_attachment=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', type=int, default=8506, help='Port number')
    parser.add_argument('-d', '--directory', type=str, default='.', help='Parent directory location')
    parser.add_argument('-b', '--bucket', type=str, default='.', help='S3 bucket name')
    args = parser.parse_args()
    print(args)

    app.run(host="0.0.0.0", debug=True, port=args.port)
