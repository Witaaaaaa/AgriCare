from flask import Flask, request, render_template, redirect, url_for, send_from_directory, Response, jsonify
from werkzeug.utils import secure_filename
import torch
from pathlib import Path
import os
import glob
import pathlib
import cv2
import base64
import logging

# Ensure pathlib settings are only changed for Windows
if os.name == 'nt':
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load YOLOv5 model
model_path = Path('D:/SKRIPSI/AgriCare/yolov5/best.pt').as_posix()
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
model.eval()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Define class names
class_names = ['blast', 'blight', 'brown spot', 'healthy', 'tungro']

def gen_frames():
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow backend
    if not camera.isOpened():
        raise RuntimeError("Could not start camera.")
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Perform detection
            results = model(frame)
            print("Detection results:", results)  # Debugging print statement
            
            # Extract bounding boxes and labels
            for *xyxy, conf, cls in results.xyxy[0].numpy():
                label = f'{class_names[int(cls)]} {conf:.2f}'
                print("Detected label:", label)  # Debugging print statement
                print("Bounding box coordinates:", xyxy)  # Debugging print statement
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 255), 2)
                cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def gen_frames_with_detection():
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow backend
    if not camera.isOpened():
        raise RuntimeError("Could not start camera.")
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Perform detection
            results = model(frame)
            
            # Extract bounding boxes and labels
            for *xyxy, conf, cls in results.xyxy[0].numpy():
                label = f'{class_names[int(cls)]} {conf:.2f}'
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 255), 2)
                cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Run inference
            results = model(filepath)
            
            # Save results
            results_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'results')
            os.makedirs(results_dir, exist_ok=True)
            results.save(save_dir=results_dir)

            # Get the latest result directory
            actual_results_dir = max(glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], 'results*')), key=os.path.getmtime)

            # Find the result image
            result_image_path = next((f for f in glob.glob(os.path.join(actual_results_dir, '*')) if f.endswith(('.jpg', '.jpeg', '.png'))), None)

            if result_image_path and os.path.exists(result_image_path):
                label = results.pandas().xyxy[0]['name'].iloc[0] if not results.pandas().xyxy[0].empty else 'No objects detected'
                result_image_filename = os.path.basename(result_image_path)  # Ambil nama file asli
                result_image_url = url_for('uploaded_file', filename=result_image_filename)  # Kirim file asli ke frontend

                return render_template('index.html', filename=result_image_url, label=f'Penyakit: {label}')
            else:
                return 'Error in processing the image.'

    return render_template('index.html')

@app.route('/home')
def home():
    return redirect(url_for('index'))

@app.route('/check', methods=['GET'])
def check():
    return render_template('check.html')

@app.route('/check', methods=['POST'])
def check_post():
    logging.info("Received request to /check")
    try:
        if 'file' in request.files:  # Handle file uploads
            file = request.files['file']
            logging.info(f"File received: {file.filename}")
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.abspath(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                file.save(filepath)
                logging.info(f"File saved to: {filepath}")

                # Run inference with error handling
                try:
                    results = model(filepath)
                    logging.info(f"Inference successful for {filepath}")
                except Exception as e:
                    logging.error(f"❌ Error during model inference: {e}")
                    return jsonify({'error': f'Model inference failed: {str(e)}'}), 500

                # Save results manually without altering original colors
                try:
                    img = cv2.imread(filepath)  # Load the original image
                    for *xyxy, conf, cls in results.xyxy[0].numpy():
                        label = f'{class_names[int(cls)]} {conf:.2f}'
                        cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 255), 2)
                        cv2.putText(img, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'detected_result_0.jpg')
                    cv2.imwrite(img_path, img)
                    logging.info(f"✅ Detection result saved: {img_path}")
                except Exception as e:
                    logging.error(f"❌ Error saving detection result: {e}")
                    return jsonify({'error': f'Failed to save detection result: {str(e)}'}), 500

                # Generate correct URL for the detection result
                result_image_url = url_for('uploaded_file', filename='detected_result_0.jpg')
                logging.info(f"Serving detection result: {result_image_url}")

                return jsonify({
                    'image': result_image_url,
                    'detections': [
                        {'name': detection['name'] if detection['name'].startswith("Jenis Penyakit: ") else f"Jenis Penyakit: {detection['name']}", 'confidence': detection['confidence']}
                        for detection in results.pandas().xyxy[0].to_dict(orient='records')
                    ]
                })

        elif request.is_json:  # Handle JSON data (e.g., from captureImage)
            data = request.get_json()
            image_data = data.get('image')
            if image_data:
                # Decode the base64 image
                image_data = image_data.split(",")[1]
                image_data = base64.b64decode(image_data)
                image_path = os.path.abspath(os.path.join(app.config['UPLOAD_FOLDER'], 'captured_image.png'))
                with open(image_path, 'wb') as f:
                    f.write(image_data)
                logging.info(f"Image data saved to {image_path}")

                # Run inference
                try:
                    results = model(image_path)
                    logging.info(f"Inference successful for {image_path}")
                except Exception as e:
                    logging.error(f"❌ Error during model inference: {e}")
                    return jsonify({'error': f'Model inference failed: {str(e)}'}), 500

                # Save results manually without altering original colors
                try:
                    img = cv2.imread(image_path)  # Load the original image
                    for *xyxy, conf, cls in results.xyxy[0].numpy():
                        label = f'{class_names[int(cls)]} {conf:.2f}'
                        cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 255), 2)
                        cv2.putText(img, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'detected_result_0.jpg')
                    cv2.imwrite(img_path, img)
                    logging.info(f"✅ Detection result saved: {img_path}")
                except Exception as e:
                    logging.error(f"❌ Error saving detection result: {e}")
                    return jsonify({'error': f'Failed to save detection result: {str(e)}'}), 500

                # Generate correct URL for the detection result
                result_image_url = url_for('uploaded_file', filename='detected_result_0.jpg')
                logging.info(f"Serving detection result: {result_image_url}")

                return jsonify({
                    'image': result_image_url,
                    'detections': [
                        {'name': detection['name'] if detection['name'].startswith("Jenis Penyakit: ") else f"Jenis Penyakit: {detection['name']}", 'confidence': detection['confidence']}
                        for detection in results.pandas().xyxy[0].to_dict(orient='records')
                    ]
                })

        logging.error("Invalid request: No file or JSON data provided")
        return jsonify({'error': 'Invalid request'}), 400
    except Exception as e:
        logging.exception("Exception occurred while processing the request")
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    # Correct the directory path to serve files from the correct location
    uploads_dir = os.path.abspath(app.config['UPLOAD_FOLDER'])
    logging.info(f"Serving file: {os.path.join(uploads_dir, filename)}")
    return send_from_directory(uploads_dir, filename)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/detection', methods=['POST'])
def detection():
    return render_template('detection.html')

@app.route('/camera', methods=['POST'])
def camera():
    return render_template('camera.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_with_detection')
def video_feed_with_detection():
    return Response(gen_frames_with_detection(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture():
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not camera.isOpened():
        return jsonify({'error': 'Could not start camera.'}), 500
    success, frame = camera.read()
    if not success:
        return {'error': 'Failed to capture image.'}, 500
    camera.release()

    # Perform detection
    results = model(frame)
    print("Detection results:", results)  # Debugging print statement

    # Extract bounding boxes and labels
    for *xyxy, conf, cls in results.xyxy[0].numpy():
        label = f'{class_names[int(cls)]} {conf:.2f}'
        print("Detected label:", label)  # Debugging print statement
        print("Bounding box coordinates:", xyxy)  # Debugging print statement
        cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 255), 2)
        cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    ret, buffer = cv2.imencode('.jpg', frame)
    frame = buffer.tobytes()
    encoded_image = base64.b64encode(frame).decode('utf-8')
    return {'image': f'data:image/jpeg;base64,{encoded_image}'}

@app.route('/live_detection', methods=['POST'])
def live_detection():
    data = request.get_json()
    image_data = data.get('image')
    if image_data:
        # Decode the image data
        image_data = image_data.split(",")[1]
        image_data = base64.b64decode(image_data)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'live_camera_image.png')
        with open(image_path, 'wb') as f:
            f.write(image_data)

        # Run inference
        results = model(image_path)
        
        # Extract detections
        detections = results.pandas().xyxy[0].to_dict(orient='records')
        return {'detections': detections}
    return {'error': 'Invalid request'}, 400

@app.route('/save_video_frame', methods=['POST'])
def save_video_frame():
    try:
        data = request.get_json()
        image_data = data.get('image')
        if image_data:
            # Decode the base64 image
            image_data = image_data.split(",")[1]
            image_data = base64.b64decode(image_data)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'video_frame.png')
            with open(image_path, 'wb') as f:
                f.write(image_data)
            return jsonify({'success': True, 'message': 'Frame saved successfully!', 'path': image_path})
        return jsonify({'success': False, 'error': 'No image data provided.'}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(debug=True, host='0.0.0.0', port=5000)