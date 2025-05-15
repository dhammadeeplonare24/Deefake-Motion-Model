from flask import Flask, render_template, request, send_file, abort
import os
from werkzeug.utils import secure_filename
from models.face_swap import swap_faces
from models.lip_sync import generate_lip_sync
from models.motion_transfer import animate_motion

app = Flask(__name__)

# Folder paths
UPLOAD_FOLDER = os.path.join('static', 'uploads')
RESULT_FOLDER = os.path.join('static', 'results')

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        task = request.form.get('task')
        source = request.files.get('source')
        target = request.files.get('target')
        audio = request.files.get('audio')

        if not source or not target:
            return "Source and target files are required!", 400

        # Save source and target files
        source_filename = secure_filename(source.filename)
        target_filename = secure_filename(target.filename)

        source_path = os.path.join(UPLOAD_FOLDER, source_filename)
        target_path = os.path.join(UPLOAD_FOLDER, target_filename)

        source.save(source_path)
        target.save(target_path)

        # Process according to task
        result_path = None

        if task == 'face_swap':
            result_path = swap_faces(source_path, target_path)

        elif task == 'lip_sync':
            if not audio:
                return "Audio file is required for lip syncing!", 400
            audio_filename = secure_filename(audio.filename)
            audio_path = os.path.join(UPLOAD_FOLDER, audio_filename)
            audio.save(audio_path)
            result_path = generate_lip_sync(target_path, audio_path)

        elif task == 'motion_transfer':
            result_path = animate_motion(source_path, target_path)

        else:
            return "Invalid task selected!", 400

        # Check if the result file was created
        if not result_path or not os.path.exists(result_path):
            return "Error: Result file was not generated!", 500

        return send_file(result_path, as_attachment=True)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
