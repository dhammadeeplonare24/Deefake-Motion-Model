import os

def generate_lip_sync(video_path, audio_path):
    result_path = os.path.join('static', 'results', 'lip_sync_result.mp4')
    temp_folder = os.path.join('Wav2Lip', 'temp')

    # ✅ Ensure temp folder exists
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    # ✅ Proper formatting for file paths (Windows safe)
    video_path = os.path.normpath(video_path)
    audio_path = os.path.normpath(audio_path)
    result_path = os.path.normpath(result_path)

    # ✅ Command to run Wav2Lip inference
    command = f"python Wav2Lip/inference.py " \
              f"--checkpoint_path Wav2Lip/checkpoints/wav2lip_gan.pth " \
              f"--face \"{video_path}\" --audio \"{audio_path}\" --outfile \"{result_path}\""

    print("Running Wav2Lip command:\n", command)
    os.system(command)

    return result_path
