from skimage.metrics import structural_similarity as ssim
import math
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import base64
import cv2
from PIL import Image
import tempfile
import math
import subprocess
import os
import time
import ffmpeg_downloader as ffdl

st.set_page_config(layout="wide")

image_prompt = "You are an expert video narrator. Describe 1- surrounding and 2. what is happening. DO NOT EVER directly refer to it as 'image', 'photos', 'frames', 'seqence' or other synonyms. Sentences should be concise. Focus on continuity and cohesiveness of narration. Describe like you are observing with your own eyes and you are there. Maximum sentences: {max_sentence_length}"""

narration_prompt = """You are an expert video narrator. Generate a coherent narration from given descriptions from chunks of the same video. Sentences should be short. Use simple english. If an entity is repeated mutliple times, assume its the same entity, focus on continuity. Do not EVER repeat same information. Maximum words: {max_word_length}"""


def get_image_description(client, base64_image, image_prompt, max_sentence_length=2):
    """
    Generates a description of an image using OpenAI's API.

    Args:
    client (OpenAI): OpenAI client.
    base64_image (str): Base64 blob of the image.
    image_prompt (str): Prompt for the image description.

    Returns:
    str: Generated description.
    """

    if not base64_image:
        return None
    image_prompt = image_prompt.format(
        max_sentence_length=max_sentence_length)
    try:
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": image_prompt}
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300
        )
        description = response.choices[0].message.content
        return description

    except Exception as e:
        print("Error encountered:\n", e)
        return None


def test_api_key(client):
    try:
        response = client.models.list()
        return True
    except Exception as e:
        return False


def generate_narration(client, descriptions, narration_prompt, max_word_length=100):
    """
    Generates a narration from a list of descriptions.

    Args:
    client (OpenAI): OpenAI client.
    descriptions (list): List of descriptions.
    narration_prompt (str): Prompt for the narration.

    Returns:
    str: Generated narration.
    """
    # Generate the narration from the descriptions
    description = ""
    for index, text in enumerate(descriptions):
        description += f"{index + 1}: {text}\n"

    try:
        narration_prompt = narration_prompt.format(
            max_word_length=max_word_length)
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": narration_prompt}
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": description},
                    ]
                }
            ],
            max_tokens=300
        )
        description = response.choices[0].message.content
        return description

    except Exception as e:
        return None


def get_audio(client, text):
    """
    Converts text to speech using OpenAI's API.

    Args:
    client (OpenAI): OpenAI client.
    text (str): Text to be converted to speech.

    Returns:
    str: Path to the temporary audio file.
    """

    # Get the audio response from OpenAI
    audio_response = client.audio.speech.create(
        response_format="opus",
        speed=1,
        model="tts-1",
        voice="nova",
        input=text,
    )

    # Convert the audio response to a base64 blob
    # audio_blob = base64.b64encode(audio_response.content).decode('utf-8')

    # save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.opus') as temp_file:
        temp_file.write(audio_response.content)
        return temp_file.name


def trim_sentence(sentence, threshold):
    """
    Trims the sentence if the number of words exceeds the given threshold.

    Args:
    sentence (str): Sentence to be trimmed.
    threshold (int): Maximum number of words in the sentence.

    Returns:
    str: Trimmed sentence.
    """
    if threshold < 20:
        threshold = 20

    # Split the sentence into words
    words = sentence.split()

    # Check if the number of words exceeds the threshold
    if len(words) > threshold:
        # Find the last sentence within the threshold
        trimmed_sentence = ' '.join(words[:threshold])
        last_period = trimmed_sentence.rfind('.')

        # If a period is found, trim to the last complete sentence
        if last_period != -1:
            return trimmed_sentence[:last_period + 1]
        else:
            # If no period is found, return the words up to the threshold
            return trimmed_sentence

    # Return the original sentence if it doesn't exceed the threshold
    return sentence


def image_to_base64(image):
    """
    Converts an image to a base64 blob.

    Args:
    image (PIL.Image): Image to be converted to a base64 blob.

    Returns:
    str: Base64 blob of the image.
    """
    # Convert the image to a base64 blob
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpeg') as temp_file:
        image.save(temp_file.name, "JPEG")
        with open(temp_file.name, "rb") as image_file:
            base64_blob = base64.b64encode(image_file.read()).decode('utf-8')
    return base64_blob


def bytes_to_temp_file(bytes, suffix):
    """
    Converts a stream of bytes to a temporary file.

    Args:
    bytes (BytesIO): Stream of bytes.
    suffix (str): Suffix of the temporary file.

    Returns:
    str: Path to the temporary file.
    """
    # Save the bytes to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(bytes)
        return temp_file.name


def add_audio_to_video(video_path, audio_path):
    """
    Adds audio to a video clip and modifies the audio speed to match the video length using ffmpeg.
    Writes the result to a file.

    Args:
        video_path (str): Path to the video file.
        audio_path (str): Path to the audio file.

    Returns:
        str: Path to the output video file with adjusted audio.
    """
    # Create a temporary file for the output video
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')

    # Run ffmpeg to get the duration of the video in seconds
    video_duration = subprocess.check_output(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1:nokey=1', video_path]).strip()

    # Run ffmpeg to get the duration of the audio in seconds
    audio_duration = subprocess.check_output(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1:nokey=1', audio_path]).strip()

    # Calculate the tempo adjustment factor
    duration_ratio = float(audio_duration) / float(video_duration)

    if duration_ratio < 0.8:
        duration_ratio = 0.8
    elif duration_ratio > 1.3:
        duration_ratio = 1.3

    # Use ffmpeg to adjust audio speed and merge with video
    subprocess.call([
        'ffmpeg', '-i', video_path, '-i', audio_path,
        '-filter_complex', f'[1:a]atempo={duration_ratio}[adjusted]',
        '-map', '0:v', '-map', '[adjusted]', '-c:v', 'copy',
        '-c:a', 'aac', '-strict', '-2', '-hide_banner', '-loglevel', 'error', temp_video.name, '-y'
    ])

    # Return the path of the temporary video file
    return temp_video.name


def extract_frames_and_create_grids(video_stream, rows=3, columns=3, sample_interval=1.0, border_width=10):
    """
    Extracts frames from a video stream and creates image grids with a specified number of rows and columns.
    Each image in the grid will have a sequence number on the top left corner with a black background and white font.
    Adds a border around each image in the grid.

    Args:
    video_stream (BytesIO): Stream of bytes from the uploaded video file.
    rows (int): Number of rows in each grid.
    columns (int): Number of columns in each grid.
    sample_interval (float): Interval in seconds at which frames are sampled.
    border_width (int): Width of the border around each image.

    Returns:
    (list, float): List of image grids and video duration in seconds.
    """

    temp_video_path = bytes_to_temp_file(video_stream, ".mp4")

    # Create a VideoCapture object
    video = cv2.VideoCapture(temp_video_path)

    # Calculate frame parameters
    fps = video.get(cv2.CAP_PROP_FPS)

    # Extract frames
    frames = []
    sequence_number = 1
    image_width = 320
    image_height = 240
    while True:
        success, frame = video.read()
        if not success:
            break
        current_time = video.get(cv2.CAP_PROP_POS_MSEC) / \
            1000  # current time in seconds
        if current_time % sample_interval < 1 / fps:
            frame = cv2.resize(frame, (image_width, image_height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame)

            # Draw sequence number with background
            draw = ImageDraw.Draw(pil_frame)
            font_size = int(pil_frame.height * 0.2)
            font = ImageFont.truetype("arial.ttf", font_size)
            text = str(sequence_number)
            text_bbox = draw.textbbox((0, 0), text, font=font)
            draw.rectangle(text_bbox, fill="black")
            draw.text((0, 0), text, font=font, fill="white")

            frames.append(pil_frame)
            sequence_number += 1

    # Release the video object
    video.release()

    # Create grids
    grids = []
    grid_size = rows * columns
    for i in range(0, len(frames), grid_size):
        grid_frames = frames[i:i + grid_size]
        grid_width = columns * image_width + border_width * (columns + 1)
        grid_height = rows * image_height + border_width * (rows + 1)
        grid = Image.new('RGB', (grid_width, grid_height), "black")

        for j, frame in enumerate(grid_frames):
            x = (j % columns) * image_width + \
                border_width * ((j % columns) + 1)
            y = (j // columns) * image_height + \
                border_width * ((j // columns) + 1)
            grid.paste(frame, (x, y))

        grids.append(grid)

    # video duration with ffmpeg
    video_duration = subprocess.check_output(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1:nokey=1', temp_video_path]).strip()

    return grids, float(video_duration)


def extract_frames_and_create_grids1(video_stream, rows=3, columns=3, border_width=10, similarity_threshold=0.3):
    """
    Extracts frames from a video stream and creates image grids with a specified number of rows and columns.
    Only samples frames that are significantly different from the previous frame.
    Each image in the grid will have a sequence number on the top left corner with a black background and white font.
    Adds a border around each image in the grid.

    Args:
    video_stream (BytesIO): Stream of bytes from the uploaded video file.
    rows (int): Number of rows in each grid.
    columns (int): Number of columns in each grid.
    border_width (int): Width of the border around each image.
    similarity_threshold (float): Threshold for frame similarity (lower values mean more difference is required).

    Returns:
    (list, float): List of image grids and video duration in seconds.
    """
    temp_video_path = bytes_to_temp_file(video_stream, ".mp4")

    # Create a VideoCapture object
    video = cv2.VideoCapture(temp_video_path)

    # Calculate video duration with ffmpeg
    video_duration = subprocess.check_output(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1:nokey=1', temp_video_path]).strip()
    video_duration = float(video_duration)

    # calculate similarity threshold based on video duration. lower values for longer videos.
    similarity_threshold = max(similarity_threshold -
                               (0.0001 * video_duration), 0.1)

    print("similarity_threshold", similarity_threshold)

    # Extract frames
    frames = []
    previous_frame = None
    sequence_number = 1
    image_width = 320
    image_height = 240
    while True:
        success, frame = video.read()
        if not success:
            break

        frame_resized = cv2.resize(frame, (image_width, image_height))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        # Check if the frame is different enough from the previous frame
        if previous_frame is not None:
            frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            prev_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
            s = ssim(frame_gray, prev_frame_gray)

            if s > similarity_threshold:
                continue  # Skip this frame, not different enough

        previous_frame = frame_rgb
        pil_frame = Image.fromarray(frame_rgb)

        # Draw sequence number with background
        draw = ImageDraw.Draw(pil_frame)
        font_size = int(pil_frame.height * 0.1)
        font = ImageFont.truetype("arial.ttf", font_size)
        text = str(sequence_number)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        draw.rectangle(text_bbox, fill="black")
        draw.text((0, 0), text, font=font, fill="white")

        frames.append(pil_frame)
        sequence_number += 1

    # Release the video object
    video.release()

    # Create grids
    grids = []
    grid_size = rows * columns
    for i in range(0, len(frames), grid_size):
        grid_frames = frames[i:i + grid_size]
        grid_width = columns * image_width + border_width * (columns + 1)
        grid_height = rows * image_height + border_width * (rows + 1)
        grid = Image.new('RGB', (grid_width, grid_height), "black")

        for j, frame in enumerate(grid_frames):
            x = (j % columns) * image_width + \
                border_width * ((j % columns) + 1)
            y = (j // columns) * image_height + \
                border_width * ((j // columns) + 1)
            grid.paste(frame, (x, y))

        grids.append(grid)

    # # Clean up temporary video file
    # os.remove(temp_video_path)

    return grids, video_duration, temp_video_path


class VideoProcessor(VideoProcessorBase):

    def __init__(self) -> None:
        self.frames = []
        print("Recording Started")

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frames.append(img)
        return frame

    def on_ended(self):
        filename = "webcam_video.avi"
        # Remove the existing video file if it exists
        if os.path.exists(filename):
            os.remove(filename)

        # Create a file to store the video
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(filename, fourcc, 30.0,
                              self.frames[0].shape[:2][::-1])
        for frame in self.frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        out.release()

        print("Recording Ended")
        print("Video saved to", filename)


def get_webcam_video():
    # Initialize video recorder
    webrtc_ctx = webrtc_streamer(
        key="webcam-video",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )


def video_processor(client, filename, video_stream):
    process_start_time = time.time()
    with st.status(label='Processing Video...', expanded=False) as status:
        try:
            # 2. Frame Extraction and Image Grid Creation
            status.update(
                label="Sampling Video Frames...", expanded=False)
            grids, video_duration, temp_video_path = extract_frames_and_create_grids1(
                video_stream=video_stream,
                rows=4,
                columns=4,
            )

            for grid in grids:
                st.image(
                    f"data:image/jpeg;base64,{image_to_base64(grid)}")

            # 3. Narration Generation
            status.update(
                label="Generating Narration...", expanded=True)
            description_start_time = time.time()
            max_sentence_length = math.ceil(video_duration / 5)
            descriptions = [get_image_description(
                client, image_to_base64(grid), image_prompt, max_sentence_length=max_sentence_length) for grid in grids]
            st.write(
                f"Time taken for description generation: {(time.time()-description_start_time):.2f} seconds")
            st.write("Descriptions:", descriptions)

            # max words based on video duration assuming 4 words per second
            narration_start_time = time.time()
            max_word_length = math.ceil(video_duration * 4)
            narration = generate_narration(
                client, descriptions, narration_prompt, max_word_length) if len(descriptions) > 1 else descriptions[0]
            st.markdown(
                f"__Time taken for narration generation:__ {(time.time()-narration_start_time):.2f} seconds")

            # if the word length in narration are more than max_word_length, trim at the last full stop
            narration = trim_sentence(narration, max_word_length)

            st.markdown(f"__Narration:__ {narration}")

            # 4. Text-to-Speech Conversion
            status.update(label="Generating Audio...")
            audio_path = get_audio(client, narration)

            status.update(
                label="Finalizing Narrated Video...")
            # 5. Audio and Video Integration
            final_video = add_audio_to_video(
                temp_video_path, audio_path)

            if final_video:
                status.update(label="Video processed successfully",
                              state="complete", expanded=False)
            else:
                status.update(label="Error Processing Video",
                              state="error")
        except Exception as e:
            status.update(label="Error Processing Video",
                          state="error")
            raise e
    if final_video:
        # 6. Display the final video
        st.video(final_video)
        st.download_button(
            label="Download Video", data=final_video, file_name=f"{os.path.splitext(filename)[0]}_narrated.mp4")
    st.write(
        f"Processing time: {(time.time() - process_start_time):.2f} seconds")


def change_event(value):
    st.session_state.event = value


def main():
    # centered title
    st.markdown(
        "<h1 style='text-align: center;'>Video Narration Assistant</h1>", unsafe_allow_html=True)

    # install ffmpeg
    try:
        subprocess.run("ffmpeg -version",
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print("FFmpeg not found. Installing FFmpeg:", e)
        try:
            subprocess.run("ffdl install -y",
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except:
            print("FFmpeg installation failed:", e)
            st.info(
                "FFmpeg installation failed. Please install FFmpeg manually.")
            return

    api_key = st.text_input("Enter your OpenAI API Key",
                            type="password", key="api_key")
    if api_key:
        # add two columns
        col1, col2 = st.columns(2, gap="large")

        # initialize states
        video_stream = None
        webcam_video_path = None
        filename = None
        video_file = None

        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        if not test_api_key(client):
            st.error("Invalid API Key")
            return

        st.session_state.event = 0
        st.session_state.video_frames = []
        st.session_state.video_recorded = False

        with col1:
            # Video Upload
            st.subheader("Upload a Video")
            video_file = st.file_uploader("Upload a Video", type=[
                "mp4", "avi", "mov", "mkv"], label_visibility="hidden", on_change=lambda: change_event(0))

            st.divider()
            # Webcam recording functionality
            # st.subheader("Record from Webcam")
            # webcam_video_path = get_webcam_video()
            # if webcam_video_path:
            #     st.video(webcam_video_path)

        with col2:
            if video_file and st.session_state.event == 0:
                filename = video_file.name
                video_stream = video_file.getvalue()

            elif webcam_video_path and st.session_state.event == 1:
                filename = "webcam_video"
                video_stream = open(webcam_video_path, "rb").read()

            if video_stream:
                video_processor(client, filename, video_stream)
            else:
                st.markdown("<br/>"*3, unsafe_allow_html=True)
                st.info("Upload a video to get started.")
    else:
        st.info("Enter your OpenAI API Key to get started.")


if __name__ == "__main__":
    main()
