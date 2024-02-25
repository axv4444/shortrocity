from pydub import AudioSegment
import numpy as np
import math
import cv2
import os

import text

def get_audio_duration(audio_file):
    return len(AudioSegment.from_file(audio_file))

# Helper function to create a blurred background
def create_blurred_background(image, target_width, target_height):
    # Calculate the aspect ratio of the image
    image_height, image_width = image.shape[:2]
    image_aspect = image_width / image_height
    target_aspect = target_width / target_height
    
    # Determine the scaling factors for resizing the image and creating the blurred background
    if image_aspect > target_aspect:
        # For wider images: Scale based on width
        scale_width = target_width
        scale_height = int(scale_width / image_aspect)
        resize_scale = target_width / image_width
    else:
        # For taller images: Scale based on height
        scale_height = target_height
        scale_width = int(scale_height * image_aspect)
        resize_scale = target_height / image_height
    
    # Resize the image to fit within the target frame
    resized_image = cv2.resize(image, (scale_width, scale_height), interpolation=cv2.INTER_LINEAR)
    
    # Create a blurred background by scaling the original image and applying a Gaussian blur
    blur_size = (int(image_width * resize_scale * 1.5), int(image_height * resize_scale * 1.5))
    blurred_image = cv2.resize(image, blur_size, interpolation=cv2.INTER_LINEAR)
    blurred_image = cv2.GaussianBlur(blurred_image, (0, 0), sigmaX=20, sigmaY=20)
    
    # Create a canvas for the target frame
    canvas = np.zeros((target_height, target_width, 3), dtype='uint8')
    
    # Calculate the position for the blurred image to be centered
    blur_x_offset = (target_width - blurred_image.shape[1]) // 2
    blur_y_offset = (target_height - blurred_image.shape[0]) // 2
    
    # Place the blurred image on the canvas
    canvas[max(blur_y_offset, 0):max(blur_y_offset, 0)+blurred_image.shape[0], max(blur_x_offset, 0):max(blur_x_offset, 0)+blurred_image.shape[1]] = blurred_image[max(-blur_y_offset, 0):max(-blur_y_offset, 0)+target_height, max(-blur_x_offset, 0):max(-blur_x_offset, 0)+target_width]
    
    # Overlay the resized image on top of the blurred background
    image_x_offset = (target_width - scale_width) // 2
    image_y_offset = (target_height - scale_height) // 2
    canvas[image_y_offset:image_y_offset+scale_height, image_x_offset:image_x_offset+scale_width] = resized_image
    
    return canvas

def resize_image(image, target_width, target_height):
    # Calculate the aspect ratio of the image and the target frame
    image_height, image_width = image.shape[:2]
    image_aspect = image_width / image_height
    target_aspect = target_width / target_height

    # Resize the image to maintain aspect ratio
    if image_aspect > target_aspect:
        # Image is wider than target, fit to width
        new_width = target_width
        new_height = int(target_width / image_aspect)
    else:
        # Image is taller than target, fit to height
        new_height = target_height
        new_width = int(target_height * image_aspect)

    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Create a blurred version of the image to fill the background
    background_width = target_width
    background_height = target_height
    large_blur_size = (background_width, background_height)
    blurred_background = cv2.resize(image, large_blur_size)
    blurred_background = cv2.GaussianBlur(blurred_background, (0, 0), 10)

    # Create a blank canvas and place the blurred background
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)# Center the blurred background on the canvas
    canvas[0:target_height, 0:target_width] = blurred_background
    
    overlay_x = (target_width - new_width) // 2
    overlay_y = (target_height - new_height) // 2

    # Place the resized image on top of the blurred background
    canvas[overlay_y:overlay_y+new_height, overlay_x:overlay_x+new_width] = resized_image

    return canvas

def create(narrations, output_dir, output_filename):
    # Define the dimensions and frame rate of the video
    width, height = 1080, 1920  # Change as needed for your vertical video
    frame_rate = 30  # Adjust as needed

    fade_time = 1000

    # Create a VideoWriter object to save the video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec as needed
    temp_video = os.path.join(output_dir, "temp_video.mp4")  # Output video file name
    out = cv2.VideoWriter(temp_video, fourcc, frame_rate, (width, height))

    # List of image file paths to use in the video
    image_paths = os.listdir(os.path.join(output_dir, "images"))  # Replace with your image paths
    image_count = len(image_paths)

    for i in range(image_count):
        if i < image_count - 1:
            image1_path = os.path.join(output_dir, "images", f"image_{i+1}.jpg")
            image2_path = os.path.join(output_dir, "images", f"image_{i+2}.jpg")
        else:
            # Loop back to the first image if at the last image
            image1_path = os.path.join(output_dir, "images", f"image_{i+1}.jpg")
            image2_path = os.path.join(output_dir, "images", f"image_1.jpg")
        
        image1 = cv2.imread(image1_path)
        image2 = cv2.imread(image2_path)
        
        # Resize and prepare images using the updated function
        image1_prepared = resize_image(image1, width, height)
        image2_prepared = resize_image(image2, width, height)

        narration = os.path.join(output_dir, "narrations", f"narration_{i+1}.mp3")
        duration = get_audio_duration(narration)
        
        # Write frames for the first image
        # Assuming duration and fade_time are defined
        for _ in range(int((duration - fade_time) * frame_rate / 1000)):
            out.write(image1_prepared)
        
        # Fade transition between image1 and image2
        for alpha in np.linspace(0, 1, int(fade_time * frame_rate / 1000)):
            blended_image = cv2.addWeighted(image1_prepared, 1 - alpha, image2_prepared, alpha, 0)
            out.write(blended_image)

    # Release the VideoWriter and close the window if any
    out.release()
    cv2.destroyAllWindows()

    text.add_narration_to_video(narrations, temp_video, output_dir, output_filename)

    os.remove(temp_video)
