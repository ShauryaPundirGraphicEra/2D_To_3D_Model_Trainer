import cv2
import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image

# Load the model
@st.cache_resource
def load_model():
    return YOLO("C:/Users/ACER/PycharmProjects/YOLO-Object-Detection-Course/yolo11m-pose.pt")

model = load_model()

# File uploader
st.title("Human Pose Estimation")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
threshold = st.slider("Threshold", 0.0, 1.0, 0.5)


# Skeleton connection pairs (adjust for the actual pose model used)
POSE_SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Head -> Shoulders -> Elbows -> Wrists
    (5, 6), (6, 7), (7, 8), (8, 9),  # Hips -> Knees -> Ankles
    (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)  # Body connections
]


if uploaded_file is not None:
    # Load and preprocess image
    image = Image.open(uploaded_file)
    if image.mode == "RGBA":
        image = image.convert("RGB")
    image_np = np.array(image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Perform inference
    results = model(image_np)  # Returns a list of results

    # Plot the results on the image
    plotted_img = results[0].plot()

    # Convert numpy array back to PIL Image
    result_image = Image.fromarray(plotted_img)

    # Display the result in Streamlit
    st.image(result_image, caption='Processed Image', use_column_width=True)



#
# import cv2
# import streamlit as st
# from ultralytics import YOLO
# import numpy as np
# from PIL import Image
#
#
# # Load the model
# @st.cache_resource
# def load_model():
#     return YOLO("C:/Users/ACER/PycharmProjects/YOLO-Object-Detection-Course/yolo11m-pose.pt")
#
#
# model = load_model()
#
# # Streamlit interface
# st.title("Human Pose Estimation for Video")
# uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
# threshold = st.slider("Threshold", 0.0, 1.0, 0.5)
#
# if uploaded_video is not None:
#     # Save uploaded video to a temporary file
#     temp_file = "temp_video.mp4"
#     with open(temp_file, "wb") as f:
#         f.write(uploaded_video.read())
#
#     # Open the video file
#     video_cap = cv2.VideoCapture(temp_file)
#     if not video_cap.isOpened():
#         st.error("Error loading video!")
#     else:
#         # Prepare output video writer
#         fps = int(video_cap.get(cv2.CAP_PROP_FPS))
#         width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         out_video = cv2.VideoWriter(
#             "output_video.mp4",
#             cv2.VideoWriter_fourcc(*"mp4v"),
#             fps,
#             (width, height)
#         )
#
#         stframe = st.empty()  # Placeholder for displaying frames in Streamlit
#
#         while video_cap.isOpened():
#             ret, frame = video_cap.read()
#             if not ret:
#                 break  # End of video
#
#             # Convert BGR to RGB for the model
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#             # Perform inference
#             results = model(frame_rgb)
#
#             # Get the processed frame
#             processed_frame = results[0].plot()
#
#             # Write the processed frame to the output video
#             out_video.write(cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))
#
#             # Display the frame in Streamlit
#             stframe.image(processed_frame, channels="RGB", use_column_width=True)
#
#         # Release resources
#         video_cap.release()
#         out_video.release()
#
#         # Provide a link to download the output video
#         st.success("Video processing complete!")
#         with open("output_video.mp4", "rb") as f:
#             st.download_button("Download Processed Video", f, file_name="output_video.mp4")
