import cv2
import time
import numpy as np
from ultralytics import YOLO
import streamlit as st
from draw_utils import draw_label, draw_rounded_rectangle, get_box_details  # Ensure these functions exist
from deep_sort import DeepSort  # Ensure DeepSort is correctly imported and installed
import tempfile

from twilio.rest import Client

# Replace these with your actual Twilio credentials
account_sid = "......"
auth_token = "......."
whatsapp_number = "whatsapp:....."  # e.g., "whatsapp:+14155238886"
user_whatsapp_number = "whatsapp:......"  # e.g., "whatsapp:+919876543210"

# Initialize Twilio Client
twilio_client = Client(account_sid, auth_token)

# Function to send WhatsApp notifications
def send_whatsapp_notification(count, object_type):
    message = f" {count} '{object_type}' have been detected on the live feed."
    twilio_client.messages.create(
        body=message,
        from_=whatsapp_number,
        to=user_whatsapp_number
    )
model = YOLO('yolov8n.pt')
deep_sort_weights = 'deep_sort/deep/checkpoint/ckpt.t7'
# Tracking variables
details = []
prev_details = {}
unique_track_ids = set()

def track_video(frame, model, object_, detection_threshold, tracker, frame_no):
    og_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame)

    bboxes_xywh = []
    confs = []

    class_names = list(model.names.values())
    cls, xyxy, conf, xywh = get_box_details(results[0].boxes)  # Ensure `get_box_details` is defined

    for c, b, co in zip(cls, xywh, conf.cpu().numpy()):
        if class_names[int(c)] == object_ and co >= detection_threshold:
            bboxes_xywh.append(b.cpu().numpy())
            confs.append(co)

    bboxes_xywh = np.array(bboxes_xywh, dtype=float)

    new_ids = set()  # To track new unique IDs
    if len(bboxes_xywh) >= 1:
        tracks = tracker.update(bboxes_xywh, confs, og_frame)

        ids = []
        for track in tracker.tracker.tracks:
            track_id = track.track_id
            x1, y1, x2, y2 = track.to_tlbr()  # Bounding box coordinates
            w, h = x2 - x1, y2 - y1

            # Color selection based on track_id
            color = [(255, 0, 0), (0, 0, 255), (0, 255, 0)][track_id % 3]

            draw_rounded_rectangle(og_frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 1, 15)
            draw_label(og_frame, f"{object_}-{track_id}", (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color,
                       (255, 255, 255))

            if track_id not in prev_details:
                prev_details[track_id] = [time.time(), color]
                new_ids.add(track_id)  # Add new unique ID

            unique_track_ids.add(track_id)
            ids.append(track_id)

        # Update details for IDs that have completed tracking
        ids_done = set(prev_details.keys()) ^ set(ids)
        for id in ids_done:
            details.append([object_, id, time.time() - prev_details[id][0], prev_details[id][1], frame_no - 1])
            del prev_details[id]

    og_frame = cv2.cvtColor(og_frame, cv2.COLOR_BGR2RGB)
    return og_frame, len(new_ids)  # Return the count of new IDs

# Streamlit UI
st.title('Real-Time Object Tracking')
st.sidebar.title("Tracker Options")
object_ = st.sidebar.selectbox('Select object to track', list(model.names.values()), placeholder='Select any..')
detection_threshold = st.sidebar.slider('Detection Threshold', 0.1, 1.0, 0.5)
input_source = st.sidebar.radio("Select Input Source", ('Video File', 'Webcam'))
st.sidebar.markdown('---')

# Tracker configuration
tracker = DeepSort(model_path=deep_sort_weights, max_age=70, n_init=5)

# Initialize session state for count tracking
if 'total_count' not in st.session_state:
    st.session_state.total_count = 0

# Initialize a variable to store the last displayed count
if 'last_displayed_count' not in st.session_state:
    st.session_state.last_displayed_count = -1  # Set to -1 initially to force display on first run

# Initialize capture based on input source
cap = None
if input_source == 'Video File':
    video_file = st.sidebar.file_uploader('Upload your video file', type=['mp4', 'mov', 'avi', 'm4v'])
    if video_file:
        t_file = tempfile.NamedTemporaryFile(delete=False)
        t_file.write(video_file.read())
        t_file.close()  # Close the temporary file to ensure it can be accessed by OpenCV
        cap = cv2.VideoCapture(t_file.name)
elif input_source == 'Webcam':
    cap = cv2.VideoCapture(0)  # Use the first camera

stframe = st.empty()
frame_no = 0

# Create a placeholder for the total count display
count_placeholder = st.empty()

# Process video or webcam feed
if cap is not None and cap.isOpened():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("No more frames to read.")
            break
        frame, new_ids = track_video(frame, model, object_, detection_threshold, tracker, frame_no)

        # Update total count only if new unique IDs were detected
        if new_ids:
            st.session_state.total_count = len(unique_track_ids)  # Update total count if there are new IDs

        # Display the processed frame in Streamlit
        stframe.image(frame, channels='RGB', use_column_width=True)  # Use RGB for Streamlit

        # Update the total count in the placeholder if it has changed
        if st.session_state.total_count != st.session_state.last_displayed_count:
            count_placeholder.markdown(
                f"<div style='color: red; font-size: 24px; font-weight: bold;'>Total Count of {object_}</div><div style='color:white; font-size:24px; font-weight: bold;'> {st.session_state.total_count}</div>",
                unsafe_allow_html=True
            )
            st.session_state.last_displayed_count = st.session_state.total_count  # Update last displayed count
            if st.session_state.total_count > 0:
                send_whatsapp_notification(st.session_state.total_count, object_)
        frame_no += 1
    cap.release()
else:
    st.write(" ")
