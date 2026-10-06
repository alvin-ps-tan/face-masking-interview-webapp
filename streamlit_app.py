"""
Week 11 sample project -- the live interview as a Streamlit app.

The same pipeline as notebook_llm.ipynb, but in the browser:
  1. take a photo of the INTERVIEWEE with the camera (it is kept in memory only, never saved to disk);
  2. the REPORTERS come from the repository: data/reporter_enrolment/<Name>/<photo>;
  3. start the live interview: every camera frame is processed and sent back with the interviewee masked
     and every face named. Tick "Record" to save the processed interview as an MP4 you can download.

Run it on your own computer, from this folder:
    pip install -r requirements.txt
    streamlit run streamlit_app.py

Deploy it on Streamlit Community Cloud: see README.md in this folder. The camera works on desktop and
mobile browsers, because the cloud serves the app over https (browsers only allow the camera on https
pages, or on localhost).

(The file is NOT called streamlit.py on purpose: a file with that name would hide the real
 streamlit package, and "import streamlit" would import this file instead.)
"""
import os
import threading
import time
import uuid
from fractions import Fraction

import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from scipy import spatial
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, get_cloudflare_ice_servers, webrtc_streamer

# ---- files (all paths are relative to this script, so the app runs from any folder)
APP_FOLDER = os.path.dirname(os.path.abspath(__file__))
REPORTER_DIR = os.path.join(APP_FOLDER, "data", "reporter_enrolment")
MODEL_PATH = os.path.join(APP_FOLDER, "models", "mobilefacenet", "w600k_mbf.onnx")
RECORDING_DIR = os.path.join(APP_FOLDER, "recordings")    # one MP4 per browser session

# ---- detection (MediaPipe) and recognition (MobileFaceNet) -- the same values as the notebook
MODEL_SELECTION = 1              # MediaPipe: 1 = full-range model (faces up to ~5 m), 0 = short-range (~2 m)
MIN_DETECTION_CONFIDENCE = 0.5   # MediaPipe reports only faces it is at least this sure of
FACE_SIZE = 112                  # every face crop is resized to 112 x 112 for MobileFaceNet
THRESHOLD = 0.60                 # angular-similarity threshold from Week 7

# ---- following faces from frame to frame
IOU_SAME_FACE = 0.4              # detected box vs a box from the previous frame: IoU above this = same face
TRACK_MIN_SCORE = 0.4            # template-matching score below this = the face is lost
SEARCH_MARGIN = 0.5              # search around the last box, half a box wider on every side
SCALES = [0.95, 1.0, 1.05]       # template sizes tried in every frame
MERGE_IOU = 0.5                  # two boxes overlapping more than this are the same face
UNSEEN_SECONDS = {"interviewee": 10, "reporter": 3, "stranger": 1}   # follow an undetected face this long

# ---- masking and labels
INTERVIEWEE_ALIAS = "Mr. X"      # the name shown for the interviewee
MASK_STRANGERS = True            # also hide faces that match nobody
FACE_RECT_WIDTH = 1.2            # face rectangle (crop and mask): 120% of MediaPipe's box width
FACE_RECT_FOREHEAD = 0.3         # face rectangle: raise the top by 30% of the box height (forehead)

# name colours in BGR order: green, orange, red
COLOURS = {"reporter": (0, 170, 0), "interviewee": (0, 140, 255), "stranger": (0, 0, 230)}


# =====================================================================================================
# The face tools -- copied from notebook_llm.ipynb. Two small differences:
#   * detect_faces and make_template are given the detector and the face model to use, because the live
#     video runs in its own thread, and each thread needs its own copy of the two models;
#   * a face's "last_seen" is a time in seconds, not a frame number, because live video has no fixed frame rate.
# =====================================================================================================

def keep_inside_image(x, y, x1, y1, image):
    """Keep a rectangle inside the image. (x, y) is its top-left corner, (x1, y1) its bottom-right corner."""
    image_height, image_width = image.shape[:2]
    if x < 0:
        x = 0
    if x1 > image_width:
        x1 = image_width
    if y < 0:
        y = 0
    if y1 > image_height:
        y1 = image_height
    return x, y, x1, y1


def new_face_detector():
    """A MediaPipe face detector."""
    return mp.solutions.face_detection.FaceDetection(
        model_selection=MODEL_SELECTION, min_detection_confidence=MIN_DETECTION_CONFIDENCE)


def new_face_model():
    """The MobileFaceNet face model, run by OpenCV's deep-learning module (cv2.dnn)."""
    return cv2.dnn.readNetFromONNX(MODEL_PATH)


def detect_faces(bgr, face_detector):
    """Every face MediaPipe finds in a colour image, as a list of (x, y, w, h) boxes in pixels."""
    image_height, image_width = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)        # MediaPipe expects RGB, OpenCV gives BGR
    results = face_detector.process(rgb)

    boxes = []
    if results.detections is None:                    # no face in this image
        return boxes

    for detection in results.detections:
        # MediaPipe gives the box as fractions of the image size (0 to 1), so turn them into pixels
        box = detection.location_data.relative_bounding_box
        x = int(box.xmin * image_width)
        y = int(box.ymin * image_height)
        x1 = x + int(box.width * image_width)         # right edge of the face box
        y1 = y + int(box.height * image_height)       # bottom edge of the face box

        # a face at the edge of the image can stick out of it: keep only the part inside the image
        x, y, x1, y1 = keep_inside_image(x, y, x1, y1, bgr)
        if x1 - x < 20 or y1 - y < 20:
            continue                                  # too small (or too far outside) to use
        boxes.append((x, y, x1 - x, y1 - y))
    return boxes


def biggest_face(boxes):
    """The biggest box (largest width x height) in a list of boxes."""
    biggest = boxes[0]
    for box in boxes:
        if box[2] * box[3] > biggest[2] * biggest[3]:
            biggest = box
    return biggest


def rectangle_around_face(box, image):
    """MediaPipe's box, turned into a vertical rectangle that fits the face: FACE_RECT_WIDTH times as wide,
    with its top raised by FACE_RECT_FOREHEAD of the box height to cover the forehead."""
    x, y, w, h = box
    x1 = x + w                                    # right edge of the face box
    y1 = y + h                                    # bottom edge of the face box

    # 1. the centre of the face, and the rectangle's width
    face_center_x = (x + x1) // 2
    face_width = int(w * FACE_RECT_WIDTH)
    half_width = face_width // 2

    # 2. the rectangle around the centre, with its top raised to cover the forehead
    x = face_center_x - half_width
    x1 = x + face_width
    y = y - int(h * FACE_RECT_FOREHEAD)

    # 3. keep the rectangle inside the image
    x, y, x1, y1 = keep_inside_image(x, y, x1, y1, image)
    return x, y, x1, y1


def crop_face(bgr, box):
    """Cut the face rectangle, and resize it to 112 x 112 RGB for MobileFaceNet."""
    x, y, x1, y1 = rectangle_around_face(box, bgr)
    detected_face = bgr[y:y1, x:x1]
    detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2RGB)
    detected_face = cv2.resize(detected_face, (FACE_SIZE, FACE_SIZE))
    return detected_face


def make_template(face_rgb, face_model):
    """MobileFaceNet turns a 112 x 112 face into a template of 512 numbers."""
    # pixels scaled from 0..255 to -1..1, arranged as a batch of one face, shape (1, 3, 112, 112)
    blob = cv2.dnn.blobFromImage(face_rgb, scalefactor=1.0 / 127.5, size=(FACE_SIZE, FACE_SIZE),
                                 mean=(127.5, 127.5, 127.5))
    face_model.setInput(blob)
    template = face_model.forward()               # run the network: one row of 512 numbers
    return template[0].astype("float32")


def angular_similarity(t1, t2):
    """The Week 7 score: 1 = the templates point the same way, 0 = opposite ways."""
    cos = 1.0 - spatial.distance.cosine(t1, t2)
    cos = float(np.clip(cos, -1.0, 1.0))
    return float(1.0 - np.arccos(cos) / np.pi)


def recognise(face_rgb, gallery, face_model):
    """1:N against every enrolled template, then 1:1 against the threshold. Returns (role, label, score)."""
    template = make_template(face_rgb, face_model)

    # 1:N -- compare with every enrolled photo and remember the closest one
    best_score = 0.0
    best_entry = None
    for entry in gallery:
        score = angular_similarity(template, entry["template"])
        if score > best_score:
            best_score = score
            best_entry = entry

    # 1:1 -- accept the closest photo only if it is close enough
    if best_entry is not None and best_score >= THRESHOLD:
        return best_entry["role"], best_entry["label"], best_score
    else:
        return "stranger", "Stranger", best_score


def follow_face(gray, face):
    """Look for the face's template near its last box, at a few sizes. Returns (best_score, best_box)."""
    x, y, w, h = face["box"]
    x1 = x + w
    y1 = y + h

    # 1. the search window: the last box, made bigger by SEARCH_MARGIN on every side
    margin_x = int(w * SEARCH_MARGIN)
    margin_y = int(h * SEARCH_MARGIN)
    x = x - margin_x
    y = y - margin_y
    x1 = x1 + margin_x
    y1 = y1 + margin_y
    x, y, x1, y1 = keep_inside_image(x, y, x1, y1, gray)
    window = gray[y:y1, x:x1]

    # 2. try the template at each size, and keep the best match
    best_score = -1.0
    best_box = None
    for scale in SCALES:
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        if new_w < 20 or new_h < 20:
            continue
        if new_w > window.shape[1] or new_h > window.shape[0]:
            continue
        template = cv2.resize(face["template"], (new_w, new_h))
        result = cv2.matchTemplate(window, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        if max_val > best_score:
            best_score = max_val
            best_box = (x + max_loc[0], y + max_loc[1], new_w, new_h)
    return best_score, best_box


def iou(box_a, box_b):
    """Intersection over union of two (x, y, w, h) boxes: 1 = the same box, 0 = not touching."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    overlap_left = max(ax, bx)
    overlap_top = max(ay, by)
    overlap_right = min(ax + aw, bx + bw)
    overlap_bottom = min(ay + ah, by + bh)
    overlap_w = max(0, overlap_right - overlap_left)
    overlap_h = max(0, overlap_bottom - overlap_top)
    intersection = overlap_w * overlap_h
    union = aw * ah + bw * bh - intersection
    return intersection / union


def merge_duplicates(faces):
    """One face, one box: if two boxes overlap by more than MERGE_IOU, keep the older face's label
    and give it the newer box. The list is in the order the faces appeared, so earlier = older."""
    merged = []
    for face in faces:
        twin = None
        for kept in merged:
            if iou(face["box"], kept["box"]) > MERGE_IOU:
                twin = kept
                break
        if twin is None:
            merged.append(face)
        else:
            twin["box"] = face["box"]
            twin["template"] = face["template"]
            twin["last_seen"] = max(twin["last_seen"], face["last_seen"])
    return merged


def pixelate_face(frame, box):
    """Hide a face: pixelate an ellipse that fills the face rectangle."""
    x, y, x1, y1 = rectangle_around_face(box, frame)
    region = frame[y:y1, x:x1]
    region_height, region_width = region.shape[:2]
    if region_width < 8 or region_height < 8:
        return

    # 1. pixelate: shrink the region to 8 x 8 blocks, then blow it back up with sharp edges
    small = cv2.resize(region, (8, 8), interpolation=cv2.INTER_AREA)
    pixelated = cv2.resize(small, (region_width, region_height), interpolation=cv2.INTER_NEAREST)

    # 2. an ellipse mask that fills the rectangle: white inside the ellipse, black outside
    mask = np.zeros_like(region)
    centre = (region_width // 2, region_height // 2)
    axes = (region_width // 2, region_height // 2)
    cv2.ellipse(mask, centre, axes, 0, 0, 360, (255, 255, 255), -1)

    # 3. pixelated pixels inside the ellipse, the original pixels outside it
    frame[y:y1, x:x1] = np.where(mask == 255, pixelated, region)


def draw_label(frame, face):
    """Write the face's name just below the face, in its role colour, on a translucent grey band."""
    colour = COLOURS[face["role"]]
    name = face["label"]
    image_height, image_width = frame.shape[:2]

    # 1. where the face is: the same rectangle that is cropped and pixelated
    x, y, x1, y1 = rectangle_around_face(face["box"], frame)

    # 2. centre the name just below the face
    (text_w, text_h), baseline = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    padding = 6
    face_center_x = (x + x1) // 2
    text_x = face_center_x - text_w // 2
    text_y = y1 + padding + text_h + 4
    if text_x < padding:
        text_x = padding
    if text_y + baseline + padding > image_height:
        text_y = image_height - baseline - padding

    # 3. the translucent grey band behind the name: 60% dark grey, 40% the picture underneath
    band_x, band_y = text_x - padding, text_y - text_h - padding
    band_x1, band_y1 = text_x + text_w + padding, text_y + baseline + padding
    band_x, band_y, band_x1, band_y1 = keep_inside_image(band_x, band_y, band_x1, band_y1, frame)
    band = frame[band_y:band_y1, band_x:band_x1]
    grey = np.full_like(band, 70)
    frame[band_y:band_y1, band_x:band_x1] = cv2.addWeighted(grey, 0.6, band, 0.4, 0)

    # 4. the name on top of the band
    cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2, cv2.LINE_AA)


# =====================================================================================================
# The live video: streamlit-webrtc sends every camera frame to InterviewProcessor.recv(),
# which runs the notebook's seven steps and returns the masked, labelled frame to the browser.
# If a recording path is given, every processed frame is also written to an MP4 file -- starting with the
# FIRST processed frame, so the recording never contains the seconds while the camera is still connecting.
# =====================================================================================================

class InterviewProcessor(VideoProcessorBase):

    def __init__(self, gallery, recording_path):
        self.gallery = gallery                    # the reporters, plus the interviewee from the photo
        self.face_detector = new_face_detector()  # this thread's own copy of the two models
        self.face_model = new_face_model()
        self.faces = []                           # every face we are following (one dictionary per face)
        self.count_recognitions = 0

        # ---- the recording (None = do not record)
        self.recording_path = recording_path
        self.video_file = None                    # opened on the first processed frame
        self.video_stream = None
        self.recording_start = 0.0
        self.last_timestamp = -1
        self.lock = threading.Lock()              # recv() and on_ended() run in different threads

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")  # the camera frame as an OpenCV (BGR) image
        image = self.process(image)
        self.record_frame(image)
        return av.VideoFrame.from_ndarray(image, format="bgr24")

    def record_frame(self, image):
        """Add one processed frame to the MP4. The file is opened on the very first processed frame."""
        if self.recording_path is None:
            return
        with self.lock:
            height, width = image.shape[:2]
            width = width - width % 2             # the H.264 video format needs an even width and height
            height = height - height % 2

            if self.video_file is None:
                # the FIRST processed frame: start the recording now
                # (written to a temporary name, and renamed only when it is complete -- see on_ended)
                self.video_file = av.open(self.recording_path + ".tmp", mode="w", format="mp4")
                self.video_stream = self.video_file.add_stream("libx264", rate=30)
                self.video_stream.width = width
                self.video_stream.height = height
                self.video_stream.pix_fmt = "yuv420p"
                self.video_stream.codec_context.time_base = Fraction(1, 1000)   # timestamps in milliseconds
                self.recording_start = time.time()

            # each frame is stamped with the real time since the recording started,
            # because live video has no fixed frame rate
            timestamp = int((time.time() - self.recording_start) * 1000)
            if timestamp <= self.last_timestamp:
                timestamp = self.last_timestamp + 1        # every frame needs a later timestamp than the last
            self.last_timestamp = timestamp

            video_frame = av.VideoFrame.from_ndarray(image[:height, :width], format="bgr24")
            video_frame.pts = timestamp
            video_frame.time_base = Fraction(1, 1000)
            for packet in self.video_stream.encode(video_frame):
                self.video_file.mux(packet)

    def on_ended(self):
        """Called by streamlit-webrtc when the interview stops (STOP pressed): finish the MP4 file."""
        with self.lock:
            if self.video_file is None:
                return                            # nothing was recorded
            for packet in self.video_stream.encode():  # flush the frames still inside the encoder
                self.video_file.mux(packet)
            self.video_file.close()
            self.video_file = None
            os.replace(self.recording_path + ".tmp", self.recording_path)   # the finished recording

    def process(self, frame):
        """The notebook's render loop for ONE frame (steps 1 to 7). Returns the processed frame."""
        now = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.faces

        # ---- STEP 1: remember where every face was in the previous frame
        for face in faces:
            face["previous_box"] = face["box"]

        # ---- STEP 2: FOLLOW every face with its rolling template
        for face in faces:
            score, box = follow_face(gray, face)
            if score >= TRACK_MIN_SCORE:
                face["box"] = box
                face["found"] = True
            else:
                face["found"] = False             # lost -- unless a detected box rescues it in step 3

        # ---- STEP 3: DETECT faces. For each box: an old face or a new face?
        # are the reporter AND the interviewee both being followed? then nobody new is recognised
        reporter_in_frame = False
        interviewee_in_frame = False
        for face in faces:
            if face["role"] == "reporter":
                reporter_in_frame = True
            if face["role"] == "interviewee":
                interviewee_in_frame = True
        both_in_frame = reporter_in_frame and interviewee_in_frame

        detected_boxes = detect_faces(frame, self.face_detector)
        new_faces = []
        for box in detected_boxes:
            # which face from the previous frame does this box overlap the most?
            best_iou = 0.0
            best_face = None
            for face in faces:
                overlap = iou(box, face["previous_box"])
                if overlap > best_iou:
                    best_iou = overlap
                    best_face = face

            if best_iou > IOU_SAME_FACE:
                # an OLD face: do not recognise it again, just move its box onto the detected box
                best_face["box"] = box
                best_face["found"] = True
                best_face["last_seen"] = now
            elif both_in_frame:
                pass                              # a NEW box, but both people are already labelled: skip it
            else:
                # a NEW face: crop it and recognise it, 1:N then 1:1
                role, label, score = recognise(crop_face(frame, box), self.gallery, self.face_model)
                self.count_recognitions = self.count_recognitions + 1
                new_face = {"box": box, "role": role, "label": label, "score": score,
                            "found": True, "last_seen": now}
                new_faces.append(new_face)

        for new_face in new_faces:
            faces.append(new_face)

        # ---- STEP 4: FORGET faces that were lost, or that no detected box has matched for too long
        kept_faces = []
        for face in faces:
            seconds_unseen = now - face["last_seen"]
            if not face["found"]:
                continue
            elif seconds_unseen > UNSEEN_SECONDS[face["role"]]:
                continue
            else:
                kept_faces.append(face)
        faces = kept_faces

        # ---- STEP 5: ROLL the templates: cut every face out of this frame, ready for the next frame
        for face in faces:
            x, y, w, h = face["box"]
            face["template"] = gray[y:y + h, x:x + w].copy()

        # ---- STEP 6: MERGE two boxes on the same face into one
        faces = merge_duplicates(faces)
        self.faces = faces

        # ---- STEP 7: MASK the interviewee (and strangers), then write every NAME
        for face in faces:
            if face["role"] == "interviewee":
                pixelate_face(frame, face["box"])
            if face["role"] == "stranger" and MASK_STRANGERS:
                pixelate_face(frame, face["box"])
        for face in faces:
            draw_label(frame, face)
        return frame


# =====================================================================================================
# Enrolment: the reporters from the repository, the interviewee from the camera
# =====================================================================================================

@st.cache_resource(show_spinner="Loading the reporters' faces...")   # build the gallery once, not on every click
def load_reporter_gallery():
    """One entry per reporter photo in data/reporter_enrolment/<Name>/ -- the folder name is the label."""
    face_detector = new_face_detector()
    face_model = new_face_model()
    gallery = []
    for person in sorted(os.listdir(REPORTER_DIR)):
        person_folder = os.path.join(REPORTER_DIR, person)
        if not os.path.isdir(person_folder):
            continue
        for file_name in sorted(os.listdir(person_folder)):
            bgr = cv2.imread(os.path.join(person_folder, file_name))
            if bgr is None:
                continue                          # not an image
            boxes = detect_faces(bgr, face_detector)
            if len(boxes) == 0:
                continue                          # no face found
            template = make_template(crop_face(bgr, biggest_face(boxes)), face_model)
            gallery.append({"person": person, "role": "reporter", "label": person, "template": template})
    return gallery


@st.cache_resource(show_spinner=False)
def load_photo_models():
    """A detector and a face model for the interviewee's photo (the live video has its own copies)."""
    return new_face_detector(), new_face_model()


def read_secret(name):
    """A value from the app's Secrets, or None if it is not there (for example when running locally)."""
    try:
        return st.secrets[name]
    except Exception:
        return None


def with_tcp_versions(servers):
    """Add a TCP version of every TURN address: firewalls often block UDP traffic but let TCP through."""
    result = []
    for server in servers:
        urls = server["urls"]
        if isinstance(urls, str):                 # one address, or a list of addresses
            urls = [urls]
        all_urls = []
        for url in urls:
            all_urls.append(url)
            if url.startswith("turn:") and "transport=" not in url:
                all_urls.append(url + "?transport=tcp")
        new_server = dict(server)
        new_server["urls"] = all_urls
        result.append(new_server)
    return result


def video_relay_settings():
    """How the browser's video reaches this app (WebRTC). Returns (rtc_configuration, ok, message).

    On your own computer the free Google STUN server is enough. On Streamlit Cloud the app sits behind a
    firewall, so the video must be relayed by a TURN server. Its login comes from the app's Secrets
    (see secrets_example.toml), in one of two forms:
      1. CLOUDFLARE_TURN_KEY_ID and CLOUDFLARE_TURN_KEY_API_TOKEN -- Cloudflare's TURN service (free tier);
         we ask Cloudflare for a fresh TURN login each time;
      2. a [turn] section with the fixed login of any other TURN service.
    """
    servers = [{"urls": ["stun:stun.l.google.com:19302"]}]
    turn = read_secret("turn")
    cloudflare_id = read_secret("CLOUDFLARE_TURN_KEY_ID")
    cloudflare_token = read_secret("CLOUDFLARE_TURN_KEY_API_TOKEN")

    try:
        if cloudflare_id and cloudflare_token:
            cloudflare_servers = get_cloudflare_ice_servers(cloudflare_id, cloudflare_token)
            if isinstance(cloudflare_servers, dict):      # one server, or a list of servers
                cloudflare_servers = [cloudflare_servers]
            servers = servers + cloudflare_servers
            source = "Cloudflare"
        elif turn is not None:
            servers.append({"urls": list(turn["urls"]), "username": turn["username"],
                            "credential": turn["credential"]})
            source = "the [turn] Secrets"
        else:
            return ({"iceServers": servers}, True,
                    "STUN only -- fine on your own computer; on Streamlit Cloud add a TURN server (README.md).")
    except Exception as error:
        return ({"iceServers": servers}, False,
                "Could not get a TURN login: " + str(error) + " -- check the app's Secrets.")

    servers = with_tcp_versions(servers)
    turn_addresses = []
    for server in servers:
        for url in server["urls"]:
            if url.startswith("turn"):
                turn_addresses.append(url.split("?")[0])
    return {"iceServers": servers}, True, "TURN relay from " + source + ": " + turn_addresses[0]


def show_reporters(reporter_gallery):
    """The reporters the app knows (from the repository), folded away in an expander."""
    names = []
    for entry in reporter_gallery:
        if entry["person"] not in names:
            names.append(entry["person"])
    with st.expander("Reporters known to the app (" + str(len(names)) + ")", icon=":material/badge:"):
        st.write(", ".join(names))
        st.caption("To add a reporter, add a folder data/reporter_enrolment/<Name>/ with a clear, "
                   "front-facing photo of them.")


def photograph_interviewee():
    """STEP 1: take the interviewee's photo. Returns their face template, or None if there is none yet."""
    with st.container(border=True):
        st.subheader("1. Photograph the interviewee")
        st.write("Ask the interviewee to look straight at the camera, in good light, and take the photo. "
                 "Only their face template is kept, for this browser session; the photo is never saved.")

        camera_column, result_column = st.columns([3, 2], vertical_alignment="center")
        with camera_column:
            photo = st.camera_input("Interviewee photo", label_visibility="collapsed")

        with result_column:
            if photo is None:
                st.info("No photo yet.", icon=":material/photo_camera:")
                return None

            # the photo arrives as JPEG bytes: decode them into an OpenCV (BGR) image
            photo_bytes = np.frombuffer(photo.getvalue(), dtype=np.uint8)
            bgr = cv2.imdecode(photo_bytes, cv2.IMREAD_COLOR)

            face_detector, face_model = load_photo_models()
            boxes = detect_faces(bgr, face_detector)
            if len(boxes) == 0:
                st.error("No face found. Please retake the photo, facing the camera.", icon=":material/error:")
                return None
            if len(boxes) > 1:
                st.warning("More than one face: the biggest one is used.", icon=":material/warning:")

            interviewee_crop = crop_face(bgr, biggest_face(boxes))         # 112 x 112 RGB
            st.image(interviewee_crop, width=140)
            st.success("Enrolled as **" + INTERVIEWEE_ALIAS + "**: masked in the video.",
                       icon=":material/check_circle:")
            return make_template(interviewee_crop, face_model)


def live_interview(gallery, rtc_configuration):
    """STEP 2: the live interview. Returns (the video stream, the recording's file path or None)."""
    with st.container(border=True):
        st.subheader("2. Live interview")
        st.markdown("Names on screen: :green[**reporter**] · :orange[**" + INTERVIEWEE_ALIAS + "** (masked)] · "
                    ":red[**Stranger** (masked)]")
        record = st.toggle("Record this interview", value=True)

        # every browser session records to its own file, so two people never overwrite each other's video
        if "session_id" not in st.session_state:
            st.session_state["session_id"] = uuid.uuid4().hex[:8]
        os.makedirs(RECORDING_DIR, exist_ok=True)
        recording_path = os.path.join(RECORDING_DIR, "interview_" + st.session_state["session_id"] + ".mp4")
        if not record:
            recording_path = None

        stream = webrtc_streamer(
            key="interview",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=lambda: InterviewProcessor(gallery, recording_path),
            # ask for about 640 x 480 (phones and laptops pick their nearest size)
            media_stream_constraints={"video": {"width": {"ideal": 640}, "height": {"ideal": 480}},
                                      "audio": False},
            rtc_configuration=rtc_configuration,
            async_processing=True,                # drop frames instead of lagging behind when busy
            # how the video and its buttons look ("Switch camera" picks e.g. a phone's back camera)
            video_html_attrs={"autoPlay": True, "controls": False, "playsInline": True, "muted": True,
                              "style": {"width": "100%", "borderRadius": "8px"}},
            translations={"start": "Start interview", "stop": "Stop interview", "select_device": "Switch camera"},
        )

        with st.expander("Tips for a good interview", icon=":material/lightbulb:"):
            st.markdown("- Both people face the camera for the first few seconds, so they are recognised.\n"
                        "- Sit within about 2 metres of the camera, in good light.\n"
                        "- On a phone, use **Switch camera** to film with the back camera.\n"
                        "- The recording starts with the first processed frame, and appears below "
                        "when you press **Stop interview**.")
    return stream, recording_path


def show_recording(recording_path):
    """STEP 3: play the recorded interview, with a download button."""
    with st.container(border=True):
        st.subheader("3. Your recording")
        with open(recording_path, "rb") as video_file:
            video_bytes = video_file.read()
        st.video(video_bytes)
        st.download_button("Download the masked interview (MP4)", video_bytes, file_name="masked_interview.mp4",
                           mime="video/mp4", type="primary", icon=":material/download:", width="stretch")


def main():
    st.set_page_config(page_title="Masked Interview", page_icon="🎥", layout="centered")
    st.title("Masked interview")
    st.write("Film an interview in which the **interviewee can never be identified**, while every face on "
             "screen is still named.")

    # ---- how the live video will connect (written to the app's logs once per visitor; shown only if broken)
    rtc_configuration, relay_ok, relay_message = video_relay_settings()
    if "relay_logged" not in st.session_state:
        print("[video connection]", relay_message, flush=True)
        st.session_state["relay_logged"] = True
    if not relay_ok:
        st.error("Live video: " + relay_message, icon=":material/error:")

    reporter_gallery = load_reporter_gallery()
    show_reporters(reporter_gallery)

    # ---- STEP 1: the interviewee's photo
    interviewee_template = photograph_interviewee()
    if interviewee_template is None:
        st.caption("The live interview appears here once the interviewee's photo is taken.")
        return

    # the gallery for the live interview: every reporter, plus the interviewee
    gallery = []
    for entry in reporter_gallery:
        gallery.append(entry)
    gallery.append({"person": "interviewee", "role": "interviewee", "label": INTERVIEWEE_ALIAS,
                    "template": interviewee_template})

    # ---- STEP 2: the live interview
    stream, recording_path = live_interview(gallery, rtc_configuration)

    # ---- STEP 3: the recording, once an interview has been recorded and stopped
    interview_running = stream.state.playing
    if recording_path is not None and not interview_running and os.path.exists(recording_path):
        show_recording(recording_path)

    st.caption("Week 11 sample project · MediaPipe finds the faces, MobileFaceNet recognises them, "
               "and the interviewee is pixelated in every frame.")


if __name__ == "__main__":
    main()
