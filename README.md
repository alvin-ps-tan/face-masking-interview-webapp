# Live interview with face masking — Streamlit app

The Week 11 sample project as a web app. It takes a photo of the interviewee, then streams the camera,
masking the interviewee and naming every face in every frame. You can also record the processed interview
as an MP4 file.

| File | What it is |
|---|---|
| `streamlit_app.py` | the app (the same pipeline as `../notebook_llm.ipynb`) |
| `data/reporter_enrolment/<Name>/` | the reporters' photos — the folder name is the name shown on screen |
| `models/mobilefacenet/w600k_mbf.onnx` | the MobileFaceNet face model (13.6 MB, InsightFace, non-commercial use) |
| `requirements.txt` | the Python packages, with the versions tested together |
| `packages.txt` | two Linux libraries that OpenCV needs on Streamlit Cloud |
| `secrets_example.toml` | the format of the TURN login (step 2 below) |

## Run it on your own computer

```
cd week_11_project_specifications_walkthrough/sample_project/streamlit
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The browser opens at `http://localhost:8501`. The camera works there because `localhost` counts as secure.
A phone on the same Wi-Fi **cannot** use its camera via `http://<laptop-ip>:8501`: browsers only allow the
camera on `https://` pages. Deploy (below) to use phones.

## Deploy on Streamlit Community Cloud

1. **Push this folder to GitHub.** Only this folder is needed. Do **not** push `../models/resnet100_glint360/`
   (261 MB — GitHub refuses files over 100 MB); this app does not use it.
2. **Get a free TURN server login.** On the cloud, the live video has to be relayed by a TURN server —
   without one the video keeps "connecting" and never starts. Without a payment method: sign up at
   **expressturn.com** with an email (free plan: 1,000 GB a month) and copy the server address, username
   and password from its dashboard. (Cloudflare's TURN service also works, but needs a payment method on
   the Cloudflare account.) `secrets_example.toml` shows where the values go. (The Hugging Face TURN option
   built into `streamlit-webrtc` no longer works: its login service is down and its TURN server address no
   longer exists.)
3. On [share.streamlit.io](https://share.streamlit.io), **Create app** and choose:
   - the repository and branch;
   - **Main file path**: `streamlit_app.py` (the path of this file inside your repository);
   - **Advanced settings → Python version: 3.10 or 3.11**;
   - **Advanced settings → Secrets**: one of the options in `secrets_example.toml` (e.g. the `[turn]` section
     with your ExpressTURN address, username and password).
     (Already deployed? Open the app's **⋮ → Settings → Secrets**, paste it, and save: the app restarts.)
4. **Deploy.** The first start takes a few minutes while the packages install. Open the app's
   `https://...streamlit.app` address on a laptop or a phone, and allow the camera when the browser asks.

**If the live video keeps "connecting" and never starts**, check the app's logs (**Manage app** at the bottom
right of the app). Each visitor adds a `[video connection]` line saying which relay the app uses: it should
name your TURN server, not "STUN only". Lines with `gl_context ... eglMakeCurrent` are harmless (MediaPipe
looks for a graphics card, finds none, and uses the CPU). Lines with `aioice ... Transaction.__retry` appear
when a video connection attempt fails or is closed — a sign the TURN login is missing or wrong.

## Using the app

1. **Photograph the interviewee**: they look straight at the camera, you press *Take photo*. Their face crop
   appears with *Enrolled as Mr. X*. Only their face template is kept, in memory, for this browser session.
2. **Live interview**: press **Start interview**. Use **Switch camera** to change camera (for example a
   phone's back camera). The interviewee is pixelated and named *Mr. X*; reporters are named in green; anyone
   else is a masked *Stranger*. Press **Stop interview** to end.
3. If *Record this interview* was on, the processed interview appears below the video, with a download
   button. The recording starts with the first processed frame.

The app's colours come from `.streamlit/config.toml` (one line: the colour of buttons and highlights).

## Good to know

- **The video goes to the server.** Each camera frame is sent to the app, masked there, and sent back, so the
  unmasked video does pass through the server.
- **One shared CPU.** The free cloud tier is fine for one or two interviews at a time, not a whole class at once.
- **Recordings are temporary.** On the cloud they are deleted whenever the app restarts — download them.
- **Add a reporter**: add a folder `data/reporter_enrolment/<Name>/` with a clear, front-facing photo, and push.
  A selfie taken with the same kind of camera as the interview works best.
