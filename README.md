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
2. **Get a free TURN server login.** On the cloud, the live video has to be relayed by a TURN server.
   Sign up for a free TURN service (for example Metered, or Twilio), create a TURN credential, and note its
   server URLs, username and password.
3. On [share.streamlit.io](https://share.streamlit.io), **Create app** and choose:
   - the repository and branch;
   - **Main file path**: `week_11_project_specifications_walkthrough/sample_project/streamlit/streamlit_app.py`;
   - **Advanced settings → Python version: 3.10**;
   - **Advanced settings → Secrets**: paste the text of `secrets_example.toml`, with your own TURN values.
4. **Deploy.** The first start takes a few minutes while the packages install. Open the app's
   `https://...streamlit.app` address on a laptop or a phone, and allow the camera when the browser asks.

## Using the app

1. **Photograph the interviewee**: they look straight at the camera, you press *Take photo*. Only their face
   template is kept, in memory, for this browser session.
2. **Live interview**: press **START**. Use **SELECT DEVICE** to switch camera (for example a phone's back
   camera). The interviewee is pixelated and shown as *Mr. X*; reporters are named; anyone else is a masked
   *Stranger*. Press **STOP** to end.
3. If *Record* was ticked, the processed interview appears below the video, with a download button. The
   recording starts with the first processed frame.

## Good to know

- **The video goes to the server.** Each camera frame is sent to the app, masked there, and sent back, so the
  unmasked video does pass through the server.
- **One shared CPU.** The free cloud tier is fine for one or two interviews at a time, not a whole class at once.
- **Recordings are temporary.** On the cloud they are deleted whenever the app restarts — download them.
- **Add a reporter**: add a folder `data/reporter_enrolment/<Name>/` with a clear, front-facing photo, and push.
  A selfie taken with the same kind of camera as the interview works best.
