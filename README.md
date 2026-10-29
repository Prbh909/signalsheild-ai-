# SignalShield AI — GitHub Pages Edition

A static, browser-based Class XII AI project exploring toxicity signals in text and speech.

## Features

- Text analysis with `Xenova/toxic-bert` through Transformers.js.
- Six toxicity-related scores: toxicity, severe toxicity, obscene language, threat, insult, and identity-related hostility.
- Microphone recording with playback and downloadable audio.
- Live speech-to-text while recording when the browser supports the Web Speech API.
- Audio-file transcription with the multilingual `onnx-community/whisper-tiny` model (WAV/MP3/M4A/OGG/WebM supported when the browser can decode the format; max 25 MB and 3 minutes in this demo).
- Editable transcript preview and a one-click route from transcript to text analysis.
- Copy/download report, human-review decision and notes saved locally in the browser, and light/dark theme.
- Responsive layout for desktop and mobile.

## Publish on GitHub Pages

1. Download and unzip `SignalShield_GitHub_Pages.zip`.
2. Create a new repository on GitHub (for example, `signalshield-ai`).
3. Upload **the contents** of this folder to the repository root: `index.html`, `style.css`, `app.js`, and `README.md`.
4. In the repository, open **Settings → Pages**.
5. Under **Build and deployment**, select **Deploy from a branch**.
6. Choose the `main` branch and `/(root)` folder, then click **Save**.
7. Wait for GitHub Pages to publish the site. Open the URL shown on the Pages settings screen.
8. Use the site over HTTPS. Allow microphone access when prompted.

## How to use audio

### Record speech
1. Open the **Audio lab** section and click **Start recording**.
2. Allow microphone access and speak clearly.
3. Click **Stop**. The audio recording appears in the player and can be downloaded.
4. If browser speech recognition is supported, the transcript appears automatically. Otherwise, use the uploaded-file transcription option or type the transcript manually.
5. Review/edit the transcript and click **Analyze transcript**.

### Transcribe an existing audio file
1. Select an audio file in the **Transcribe an audio file** card.
2. Click **Transcribe audio with AI** and wait for the speech model to load/run.
3. Review and correct the transcript, then click **Analyze transcript**.

The first text and speech model runs can download substantial model files from Hugging Face. Use a modern browser, a stable connection, and short clips. If the audio model runs out of memory, use a shorter/smaller clip.

## Important limitations and privacy

- This is an educational demo, **not a production moderation or security system**.
- `toxic-bert` is a general toxicity classifier, not a definitive hate-speech detector. The displayed overall score is a simple maximum-score heuristic, not a scientifically validated hate-speech probability.
- Context, sarcasm, reclaimed language, dialects, spelling variations, and languages not well represented in the model can cause false positives or false negatives. Always use human review.
- Uploaded-file transcription runs with a browser-loaded Whisper model after its model files download. The browser's built-in speech recognition used during live recording may rely on an external service, depending on browser/platform. Do not record private or sensitive conversations without permission.
- Review history and theme preference are stored in local browser storage on that device only. There is no server, shared database, secure login, or admin dashboard in this GitHub Pages edition. Do not present the local review field as a secure account system.
- GitHub Pages serves static files; it cannot run a Python backend. A real secure login/shared moderation queue would require a separate backend and database.

## Files

- `index.html` — page structure and model import map
- `style.css` — responsive styling and themes
- `app.js` — text model, audio recording, transcription, analysis, and report interactions

## Credits

- Transformers.js: https://huggingface.co/docs/transformers.js
- Toxic-BERT model: https://huggingface.co/Xenova/toxic-bert
- Whisper model: https://huggingface.co/onnx-community/whisper-tiny
- Browser audio recording: MediaRecorder API
