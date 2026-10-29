import { pipeline, env } from '@huggingface/transformers';

// SignalShield AI: static, browser-based educational demo.
env.allowLocalModels = false;
env.useBrowserCache = true;

const $ = (id) => document.getElementById(id);
const state = {
  classifier: null, asr: null, modelPromise: null, asrPromise: null,
  lastReport: null, lastText: '', recording: false, recorder: null,
  stream: null, chunks: [], audioBlob: null, audioUrl: null,
  recognition: null, recognitionFinal: '', selectedAudioFile: null,
  toastTimer: null
};

const LABEL_INFO = [
  { key: 'toxic', name: 'Toxicity', description: 'General hostile or abusive language' },
  { key: 'severe_toxic', name: 'Severe toxicity', description: 'A stronger toxicity signal' },
  { key: 'obscene', name: 'Obscene language', description: 'Potentially vulgar or explicit language' },
  { key: 'threat', name: 'Threat', description: 'Language that may express a threat' },
  { key: 'insult', name: 'Insult', description: 'Potentially insulting language' },
  { key: 'identity_hate', name: 'Identity-related signal', description: 'Potential hostility related to identity' }
];
const PROFILE_THRESHOLDS = { balanced: 0.55, cautious: 0.35, strict: 0.22 };

function toast(message) {
  const node = $('toast');
  node.textContent = message;
  node.classList.add('show');
  clearTimeout(state.toastTimer);
  state.toastTimer = setTimeout(() => node.classList.remove('show'), 2800);
}
function setStatus(message, kind = 'normal') {
  $('modelBadge').innerHTML = `<span class="status-dot"></span>${escapeHtml(message)}`;
  $('modelBadge').style.color = kind === 'ready' ? 'var(--mint)' : kind === 'error' ? 'var(--danger)' : 'var(--muted)';
}
function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, ch => ({ '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;' }[ch]));
}
function updateTextCount() {
  $('charCount').textContent = `${$('sourceText').value.length.toLocaleString()} / 12,000 characters`;
}
function updateTranscriptUI() {
  const text = $('transcriptText').value.trim();
  $('transcriptCount').textContent = `${$('transcriptText').value.length.toLocaleString()} characters`;
  $('useTranscriptBtn').disabled = !text;
  $('analyzeTranscriptBtn').disabled = !text;
}
function showOnly(which) {
  ['emptyState','loadingState','reportState','errorState'].forEach(id => $(id).hidden = id !== which);
}
function setLoading(title, message, percent = 20) {
  showOnly('loadingState');
  $('loadingTitle').textContent = title;
  $('loadingMessage').textContent = message;
  $('progressBar').style.width = `${percent}%`;
}
function normalizeScores(output) {
  let rows = output;
  if (Array.isArray(rows) && rows.length && Array.isArray(rows[0])) rows = rows[0];
  const map = {};
  (rows || []).forEach(item => { map[String(item.label || '').toLowerCase()] = Number(item.score) || 0; });
  return LABEL_INFO.map(info => ({ ...info, score: map[info.key] ?? 0 }));
}
function getRisk(scores) {
  const relevant = scores.filter(s => ['toxic','severe_toxic','threat','insult','identity_hate','obscene'].includes(s.key));
  // A transparent heuristic for a student demo; not a validated hate-speech metric.
  return Math.max(0, ...relevant.map(s => s.score));
}
function classifyBand(score, threshold) {
  if (score >= threshold) return { label: 'Review recommended', summary: 'The model found one or more elevated toxicity-related signals. A person should review the full context.' };
  if (score >= Math.max(0.12, threshold * 0.55)) return { label: 'Some signals', summary: 'The model found a few possible signals. Context, sarcasm, dialect, and transcription quality can change how this should be interpreted.' };
  return { label: 'Lower signal', summary: 'The model returned lower toxicity-related scores. This does not guarantee that the message is safe or free of hate speech.' };
}
async function getClassifier() {
  if (state.classifier) return state.classifier;
  if (!state.modelPromise) {
    setStatus('Loading text model…');
    state.modelPromise = pipeline('text-classification', 'Xenova/toxic-bert', {
      progress_callback: (p) => {
        if (p && typeof p.progress === 'number') {
          $('loadingMessage').textContent = `Downloading model files… ${Math.round(p.progress)}%`;
          $('progressBar').style.width = `${Math.max(8, Math.min(100, p.progress))}%`;
        }
      }
    }).then(model => { state.classifier = model; setStatus('Text model ready', 'ready'); return model; })
      .catch(err => { state.modelPromise = null; throw err; });
  }
  return state.modelPromise;
}
async function analyzeText(text) {
  const trimmed = text.trim();
  if (!trimmed) { toast('Enter or transcribe some text first.'); return; }
  if (trimmed.length > 12000) { toast('Please keep the text under 12,000 characters.'); return; }
  state.lastText = trimmed;
  $('analyzeBtn').disabled = true;
  setLoading('Preparing analysis…', 'Loading the toxicity classifier if this is your first run.', 15);
  try {
    const classifier = await getClassifier();
    setLoading('Reading language signals…', 'The model is estimating six toxicity-related labels.', 65);
    // Split long input to avoid tokenizer limits, then combine by maximum score per label.
    const chunks = splitText(trimmed, 1600);
    let allScores = [];
    for (let i = 0; i < chunks.length; i++) {
      $('loadingMessage').textContent = chunks.length > 1 ? `Analyzing text segment ${i + 1} of ${chunks.length}…` : 'Running the text classifier…';
      $('progressBar').style.width = `${65 + Math.round(((i + 1) / chunks.length) * 30)}%`;
      const output = await classifier(chunks[i], { top_k: null });
      const normalized = normalizeScores(output);
      if (!allScores.length) allScores = normalized;
      else allScores = allScores.map((item, index) => ({ ...item, score: Math.max(item.score, normalized[index].score) }));
    }
    renderReport(trimmed, allScores);
    $('analyzeBtn').disabled = false;
    $('analyzeBtn').innerHTML = '<span>Analyze text</span><span>→</span>';
    $('reportState').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  } catch (err) {
    console.error(err);
    showOnly('errorState');
    $('errorMessage').textContent = `The model could not load or run. Check your internet connection and reload the page. Details: ${err?.message || 'unknown error'}`;
    setStatus('Model error', 'error');
  } finally {
    $('analyzeBtn').disabled = false;
  }
}
function splitText(text, maxLength) {
  if (text.length <= maxLength) return [text];
  const parts = [];
  let rest = text;
  while (rest.length > maxLength) {
    let cut = rest.lastIndexOf(' ', maxLength);
    if (cut < maxLength * .55) cut = maxLength;
    parts.push(rest.slice(0, cut)); rest = rest.slice(cut).trimStart();
  }
  if (rest) parts.push(rest);
  return parts;
}
function renderReport(text, scores) {
  const risk = getRisk(scores);
  const threshold = PROFILE_THRESHOLDS[$('profile').value] || PROFILE_THRESHOLDS.balanced;
  const band = classifyBand(risk, threshold);
  state.lastReport = { text, scores, risk, threshold, band, createdAt: new Date().toISOString() };
  showOnly('reportState');
  $('overallValue').textContent = `${Math.round(risk * 100)}%`;
  $('ringValue').textContent = `${Math.round(risk * 100)}%`;
  $('scoreRing').style.setProperty('--score', `${Math.round(risk * 100)}%`);
  $('overallLabel').textContent = band.label;
  $('overallLabel').style.background = risk >= threshold ? '#ff7e9418' : '#7df2c415';
  $('overallLabel').style.color = risk >= threshold ? 'var(--danger)' : 'var(--mint)';
  $('reportSummary').textContent = band.summary;
  $('labelsList').innerHTML = scores.map(item => `
    <div class="label-row" title="${escapeHtml(item.description)}">
      <span class="label-name">${escapeHtml(item.name)}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${Math.round(item.score * 100)}%"></div></div>
      <span class="label-score">${Math.round(item.score * 100)}%</span>
    </div>`).join('');
  $('reviewDecision').value = 'unreviewed';
  $('reviewNotes').value = '';
  $('reviewSaved').textContent = '';
}
function reportText() {
  if (!state.lastReport) return '';
  const r = state.lastReport;
  return [
    'SignalShield AI — Analysis report',
    `Generated: ${new Date(r.createdAt).toLocaleString()}`,
    `Overall toxicity signal (heuristic): ${Math.round(r.risk * 100)}%`,
    `Interpretation: ${r.band.label}`,
    `Summary: ${r.band.summary}`,
    '', 'Label scores:', ...r.scores.map(s => `- ${s.name}: ${Math.round(s.score * 100)}%`),
    '', 'Text analyzed:', r.text,
    '', 'Important: Toxic-BERT is a general toxicity classifier, not a definitive hate-speech detector. Scores are not proof of intent or a basis for automatic punishment.'
  ].join('\n');
}
async function copyText(text) {
  try { await navigator.clipboard.writeText(text); toast('Copied to clipboard.'); }
  catch { const temp = document.createElement('textarea'); temp.value = text; document.body.appendChild(temp); temp.select(); document.execCommand('copy'); temp.remove(); toast('Copied to clipboard.'); }
}
function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href = url; a.download = filename; document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1500);
}
function saveReview() {
  if (!state.lastReport) { toast('Run an analysis first.'); return; }
  const item = { ...state.lastReport, decision: $('reviewDecision').value, notes: $('reviewNotes').value, reviewedAt: new Date().toISOString() };
  try {
    const existing = JSON.parse(localStorage.getItem('signalshield_reviews') || '[]');
    existing.unshift(item);
    localStorage.setItem('signalshield_reviews', JSON.stringify(existing.slice(0, 30)));
    $('reviewSaved').textContent = 'Saved on this device only.';
    toast('Review saved locally.');
  } catch { $('reviewSaved').textContent = 'Local storage is unavailable in this browser.'; }
}

function getSpeechRecognitionConstructor() { return window.SpeechRecognition || window.webkitSpeechRecognition || null; }
async function startRecording() {
  if (state.recording) return;
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    toast('Audio recording is not supported in this browser. Try current Chrome or Edge over HTTPS.'); return;
  }
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    state.chunks = []; state.audioBlob = null; state.recognitionFinal = '';
    const mime = ['audio/webm;codecs=opus','audio/webm','audio/ogg;codecs=opus','audio/mp4'].find(t => MediaRecorder.isTypeSupported?.(t));
    state.recorder = mime ? new MediaRecorder(state.stream, { mimeType: mime }) : new MediaRecorder(state.stream);
    state.recorder.ondataavailable = e => { if (e.data && e.data.size) state.chunks.push(e.data); };
    state.recorder.onerror = e => { console.error('MediaRecorder error', e); toast('Recording encountered an error.'); };
    state.recorder.onstop = () => {
      state.audioBlob = new Blob(state.chunks, { type: state.recorder.mimeType || 'audio/webm' });
      if (state.audioUrl) URL.revokeObjectURL(state.audioUrl);
      state.audioUrl = URL.createObjectURL(state.audioBlob);
      $('audioPlayback').src = state.audioUrl; $('audioPlayback').hidden = false;
      $('audioDownloadBtn').disabled = !state.audioBlob.size;
      state.stream?.getTracks().forEach(track => track.stop()); state.stream = null;
      const transcript = state.recognitionFinal.trim();
      if (transcript) {
        $('transcriptText').value = transcript; updateTranscriptUI();
        $('recordStatus').textContent = 'Recording saved · transcript ready';
      } else {
        $('recordStatus').textContent = 'Recording saved · transcription unavailable';
        $('asrStatus').textContent = 'No transcript was captured. Try “Transcribe an audio file” to run the browser AI speech model, or type the words into the transcript preview.';
      }
      $('audioAnalyzeBtn').disabled = !transcript;
      $('waveBars').classList.remove('active');
      $('audioStartBtn').disabled = false; $('audioStopBtn').disabled = true;
      state.recording = false;
    };
    state.recorder.start(250);
    state.recording = true;
    $('audioStartBtn').disabled = true; $('audioStopBtn').disabled = false;
    $('recordStatus').textContent = 'Recording… speak now'; $('waveBars').classList.add('active');
    const Recognition = getSpeechRecognitionConstructor();
    if (Recognition) {
      try {
        state.recognition = new Recognition();
        state.recognition.lang = navigator.language || 'en-US';
        state.recognition.continuous = true;
        state.recognition.interimResults = true;
        state.recognition.onresult = event => {
          let interim = '';
          for (let i = event.resultIndex; i < event.results.length; i++) {
            const phrase = event.results[i][0]?.transcript || '';
            if (event.results[i].isFinal) state.recognitionFinal += phrase + ' ';
            else interim += phrase;
          }
          $('recordStatus').textContent = `Recording… ${((state.recognitionFinal + interim).trim()).slice(-80) || 'speak now'}`;
        };
        state.recognition.onerror = e => {
          if (e.error !== 'no-speech' && e.error !== 'aborted') console.info('Speech recognition:', e.error);
        };
        state.recognition.start();
      } catch (e) { console.info('Browser speech recognition unavailable', e); state.recognition = null; }
    } else {
      $('recordStatus').textContent = 'Recording audio… browser speech recognition unavailable';
    }
  } catch (err) {
    console.error(err);
    toast(err?.name === 'NotAllowedError' ? 'Microphone permission was denied.' : 'Could not access the microphone.');
    $('recordStatus').textContent = 'Microphone access unavailable';
  }
}
function stopRecording() {
  if (!state.recording) return;
  const recognition = state.recognition;
  state.recognition = null;
  if (recognition) {
    // Wait for the browser's final recognition results before finalizing the recording UI.
    recognition.onend = () => { if (state.recorder && state.recorder.state !== 'inactive') state.recorder.stop(); };
    try { recognition.stop(); } catch { if (state.recorder && state.recorder.state !== 'inactive') state.recorder.stop(); }
    setTimeout(() => { if (state.recorder && state.recorder.state !== 'inactive') state.recorder.stop(); }, 1800);
  } else if (state.recorder && state.recorder.state !== 'inactive') state.recorder.stop();
}
function downloadRecording() {
  if (!state.audioBlob) return;
  const ext = (state.audioBlob.type.includes('ogg') ? 'ogg' : state.audioBlob.type.includes('mp4') ? 'm4a' : 'webm');
  downloadBlob(state.audioBlob, `signalshield-recording.${ext}`);
}

async function getASR() {
  if (state.asr) return state.asr;
  if (!state.asrPromise) {
    $('asrStatus').textContent = 'Loading speech model… this can download a large model and take several minutes on slower connections.';
    state.asrPromise = pipeline('automatic-speech-recognition', 'onnx-community/whisper-tiny', {
      progress_callback: p => {
        if (p && typeof p.progress === 'number') $('asrStatus').textContent = `Downloading speech model… ${Math.round(p.progress)}%`;
      }
    }).then(model => { state.asr = model; $('asrStatus').textContent = 'Speech model ready. Select an audio clip to transcribe.'; return model; })
      .catch(err => { state.asrPromise = null; throw err; });
  }
  return state.asrPromise;
}
async function decodeAudioFile(file) {
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) throw new Error('Audio decoding is not supported in this browser.');
  const context = new AudioContextClass();
  try {
    const buffer = await context.decodeAudioData(await file.arrayBuffer());
    const channelCount = buffer.numberOfChannels;
    const length = buffer.length;
    const mono = new Float32Array(length);
    for (let c = 0; c < channelCount; c++) {
      const channel = buffer.getChannelData(c);
      for (let i = 0; i < length; i++) mono[i] += channel[i] / channelCount;
    }
    const targetRate = 16000;
    if (buffer.sampleRate === targetRate) return { audio: mono, sampling_rate: targetRate, duration: buffer.duration };
    const outLength = Math.floor(mono.length * targetRate / buffer.sampleRate);
    const resampled = new Float32Array(outLength);
    const ratio = buffer.sampleRate / targetRate;
    for (let i = 0; i < outLength; i++) {
      const pos = i * ratio; const left = Math.floor(pos); const right = Math.min(left + 1, mono.length - 1); const fraction = pos - left;
      resampled[i] = mono[left] * (1 - fraction) + mono[right] * fraction;
    }
    return { audio: resampled, sampling_rate: targetRate, duration: buffer.duration };
  } finally { await context.close(); }
}
async function transcribeFile() {
  const file = state.selectedAudioFile;
  if (!file) { toast('Choose an audio file first.'); return; }
  if (file.size > 25 * 1024 * 1024) { toast('Please choose a file smaller than 25 MB.'); return; }
  $('transcribeFileBtn').disabled = true;
  $('asrStatus').textContent = 'Preparing audio…';
  try {
    const audioData = await decodeAudioFile(file);
    if (audioData.duration > 180) { toast('For this browser demo, choose a clip under 3 minutes.'); $('transcribeFileBtn').disabled = false; return; }
    const asr = await getASR();
    $('asrStatus').textContent = `Transcribing ${Math.round(audioData.duration)} seconds of audio…`;
    const output = await asr(audioData.audio, { sampling_rate: audioData.sampling_rate, chunk_length_s: 30, stride_length_s: 5 });
    const transcript = (typeof output === 'string' ? output : Array.isArray(output) ? output.map(part => part.text || '').join(' ') : output?.text || '').trim();
    if (!transcript) throw new Error('No speech was recognized. Try a clearer or shorter audio clip.');
    $('transcriptText').value = transcript; updateTranscriptUI();
    $('asrStatus').textContent = `Transcription complete · ${transcript.length} characters. Review the words before analysis.`;
    toast('Audio transcribed. Review the transcript before analyzing.');
  } catch (err) {
    console.error(err);
    $('asrStatus').textContent = `Transcription failed: ${err?.message || 'unknown error'}. Try a shorter WAV/MP3/WebM file or type the transcript manually.`;
  } finally { $('transcribeFileBtn').disabled = !state.selectedAudioFile; }
}
function useTranscript(analyze = false) {
  const text = $('transcriptText').value.trim();
  if (!text) { toast('There is no transcript to use.'); return; }
  $('sourceText').value = text; updateTextCount();
  if (analyze) { document.querySelector('#analyzer').scrollIntoView({ behavior: 'smooth' }); analyzeText(text); }
  else { toast('Transcript copied into the analyzer.'); document.querySelector('#analyzer').scrollIntoView({ behavior: 'smooth' }); }
}

// Events
$('sourceText').addEventListener('input', updateTextCount);
$('transcriptText').addEventListener('input', updateTranscriptUI);
$('analyzeBtn').addEventListener('click', () => analyzeText($('sourceText').value));
$('retryBtn').addEventListener('click', () => analyzeText(state.lastText || $('sourceText').value));
$('clearTextBtn').addEventListener('click', () => { $('sourceText').value = ''; updateTextCount(); $('sourceText').focus(); });
document.querySelectorAll('[data-example]').forEach(btn => btn.addEventListener('click', () => { $('sourceText').value = btn.dataset.example; updateTextCount(); $('sourceText').focus(); }));
$('copyReportBtn').addEventListener('click', () => state.lastReport && copyText(reportText()));
$('downloadReportBtn').addEventListener('click', () => state.lastReport && downloadBlob(new Blob([reportText()], { type: 'text/plain;charset=utf-8' }), 'signalshield-report.txt'));
$('saveReviewBtn').addEventListener('click', saveReview);
$('audioStartBtn').addEventListener('click', startRecording);
$('audioStopBtn').addEventListener('click', stopRecording);
$('audioDownloadBtn').addEventListener('click', downloadRecording);
$('audioAnalyzeBtn').addEventListener('click', () => useTranscript(true));
$('useTranscriptBtn').addEventListener('click', () => useTranscript(false));
$('analyzeTranscriptBtn').addEventListener('click', () => useTranscript(true));
$('clearTranscriptBtn').addEventListener('click', () => { $('transcriptText').value = ''; updateTranscriptUI(); });
$('audioFileInput').addEventListener('change', event => {
  const file = event.target.files?.[0]; state.selectedAudioFile = file || null;
  $('fileName').textContent = file ? `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MB` : 'No file selected';
  $('transcribeFileBtn').disabled = !file;
  if (file) {
    const url = URL.createObjectURL(file); $('fileAudioPreview').src = url; $('fileAudioPreview').hidden = false;
    $('asrStatus').textContent = 'File ready. Click “Transcribe audio with AI”. Keep clips under 3 minutes and 25 MB.';
  } else { $('fileAudioPreview').hidden = true; $('fileAudioPreview').removeAttribute('src'); }
});
$('transcribeFileBtn').addEventListener('click', transcribeFile);
$('themeBtn').addEventListener('click', () => {
  document.body.classList.toggle('light');
  $('themeBtn').textContent = document.body.classList.contains('light') ? '☾' : '☼';
  try { localStorage.setItem('signalshield_theme', document.body.classList.contains('light') ? 'light' : 'dark'); } catch {}
});
try { if (localStorage.getItem('signalshield_theme') === 'light') { document.body.classList.add('light'); $('themeBtn').textContent = '☾'; } } catch {}
window.addEventListener('beforeunload', () => {
  if (state.stream) state.stream.getTracks().forEach(track => track.stop());
  if (state.audioUrl) URL.revokeObjectURL(state.audioUrl);
});
updateTextCount(); updateTranscriptUI();
