import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

import shutil
import re
from datetime import datetime
from flask import Flask, render_template, request, send_file, jsonify, abort
from werkzeug.utils import secure_filename
import librosa as lb
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import rdc_model
import ai_assistant

# ── Firebase Admin ──────────────────────────────────────────────────
# Supports two credential modes:
#   1. FIREBASE_SERVICE_ACCOUNT_JSON env var → JSON string of the key (for cloud deploys)
#   2. File at backend/serviceAccountKey.json (for local dev)
import json as _json
import tempfile as _tempfile
import firebase_admin
from firebase_admin import credentials, firestore

if not firebase_admin._apps:
    _sa_json_str = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
    if _sa_json_str:
        # Cloud deployment: load credentials from JSON string env var
        _sa_dict = _json.loads(_sa_json_str)
        _cred = credentials.Certificate(_sa_dict)
    else:
        # Local development: load from file
        _sa_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_KEY",
                             os.path.join(os.path.dirname(__file__), "serviceAccountKey.json"))
        if not os.path.isabs(_sa_path):
            _abs1 = os.path.abspath(_sa_path)
            _abs2 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", _sa_path))
            _sa_path = _abs1 if os.path.exists(_abs1) else _abs2
        _cred = credentials.Certificate(_sa_path)
    firebase_admin.initialize_app(_cred)

db = firestore.client()
COLLECTION = "analyses"


root_folder = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER_temp = os.path.join(root_folder, "static")
UPLOAD_FOLDER = os.path.join(UPLOAD_FOLDER_temp, "uploads")
PATIENT_AUDIO_FOLDER = os.path.join(UPLOAD_FOLDER_temp, "patient_audio")
PATIENT_REPORTS_FOLDER = os.path.join(UPLOAD_FOLDER_temp, "patient_reports")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.context_processor
def inject_firebase_config():
    return {
        'FIREBASE_API_KEY': os.getenv('FIREBASE_API_KEY', ''),
        'FIREBASE_AUTH_DOMAIN': os.getenv('FIREBASE_AUTH_DOMAIN', ''),
        'FIREBASE_PROJECT_ID': os.getenv('FIREBASE_PROJECT_ID', ''),
        'FIREBASE_STORAGE_BUCKET': os.getenv('FIREBASE_STORAGE_BUCKET', ''),
        'FIREBASE_MESSAGING_SENDER_ID': os.getenv('FIREBASE_MESSAGING_SENDER_ID', ''),
        'FIREBASE_APP_ID': os.getenv('FIREBASE_APP_ID', ''),
    }

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PATIENT_AUDIO_FOLDER, exist_ok=True)
os.makedirs(PATIENT_REPORTS_FOLDER, exist_ok=True)


# ── Firestore helpers ───────────────────────────────────────────────
def save_analysis_firestore(patient_name, prediction, confidence, audio_filename, report_filename, doctor_uid=None):
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    db.collection(COLLECTION).add({
        "patient_name": patient_name,
        "date": date_str,
        "prediction": prediction,
        "confidence": confidence,
        "audio_file_path": audio_filename,
        "report_path": report_filename,
        "doctor_uid": doctor_uid or "",
        "created_at": firestore.SERVER_TIMESTAMP
    })


def clean_prediction(raw):
    """Return (disease_name, probability_pct) from the verbose model string."""
    import re as _re
    s = str(raw)
    # Extract percentage
    pct_match = _re.search(r'([\d.]+)\s*%', s)
    probability = round(float(pct_match.group(1)), 1) if pct_match else 0.0
    # Strip prefixes and probability from name
    s = _re.sub(r'respiratory\s+disorder\s+detected\s*:?', '', s, flags=_re.I)
    s = _re.sub(r'predicted?\s+disorder\s+detected\s*:?', '', s, flags=_re.I)
    s = _re.sub(r'prediction\s*:?', '', s, flags=_re.I)
    s = _re.sub(r'with\s+probability', '', s, flags=_re.I)
    s = _re.sub(r'[\d.]+\s*%', '', s)
    s = _re.sub(r'[,;]?\s*second.*', '', s, flags=_re.I)
    return s.strip(), probability


# ── Routes ──────────────────────────────────────────────────────────

@app.route("/login")
def login():
    return render_template("login.html")


@app.route("/")
def index():
    for f in os.listdir(UPLOAD_FOLDER):
        os.remove(os.path.join(UPLOAD_FOLDER, f))
    return render_template("index.html", ospf=1)


@app.route("/", methods=['POST'])
def analyse():
    plt.close('all')

    name = request.form.get("name", "Unknown")
    doctor_uid = request.form.get("doctor_uid", "")
    lungSounds = request.files["lungSounds"]
    filename = secure_filename(lungSounds.filename)

    # Save a permanent copy for history audio playback
    patient_audio_path = os.path.join(PATIENT_AUDIO_FOLDER, filename)
    lungSounds.save(patient_audio_path)

    # Copy to uploads for immediate display
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    shutil.copy(patient_audio_path, upload_path)

    url2 = os.path.join("static", "uploads")
    url = os.path.join(url2, filename)
    absolute_url = os.path.abspath(url)

    res_list = rdc_model.classificationResults(absolute_url)

    # Parse prediction and confidence
    prediction = str(res_list[0]) if res_list else "Unknown"
    disease_name, probability = clean_prediction(prediction)
    confidence = probability  # use the actual model probability

    audio1, sample_rate1 = lb.load(url, mono=True)

    # ── Graph 1: Waveform ────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(8, 3))
    fig1.patch.set_facecolor('#0B0B18'); ax1.set_facecolor('#15152A')
    librosa.display.waveshow(audio1, sr=sample_rate1, ax=ax1, color='#A78BFA',
                              max_points=50000, x_axis='time')
    ax1.set_xlabel('Time (s)', color='#7C6F9F'); ax1.set_ylabel('Amplitude', color='#7C6F9F')
    ax1.tick_params(colors='#7C6F9F')
    for spine in ax1.spines.values(): spine.set_edgecolor('#1C1C35')
    plt.tight_layout()
    fig1.savefig("./static/uploads/outSoundWave.png", dpi=120, facecolor='#0B0B18', bbox_inches='tight')
    plt.close(fig1)

    # ── Graph 2: MFCC Spectrogram ────────────────────────────────────
    mfccs = lb.feature.mfcc(y=audio1, sr=sample_rate1, n_mfcc=40)
    fig2, ax2 = plt.subplots(figsize=(8, 3))
    fig2.patch.set_facecolor('#0B0B18'); ax2.set_facecolor('#15152A')
    img2 = librosa.display.specshow(mfccs, x_axis='time', ax=ax2, sr=sample_rate1, cmap='magma')
    fig2.colorbar(img2, ax=ax2)
    ax2.set_xlabel('Time (s)', color='#7C6F9F'); ax2.set_ylabel('MFCC Coefficients', color='#7C6F9F')
    ax2.tick_params(colors='#7C6F9F')
    for spine in ax2.spines.values(): spine.set_edgecolor('#1C1C35')
    plt.tight_layout()
    fig2.savefig("./static/uploads/outSoundMFCC.png", dpi=120, facecolor='#0B0B18', bbox_inches='tight')
    plt.close(fig2)

    # ── Graph 3: Mel Spectrogram ─────────────────────────────────────
    mel_spec = lb.feature.melspectrogram(y=audio1, sr=sample_rate1, n_mels=128)
    mel_spec_db = lb.power_to_db(mel_spec, ref=np.max)
    fig3, ax3 = plt.subplots(figsize=(8, 3))
    fig3.patch.set_facecolor('#0B0B18'); ax3.set_facecolor('#15152A')
    img3 = librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel',
                                     ax=ax3, sr=sample_rate1, cmap='inferno')
    fig3.colorbar(img3, ax=ax3, format='%+2.0f dB')
    ax3.set_xlabel('Time (s)', color='#7C6F9F'); ax3.set_ylabel('Mel Frequency', color='#7C6F9F')
    ax3.tick_params(colors='#7C6F9F')
    for spine in ax3.spines.values(): spine.set_edgecolor('#1C1C35')
    plt.tight_layout()
    fig3.savefig("./static/uploads/outSoundMel.png", dpi=120, facecolor='#0B0B18', bbox_inches='tight')

    # ── Generate PDF Report ──────────────────────────────────────────
    from matplotlib.backends.backend_pdf import PdfPages
    report_filename = f"report_{os.path.splitext(filename)[0]}.pdf"
    report_path = os.path.join(PATIENT_REPORTS_FOLDER, report_filename)

    with PdfPages(report_path) as pdf:

        # ── Cover Page ───────────────────────────────────────────────
        fig0, ax0 = plt.subplots(figsize=(8.27, 11.69))  # A4 portrait
        fig0.patch.set_facecolor('#F8FAFF')
        ax0.axis('off')

        # Header bar — span 0.80 to 1.0, center at 0.90
        ax0.axhspan(0.80, 1.0, color='#059669', alpha=1.0)
        ax0.text(0.5, 0.935, 'Respiratory AI', transform=ax0.transAxes,
                 ha='center', va='center', fontsize=26, fontweight='bold',
                 color='white', fontfamily='DejaVu Sans')
        ax0.text(0.5, 0.870, 'Smart Diagnostics — Respiratory Analysis Report',
                 transform=ax0.transAxes, ha='center', va='center',
                 fontsize=11, color=(1, 1, 1, 0.82),
                 fontstyle='italic', fontfamily='DejaVu Sans')

        # Divider
        ax0.axhline(0.80, color='#6EE7B7', linewidth=0.8)

        report_date = datetime.now().strftime("%d %B %Y, %I:%M %p")

        # Patient info block — start lower to give header breathing room
        info = [
            ('Patient Name',    name),
            ('Diagnosis',       disease_name),
            ('Probability',     f'{probability:.1f}%'),
            ('Analysis Date',   report_date),
            ('Audio File',      filename),
        ]
        y = 0.76
        for label, value in info:
            ax0.text(0.10, y, label + ':',   transform=ax0.transAxes,
                     ha='left', va='top', fontsize=11, fontweight='bold',
                     color='#374151', fontfamily='DejaVu Sans')
            ax0.text(0.40, y, value,         transform=ax0.transAxes,
                     ha='left', va='top', fontsize=11,
                     color='#111827', fontfamily='DejaVu Sans')
            y -= 0.075

        # Probability score visual bar
        ax0.text(0.10, y - 0.01, 'Probability Score:', transform=ax0.transAxes,
                 ha='left', va='top', fontsize=10, fontweight='bold', color='#374151')
        bar_x, bar_y, bar_w, bar_h = 0.10, y - 0.065, 0.80, 0.030
        fill_w = bar_w * max(0.001, probability / 100)
        ax0.add_patch(plt.Rectangle((bar_x, bar_y), bar_w, bar_h,
                                    transform=ax0.transAxes, color='#E5E7EB', zorder=2))
        ax0.add_patch(plt.Rectangle((bar_x, bar_y), fill_w, bar_h,
                                    transform=ax0.transAxes, color='#059669', zorder=3))
        # Label centered inside the full bar
        ax0.text(bar_x + bar_w / 2, bar_y + bar_h / 2,
                 f'{probability:.1f}%', transform=ax0.transAxes,
                 va='center', ha='center', fontsize=10, color='white', fontweight='bold', zorder=4)

        # Footer
        ax0.text(0.5, 0.04, 'Generated by Respiratory AI · For clinical reference only',
                 transform=ax0.transAxes, ha='center', va='bottom',
                 fontsize=8, color='#9CA3AF', fontstyle='italic')
        ax0.axhline(0.06, color='#E5E7EB', linewidth=0.6)

        pdf.savefig(fig0, facecolor='#F8FAFF', bbox_inches='tight')
        plt.close(fig0)

        # ── Spectrograms ─────────────────────────────────────────────
        pdf.savefig(fig1, facecolor='#0B0B18')
        pdf.savefig(fig2, facecolor='#0B0B18')
        pdf.savefig(fig3, facecolor='#0B0B18')

    plt.close('all')

    # ── Save to Firestore ────────────────────────────────────────────
    try:
        save_analysis_firestore(name, prediction, confidence, filename, report_filename, doctor_uid)
    except Exception as e:
        print(f"[Firestore] Warning: Could not save analysis — {e}")

    return render_template("index.html", ospf=0, n=name, lungSounds=url, res=res_list)


@app.route("/history")
def history():
    return render_template("history.html")


@app.route("/history/audio/<filename>")
def patient_audio(filename):
    safe_name = secure_filename(filename)
    audio_path = os.path.join(PATIENT_AUDIO_FOLDER, safe_name)
    if not os.path.exists(audio_path):
        abort(404)
    return send_file(audio_path, mimetype="audio/wav")


@app.route("/history/report/<filename>")
def patient_report(filename):
    safe_name = secure_filename(filename)
    report_path = os.path.join(PATIENT_REPORTS_FOLDER, safe_name)
    if not os.path.exists(report_path):
        abort(404)
    dl = request.args.get('download') == '1'
    return send_file(report_path, mimetype="application/pdf", as_attachment=dl)


# ── Dynamic report generator (works for ALL Firestore records) ──────
@app.route("/history/report/generate/<doc_id>")
def generate_patient_report(doc_id):
    """Generate a fresh PDF from Firestore data — works for old and new records."""
    import io
    from matplotlib.backends.backend_pdf import PdfPages

    # ── Fetch from Firestore ─────────────────────────────────────────
    try:
        doc = db.collection(COLLECTION).document(doc_id).get()
        if not doc.exists:
            abort(404)
        data = doc.to_dict()
    except Exception as e:
        print(f"[Generate Report] Firestore error: {e}")
        abort(500)

    patient_name  = data.get('patient_name', 'Unknown')
    prediction    = data.get('prediction', '')
    date_str      = data.get('date', datetime.now().strftime("%Y-%m-%d %H:%M"))
    audio_fname   = data.get('audio_file_path', '')

    disease_name, probability = clean_prediction(prediction)

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:

        # ── Cover Page ───────────────────────────────────────────────
        fig0, ax0 = plt.subplots(figsize=(8.27, 11.69))
        fig0.patch.set_facecolor('#F8FAFF')
        ax0.axis('off')

        # Header bar — span 0.80 to 1.0, center at 0.90
        ax0.axhspan(0.80, 1.0, color='#059669', alpha=1.0)
        ax0.text(0.5, 0.935, 'Respiratory AI', transform=ax0.transAxes,
                 ha='center', va='center', fontsize=28, fontweight='bold',
                 color='white', fontfamily='DejaVu Sans')
        ax0.text(0.5, 0.870, 'Smart Diagnostics — Respiratory Analysis Report',
                 transform=ax0.transAxes, ha='center', va='center',
                 fontsize=12, color=(1, 1, 1, 0.82),
                 fontstyle='italic', fontfamily='DejaVu Sans')

        ax0.axhline(0.80, color='#6EE7B7', linewidth=0.8)

        info = [
            ('Patient Name',  patient_name),
            ('Diagnosis',     disease_name),
            ('Probability',   f'{probability:.1f}%'),
            ('Analysis Date', date_str),
            ('Audio File',    audio_fname),
        ]
        y = 0.76
        for label, value in info:
            ax0.text(0.10, y, label + ':',
                     transform=ax0.transAxes, ha='left', va='top',
                     fontsize=12, fontweight='bold', color='#374151',
                     fontfamily='DejaVu Sans')
            ax0.text(0.40, y, value,
                     transform=ax0.transAxes, ha='left', va='top',
                     fontsize=12, color='#111827', fontfamily='DejaVu Sans')
            y -= 0.08

        # Probability bar
        y -= 0.02
        ax0.text(0.10, y, 'Probability Score:',
                 transform=ax0.transAxes, ha='left', va='top',
                 fontsize=10, fontweight='bold', color='#374151')
        bx, by, bw, bh = 0.10, y - 0.055, 0.80, 0.030
        fill_w = bw * max(0.001, probability / 100)
        ax0.add_patch(plt.Rectangle((bx, by), bw, bh,
                                    transform=ax0.transAxes,
                                    color='#E5E7EB', zorder=2))
        ax0.add_patch(plt.Rectangle((bx, by), fill_w, bh,
                                    transform=ax0.transAxes,
                                    color='#059669', zorder=3))
        # Label centered inside the full bar
        ax0.text(bx + bw / 2, by + bh / 2,
                 f'{probability:.1f}%',
                 transform=ax0.transAxes, va='center', ha='center',
                 fontsize=10, color='white', fontweight='bold', zorder=4)

        ax0.axhline(0.06, color='#E5E7EB', linewidth=0.6)
        ax0.text(0.5, 0.035,
                 'Generated by Respiratory AI  ·  For clinical reference only',
                 transform=ax0.transAxes, ha='center', va='bottom',
                 fontsize=8, color='#9CA3AF', fontstyle='italic')

        pdf.savefig(fig0, facecolor='#F8FAFF', bbox_inches='tight')
        plt.close(fig0)

        # ── Spectrograms (regenerate from stored audio if available) ─
        audio_path = os.path.join(PATIENT_AUDIO_FOLDER, secure_filename(audio_fname))
        if audio_fname and os.path.exists(audio_path):
            try:
                audio1, sr1 = lb.load(audio_path, mono=True)

                fig1, ax1 = plt.subplots(figsize=(8, 3))
                fig1.patch.set_facecolor('#0B0B18'); ax1.set_facecolor('#15152A')
                librosa.display.waveshow(audio1, sr=sr1, ax=ax1, color='#A78BFA',
                                         max_points=50000, x_axis='time')
                ax1.set_xlabel('Time (s)', color='#7C6F9F')
                ax1.set_ylabel('Amplitude', color='#7C6F9F')
                ax1.tick_params(colors='#7C6F9F')
                ax1.set_title(f'Waveform  —  {patient_name}  |  {disease_name}  ({probability:.1f}%)',
                              color='#C4B5FD', fontsize=9, pad=6)
                for sp in ax1.spines.values(): sp.set_edgecolor('#1C1C35')
                plt.tight_layout()
                pdf.savefig(fig1, facecolor='#0B0B18', bbox_inches='tight')
                plt.close(fig1)

                mfccs = lb.feature.mfcc(y=audio1, sr=sr1, n_mfcc=40)
                fig2, ax2 = plt.subplots(figsize=(8, 3))
                fig2.patch.set_facecolor('#0B0B18'); ax2.set_facecolor('#15152A')
                img2 = librosa.display.specshow(mfccs, x_axis='time',
                                                ax=ax2, sr=sr1, cmap='magma')
                fig2.colorbar(img2, ax=ax2)
                ax2.set_xlabel('Time (s)', color='#7C6F9F')
                ax2.set_ylabel('MFCC Coefficients', color='#7C6F9F')
                ax2.tick_params(colors='#7C6F9F')
                ax2.set_title(f'MFCC Spectrogram  —  {patient_name}',
                              color='#C4B5FD', fontsize=9, pad=6)
                for sp in ax2.spines.values(): sp.set_edgecolor('#1C1C35')
                plt.tight_layout()
                pdf.savefig(fig2, facecolor='#0B0B18', bbox_inches='tight')
                plt.close(fig2)

                mel = lb.feature.melspectrogram(y=audio1, sr=sr1, n_mels=128)
                mel_db = lb.power_to_db(mel, ref=np.max)
                fig3, ax3 = plt.subplots(figsize=(8, 3))
                fig3.patch.set_facecolor('#0B0B18'); ax3.set_facecolor('#15152A')
                img3 = librosa.display.specshow(mel_db, x_axis='time', y_axis='mel',
                                                ax=ax3, sr=sr1, cmap='inferno')
                fig3.colorbar(img3, ax=ax3, format='%+2.0f dB')
                ax3.set_xlabel('Time (s)', color='#7C6F9F')
                ax3.set_ylabel('Mel Frequency', color='#7C6F9F')
                ax3.tick_params(colors='#7C6F9F')
                ax3.set_title(f'Mel Spectrogram  —  {patient_name}',
                              color='#C4B5FD', fontsize=9, pad=6)
                for sp in ax3.spines.values(): sp.set_edgecolor('#1C1C35')
                plt.tight_layout()
                pdf.savefig(fig3, facecolor='#0B0B18', bbox_inches='tight')
                plt.close(fig3)

            except Exception as e:
                print(f"[Generate Report] Audio processing error: {e}")

    plt.close('all')
    buf.seek(0)

    dl = request.args.get('download') == '1'
    safe_pt = patient_name.replace(' ', '_').replace('/', '_')
    safe_dis = disease_name.replace(' ', '_')
    fname = f"RespiratoryAI_{safe_pt}_{safe_dis}_{probability:.0f}pct.pdf"
    return send_file(buf, mimetype="application/pdf",
                     as_attachment=dl,
                     download_name=fname)


# ── AI Assistant endpoint ────────────────────────────────────────────
@app.route("/doctor-ai", methods=["POST"])
def doctor_ai():
    """
    POST /doctor-ai
    Body: { "question": "...", "doctor_email": "..." }
    Returns: { "answer": "...", "chart": {"type": "pie|bar", "data": {...}} | null }
    """
    body = request.get_json(silent=True) or {}
    question = (body.get("question") or "").strip()
    doctor_email = (body.get("doctor_email") or "").strip()

    if not question or not doctor_email:
        return jsonify({"error": "Missing question or doctor_email"}), 400

    # ── Fetch this doctor's records ───────────────────────────────────
    try:
        snap = db.collection(COLLECTION)\
                 .where("doctor_uid", "==", doctor_email)\
                 .get()
        raw_records = [doc.to_dict() for doc in snap]
    except Exception as e:
        print(f"[doctor-ai] Firestore error: {e}")
        return jsonify({"error": "Could not fetch records from Firestore."}), 500

    # ── Process records ───────────────────────────────────────────────
    records = ai_assistant.process_records(raw_records)

    # ── Run smart analytics ───────────────────────────────────────────
    try:
        answer, chart = ai_assistant.smart_answer(question, records)
    except Exception as e:
        print(f"[doctor-ai] Analytics error: {e}")
        answer = "Sorry, I encountered an error processing your question."
        chart = None

    return jsonify({"answer": answer, "chart": chart})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)