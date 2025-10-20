from flask import Flask, render_template, request, redirect, url_for, flash
import os
import io
import base64
from PIL import Image
import numpy as np
import cv2
import pytesseract
import imutils
import geocoder
from twilio.rest import Client

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev_secret_key")

# Twilio credentials - use environment variables in production
ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")
FROM_PHONE = os.environ.get("TWILIO_FROM_PHONE", "")

# Initialize global variables (keeps simple state for demo)
TARGET_PLATE_NUMBERS = []
RECIPIENT_NAME = ""
RECIPIENT_PHONE_NUMBER = ""

# Optionally set pytesseract path through env var TESSERACT_CMD
if os.environ.get("TESSERACT_CMD"):
    pytesseract.pytesseract.tesseract_cmd = os.environ.get("TESSERACT_CMD")


def send_sms(message, to_phone):
    """Send SMS using Twilio (no-op if credentials missing)"""
    if not (ACCOUNT_SID and AUTH_TOKEN and FROM_PHONE):
        app.logger.warning("Twilio credentials are not set. Skipping SMS.")
        return None
    client = Client(ACCOUNT_SID, AUTH_TOKEN)
    msg = client.messages.create(to=to_phone, from_=FROM_PHONE, body=message)
    app.logger.info(f"Message sent: {msg.sid}")
    return msg.sid


def detect_license_plate_from_image(image):
    """
    Detect license plate-like rectangular regions and run OCR.
    Returns list of detected text strings.
    Image expected as a BGR (OpenCV) numpy array.
    """
    image = imutils.resize(image, width=600)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 150)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

    detected_plates = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.018 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:  # likely rectangle -> candidate plate
            x, y, w, h = cv2.boundingRect(approx)
            if w < 50 or h < 10:
                continue
            license_plate = gray[y:y + h, x:x + w]
            # optional preprocessing for OCR
            _, thresh = cv2.threshold(license_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            plate_text = pytesseract.image_to_string(thresh, config='--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            plate_text = plate_text.strip()
            if plate_text:
                detected_plates.append(plate_text)
    # unique and return
    detected_unique = []
    for p in detected_plates:
        if p not in detected_unique:
            detected_unique.append(p)
    return detected_unique


def get_location():
    """Get approximate location using IP-based geocoder"""
    try:
        g = geocoder.ip('me')
        if g.latlng:
            latitude = g.latlng[0]
            longitude = g.latlng[1]
            place_name = g.city if g.city else "Unknown place"
        else:
            latitude = longitude = "Unknown"
            place_name = "Unknown place"
    except Exception:
        latitude = longitude = "Unknown"
        place_name = "Unknown place"
    return latitude, longitude, place_name


@app.route('/')
def login():
    """Page to set target plate(s) and recipient info"""
    return render_template('login.html')


@app.route('/set_target', methods=['POST'])
def set_target():
    global TARGET_PLATE_NUMBERS, RECIPIENT_NAME, RECIPIENT_PHONE_NUMBER
    plate_numbers = request.form.get('plate_number', '').strip()
    phone_number = request.form.get('phone_number', '').strip()
    name = request.form.get('name', '').strip()

    if not plate_numbers or not phone_number or not name:
        flash("Please enter the license plate numbers, phone number, and name.", 'error')
        return redirect(url_for('login'))

    TARGET_PLATE_NUMBERS = [p.strip().upper().replace(" ", "") for p in plate_numbers.split(",") if p.strip()]
    RECIPIENT_NAME = name
    RECIPIENT_PHONE_NUMBER = phone_number
    flash(f"Target Plates: {', '.join(TARGET_PLATE_NUMBERS)} set for {RECIPIENT_NAME}.", 'success')

    return redirect(url_for('detect_license_plate'))


@app.route('/detect_license_plate')
def detect_license_plate():
    """Render detection page (webcam + upload)"""
    return render_template('detect_license_plate.html', targets=TARGET_PLATE_NUMBERS, recipient=RECIPIENT_NAME)


def _read_image_from_base64(data_url):
    """Convert a data URL (base64) to OpenCV BGR image"""
    header, encoded = data_url.split(",", 1)
    data = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(data)).convert("RGB")
    arr = np.array(image)
    # PIL gives RGB, OpenCV uses BGR
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _read_image_from_file_storage(file_storage):
    """Read Flask FileStorage into OpenCV BGR image"""
    data = file_storage.read()
    image = Image.open(io.BytesIO(data)).convert("RGB")
    arr = np.array(image)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


@app.route('/upload_image', methods=['POST'])
def upload_image():
    """
    Accepts an uploaded file (multipart/form-data 'image_file')
    or a JSON/form field 'image_data' with a data URL from canvas.
    Runs plate detection, checks matches, sends SMS if match found.
    """
    image = None
    # prefer uploaded file
    if 'image_file' in request.files and request.files['image_file'].filename:
        try:
            image = _read_image_from_file_storage(request.files['image_file'])
        except Exception as e:
            flash(f"Failed to read uploaded image: {e}", 'error')
            return redirect(url_for('detect_license_plate'))
    else:
        image_data = request.form.get('image_data', '') or request.json.get('image_data') if request.is_json else ''
        if image_data:
            try:
                image = _read_image_from_base64(image_data)
            except Exception as e:
                flash(f"Failed to decode image data: {e}", 'error')
                return redirect(url_for('detect_license_plate'))

    if image is None:
        flash("No image received.", "error")
        return redirect(url_for('detect_license_plate'))

    detected = detect_license_plate_from_image(image)
    # normalize detected strings
    detected_norm = [d.upper().replace(" ", "") for d in detected]

    # check for matches
    matches = []
    for d, dn in zip(detected, detected_norm):
        for target in TARGET_PLATE_NUMBERS:
            if target and target == dn:
                matches.append(d)
    latitude, longitude, place_name = get_location()

    sms_sid = None
    if matches and RECIPIENT_PHONE_NUMBER:
        message = f"Alert: Detected plate(s) {', '.join(matches)} at {place_name} (lat: {latitude}, lon: {longitude})"
        sms_sid = send_sms(message, RECIPIENT_PHONE_NUMBER)
        flash("Match found. Notification sent (if Twilio configured).", "success")
    elif matches:
        flash("Match found but recipient phone not configured.", "warning")
    else:
        flash("No matching plates found.", "info")

    return render_template('result.html',
                           detected=detected,
                           matches=matches,
                           latitude=latitude,
                           longitude=longitude,
                           place_name=place_name,
                           sms_sid=sms_sid)


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
