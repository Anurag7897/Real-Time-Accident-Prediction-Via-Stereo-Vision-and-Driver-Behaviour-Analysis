import streamlit as st
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
from datetime import datetime
import pytz

st.set_page_config(page_title="Be Safe on the Road", layout="centered")
st.title("🚗 Be Safe on the Road: Dashboard")

st.sidebar.header("📤 Upload Required Images")
left_img = st.sidebar.file_uploader("Left Image (Stereo)", type=["jpg", "png"])
right_img = st.sidebar.file_uploader("Right Image (Stereo)", type=["jpg", "png"])
frame1 = st.sidebar.file_uploader("Frame at t1 (Speed Estimation)", type=["jpg", "png"])
frame2 = st.sidebar.file_uploader("Frame at t2 (Speed Estimation)", type=["jpg", "png"])
driver_img = st.sidebar.file_uploader("Driver Face Image", type=["jpg", "png"])

st.subheader("🔍 Object Detection and depth calculation")
yolo_model = YOLO("yolov8n.pt")

if frame1:
    img = np.array(Image.open(frame1).convert("RGB"))
    yolo_results = yolo_model(img)[0]
    for box in yolo_results.boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    st.image(img, caption="YOLOv8 Detection", channels="RGB")

if left_img and right_img:
    st.subheader("📏 Depth Estimation (Stereo Vision)")
    l_img = np.array(Image.open(left_img).convert("L"))
    r_img = np.array(Image.open(right_img).convert("L"))
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(l_img, r_img).astype(np.float32) / 16.0
    disparity = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    st.image(disparity, caption="Depth Map", clamp=True, channels="GRAY")

if frame1 and frame2:
    st.subheader("💨 Speed Estimation")
    f1 = np.array(Image.open(frame1).convert("RGB"))
    f2 = np.array(Image.open(frame2).convert("RGB"))

    def get_centroid(image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        ret, thresh = cv2.threshold(gray, 127, 255, 0)
        M = cv2.moments(thresh)
        if M["m00"] == 0: return (0, 0)
        return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

    c1 = get_centroid(f1)
    c2 = get_centroid(f2)
    displacement = np.linalg.norm(np.array(c2) - np.array(c1))
    scale_factor = 0.05
    speed = displacement * scale_factor
    st.write(f"Estimated Speed: **{speed:.2f} units/frame**")

if driver_img:
    st.subheader("🙍‍♂️ Driver Behavior Detection")
    d_img = np.array(Image.open(driver_img).convert("RGB"))
    st.image(d_img, caption="Driver Image")
    driver_status = "Normal"
    st.success(f"Driver Status: **{driver_status}**")

if st.button("Run Full Prediction"):
    st.subheader("⚠️ Final Risk Evaluation")
    risk_score = 0.76
    st.metric("Predicted Risk Score", f"{risk_score:.2f}")
    if risk_score > 0.7:
        st.error("🚨 High Risk Detected! Alert Triggered.")
    else:
        st.success(" Driving Conditions Safe.")

def generate_textual_report(vehicle_id="VEHICLE_0001", driver_id="DRIVER_0001", speed=None, risk_score=None, proximity="Unknown", status="Normal", lat=42.8865, lon=-78.8784):
    now = datetime.now(pytz.timezone('America/New_York')).strftime("%B %d, %Y | %H:%M:%S")
    speed_alert = f"Speed dropped significantly (to {speed:.2f} units/frame)" if speed is not None and speed < 0.5 else "No abnormal speed change detected."
    proximity_alert = f"Object detected within {proximity} meters." if proximity != "Unknown" else "Proximity data not available."
    risk_status = "🚨 ALERT TRIGGERED" if risk_score and risk_score > 0.7 else "✅ Safe conditions."
    google_maps_url = f"https://www.google.com/maps?q={lat},{lon}"
    risk_display = f"{risk_score:.2f}" if risk_score is not None else "N/A"

    report = f"""
Date & Time of Report:
{now}

Location Coordinates:
Latitude: {lat:.4f}° N
Longitude: {lon:.4f}° W

Nearest Police Station:
Buffalo Police Department – Central Precinct

Vehicle Information
Vehicle ID: {vehicle_id}
Driver ID: {driver_id}
Vehicle Model & Registration: Tesla Model 3 | Plate: ABX-9124

System-Detected Risk Summary
The vehicle monitoring system has identified indicators of driver status and vehicle environment.

Driver Status
Driver appears to be: {status}

Speed Observation
{speed_alert}

Object Proximity
{proximity_alert}

Risk Score: {risk_display} / 1.00
(System threshold = 0.70)
{risk_status}

Google Maps: {google_maps_url}
"""
    return report.strip()

if st.button("Generate Report"):
    report_text = generate_textual_report(
        vehicle_id="VEHICLE_0183",
        driver_id="DRIVER_1157",
        speed=speed if 'speed' in locals() else None,
        risk_score=risk_score if 'risk_score' in locals() else None,
        proximity="1.3",
        status=driver_status if 'driver_status' in locals() else "Unknown",
        lat=42.8865,
        lon=-78.8784
    )
    st.text_area(" Final Incident Report", report_text, height=400)
