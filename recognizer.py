import cv2
import boto3
import time

# Initialize Rekognition client
rekognition = boto3.client('rekognition', region_name='us-west-2')

# Open webcam (0 = default camera, 1 = external USB cam, etc.)
cap = cv2.VideoCapture(0)

frame_number = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from camera")
        break

    frame_number += 1

    # Encode frame to JPEG
    _, buffer = cv2.imencode('.jpg', frame)
    byte_frame = buffer.tobytes()

    # Call Rekognition for label detection
    response = rekognition.detect_labels(
        Image={'Bytes': byte_frame},
        MaxLabels=15,
        MinConfidence=10
    )

    # Print results for this frame
    print(f"\nFrame {frame_number}:")
    tongue_label = next(
        (label for label in response['Labels'] if label['Name'].lower() == 'tongue'),
        None
    )
    if tongue_label:
        print(f"Tongue: {tongue_label['Confidence']:.2f}%")
    else:
        print("Tongue not detected")

    # Show the live camera feed in a window
    cv2.imshow("Webcam", frame)

    # Wait 1 second between frames to control API usage
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty("Webcam", cv2.WND_PROP_VISIBLE) < 1:
        break
    time.sleep(0.5)  # ~1 fps (adjust/remove if you want faster)

cap.release()
cv2.destroyAllWindows()