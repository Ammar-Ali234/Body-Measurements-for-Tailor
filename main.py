import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_distance(p1, p2):
    return np.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))

def pixels_to_cm(pixel_distance, calibration_factor=125):
    return pixel_distance * calibration_factor

def calculate_arm_length(shoulder, elbow, wrist):
    # Calculate upper arm (shoulder to elbow) and forearm (elbow to wrist) lengths
    upper_arm = calculate_distance(shoulder, elbow)
    forearm = calculate_distance(elbow, wrist)
    # Total arm length
    return pixels_to_cm(upper_arm + forearm)

def measure_body(image_path):
    image = cv2.imread(image_path)
    
    height, width = image.shape[:2]
    if width > 1280:
        scale = 1280 / width
        width = 1280
        height = int(height * scale)
        image = cv2.resize(image, (width, height))

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(
        static_image_mode=True,
        min_detection_confidence=0.5) as pose:

        results = pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            # Get all required landmarks
            # Shoulders
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            
            # Elbows
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            
            # Wrists
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            #shirt
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            waist_left = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            #Waist
            waist_right = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            waist_left = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

            # Calculate measurements
            # Shoulder width
            shoulder_distance_pixels = calculate_distance(left_shoulder, right_shoulder)
            shoulder_distance_cm = pixels_to_cm(shoulder_distance_pixels)

            # Chest width (using offset from shoulders)
            left_chest = [left_shoulder[0], left_shoulder[1] + 0.1]
            right_chest = [right_shoulder[0], right_shoulder[1] + 0.1]
            chest_distance_pixels = calculate_distance(left_chest, right_chest)
            chest_distance_cm = pixels_to_cm(chest_distance_pixels * 0.8)

            # Arm lengths
            left_arm_cm = calculate_arm_length(left_shoulder, left_elbow, left_wrist)
            right_arm_cm = calculate_arm_length(right_shoulder, right_elbow, right_wrist)

            #Waist
            waist_pixels = calculate_distance(waist_left, waist_right)
            waist_cm = pixels_to_cm(waist_pixels*1.4)

            #shirt
            shirt_pixels = calculate_distance(left_shoulder, waist_left)
            shirt_cm = pixels_to_cm(shirt_pixels)
            # Draw measurements on image
            # Draw pose landmarks
            mp_drawing.draw_landmarks(
                image_bgr, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=1)
            )

            # Convert landmark points to pixel coordinates
            def to_pixel_coords(point):
                return tuple(np.multiply(point, [width, height]).astype(int))

            # Draw lines for measurements
            # Shoulder line
            cv2.line(image_bgr, to_pixel_coords(left_shoulder), to_pixel_coords(right_shoulder), (0, 255, 0), 2)
            
            # Chest line
            cv2.line(image_bgr, to_pixel_coords(left_chest), to_pixel_coords(right_chest), (0, 255, 255), 2)

            #Waist
            cv2.line(image_bgr, to_pixel_coords(waist_right), to_pixel_coords(waist_left), (0, 255, 255), 2)

            #shirt
            cv2.line(image_bgr, to_pixel_coords(left_shoulder), to_pixel_coords(waist_left), (0, 128, 255), 2)
            
            # Left arm lines
            cv2.line(image_bgr, to_pixel_coords(left_shoulder), to_pixel_coords(left_elbow), (255, 0, 0), 2)
            cv2.line(image_bgr, to_pixel_coords(left_elbow), to_pixel_coords(left_wrist), (255, 0, 0), 2)
            
            # Right arm lines
            cv2.line(image_bgr, to_pixel_coords(right_shoulder), to_pixel_coords(right_elbow), (0, 0, 255), 2)
            cv2.line(image_bgr, to_pixel_coords(right_elbow), to_pixel_coords(right_wrist), (0, 0, 255), 2)

            # Display measurements on image
            measurements = [
                (f"Shoulder: {shoulder_distance_cm:.1f} cm", (int(width/1.5) + 100, 30), (0, 255, 0)),
                (f"Chest: {chest_distance_cm:.1f} cm", (int(width/1.5) + 100, 60), (0, 255, 255)),
                (f"Left Arm: {left_arm_cm:.1f} cm", (int(width/1.5) + 100, 90), (255, 0, 0)),
                (f"Right Arm: {right_arm_cm:.1f} cm", (int(width/1.5) + 100, 120), (0, 0, 255)),
                (f"shirt length: {shirt_cm:.1f} cm", (int(width/1.5) + 100, 150), (0, 128, 255)),
                (f"Waist: {waist_cm:.1f} cm", (int(width/1.5) + 100, 180), (150, 128, 255)),
            ]

            for text, position, color in measurements:
                cv2.putText(image_bgr, text, position,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Print measurements to terminal
            print(f"Shoulder width: {shoulder_distance_cm:.1f} cm")
            print(f"Chest width: {chest_distance_cm:.1f} cm")
            print(f"Left arm length: {left_arm_cm:.1f} cm")
            print(f"Right arm length: {right_arm_cm:.1f} cm")
            print(f"shirt lenght: {shirt_cm:.1f} cm")
            print(f"Waist: {waist_cm:.1f} cm")

            # Show image
            cv2.imshow("Body Measurements", image_bgr)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # Save the annotated image
            output_path = "output_" + image_path.split("/")[-1]
            cv2.imwrite(output_path, image_bgr)
            print(f"Annotated image saved as: {output_path}")

        else:
            print("No pose landmarks detected in the image.")

# Example usage
image_path = "myself.jpg"  # Replace with your image path
measure_body(image_path)