import cv2

def detect_cars(video_path, cascade_path):
    # Load the video
    cap = cv2.VideoCapture(video_path)
    
    # Load the trained car cascade classifier
    car_cascade = cv2.CascadeClassifier(cascade_path)

    # Check if video capture is successful
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        # Read frames from the video
        ret, frame = cap.read()

        if not ret:
            break  # Break the loop if no frame is read

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect cars in the frame
        cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)

        # Draw rectangles around the detected cars
        for (x, y, w, h) in cars:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        # Display the frame
        cv2.imshow('Car Detection', frame)

        # Check for key press (ESC to exit)
        if cv2.waitKey(25) == 27:
            break

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Specify the paths to the video file and the cascade classifier
video_path = 'video1.avi'
cascade_path = 'cars.xml'

# Call the function to detect cars in the video
detect_cars(video_path, cascade_path)
