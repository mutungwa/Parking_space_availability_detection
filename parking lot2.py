import cv2
import numpy as np

# Load a pre-trained car detector model
car_cascade = cv2.CascadeClassifier('cars.xml')  # Ensure this path is correct

# Define the regions of interest (ROIs) for each parking space
# Example format: [(x, y, w, h), (x, y, w, h), ...]
parking_spaces = [
    (115, 250, 95, 200),   # Example values, adjust accordingly
    (355, 250, 95, 200),
    (600, 250, 95, 200),
    # Add more parking spaces as needed
]

def detect_cars_in_roi(gray, roi):
    x, y, w, h = roi
    roi_gray = gray[y:y+h, x:x+w]
    cars = car_cascade.detectMultiScale(roi_gray, 1.1, 1)
    return len(cars) > 0  # Return True if any car is detected

def process_parking_lot_image(image_path):
    # Read the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    for roi in parking_spaces:
        occupied = detect_cars_in_roi(gray, roi)
        x, y, w, h = roi
        if occupied:
            color = (0, 0, 255)  # Red for occupied
            status = "Occupied"
        else:
            color = (0, 255, 0)  # Green for empty
            status = "Empty"
        
        # Draw the rectangle around the parking space and put the status text
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, status, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    # Display the result
    cv2.imshow('Parking Lot', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save the processed image
    output_path = image_path.replace('.jpg', '_processed.jpg')
    cv2.imwrite(output_path, img)

    print(f'Processed image saved as: {output_path}')

image_path = r'C:\Users\Administrator\Downloads\parking lot\p6.jpeg'
process_parking_lot_image(image_path)
