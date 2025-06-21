import cv2
import time

def test_camera():
    print("Testing camera...")
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera!")
        return
    
    print("Camera opened successfully!")
    
    # Create window
    cv2.namedWindow("Camera Test", cv2.WINDOW_AUTOSIZE)
    cv2.setWindowProperty("Camera Test", cv2.WND_PROP_TOPMOST, 1)
    
    print("Window created. Look for 'Camera Test' window...")
    time.sleep(2)
    
    # Read a few frames
    for i in range(100):
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
            
        # Add text to frame
        cv2.putText(frame, f"Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow("Camera Test", frame)
        
        # Wait for key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Test completed!")

if __name__ == "__main__":
    test_camera() 