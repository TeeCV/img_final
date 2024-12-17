import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageChops
from torchvision.models import vgg16
from torchvision import transforms
import matplotlib.pyplot as plt
import mediapipe as mp

class ChessboardFrameExtractorV3:
    def __init__(self, video_path, output_folder, frame_interval=180):
        self.video_path = video_path
        self.output_folder = output_folder
        self.frame_interval = frame_interval

        # Initialize VGG16 model for feature extraction
        # self.model = vgg16(pretrained=True).features.eval()
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available
        # self.model = self.model.to(self.device)

        # # Define image transformations
        # self.transform = transforms.Compose([
        #     transforms.Resize((224, 224)),  # Resize to VGG16 input size
        #     transforms.ToTensor(),         # Convert image to tensor
        #     transforms.Normalize(          # Normalize with ImageNet mean and std
        #         mean=[0.485, 0.456, 0.406],
        #         std=[0.229, 0.224, 0.225]
        #     )
        # ])

        self.mp_hands = mp.solutions.hands

        os.makedirs(output_folder, exist_ok=True)

    def extract_features(self, frame):
        """Extract deep features from the frame using VGG16."""
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert to PIL format
        image = self.transform(image).unsqueeze(0).to(self.device)  # Transform and send to GPU
        with torch.no_grad():
            features = self.model(image)
        return features.view(-1).cpu().numpy()  # Flatten and move to CPU

    def robust_compare_chessboards(self, image1, image2):
        # Load images
        # image1 = cv2.imread(image1_path)
        # image2 = cv2.imread(image2_path)

        # Resize images to ensure alignment
        height, width = image1.shape[:2]
        image2 = cv2.resize(image2, (width, height))

        # Convert to grayscale
        gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)

        # Apply Gaussian blur to reduce noise
        gray1_blurred = cv2.GaussianBlur(gray1, (5, 5), 0)
        gray2_blurred = cv2.GaussianBlur(gray2, (5, 5), 0)

        # Compute absolute difference
        diff = cv2.absdiff(gray1_blurred, gray2_blurred)

        # Threshold the difference to ignore small changes
        _, diff_thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)

        # Count non-zero pixels in the thresholded difference
        diff_score = cv2.countNonZero(diff_thresh)

        return 5000 < diff_score < 19_000

    def is_hand_in_image(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        with self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5
        ) as hands:
            results = hands.process(image_rgb)
            return results.multi_hand_landmarks is not None

    def detect_and_crop_chessboard(self, frame):
        height, width = frame.shape[:2]
        border_x = int(width * 0.05)
        border_y = int(height * 0.05)
        image_noborder = frame[border_y:height-border_y, border_x:width-border_x]
        gray = cv2.cvtColor(image_noborder, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        chessboard_contour = None
        for contour in contours[:5]:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h
                if 0.8 <= aspect_ratio <= 1.2:
                    chessboard_contour = approx
                    break

        if chessboard_contour is not None:
            x, y, w, h = cv2.boundingRect(chessboard_contour)
            padding = 5
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image_noborder.shape[1] - x, w + 2 * padding)
            h = min(image_noborder.shape[0] - y, h + 2 * padding)
            cropped = image_noborder[y:y+h, x:x+w]
            return cropped, True
        return None, False

    def is_significant_change(self, current_frame, previous_frame, pixel_threshold=10_000):
        if previous_frame is None:
            return True

        # Convert frames to PIL Images
        current_image = Image.fromarray(cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB))
        previous_image = Image.fromarray(cv2.cvtColor(previous_frame, cv2.COLOR_BGR2RGB))

        # Compute the absolute difference
        difference = ImageChops.difference(current_image, previous_image)

        # Threshold and count non-zero pixels
        nonzero_pixels = sum(difference.convert("L").point(lambda p: p > 50 and 1).getdata())

        # Return True if significant change detected
        return pixel_threshold < nonzero_pixels < 80_000

    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Error: Unable to open the video file.")
            return

        saved_frame_count = 0
        previous_frame = None
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        progress_bar = tqdm(total=total_frames // self.frame_interval)

        for frame_count in range(0, total_frames, self.frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            ret, frame = cap.read()

            if not ret or frame is None:
                break

            cropped_frame, success = self.detect_and_crop_chessboard(frame)
            if not success or cropped_frame is None:
                progress_bar.update(1)
                continue

            if self.is_hand_in_image(cropped_frame):
                progress_bar.update(1)
                continue

            if previous_frame is not None:
                # if not self.is_significant_change(cropped_frame, previous_frame):
                #     progress_bar.update(1)
                #     continue
                if not self.robust_compare_chessboards(cropped_frame, previous_frame):
                    progress_bar.update(1)
                    continue


            current_size = cropped_frame.shape[:2]
            if current_size[0] < 500 or current_size[1] < 500:
                progress_bar.update(1)
                continue

            frame_filename = os.path.join(
                self.output_folder,
                f"chess_frame_{saved_frame_count:05d}.jpg"
            )
            cv2.imwrite(frame_filename, cropped_frame)
            saved_frame_count += 1

            previous_frame = cropped_frame
            progress_bar.update(1)

        cap.release()
        progress_bar.close()
        print(f"Processing complete! Saved {saved_frame_count} frames.")
