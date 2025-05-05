#scyxz5
import cv2
import sys
import numpy as np
#used for UI
import tkinter as tk
from tkinter import filedialog, simpledialog, scrolledtext, ttk
from PIL import Image, ImageTk

from matplotlib import pyplot as plt

#processed bar
from tqdm import tqdm
import threading
import time


#core image processing functions
def select_frames(video_path, algorithm_name="sift", max_frames=100):
    #Select key frames from video using feature matching
    cap = cv2.VideoCapture(video_path)
    # //test
    if not cap.isOpened():
        raise ValueError("Cannot open video file!")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #max_frames = total_frames;

    #store selected key frames
    frames = []
    #store descriptors from previous frame for matching,,,,,
    # #used for deciding the derection
    prev_descriptors = None

    # initialize feature detector
    #SIFT/PRB/AKAZE
    if algorithm_name == "sift":
        detector = cv2.SIFT_create() #Scale Invariant Feature Transform
        index_params = dict(algorithm=0, trees=5)  # FLANN_INDEX_KDTREE
        # norm_type = cv2.NORM_L2
    elif algorithm_name == "orb":
        detector = cv2.ORB_create() #Oriented FAST and Rotated BRIEF
        index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)  # FLANN_INDEX_LSH
        #norm_type = cv2.NORM_HAMMING
    elif algorithm_name == "akaze":
        detector = cv2.AKAZE_create()
        index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)  # FLANN_INDEX_LSH
        #norm_type = cv2.NORM_HAMMING #Hamming distance -> binary descriptors
    else:
        raise ValueError("Unsupported algorithm!")

    search_params = dict(checks=30)
    flann = cv2.FlannBasedMatcher(index_params, search_params)  # pre initial FLANN

    #progress bar, using video length / max_frames => upper bound
    progress_bar = tqdm(total=min(max_frames, total_frames), desc="\nSelecting keyframes")

    while len(frames) < max_frames and cap.isOpened():
        ret, frame = cap.read()  #read next frame from video
        if not ret:
            break

        #feature detection and matching
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #locations of distinctive features, feature vectors
        kp, desc = detector.detectAndCompute(gray, None)

        #(skip frames with no features
        if prev_descriptors is not None and desc is not None and len(prev_descriptors) >= 2 and len(desc) >= 2:            #initialize FLANN matcher
            ##FLANN
            #flann = cv2.FlannBasedMatcher(index_params, search_params)
            #matches = flann.knnMatch(prev_descriptors, desc, k=2)
            try:
                matches = flann.knnMatch(prev_descriptors, desc, k=2)
                # Lowe's Ratio testing
                good = []
                # for m, n in matches:
                #     if m.distance < 0.7 * n.distance:
                #         good.append(m)
                for pair in matches:
                    if len(pair) == 2:  # ensuring that each matching pair has 2 results
                        m, n = pair
                        if m.distance < 0.7 * n.distance:
                            good.append(m)

                # Add frame if significant changes detected
                #if len(good) > 50:
                if len(good) > 30:
                    frames.append(frame)
                    progress_bar.update(1)

            except cv2.error as e:
                print(f"FLANN matching error: {e}")
                continue

        #maintain last valid descriptors,, if current is None
        prev_descriptors = desc if desc is not None else prev_descriptors

    # cleanup resources
    cap.release()
    progress_bar.close()
    return frames

# denoising using convolution filter
def deblur_frames(frames):
    deblurred = []
    for frame in tqdm(frames, desc="\nNoise reduction"):
        b, g, r = cv2.split(frame)
        psf = np.ones((5, 5)) / 25

        b_restored = cv2.filter2D(b, -1, psf)
        g_restored = cv2.filter2D(g, -1, psf)
        r_restored = cv2.filter2D(r, -1, psf)

        restored = cv2.merge([b_restored, g_restored, r_restored])
        deblurred.append(restored)
    return deblurred

#color consistency adjustment ( using histogram matching
def color_correction(frames):
    #empty
    if not frames:
        return []

    # Use first frame as color reference
    reference = frames[0]
    corrected = [reference]
    ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB)
    #channel separation for reference image
    l_ref, a_ref, b_ref = cv2.split(ref_lab)

    for frame in tqdm(frames[1:], desc="\nColor correction"):
        # Convert to color space for better color adjustment
        frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_frame, a_frame, b_frame = cv2.split(frame_lab) #lightness /Green-Red /Blue-Yellow

        # Match luminance channel histogram
        l_matched = cv2.equalizeHist(l_frame, l_ref) #reference histogram
        matched_lab = cv2.merge([l_matched, a_frame, b_frame]) #luminance reconstruct
        matched_bgr = cv2.cvtColor(matched_lab, cv2.COLOR_LAB2BGR) #back to BGR color
        corrected.append(matched_bgr)

    return corrected

# Lens distortion correction ( using camera matrix
# f, 0, c  focal length
# 0, f, c  Center point set to image middle
#todo: not obiviously improved
def geometric_correction(frames):
    camera_matrix = np.array([
        [1000, 0, frames[0].shape[1] / 2],
        [0, 1000, frames[0].shape[0] / 2],
        [0, 0, 1]
    ], dtype=np.float32)
    dist_coeffs = np.zeros(4)

    corrected = []
    for frame in tqdm(frames, desc="\nGeometric correction"):
        h, w = frame.shape[:2]
        #Alpha=1 keep all pixels, 0 crop black edges)
        # Calculate optimal camera matrix
        new_cam_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), 1, (w, h)
        )
        #actual distortion removal
        undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_cam_matrix)
        corrected.append(undistorted)
        #frames
    return corrected

# def stitch_images(frames, algorithm_name="sift"):
# print("\n[STATUS] starts splicing " )
#     start_time = time.time()
#
#     # heartbeat thread [todo: tbd]
#     def heartbeat():
#         while not stitching_complete:
#             print("[HEARTBEAT] The splicing is still ongoing...")
#             time.sleep(5)
#
#     stitching_complete = False
#     heartbeat_thread = threading.Thread(target=heartbeat)
#     heartbeat_thread.daemon = True
#     heartbeat_thread.start()
#
#     #actual splicing
#     stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
#     status, panorama = stitcher.stitch(frames)
#
#     # stop heartbeat
#     stitching_complete = True
#     heartbeat_thread.join()
#     if status != cv2.Stitcher_OK:
#         print(f"[WARNING] Splicing failed! Status code: {status}")
#         # Testing (todo)
#         h, w = frames[0].shape[:2]
#         panorama = np.zeros((h, w * 2, 3), dtype=np.uint8)
#
#     return panorama

   # #remove black borders from processed image
def crop_black_borders(img, threshold=5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    rect = cv2.boundingRect(max_contour)
    x, y, w, h = rect
    return img[y:y + h, x:x + w]

# def stitch_images(frames, algorithm_name="sift"):
#     print("\n[scyxz5] Starting stitching...")
#     if len(frames) < 2:
#         return None
#     #detect the main direction of movement (horizontal or vertical)
#     motion_direction = detect_motion_direction(frames)
#     print(f"Detected motion direction: {'horizontal' if motion_direction == 'horizontal' else 'vertical'}")
#     #select the splicing method based on the movement direction
#     if motion_direction == "horizontal":
#         stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)  # horizontal
#     else:
#         stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)  # vertical splicing (up / down
#     status, panorama = stitcher.stitch(frames)
#
#     if status != cv2.Stitcher_OK:
#         print(f"Stitching failed with status {status}")
#         return None
#
#
#     # # feathered edge treatment
#     # if blend_strength > 0:
#     #     gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
#     #     _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
#     #     kernel = np.ones((15, 15), np.uint8)
#     #     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#     #     mask = cv2.GaussianBlur(mask, (15, 15), 0)
#     #     mask = mask.astype(np.float32) / 255.0
#     #     mask = cv2.merge([mask, mask, mask])
#     #
#     #     # application feathering
#     #     white = np.ones_like(panorama, dtype=np.uint8) * 255
#     #     panorama = (panorama * mask + white * (1 - mask)).astype(np.uint8)
#
#     return panorama

def stitch_images(frames, algorithm_name="sift"):
    print("\n[scyxz5] Starting stitching...")
    if len(frames) < 2:
        return None
    #detect the main direction of movement (horizontal or vertical)
    #todo: inverse? √
    motion_direction = detect_motion_direction(frames)
    print(f"Detected motion direction: {'horizontal' if motion_direction == 'horizontal' else 'vertical'}")
    #select the splicing method based on the movement direction
    if motion_direction == "horizontal":
        stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)  # horizontal
    else:
        stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)  # vertical splicing
    status, panorama = stitcher.stitch(frames)

    if status != cv2.Stitcher_OK:
        print(f"Stitching failed with status {status}")
        return None
    return panorama

#Through movement direction
def detect_motion_direction(frames, sample_size=5):

    #Detect the main movement direction (horizontal / vertical)
    if len(frames) < 2:
        return "horizontal"  #init horizon

    # calc the optical flow between nearby frame (movement direction)!
    prev_frame = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    h, w = prev_frame.shape
    motion_x = 0
    motion_y = 0
    for i in range(1, min(sample_size, len(frames))):
        next_frame = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        avg_flow = np.mean(flow, axis=(0, 1))  #calc the average direction of movement
        motion_x += abs(avg_flow[0])
        motion_y += abs(avg_flow[1])
        prev_frame = next_frame
    # decide the main direction of movement
    if motion_x > motion_y:
        return "horizontal"
    else:
        return "vertical"


# evaluate (using in the log function)
def calculate_alignment_error(panorama, last_frame, detector_type="sift"):
    if panorama is None or last_frame is None:
        return None

    # init the feature detector
    if detector_type == "sift":
        detector = cv2.SIFT_create()
    elif detector_type == "orb":
        detector = cv2.ORB_create()
    elif detector_type == "akaze":
        detector = cv2.AKAZE_create()
    else:
        return None

    # convent to gray
    gray_pano = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    gray_last = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)

    # detect feature points
    kp1, des1 = detector.detectAndCompute(gray_last, None)
    kp2, des2 = detector.detectAndCompute(gray_pano, None)

    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return None

    # deature matching
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k=2)

    # select the high quality matching
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 4:
        return None

    # Prepare coordinate points
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # calculate the homography matrix
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        return None

    # calculate the reprojection error
    transformed = cv2.perspectiveTransform(src_pts, H)
    errors = np.linalg.norm(transformed - dst_pts, axis=2)
    return np.mean(errors)


# UI
class PanoramaStitcherUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Panorama Stitcher")
        self.root.geometry("1200x800")

        # Variables
        self.video_path = ""
        self.algorithm = "sift" # init algorithm
        self.frames = []
        self.panorama = None # show panoramas (updating)
        self.processing = False
        self.video_playing = False #play video
        self.video_thread = None
        self.use_deblur = tk.BooleanVar(value=False)
        self.step_by_step = tk.BooleanVar(value=True)  # Step by step stiching or final
        # recording time
        self.processing_times = {}  # time recordings

        # Create UI
        self.create_widgets()

    # _______________________________________________________
    #| menu content buttons and file path|                  |
    #|                                   | video / panoramas|
    #|     log recordings(final repo)    |                  |
    # _______________________________________________________

    def create_widgets(self):
        # main container
        main_frame = tk.Frame(self.root)
        main_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        # left panel
        # controls and log
        left_frame = tk.Frame(main_frame, width=300)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)
        # controls (top)
        control_frame = tk.Frame(left_frame, padx=5, pady=5)
        control_frame.pack(fill=tk.X)
        # video selection
        tk.Label(control_frame, text="Video File:").pack(anchor=tk.W)
        self.video_entry = tk.Entry(control_frame, width=30)
        self.video_entry.pack(fill=tk.X)
        tk.Button(control_frame, text="Browse..", command=self.browse_video).pack(pady=5)
        # Right panel
        # display
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
        # algorithm selection
        tk.Label(control_frame, text="Algorithm:").pack(anchor=tk.W)
        self.algorithm_var = tk.StringVar(value="sift")
        tk.Radiobutton(control_frame, text="SIFT", variable=self.algorithm_var, value="sift").pack(anchor=tk.W)
        tk.Radiobutton(control_frame, text="ORB", variable=self.algorithm_var, value="orb").pack(anchor=tk.W)
        tk.Radiobutton(control_frame, text="AKAZE", variable=self.algorithm_var, value="akaze").pack(anchor=tk.W)

        tk.Checkbutton(control_frame, text="Noise Reduction", variable=self.use_deblur).pack(anchor=tk.W)

        tk.Checkbutton(control_frame, text="Step-by-Step Stitching", variable=self.step_by_step).pack(anchor=tk.W)

        # process buttons
        tk.Button(control_frame, text="Process Video", command=self.process_video).pack(pady=10, fill=tk.X)
        tk.Button(control_frame, text="Play Original Video", command=self.play_video).pack(pady=5, fill=tk.X)
        tk.Button(control_frame, text="Stop Video", command=self.stop_video).pack(pady=5, fill=tk.X)
        tk.Button(control_frame, text="Save Panorama", command=self.save_panorama).pack(pady=10, fill=tk.X)
        # progress bar
        self.progress = ttk.Progressbar(control_frame, orient=tk.HORIZONTAL, length=280, mode='determinate')
        self.progress.pack(pady=10)
        # log (bottom)
        log_frame = tk.Frame(left_frame)
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        # right panel
        # video display (top)
        video_display_frame = tk.Frame(right_frame, height=400)
        video_display_frame.pack(fill=tk.BOTH, expand=True)
        # video display (bottom)
        self.video_label = tk.Label(video_display_frame)
        self.video_label.pack(expand=True, fill=tk.BOTH)
        # image display (bottom)
        image_display_frame = tk.Frame(right_frame, height=400)
        image_display_frame.pack(fill=tk.BOTH, expand=True)
        # image display (bottom)
        self.image_label = tk.Label(image_display_frame)
        self.image_label.pack(expand=True, fill=tk.BOTH)
        # Redirect stdout to log
        sys.stdout = TextRedirector(self.log_text, "output")
        sys.stderr = TextRedirector(self.log_text, "errorMesg")

    #select video
    def browse_video(self):
        self.video_path = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[("Video files", "*.mp4 *.avi *.mov")]
        )
        self.video_entry.delete(0, tk.END)
        self.video_entry.insert(0, self.video_path)
        self.log(f"Selected video: {self.video_path}")
    #Log!!!!
    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        #print(message)  # Also print to console

    def process_video(self):
        if not self.video_path:
            self.log("Please select a video file first!")
            return

        if self.processing:
            self.log("Already processing!")
            return


        self.algorithm = self.algorithm_var.get()
        self.log(f"\nStarting panorama stitching with {self.algorithm} algorithm...")

        #step by step or not
        if not self.step_by_step.get():
            self.progress["mode"] = 'indeterminate'
            self.progress.start()
        else:
            self.progress["mode"] = 'determinate'
            self.progress.stop()

        #ensure the start processing in a separate thread
        self.processing = True
        processing_thread = threading.Thread(target=self._process_video_thread)
        processing_thread.daemon = True
        processing_thread.start()

    def _process_video_thread(self):
        try:
            self.processing_times = {}

            # select keyframes
            start_time = time.time()
            self.log("Selecting keyframes...")
            self.frames = select_frames(self.video_path, self.algorithm)
            self.processing_times['Selecting keyframes'] = time.time() - start_time
            if not self.frames:
                self.log("No valid frames extracted!")
                return

            # basic info
            if self.frames:
                self.first_frame_size = (self.frames[0].shape[1], self.frames[0].shape[0])
                self.num_frames_used = len(self.frames)

            # pre processing
            self.log("Preprocessing frames...")
            #self.frames = deblur_frames(self.frames)
            #self.processing_times['Deblurring processing'] = time.time() - start_time

            # blurring processing
            if self.use_deblur.get():
                start_time = time.time()
                self.frames = deblur_frames(self.frames)
                self.processing_times['Noise reduction'] = time.time() - start_time
                start_time = time.time()
            else:
                start_time = time.time()

            # color_correction
            #start_time = time.time()
            self.frames = color_correction(self.frames)
            self.processing_times['Color correction'] = time.time() - start_time

            #geometric_correction
            start_time = time.time()
            self.frames = geometric_correction(self.frames)
            self.processing_times['Geometric correction'] = time.time() - start_time
            self.log("Preprocessing complete")

            # stitching in batches
            batch_size = 10
            self.progress["maximum"] = len(self.frames)

            # for end_idx in range(batch_size, len(self.frames) + batch_size, batch_size):
            #     end_idx = min(end_idx, len(self.frames))
            #     current_batch = self.frames[:end_idx]
            #     self.log(f"\nStitching first {end_idx} frames...")
            #
            #     current_pano = stitch_images(current_batch, self.algorithm)
            #
            #     if current_pano is not None:
            #         self.panorama = current_pano
            #         self.progress["value"] = end_idx
            #         self.update_panorama_display(self.panorama)
            #         time.sleep(0.3)  # small delay for UI update

            if self.step_by_step.get():
                for end_idx in range(batch_size, len(self.frames) + batch_size, batch_size):
                    end_idx = min(end_idx, len(self.frames))
                    current_batch = self.frames[:end_idx]
                    self.log(f"\nStitching first {end_idx} frames...")

                    current_pano = stitch_images(current_batch, self.algorithm)

                    if current_pano is not None:
                        self.panorama = current_pano
                        self.progress["value"] = end_idx
                        self.update_panorama_display(self.panorama)
                        time.sleep(0.3)  # small delay for UI update
            else:
                # final stitching
                start_time = time.time()
                self.log("\nPerforming final stitching...")
                self.panorama = stitch_images(self.frames, self.algorithm)
                self.processing_times['Panoramic stitching'] = time.time() - start_time

                # the data of final feature of picture
                if self.panorama is not None:
                    self.panorama_size = (self.panorama.shape[1], self.panorama.shape[0])

            if self.panorama is not None:
                self.update_panorama_display(self.panorama)
                self.log("Stitching completed successfully!")
                self.log(f"Final panorama size: {self.panorama.shape[1]}x{self.panorama.shape[0]}")
            else:
                self.log("Stitching failed")

            # # final stitching
            # start_time = time.time()
            # self.log("\nPerforming final stitching...")
            # self.panorama = stitch_images(self.frames, self.algorithm)
            # self.processing_times['Panoramic stitching'] = time.time() - start_time
            #
            # # the data of final feature of picture
            # if self.panorama is not None:
            #     self.panorama_size = (self.panorama.shape[1], self.panorama.shape[0])
            #
            # if self.panorama is not None:
            #     self.update_panorama_display(self.panorama)
            #     self.log("Stitching completed successfully!")
            #     self.log(f"Final panorama size: {self.panorama.shape[1]}x{self.panorama.shape[0]}")
            # else:
            #     self.log("Stitching failed")

            # evaluation
            evaluation_metrics = {}
            if self.panorama is not None:
                start_time = time.time()
                evaluation_metrics['Alignment error'] = calculate_alignment_error(
                    self.panorama, self.frames[-1], self.algorithm
                )
                self.processing_times['Quality assessment'] = time.time() - start_time

                # final result
                self.update_panorama_display(self.panorama)
                self.log("Completed!")

                # analysis
                self.show_analysis_report(evaluation_metrics)
            else:
                self.log("Stitching Failed")

        except Exception as e:
            self.log(f"Error during processing: {str(e)}")
        finally:
            self.processing = False

    def show_analysis_report(self, metrics):
        #showing analysis and evaluation report
        self.log("\n=== Analysis & Evaluation ===")

        # Basic information
        self.log("\nBasic Information:")
        if hasattr(self, 'first_frame_size'):
            self.log(f"  - First frame size: {self.first_frame_size[0]}x{self.first_frame_size[1]} pixels")
        if hasattr(self, 'panorama_size'):
            self.log(f"  - Panorama size: {self.panorama_size[0]}x{self.panorama_size[1]} pixels")
        if hasattr(self, 'num_frames_used'):
            self.log(f"  - Number of frames used: {self.num_frames_used}")

        # processing time statistics
        total_time = sum(self.processing_times.values())
        self.log(f"\nProcessing Time Statistics (Total: {total_time:.2f} seconds):")

        for step, t in self.processing_times.items():
            percentage = t / total_time * 100 if total_time > 0 else 0
            self.log(f"  - {step}: {t:.2f} seconds ({percentage:.1f}%)")

        # performance recommendations
        self.log("\nPerformance Optimization Suggestions:")
        slowest_step = max(self.processing_times, key=self.processing_times.get)
        self.log(f"  - Most time-consuming step: 【{slowest_step}】Prioritize optimization")

        if self.algorithm == "sift" and self.processing_times.get('keyframe_selection', 0) > 10:
            self.log("  - Detected long keyframe selection time, consider using ORB algorithm for acceleration")

        self.log("\n===========================")


    def update_panorama_display(self, image):
        # convert OpenCV image to PIL format
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        # resize to fit display area
        display_width = self.image_label.winfo_width() or 800
        display_height = self.image_label.winfo_height() or 400
        if display_width > 0 and display_height > 0:
            ratio = min(display_width / pil_image.width, display_height / pil_image.height)
            new_size = (int(pil_image.width * ratio), int(pil_image.height * ratio))
            pil_image = pil_image.resize(new_size, Image.LANCZOS)
        # update display
        tk_image = ImageTk.PhotoImage(pil_image)
        self.image_label.config(image=tk_image)
        self.image_label.image = tk_image  # keep reference

    def play_video(self):
        if not self.video_path:
            self.log("Please select a video file first!")
            return

        if self.video_playing:
            self.log("Video is already playing!")
            return

        self.video_playing = True
        self.video_thread = threading.Thread(target=self._play_video_thread)
        self.video_thread.daemon = True
        self.video_thread.start()

    def _play_video_thread(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.log("Cannot open video file!")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = int(1000 / fps)

        while self.video_playing:
            ret, frame = cap.read()
            if not ret:
                break

            # convert and display frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            # resize,,, to fit display area
            display_width = self.image_label.winfo_width() or 800
            display_height = self.image_label.winfo_height() or 600

            if display_width > 0 and display_height > 0:
                ratio = min(display_width / pil_image.width, display_height / pil_image.height)
                new_size = (int(pil_image.width * ratio), int(pil_image.height * ratio))
                pil_image = pil_image.resize(new_size, Image.LANCZOS)

            tk_image = ImageTk.PhotoImage(pil_image)

            # update in main thread
            self.root.after(0, lambda: self._update_video_display(tk_image))

            #  control the playback speed
            time.sleep(1 / fps)

        cap.release()
        self.video_playing = False
        self.log("Video playback finished")

    def _update_video_display(self, tk_image):
        self.image_label.config(image=tk_image)
        self.image_label.image = tk_image

    def stop_video(self):
        self.video_playing = False
        if self.video_thread:
            self.video_thread.join(timeout=1)
        self.log("Video playback stopped")

        # show the panorama again ( if exist
        if self.panorama is not None:
            self.update_panorama_display(self.panorama)

    def save_panorama(self):
        if self.panorama is None:
            self.log("No panorama to save!")
            return

        save_path = filedialog.asksaveasfilename(
            title="Save panorama as",
            defaultextension=".png",
            filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")]
        )

        if save_path:
            cv2.imwrite(save_path, self.panorama)
            self.log(f"Panorama saved to: {save_path}")


#  redirect result or error to the log
class TextRedirector:
    def __init__(self, widget, tag="output"):
        self.widget = widget
        self.tag = tag

    def write(self, str):
        self.widget.insert(tk.END, str, (self.tag,))
        self.widget.see(tk.END)

    def flush(self):
        pass


# Begin! It!!!! Now!!!!!! Yeah!!!!!:)
def main():
    root = tk.Tk()
    app = PanoramaStitcherUI(root)

    #text colors (seperate the normal output and the error messsage)
    app.log_text.tag_config("output", foreground="black")
    app.log_text.tag_config("errorMesg", foreground="red")

    root.mainloop()


if __name__ == "__main__":
    main()
