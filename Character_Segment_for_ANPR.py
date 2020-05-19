# Import all packages and Libraries
import cv2
import os
import numpy as np
from skimage import morphology

for subdir, dirs, files in os.walk(r'put your file path here'):
    for filename in files:
        filepath = subdir + os.sep + filename

        if filepath.endswith(".png"):

            # ------------------------------ Pre-Processing -----------------------------
            # Read Original Image
            img = cv2.imread(filepath)
            # Resize Original Image
            r1_img = cv2.resize(img, (400, 100))
            # Convert to Grayscale
            g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Resize Grayscale image
            r_img = cv2.resize(g_img, (400, 100))
            # Perform Bilateral Filtering
            b_img = cv2.bilateralFilter(r_img, 9, 70, 70)
            # Perform Canny Edge Detection
            ce_img = cv2.Canny(b_img, 30, 130)
            # ---------------------------------------------------------------------------

            # -------------------------------- Filtering --------------------------------

            """____Redundant Lines Removal____________________________________________"""

            # Remove horizontal lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
            remove_horizontal = cv2.morphologyEx(ce_img, cv2.MORPH_OPEN, horizontal_kernel, iterations=8)
            cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]

            for c in cnts:
                cv2.drawContours(ce_img, [c], -1, (0, 0, 0), 5)

            # Remove vertical lines
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
            remove_vertical = cv2.morphologyEx(ce_img, cv2.MORPH_OPEN, vertical_kernel, iterations=8)
            cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]

            for c in cnts:
                cv2.drawContours(ce_img, [c], -1, (0, 0, 0), 5)

            """____Blob Removal_______________________________________________________"""

            br_img = morphology.remove_small_objects(ce_img.astype(bool), min_size=50,
                                                     connectivity=3).astype(int)
            mask_x, mask_y = np.where(br_img == 0)
            ce_img[mask_x, mask_y] = 0

            """_______________________________________________________________________"""

            # ------------------------------ Find Contours ------------------------------

            contours, hierarchy = cv2.findContours(ce_img.copy(), cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)
            sorted_contours = sorted(contours,
                                     key=lambda contour: cv2.boundingRect(contour)[0])

            # ---------------------------------------------------------------------------

            List_w = []
            List_h = []

            # ------------------------- Draw Bounding Rectangles ------------------------

            for i, ctr in enumerate(sorted_contours):
                x, y, w, h = cv2.boundingRect(ctr)

                roi = r_img[y:y + h, x:x + w]
                area = w * h
                perimeter = 2 * (w + h)
                aspect_ratio = w / h

                if 350 < area < 2800 and 80 < perimeter < 475 and 0.15 < aspect_ratio < 2.2 and w < 100 and h < 70:
                    rect = cv2.rectangle(r1_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    crop_img = r1_img[y:y + h, x:x + w]

                    List_w.append(w)
                    List_h.append(h)

                    cv2.imshow("Image", img)
                    cv2.imshow("Canny Edge", ce_img)
                    cv2.imshow('rect', rect)
                    cv2.imshow("Cropped Characters", crop_img)

            # ---------------------------------------------------------------------------

                cv2.waitKey(0)
                cv2.destroyAllWindows()

                # May use cv2.destroyAllWindows instead of cv2.destroyAllWindows() to avoid
                # repetitive reopening of windows in every new iteration

            print(List_w)
            print(List_h)
