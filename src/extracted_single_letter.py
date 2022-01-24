import os
import os.path
import cv2
import glob
import imutils

# Define the path to the directory containing the images
DATASETS_PATH_1 = "../captcha/dataset1/"
DATASETS_PATH_2 = "../captcha/dataset2"
OUTPUT_PATH = "../captcha/data"

# Get the list of all the images in the directory
captcha_1 = glob.glob(os.path.join(DATASETS_PATH_1, "*"))
captcha_2 = glob.glob(os.path.join(DATASETS_PATH_2, "*"))
captcha_image_files = captcha_1 + captcha_2
counts = {}

for i, captcha_image_files in enumerate(captcha_image_files):
    filename = os.path.basename(captcha_image_files)
    print("[INFO] processing image {} / {} ".format(i + 1, len(captcha_1 + captcha_2)))
    # Since the filename contains the captcha text (i.e. "2A2X.png" has the text "2A2X"),
    captcha_correct_text = os.path.splitext(filename)[0]
    # Load the image and convert it to grayscale
    image = cv2.imread(captcha_image_files)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Add some extra padding around the image
    gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
    # threshold the image (convert it to pure black and white)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # find the contours (continuous blobs of pixels) the image
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Hack for compatibility with different OpenCV versions
    contours = contours[1] if imutils.is_cv3() else contours[0]
    letter_image_regions = []

    for contour in contours:
        # Get the rectangle that contains the contour
        (x, y, w, h) = cv2.boundingRect(contour)

        # Filter out width and height values less than 3
        if w < 3 or h < 3:
            continue

        # Compare the width and height of the contour to detect letters that
        # are conjoined into one chunk
        if w / h > 1.25:
            # This contour is too wide to be a single letter!
            # Split it in half into two letter regions!
            half_width = int(w / 2)
            letter_image_regions.append((x, y, half_width, h))
            letter_image_regions.append((x + half_width, y, half_width, h))
        else:
            # This is a normal letter by itself
            letter_image_regions.append((x, y, w, h))

    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

    # Save out each letter as a single image
    for letter_bounding_box, letter_text in zip(letter_image_regions, captcha_correct_text):
        # Grab the coordinates of the letter in the image
        x, y, w, h = letter_bounding_box

        # Extract the letter from the original image with a 2-pixel margin around the edge
        letter_image = gray[y - 2:y + h + 2, x - 2:x + w + 2]

        # Get the folder to save the image in
        save_path = os.path.join(OUTPUT_PATH, letter_text)

        # If the output directory does not exist, create it
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Write the letter image to a file
        count = counts.get(letter_text, 1)
        p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
        cv2.imwrite(p, letter_image)

        # Increment the count for the current key
        counts[letter_text] = count + 1
