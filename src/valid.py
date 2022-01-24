import os
import cv2
import joblib
import glob
import imutils

IMAGE_PATH = "../captcha/dataset2"
MODEL_PATH = "../data/captcha_model.pkl"

clf = joblib.load(MODEL_PATH)
captcha_1 = glob.glob(os.path.join(IMAGE_PATH, "*"))
captcha_1 = captcha_1[:10]


def split_image(image_file):
    filename = os.path.basename(image_file)
    correct_text = os.path.splitext(filename)[0]
    image = cv2.imread(image_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[1] if imutils.is_cv3() else contours[0]
    letter_image_regions = []
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if w < 3 or h < 3:
            continue
        if w / h > 1.25:
            half_width = int(w / 2)
            letter_image_regions.append((x, y, half_width, h))
            letter_image_regions.append((x + half_width, y, half_width, h))
        else:
            letter_image_regions.append((x, y, w, h))
    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])
    letter_images = []
    for letter_bounding_box in letter_image_regions:
        x, y, w, h = letter_bounding_box
        letter_image = gray[y - 2:y + h + 2, x - 2:x + w + 2]
        letter_image = cv2.resize(letter_image, (28, 28))
        letter_images.append(letter_image)
    return correct_text, letter_images


for i in captcha_1:
    labels, images = split_image(i)
    results = []
    for j in images:
        result = clf.predict(j.reshape(1, -1))
        results.append(result[0])
    results = "".join(results)
    print(f'original: {labels}, predict: {results}')
