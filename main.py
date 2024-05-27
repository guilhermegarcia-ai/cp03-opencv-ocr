import cv2
import pytesseract
from PIL import Image
from google.colab.patches import cv2_imshow
import imutils

# Load passport image
image = cv2.imread('cnh.webp', 0)  # 0 for greyscale

# Apply threshold
ret, thresholded = cv2.threshold(image, 130, 255, cv2.THRESH_BINARY)

# Remove noise
blurred_image = cv2.GaussianBlur(thresholded, (3, 3), 1)

# Apply dilation to enhance text
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dilated_image = cv2.dilate(blurred_image, kernel, iterations=1)

# Define ROIs
rois = {
    "full_name": (166, 145, 170, 22),
    "doc_number": (422, 199, 90, 22),
    "rg": (169, 483, 135, 20),
    "cpf": (420, 248, 155, 23),
    "date_birth": (607, 252, 109, 20),
    "category_hab": (684, 429, 17, 20),
    "validate_date": (411, 481, 113, 22),
    "emission_date": (581, 831, 114, 21),
    "notes": (173, 579, 26, 22),
    "signature": (269, 750, 177, 30),
    "person_photo": (177, 193, 186, 242)
}

# Process and display each ROI
for roi_name, (x, y, w, h) in rois.items():
    roi = dilated_image[y:y+h, x:x+w]
    
    print(f'{roi_name.capitalize()} image:')
    cv2_imshow(roi)
    cv2.imwrite(f'/cnh_parts/{roi_name}.png', roi)
    cv2.waitKey(0)

    # Load images for Tesseract and extract
    if roi_name not in ['signature', 'person_photo']:
        im_roi = Image.open(f'/cnh_parts/{roi_name}.png')
        txt_roi = pytesseract.image_to_string(im_roi, lang='por')
        print(f'{roi_name.capitalize()} pytesseract: {txt_roi}')

    cv2.waitKey(0)

# Process signature with Canny edge detection
signature_x, signature_y, signature_w, signature_h = rois['signature']
signature_roi = dilated_image[signature_y:signature_y+signature_h, signature_x:signature_x+signature_w]

# Apply Canny edge detection
signature_edges = cv2.Canny(signature_roi, 50, 150)

print('Signature image with edges:')
cv2_imshow(signature_edges)
cv2.imwrite('/cnh_parts/signature_edges.png', signature_edges)
cv2.waitKey(0)

# Draw rectangles on the original image
for roi_name, (x, y, w, h) in rois.items():
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 5)

# Original image with ROIs
cv2_imshow(imutils.resize(image, width=300))
cv2.waitKey(0)
cv2.destroyAllWindows()
