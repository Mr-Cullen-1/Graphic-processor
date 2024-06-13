from django.shortcuts import render
from django.core.files.storage import default_storage
from .forms import UploadFileForm
import cv2
import numpy as np
from pdf2image import convert_from_bytes
import os
from io import BytesIO

def convert_pdf_to_images(pdf_bytes):
    images = convert_from_bytes(pdf_bytes)
    image_files = []
    for image in images:
        image_file = BytesIO()
        image.save(image_file, format='PNG')
        image_file.seek(0)
        image_files.append(image_file)
    return image_files

def load_image(image_file):
    image_array = np.frombuffer(image_file.read(), np.uint8)
    return cv2.imdecode(image_array, cv2.IMREAD_COLOR)

def process_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    curve_contour = max(contours, key=cv2.contourArea)
    return curve_contour

def extract_coordinates(contour):
    coordinates = [(point[0][0], point[0][1]) for point in contour]
    return coordinates

def save_result_image(image, curve_contour):
    result_path = os.path.join('static', 'images', 'result.png')
    cv2.drawContours(image, [curve_contour], -1, (0, 255, 0), 2)
    cv2.imwrite(result_path, image)
    return result_path

def upload_file(request):
    coordinates = None
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            file = request.FILES['file']
            if file.name.endswith('.pdf'):
                image_files = convert_pdf_to_images(file.read())
                for image_file in image_files:
                    image = load_image(image_file)
                    curve_contour = process_image(image)
                    coordinates = extract_coordinates(curve_contour)
                    result_image_path = save_result_image(image, curve_contour)
            else:
                image = load_image(file)
                curve_contour = process_image(image)
                coordinates = extract_coordinates(curve_contour)
                result_image_path = save_result_image(image, curve_contour)
    else:
        form = UploadFileForm()
    return render(request, 'upload/upload.html', {
        'form': form,
        'coordinates': coordinates,
        'result_image_path': result_image_path if coordinates else None
    })
