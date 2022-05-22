# teseract
img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

custom_config = r'--oem 3 --psm 6'
print(pytesseract.image_to_data(img_rgb, config=custom_config))
print("Next Frame")

