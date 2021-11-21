import cv2 as cv
import imutils
import numpy as np
import pytesseract


class detect_using_haar:
    def __init__(self):
        self.license_cascade = cv.CascadeClassifier(
            cv.data.haarcascades + 'haarcascade_russian_plate_number.xml')

    def fromVideo(self):
        cap = cv.VideoCapture('Video Analytics based License Plate Recognition - Entry_ Exit.mp4')

        while (cap.isOpened()):
            check, frame = cap.read()
            gray_image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            filter_ = cv.bilateralFilter(gray_image, 20, 17, 17)
            # edge = cv.Canny(filter_, 200, 255)
            license = self.license_cascade.detectMultiScale(filter_, 1.1, 10)
            print(license)

            for x, y, w, h in license:
                cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)

            cv.imshow("Image",frame)

            if cv.waitKey(25) & 0xFF == ord('q'):
                break

        cap.release()


    def fromImage(self):
        image = cv.imread('car6.jpg')
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        filter_ = cv.bilateralFilter(gray_image, 20, 17, 17)

        edge = cv.Canny(filter_, 230, 255)

        license = self.license_cascade.detectMultiScale(filter_, 1.1, 10)

        for x,y,w,h in license:
            cv.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv.imshow('Image', image)

        cv.waitKey(0)
        cv.destroyAllWindows()



class detect_using_imageProc:

    def fromImage(self):
        image = cv.imread('car5.jpg')
        # roi = image[0:230, 120:300]
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        thresh = cv.threshold(gray_image, 225, 255, cv.THRESH_BINARY_INV)[1]
        # blurring the image
        filter_ = cv.bilateralFilter(gray_image, 11, 17, 17)
        edged = cv.Canny(filter_, 200, 255)
        # how we want our contours to be represented => TREE
        # approx of shape
        cnts = cv.findContours(edged, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        contours = sorted(cnts, key=cv.contourArea, reverse=True)[:10]
        print(contours)
        location = None
        for contour in contours:
            approx = cv.approxPolyDP(contour, 20, True)
            if len(approx) == 4:
                location = approx

                break
        mask = np.zeros(gray_image.shape, np.uint8)
        new_image = cv.drawContours(mask, [location], 0, (255, 0, 0), -1)
        new_image = cv.bitwise_and(image, image, mask=mask)

        (x, y) = np.where(mask==255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        cropped_image = gray_image[x1:x2+1, y1:y2+1]

        cv.imwrite('licenseplate.jpg', cropped_image)

        image_to_read = cv.imread('licenseplate.jpg')

        image_text = pytesseract.image_to_string(image_to_read)

        print(image_text)

        output = image.copy()
        cv.drawContours(output, cnts, -1, (240, 0, 159), 3)
        cv.imshow("Detected", new_image)
        cv.waitKey(0)



detect_car = detect_using_haar()
detect_car.fromImage()

# detect = detect_using_imageProc()
# detect.fromImage()
