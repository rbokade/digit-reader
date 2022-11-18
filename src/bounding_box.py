import sys

import cv2


class BoundingBoxWidget:
    """
    source: https://stackoverflow.com/questions/55149171/how-to-get-roi-bounding-box-coordinates-with-mouse-clicks-instead-of-guess-che
    """
    def __init__(self, reference_image):
        self.reference_image = cv2.imread(reference_image)
        self.clone = self.reference_image.copy()
        # Bounding box reference points
        self.image_coordinates = []
        self.n_selected = 0

    def get_bounding_box(self):
        cv2.namedWindow("Reference Image")
        cv2.setMouseCallback("Reference Image", self.extract_coordinates)
        cv2.imshow("Reference Image", self.clone)
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("d"):
                break
            if key == ord("r"):
                self.image_coordinates = self.image_coordinates[
                    :len(self.image_coordinates) - 2
                ]
                self.n_selected -= 1
                self.clone = self.reference_image.copy()
        if len(self.image_coordinates) % 2 == 0:
            for selection in range(0, len(self.image_coordinates) // 2):
                roi = self.clone[
                    self.image_coordinates[0 + (selection * 2)][1]:self.image_coordinates[1 + (selection * 2)][1],
                    self.image_coordinates[0 + (selection * 2)][0]:self.image_coordinates[1 + (selection * 2)][0]
                ]
        else:
            sys.exit("Selection capture didn't get an even number of bounding points.")

            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return self.image_coordinates

    def extract_coordinates(self, event, x, y, flags, parameters):
        # Record starting (x,y) coordinates on left mouse button click
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.image_coordinates) == 0:
                self.image_coordinates = [(x, y)]
            else:
                self.image_coordinates.append((x, y))

        # Record ending (x,y) coordintes on left mouse button release
        elif event == cv2.EVENT_LBUTTONUP:
            self.image_coordinates.append((x, y))
            # Draw rectangle
            cv2.rectangle(
                self.clone,
                self.image_coordinates[0], self.image_coordinates[1],
                (0, 255, 0), 2, lineType=8
            )
            cv2.imshow("Reference Image", self.clone)
            self.n_selected += 1

    def show_image(self):
        return self.clone