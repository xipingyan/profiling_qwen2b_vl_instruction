import cv2

img = cv2.imread("../test_video/319483352-d5fbbd1a-d484-415c-88cb-9986625b7b11.jpg")
dst = cv2.resize(img, dsize=(120, 100))
cv2.imwrite("../test_video/cat_120_100.png", dst)