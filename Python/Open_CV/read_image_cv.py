import cv2

myimge=cv2.imread("ab.jpg")
show=1
if show:
    cv2.imshow('streched_image', myimge)
    ch = cv2.waitKey(1000)
    if ch & 0XFF == ord('q'):
        cv2.destroyAllWindows()
    cv2.destroyAllWindows()