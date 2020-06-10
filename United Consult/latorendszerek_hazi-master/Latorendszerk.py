import numpy as np
import glob
import cv2
from skimage.measure import label, regionprops
from matplotlib import pyplot as plt

# Root directory
root = "."

def nothing(x):
    pass

def unique_count_app(a):
    colors, count = np.unique(a.reshape(-1,a.shape[-1]), axis=0, return_counts=True)
    if count.argmax() != 0:
        return colors[count.argmax()]
    elif count[1] >= count[2]:
        return colors[1]
    else:
        return colors[2]




def fuggveny(image, szin):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    szin_szurke = int(round((0.3*szin[2]) + (0.59*szin[1]) + (0.11*szin[0])))
    image = (image == szin_szurke)*image

    l = label(image)
    out = (l == np.bincount(l.ravel())[1:].argmax() + 1).astype(int)

    out = out - 1

    out = np.uint8(out)
    
    return out

def main():
    path2images_rgb = './HW/g1/rgb/'
    path2images_depth = './HW/g1/depth/'

    cv2.namedWindow("Trackbars")
    cv2.createTrackbar("L – H", "Trackbars", 10, 179, nothing)
    cv2.createTrackbar("L – S", "Trackbars", 150, 255, nothing)
    cv2.createTrackbar("L – V", "Trackbars", 15, 255, nothing)
    cv2.createTrackbar("U – H", "Trackbars", 35, 179, nothing)
    cv2.createTrackbar("U – S", "Trackbars", 235, 255, nothing)
    cv2.createTrackbar("U – V", "Trackbars", 100, 255, nothing)
    cv2.createTrackbar("contur_area", "Trackbars", 300, 4500, nothing)

    kernel = np.ones((25, 25), np.uint8)
    kernel5 = np.ones((5,5),np.uint8)
    for img_name in glob.glob1(path2images_rgb, "*.jpg"):

        img_rgb = cv2.imread(path2images_rgb + img_name)
        img_name = img_name.replace(".jpg", ".png")
        img_depth = cv2.imread(path2images_depth + img_name)
        img_name = img_name.replace(".png", ".jpg")
        img = img_rgb



        gray_image = cv2.cvtColor(img_depth, cv2.COLOR_BGR2GRAY)
        # rescale
        gray_image = ((gray_image - gray_image.min()) / (gray_image.max() - gray_image.min())) * 255
        gray_image = np.uint8(gray_image)
        threshold = np.mean(gray_image) + 20
        img_rgb_foreground = img_rgb.copy()
        img_rgb_foreground[..., 0] = (gray_image < threshold) * img_rgb_foreground[..., 0]
        img_rgb_foreground[..., 1] = (gray_image < threshold) * img_rgb_foreground[..., 1]
        img_rgb_foreground[..., 2] = (gray_image < threshold) * img_rgb_foreground[..., 2]


        bilateral = cv2.bilateralFilter(img_rgb_foreground, 25, 75, 75)
        bilateral = cv2.cvtColor(bilateral, cv2.COLOR_BGR2RGB)
        #cv2.imshow("bilateral", bilateral)
        vectorized = bilateral.reshape((-1, 3))
        vectorized = np.float32(vectorized)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 3
        attempts = 10
        ret, label, center = cv2.kmeans(vectorized, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        kmeans = res.reshape((bilateral.shape))
        szin = unique_count_app(kmeans)

        temp = fuggveny(kmeans,szin)

        img_to_treshold = kmeans

        img_to_treshold[..., 0] = (temp != 0) * img_rgb_foreground[..., 0]
        img_to_treshold[..., 1] = (temp != 0) * img_rgb_foreground[..., 1]
        img_to_treshold[..., 2] = (temp != 0) * img_rgb_foreground[..., 2]
        hsv = cv2.cvtColor(img_to_treshold, cv2.COLOR_BGR2HSV)

        lower_repulo = np.array([0, 210, 5])    #repulo + suv
        upper_repulo = np.array([7, 255, 255])
        lower_markolo = np.array([13, 215, 130])
        upper_markolo = np.array([31, 250, 255])
        lower_suv = np.array([2, 213, 55])
        upper_suv = np.array([6, 244, 121])



        while True:
            l_h = cv2.getTrackbarPos("L – H", "Trackbars")
            l_s = cv2.getTrackbarPos("L – S", "Trackbars")
            l_v = cv2.getTrackbarPos("L – V", "Trackbars")
            u_h = cv2.getTrackbarPos("U – H", "Trackbars")
            u_s = cv2.getTrackbarPos("U – S", "Trackbars")
            u_v = cv2.getTrackbarPos("U – V", "Trackbars")
            lower_kaktusz = np.array([l_h, l_s, l_v])
            upper_kaktusz = np.array([u_h, u_s, u_v])
            contur_area = cv2.getTrackbarPos("contur_area", "Trackbars")

            key = cv2.waitKey(1)
            if key & 0xff == ord('q'):
                break

            if key == 27:
                exit(1)

            mask_repulo = cv2.inRange(hsv, lower_repulo, upper_repulo)
            morp_suv = cv2.morphologyEx(mask_repulo, cv2.MORPH_CLOSE, kernel)
            #cv2.imshow("morp_suv", morp_suv)
            result_repulo = cv2.bitwise_and(img_to_treshold, img_to_treshold, mask=morp_suv)
           # cv2.imshow("result_repulo", result_repulo)

            mask_markolo = cv2.inRange(hsv, lower_markolo, upper_markolo)
            morp_markolo = cv2.morphologyEx(mask_markolo, cv2.MORPH_CLOSE, kernel)
           # cv2.imshow("morp_markolo", morp_markolo)
            result_markolo = cv2.bitwise_and(img_to_treshold, img_to_treshold, mask=morp_markolo)
           # cv2.imshow("result_markolo", result_markolo)

            mask_kaktusz = cv2.inRange(hsv, lower_kaktusz, upper_kaktusz)
            morp_kaktusz = cv2.morphologyEx(mask_kaktusz, cv2.MORPH_CLOSE, kernel)
            dilate_kaktusz = cv2.dilate(morp_kaktusz, kernel5)
            result_kaktusz = cv2.bitwise_and(img_to_treshold, img_to_treshold, mask=dilate_kaktusz)

            contours_repulo, _ = cv2.findContours(morp_suv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            objects_repulo = [cv2.boundingRect(contour) for contour in contours_repulo if cv2.contourArea(contour) > contur_area]
            img2 = cv2.imread(path2images_rgb + img_name)
            for obj in objects_repulo:  ##repulo, suv
                x, y, width, height = obj
                x1 = x
                x2 = x1 + width
                y1 = y
                y2 = y1 + height
                cv2.rectangle(img2, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(img2, "jarmu", (x1, y1 + 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (0, 255, 0))


            contours_markolo, _ = cv2.findContours(morp_markolo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            objects_markolo = [cv2.boundingRect(contour) for contour in contours_markolo if cv2.contourArea(contour) > contur_area]
            for obj in objects_markolo:
                x, y, width, height = obj
                x1 = x
                x2 = x1 + width
                y1 = y
                y2 = y1 + height
                cv2.rectangle(img2, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img2, "Jarmu2", (x1, y1 + 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (0, 255, 0))

            contours_kaktusz, _ = cv2.findContours(dilate_kaktusz, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )
            objects_kaktusz = [cv2.boundingRect(contour) for contour in contours_kaktusz if cv2.contourArea(contour) > contur_area]
            for obj in objects_kaktusz:
                # Get corner points
                x, y, width, height = obj
                x1 = x
                x2 = x1 + width
                y1 = y
                y2 = y1 + height
                if height/width > 1:
                    cv2.rectangle(img2, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img2, "kaktusz", (x1, y1 + 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (0, 255, 0))
                else:
                    cv2.rectangle(img2, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img2, "rossz", (x1, y1 + 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (0, 255, 0))

            cv2.imshow("img2", img2)

if __name__ == '__main__':
    main()