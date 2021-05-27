import cv2

a = cv2.VideoCapture(0,cv2.CAP_DSHOW);
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# 3rd arg is no of fps
save = cv2.VideoWriter('webcam.avi', fourcc, 20.0, (640, 480))
print(a.isOpened())
while (a.isOpened()):
    ret, frame = a.read()
    if ret == True:
        print(a.get(cv2.CAP_PROP_FRAME_WIDTH))
        print(a.get(cv2.CAP_PROP_FRAME_HEIGHT))

        save.write(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', gray)
        if cv2.waitKey(1) == ord('q'):
            break
    else:
        break

a.release()
save.release()
cv2.destroyAllWindows()
