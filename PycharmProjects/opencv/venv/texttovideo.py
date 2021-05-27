import cv2
import datetime
a = cv2.VideoCapture(0,cv2.CAP_DSHOW);
print(a.get(cv2.CAP_PROP_FRAME_WIDTH))
print(a.get(cv2.CAP_PROP_FRAME_HEIGHT))
#a.set(4, 360)
#a.set(5, 240)
while (True):
    ret, frame = a.read()
    if ret == True:
        font = cv2.FONT_HERSHEY_COMPLEX
        date = str(datetime.datetime.now())
        wid = 'width:' + str(a.get(4)) + 'height:' +str(a.get(5)) + 'date and time:' +str(date)
        frame = cv2.putText(frame, wid, (10,50), font, .5, (0,255,255), 1, cv2.LINE_AA)
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    else:
        break

a.release()
cv2.destroyAllWindows()
