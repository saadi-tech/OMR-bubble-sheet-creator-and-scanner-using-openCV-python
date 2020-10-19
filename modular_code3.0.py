import cv2
import math
import imutils
import numpy as np
import json

questionsok = True
rollnum_ok = True

mcol1 = 0
mcol2 = 0
mcol3 = 0
mcol4 = 0
mcol5 = 0

dcol1 = 0
dcol2 = 0
dcol3 = 0

total_results = []
batch_results = []

green = (0, 255, 0)
red = (0, 0, 255)
black = (0, 0, 0)

def read_roll_number(question, x, y, width, image,rollnum_ok):
    cropped = image[y + 10:question - 10, x + 10:x + int(width / 1.8) - 10]
    #global rollnum_ok
    orig = cropped.copy()
    # cv2.imshow("Roll_number_section",cropped)

    cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    (T, cropped) = cv2.threshold(cropped, 180, 255, cv2.THRESH_BINARY)
    # kernel = np.ones((5, 5), np.uint8)
    # cropped = cv2.erode(cropped, kernel, iterations=1)
    #cv2.imshow("threshold",cropped)
    #cv2.waitKey(0)
    cropped = cv2.GaussianBlur(cropped, (1, 1), 0)

    canny = cv2.Canny(cropped, 30, 150)

    rectangle = None

    roll_number_circles = None

    (cropped_cnts, _) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roll_box = []

    cnt = sorted(cropped_cnts, key=cv2.contourArea)
    (a, b, c, d) = cv2.boundingRect(cnt[len(cnt) - 1])

    for i in range(len(cropped_cnts)):
        (x, y, w, h) = cv2.boundingRect(cropped_cnts[i])

        if (h > width / 4 and w > h / 3):
            # print(h,width)
            roll_box.append((x, y, w, h))
            # cv2.rectangle(orig, (x + int(h / 15), y + int(h / 28)), (x + w, y + h - int(h / 28)), red, 2)

            roll_number_circles = orig[y + int(h / 28): y + h - int(h / 28), x + int(h / 15):x + w]

            rectangle = (x, y, w, h)
            break

    # print("Roll boxes:",len(roll_box))

    print(roll_number_circles.shape [0])
    if (roll_number_circles.shape [0] > 610):
        height = roll_number_circles.shape [0]
        roll_number_circles = roll_number_circles[ height-610 : height , : ]

    #cv2.imshow("Roll-num", roll_number_circles)
    #cv2.waitKey(0)
    height, width, ch = roll_number_circles.shape
    step = height / 10

    offset = 0

    results = []
    dist = 10
    for i in range(10):

        line = roll_number_circles.copy()[offset - dist:offset + int(step) + dist, 0:width]

        if (i == 9):
            line = roll_number_circles.copy()[offset - dist:offset + int(step) + int(step / 10) + dist, 0:width]

        if (i == 0):
            line = roll_number_circles.copy()[offset:offset + dist + int(step) - int(step / 10), 0:width]

        grey = cv2.cvtColor(line, cv2.COLOR_BGR2GRAY)
        (T, thresh) = cv2.threshold(grey, 155, 255, cv2.THRESH_BINARY)
        blur = cv2.GaussianBlur(grey, (1, 1), 4)

        canny = cv2.Canny(blur, 30, 150)

        (cropped_cnts, _) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cropped_cnts = sorted(cropped_cnts, key=lambda ctr: cv2.boundingRect(ctr)[0])
        digits = []

        index = 0
        selection = []
        for j in range(len(cropped_cnts)):

            (x, y, w, h) = cv2.boundingRect(cropped_cnts[j])

            if (w > 0.65 * step and h > 0.65 * step and w < width / 4):
                # print("bubble number:",i+1," : area :",((w * w - cv2.countNonZero(thresh[y:y + h, x:x + w])) / (w * w)) * 100)

                if (((w * w - cv2.countNonZero(thresh[y:y + h, x:x + w])) / (w * w)) * 100 > 40):
                    cv2.rectangle(line, (x, y), (x + w, y + h), red, 2)

                    selection.append(index)

                else:
                    cv2.rectangle(line, (x, y), (x + w, y + h), green, 2)
                    selection.append('')
                digits.append((x, y, w, h))
                index += 1

        results.append(selection)
        # cv2.imshow("line", line)
        # cv2.waitKey(0)
        # print("DIGITS: ",len(digits))

        offset += int(step)
    # print("ROLL RESULTS", results)
    length = len(results[0])
    roll_number = ""
    # print((results))

    max_length = len(results[0])

    for i in range(length):
        '''
        for x in range(len(results)):

            if results[x].__contains__(i):
                roll_number += str(x)
        '''
        done = 0
        # print("LENGTH ROLLS:",len(results))
        for x in range(len(results)):

            if (len(results[i]) != max_length):
                rollnum_ok = False
                break

            if (len(results[x]) > i):
                if results[x][i] != '':
                    roll_number += str(x)
                    done = 1
                    break

    # print("ROLL NUMBER: ",roll_number)
    return roll_number,rollnum_ok


def get_row_answers(rects, image, h):
    # print("circles: ",len(rects))
    groups = 1
    answers = ['A', 'B', 'C', 'D', 'E']
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    (T, gray) = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    # cv2.imshow('greay',gray)
    # cv2.waitKey(0)

    gap = h * 2
    # print(gap)
    for i in range(1, len(rects)):

        if (rects[i][0] - (rects[i - 1][0] + rects[i - 1][2]) > gap):
            groups += 1
            cv2.line(image, ((rects[i - 1][0] + rects[i - 1][2]) + int(0.5 * gap), 0),
                     ((rects[i - 1][0] + rects[i - 1][2]), 3 * int(gap)), green, 2)

    # cv2.imshow("groups",image)
    # cv2.waitKey(0)
    # print("TOTAL GROUPS:  ", (groups))

    results = []
    options = int(len(rects) / groups)
    for i in range(groups):
        ind = i * options
        ans = []
        for j in range(ind, ind + options):
            (X, Y, W, H) = rects[j]
            # print(X,Y,W,H)
            cv2.rectangle(image, (X, Y), (X + W, Y + H), green, 1)

            # print("Area:",(W*W - cv2.countNonZero(gray[Y:Y+H , X:X+W]))/(W*W)*100 )

            if ((W * W - cv2.countNonZero(gray[Y:Y + H, X:X + W])) / (W * W) * 100 > 65):
                cv2.rectangle(image, (X, Y), (X + W, Y + H), red, 1)
                if (j - ind < len(answers)):
                    ans.append(answers[j - ind])
                else:
                    ans.append(answers[len(answers) - 1])
        results.append(ans)

    #cv2.imshow("img", image)
    #cv2.waitKey(0)

    return (results, groups)


def get_decimal_row_answers(image, rects, h):
    groups = 1
    answers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (T, gray) = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    ##printlen(rects))

    gap = rects[1][0] - rects[0][0]
    gap = h * 3
    for i in range(1, len(rects)):

        if (rects[i][0] - (rects[i - 1][0] + rects[i - 1][2]) > gap):
            groups += 1
            # cv2.line(image, ((rects[i - 1][0] + rects[i - 1][2]) + int(0.5 * gap), 0),
            # ((rects[i - 1][0] + rects[i - 1][2]), 3 * int(gap)), green, 2)
    # cv2.imshow("Groups",image)
    # cv2.waitKey(0)
    # print("TOTAL GROUPS:  ", (groups))

    results = []
    options = int(len(rects) / groups)
    for i in range(groups):
        ind = i * options
        ans = []
        for j in range(ind, ind + options):
            (X, Y, W, H) = rects[j]
            cv2.rectangle(image, (X, Y), (X + W, Y + H), green, 1)

            # print(X,Y,W,H)

            if ((W * W - cv2.countNonZero(gray[Y:Y + H, X:X + W])) / (W * W) * 100 > 50):
                cv2.rectangle(image, (X, Y), (X + W, Y + H), red, 1)
                if (j - ind < len(answers)):
                    ans.append(str(answers[j - ind]))
                else:
                    ans.append(str(answers[len(answers) - 1]))
        results.append(ans)

    #cv2.imshow("im",image)
    #cv2.waitKey(0)

    return (results, groups)


def update_batch(input, total_results, mcol1, mcol2, mcol3, mcol4, mcol5, dcol1, dcol2, dcol3,batch_results):


    count = 0
    if (input == 'M'):

        for i in range(mcol1):
            total_results.append(batch_results[i][0])
        for i in range(mcol2):
            total_results.append(batch_results[i][1])
        for i in range(mcol3):
            total_results.append(batch_results[i][2])
        for i in range(mcol4):
            total_results.append(batch_results[i][3])
        for i in range(mcol5):
            total_results.append(batch_results[i][4])

        mcol1 = 0
        mcol2 = 0
        mcol3 = 0
        mcol4 = 0
        mcol5 = 0

    if (input == 'D'):
        # print(dcol1,dcol2,dcol3)

        for i in range(dcol1):
            total_results.append(batch_results[i][0])
        for i in range(dcol2):
            total_results.append(batch_results[i][1])
        for i in range(dcol3):
            total_results.append(batch_results[i][2])

        dcol1 = 0
        dcol2 = 0
        dcol3 = 0
    return total_results, mcol1, mcol2, mcol3, mcol4, mcol5, dcol1, dcol2, dcol3,batch_results

def read_omr(path_to_file,rollnum_ok,questionsok, mcol1, mcol2, mcol3, mcol4, mcol5, dcol1, dcol2, dcol3,batch_results,total_results,index):
    ref_height = 0

    #path_to_file = "C:\\Users\\GAndoo\\Desktop\\OMR\\200 dpi\\200 dpi\\image5.jpg"


    # image = cv2.imread("C:\\Users\\GAndoo\\Desktop\\OMR\\200 dpi\\200 dpi\\image5.jpg")
    image = cv2.imread(path_to_file)
    # ("Dimensions:", image.shape)
    check_box = []



    (height, width, channels) = image.shape
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (T, gray) = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    (Cnts, _) = cv2.findContours(gray.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    count = 0
    dummy = image.copy()
    for i in range(len(Cnts)):
        (x, y, w, h) = cv2.boundingRect(Cnts[i])

        # abs(h - 11.25*w)<h thaaa

        if abs(h - 11.25 * w) < h and (x > width / 30 and x <  width - width/30):  # and cv2.contourArea(Cnts[i]) > 400:

            count += 1
            check_box.append((x, y, w, h))
            cv2.rectangle(dummy, (x, y), (x + w, y + h), green, 2)

    cv2.imwrite("dummy.png", dummy)

    print("Total check box: ",len(check_box))

    if (len(check_box) == 2):

        if (check_box[0][1] < height / 2):
            inverted = 0
        else:
            inverted = 1
            image = imutils.rotate(image, 180)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            (T, gray) = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
            (Cnts, _) = cv2.findContours(gray.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            count += 1

    check_box = []
    for i in range(len(Cnts)):
        (x, y, w, h) = cv2.boundingRect(Cnts[i])
        if abs(h - 11.25 * w) < h and (x > width / 30 and x <  width - width/30):
            count += 1
            check_box.append((x, y, w, h))
    print(len(check_box))
    start_pt = 0
    if (check_box[0][0] > check_box[1][0]):
        dif = check_box[0][1] - check_box[1][1]
        start_pt = check_box[0][1]
    else:
        dif = check_box[1][1] - check_box[0][1]
        start_pt = check_box[1][1]
    # print(dif)
    angle = math.degrees(math.atan(dif / abs(check_box[0][0] - check_box[1][0])))
    # print("Angle: ", angle)

    # cv2.imshow("image", image)

    # cv2.waitKey(0)

    image = imutils.rotate(image, angle + 0.1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (T, gray) = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    (Cnts, _) = cv2.findContours(gray.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imshow("Rotated: ", image)
    # cv2.imwrite('temp.png', image)
    # cv2.waitKey(0)

    # image = cv2.imread("temp.png")
    (height, width, channels) = image.shape
    # printheight,width)
    if (height < 2000 and width < 1000):
        image = imutils.resize(image, width=1654, height=2338)
        (height, width, channels) = image.shape



    row_count = 0

    current_type = 0  # 1 for mcq ,2 for dec



    types = []
    box_image = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (T, gray) = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    # cv2.imwrite('gray.png', gray)
    (Cnts, _) = cv2.findContours(gray.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    inverted = 0
    count = 0

    check_box = []
    # print("ALL CNTS:",len(Cnts))

    # print"inverted? => ",inverted)

    mcqs_rows = 0
    decimals_rows = 0
    mcq_results = []
    mcqs_in_line = 0
    mcq_box = []
    decimal_box = []
    Cnts = sorted(Cnts, key=lambda ctr: cv2.boundingRect(ctr)[1])

    big_box = []
    largest_area = 0
    largest_box = None
    for i in range(len(Cnts)):
        (x, y, w, h) = cv2.boundingRect(Cnts[i])
        cv2.rectangle(box_image, (x, y), (x + w, y + h), green, 2)

        if (h > 18 * w and h < 22 * w):
            # print("ROLL-NO box FOUND!")
            cv2.rectangle(box_image, (x, y), (x + w, y + h), red, 2)

        if (cv2.contourArea(Cnts[i]) > largest_area) and (width - w > 30):
            cv2.rectangle(box_image, (x, y), (x + w, y + h), red, 2)
            largest_area = cv2.contourArea(Cnts[i])
            largest_box = (x, y, w, h)
    #cv2.imwrite("box_test.png", box_image)
    cv2.rectangle(image, (largest_box[0], largest_box[1]),
                  (largest_box[2] + largest_box[0], largest_box[3] + largest_box[1]), red, 3)

    big_box_width = largest_box[2]
    big_box_x = largest_box[0]

    # cv2.imshow("image", image)
    # cv2.imwrite("big_box.png", image)
    # cv2.waitKey(0)

    # ("Big boxes: ",big_box)
    bigbox_x = 0
    if len(big_box) == 1:
        ref_height = big_box[0][3]
        bigbox_x = big_box[0][0]

    ref_height = largest_box[3]
    bigbox_x = largest_box[0]
    big_box_y = largest_box[1]

    try:
        first_question = 0
        for i in range(len(Cnts)):

            (x, y, w, h) = cv2.boundingRect(Cnts[i])

            if (abs(ref_height * 0.1303 - cv2.contourArea(Cnts[i])) < 400 and abs(ref_height - h * 153.4) < 1000 and x < bigbox_x and abs(w - h) < (w / 3) and x < width / 6 and y > start_pt):

                if (first_question == 0):
                    first_question = y
                if current_type != 0:

                    if (current_type == 2):

                        current_type = 1
                        (total_results, mcol1, mcol2, mcol3, mcol4, mcol5, dcol1, dcol2, dcol3,batch_results)=update_batch('D',total_results, mcol1, mcol2, mcol3, mcol4, mcol5, dcol1, dcol2, dcol3,batch_results)
                        batch_results = []
                    else:

                        current_type = 1
                else:

                    current_type = 1

                row_count += 1

                mcq_box.append((x, y))
                # cv2.rectangle(image,(x,y),(x+w,y+h),red,2)
                # cv2.imshow("boxes:",image)
                # cv2.waitKey(0)
                # cv2.rectangle(image,(x+int(2.5)*h,y-h),(width - 5*h,y+2*h),green,2)
                clr_cropped = image.copy()[y - h:y + int(3 * h), x + int(3.0 * h):width - 5 * h]
                # cropped = gray[y-h:y+int (2.5*h) , x+int(2.5*h):width-5*h]
                cropped = cv2.cvtColor(clr_cropped, cv2.COLOR_BGR2GRAY)
                kernel = np.ones((3, 3), np.uint8)
                # cropped = cv2.dilate(cropped, kernel, iterations=1)
                cropped = cv2.GaussianBlur(cropped, (3, 3), 5)
                # cropped = cv2.GaussianBlur(cropped, (1, 1), 1)

                canny = cv2.Canny(cropped, 30, 150)
                # cv2.imshow("Cropped Image",canny)
                # cv2.waitKey(0)
                (cropped_cnts, _) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cropped_cnts = sorted(cropped_cnts, key=lambda ctr: cv2.boundingRect(ctr)[0])

                circles = []
                # print("Cropped cnts: ",len(cropped_cnts))
                for j in range(len(cropped_cnts)):
                    (X, Y, W, H) = cv2.boundingRect(cropped_cnts[j])
                    cv2.rectangle(clr_cropped, (X, Y), (X + W, Y + H), red, 1)
                    if (Y < cropped.shape[0] / 2 and X > 3 * h and X < cropped.shape[1] - 4 * h and X + W <
                            cropped.shape[
                                1] - 3 * h and W > 1.5 * h and H > 1.5 * h and W < 8 * w and H < 5 * h and not (
                                    Y < h / 5 and Y + H > (clr_cropped.shape[0] - h / 5))):
                        if len(circles) > 0:
                            if X > circles[-1][0] + circles[-1][2]:
                                cv2.rectangle(clr_cropped, (X, Y), (X + W, Y + H), green, 1)
                                circles.append((X, Y, W, H))
                                mcqs_in_line += 1

                        else:
                            cv2.rectangle(clr_cropped, (X, Y), (X + W, Y + H), green, 1)
                            circles.append((X, Y, W, H))
                            mcqs_in_line += 1
                        # cv2.imshow("all CNTS:", clr_cropped)
                        # print("circles: ", len(circles))
                        # cv2.waitKey(0)

                # cv2.imshow("Cropped Image.png", clr_cropped)
                # cv2.waitKey(0)
                types.append('M')
                (results, g) = get_row_answers(circles, clr_cropped, h)

                '''if (g<5 or len(circles)>25):

                        cv2.imshow("all CNTS:", clr_cropped)


                        print("circles: ", len(circles))
                        cv2.waitKey(0)
                '''
                if (g == 5):
                    mcol1 += 1
                    mcol2 += 1
                    mcol3 += 1
                    mcol4 += 1
                    mcol5 += 1
                if (g == 4):
                    mcol1 += 1
                    mcol2 += 1
                    mcol3 += 1
                    mcol4 += 1
                if (g == 3):
                    mcol1 += 1
                    mcol2 += 1
                    mcol3 += 1
                if (g == 2):
                    mcol1 += 1
                    mcol2 += 1
                if (g == 1):
                    mcol1 += 1

                batch_results.append(results)

                # print"mcqs_in_line ",mcqs_in_line)

                mcqs_rows += 1

            if (abs(ref_height * 0.260 - cv2.contourArea(Cnts[i])) < 400 and abs(
                    ref_height - h * 153.4) < 1000 and x < bigbox_x and abs(w - 2 * h) < (w / 3) and x < width / 6):
                row_count += 1

                if (first_question == 0):
                    first_question = y

                if current_type != 0:

                    if (current_type == 1):

                        current_type = 2
                        (total_results, mcol1, mcol2, mcol3, mcol4, mcol5, dcol1, dcol2, dcol3,batch_results)=update_batch('M',total_results, mcol1, mcol2, mcol3, mcol4, mcol5, dcol1, dcol2, dcol3,batch_results)
                        batch_results = []
                    else:
                        current_type = 2
                else:

                    current_type = 2

                # cv2.rectangle(image,(x,y),(x+w,y+h),green,2)
                # cv2.rectangle(image,(x+int(2.5)*h,y-h),(width - 5*h,y+2*h),green,2)
                clr_cropped = image.copy()[int(y - 0.7 * h):y + int(3.0 * h), x + int(2.5 * h):width - 5 * h]
                # cropped = gray[y-h:y+int(2.5*h) , x+int(2.5*h):width-5*h]

                cropped = cv2.cvtColor(clr_cropped, cv2.COLOR_BGR2GRAY)
                cropped = cv2.GaussianBlur(cropped, (3, 3), 5)

                canny = cv2.Canny(cropped, 30, 150)

                (cropped_cnts, _) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # (cropped_cnts, _) = cv2.findContours(cropped.copy(), cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
                cropped_cnts = sorted(cropped_cnts, key=lambda ctr: cv2.boundingRect(ctr)[0])

                circles = []
                for j in range(len(cropped_cnts)):
                    (X, Y, W, H) = cv2.boundingRect(cropped_cnts[j])
                    cv2.rectangle(clr_cropped, (X, Y), (X + W, Y + H), red, 1)
                    if (Y < cropped.shape[0] / 2 and X > 3 * h and X < cropped.shape[1] - 4 * h and X + W <
                            cropped.shape[
                                1] - 3 * h and W > 1.5 * h and H > 1.5 * h and W < 10 * w and H < 10 * h and not (
                                    Y < h / 5 and Y + H > (clr_cropped.shape[0] - h / 5))):

                        if len(circles) > 0:
                            if X > circles[-1][0] + circles[-1][2]:
                                cv2.rectangle(clr_cropped, (X, Y), (X + W, Y + H), green, 1)
                                circles.append((X, Y, W, H))
                                mcqs_in_line += 1

                        else:
                            cv2.rectangle(clr_cropped, (X, Y), (X + W, Y + H), green, 1)
                            circles.append((X, Y, W, H))
                            mcqs_in_line += 1

                # cv2.imshow("Cropped Image", clr_cropped)
                # cv2.waitKey(0)
                # print("DECI CIRCLES:", len(circles))
                types.append('D')
                # print(len(circles))
                (results, g) = get_decimal_row_answers(clr_cropped, circles, h)

                if (g == 3):
                    dcol1 += 1
                    dcol2 += 1
                    dcol3 += 1
                if (g == 2):
                    dcol1 += 1
                    dcol2 += 1
                if (g == 1):
                    dcol1 += 1

                batch_results.append(results)
                decimal_box.append((x, y))
                decimals_rows += 1

            if (i == len(Cnts) - 1):
                if (current_type == 1):
                    (total_results, mcol1, mcol2, mcol3, mcol4, mcol5, dcol1, dcol2, dcol3,batch_results) = update_batch('M',total_results, mcol1, mcol2, mcol3, mcol4, mcol5, dcol1, dcol2, dcol3,batch_results)
                if (current_type == 2):
                    (total_results, mcol1, mcol2, mcol3, mcol4, mcol5, dcol1, dcol2, dcol3,batch_results)=update_batch('D',total_results, mcol1, mcol2, mcol3, mcol4, mcol5, dcol1, dcol2, dcol3,batch_results)
    except:

        questionsok = False

    # print"mcqs_rows", mcqs_rows)
    # print"decimals_rows  ",decimals_rows )
    roll_number = ""
    try:
        (roll_number,rollnum_ok) = read_roll_number(first_question, big_box_x, big_box_y, big_box_width, image,rollnum_ok)


    except:
        rollnum_ok = False

    DATA = {}
    if not rollnum_ok:
        # roll number not OKAY.
        DATA["roll_success"] = "false"
    else:
        DATA["roll_success"] = "true"
        DATA["roll_number"] = roll_number
    if questionsok:
        DATA["questions_success"] = "true"
    else:
        DATA["questions_success"] = "false"

    # cv2.imwrite("scanned_Results2.png", image)
    MCQS = 0
    DECIMALS = 0
    for i in range(len(mcq_results)):
        if (len(mcq_results[i]) == 3):
            DECIMALS += 1
        else:
            MCQS += 1


    final_answers = []
    for i in range(len(total_results)):
        final_answers.append(",".join(total_results[i]))

    DATA["results"] = final_answers

    output_file = "output"+str(index)+".txt"
    with open(output_file, 'w') as outfile:
        json.dump(DATA, outfile)


def reset_all(rollnum_ok,questionsok, mcol1, mcol2, mcol3, mcol4, mcol5, dcol1, dcol2, dcol3,batch_results,total_results):
    questionsok = True
    rollnum_ok = True

    mcol1 = 0
    mcol2 = 0
    mcol3 = 0
    mcol4 = 0
    mcol5 = 0

    dcol1 = 0
    dcol2 = 0
    dcol3 = 0

    total_results = []
    batch_results = []

    return (rollnum_ok,questionsok, mcol1, mcol2, mcol3, mcol4, mcol5, dcol1, dcol2, dcol3,batch_results,total_results)



#for multiple files..
from os import listdir
from os.path import isfile, join
mypath = "testing_images\\"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for i in range(len(onlyfiles)):
    (rollnum_ok, questionsok, mcol1, mcol2, mcol3, mcol4, mcol5, dcol1, dcol2, dcol3, batch_results,total_results) = reset_all(rollnum_ok, questionsok, mcol1, mcol2, mcol3, mcol4, mcol5, dcol1, dcol2, dcol3,batch_results, total_results)
    read_omr(mypath+onlyfiles[i], rollnum_ok, questionsok, mcol1, mcol2, mcol3, mcol4, mcol5, dcol1, dcol2,dcol3, batch_results, total_results,i)

