import cv2

import numpy as np
import json

font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10, 500)
fontScale = 2
fontColor = (255, 255, 255)
lineType = 5


def create_circles(circles, options, x, y, image, type, radius, gap):
    radius = radius

    gap = gap
    for i in range(circles):
        cv2.circle(image, (x, y), radius, black, 3)

        if (type == "mcq"):
            cv2.putText(image, options[i], (x - 10, y + 10), font, 1, black, 2)

        x = x + 2 * radius + gap

    return image


def write_options(options_decimals, image, x, y):
    gap = 60
    for i in range(len(options_decimals)):
        cv2.putText(image, options_decimals[i], (x - 45, y + 45), font, 1, black, lineType)
        x = x + gap
    return image


'''
def add_logo(image,logo):
    global offset

    logo = cv2.resize(logo,(600,600), interpolation = cv2.INTER_AREA)

    #image[margin+10:margin+logo.shape[0]+10, width-margin-logo.shape[0]-10:width-margin-10] = logo
    cv2.line(image,(margin,margin+logo.shape[0]),(width-margin,margin+logo.shape[0]),black,2)
    offset = margin+logo.shape[0]+100
    return image
'''


def write_roll(options_decimals, image, x, y, h):
    gap = 50
    for i in range(10):
        cv2.putText(image, str(options_decimals[i]), (x - 45 - 50, y + 45), font, 1, black, lineType)
        y = y + h
    return image


def add_mcqs(image, total, options, total_count, offset, mcq_height, height, margin):
    col4 = 0
    col3 = 0
    col1 = 0
    col2 = 0
    col5 = 0
    temp = total
    while (temp != 0):

        if temp >= 5:
            temp -= 5
            col1 += 1
            col2 += 1
            col3 += 1
            col4 += 1
            col5 += 1
        else:

            if temp == 4:
                col1 += 1
                col2 += 1
                col3 += 1
                col4 += 1
                temp -= 4
            if temp == 3:
                col1 += 1
                col2 += 1
                col3 += 1
                temp -= 3

            if temp == 2:
                col1 += 1
                col2 += 1
                temp -= 2
            if temp == 1:
                col1 += 1
                temp -= 1

    inc = 8

    dist = 0
    thick = 2
    offset += 50  # space for question 100 THI

    print("Col1: ", col1, "   Col2:  ", col2, "   col3:  ", col3)

    in_limit = 0
    if (offset + col1 * mcq_height + 20 <= height - margin):
        in_limit = 1

    for i in range(col1):
        # cv2.rectangle(image,(margin,margin+ i*mcq_height),(margin+mcq_width,margin+ i*mcq_height+mcq_height),red,2)
        cv2.rectangle(image, (margin - 100, -10 + offset + i * mcq_height + int(mcq_height / 2)),
                      (margin - 80, 10 + offset + i * mcq_height + int(mcq_height / 2)), black, -5)
        cv2.putText(image, str(total_count + i + 1),
                    (margin + 13 + 9 + dist, inc + 4 + offset + i * mcq_height + int(mcq_height / 2)), font, 1, black,
                    3)

        # cv2.line(image,(margin-100,+margin + i*mcq_height+int(mcq_height/2)),(width-margin,margin+ i*mcq_height+int(mcq_height/2)),black,2)
        image = create_circles(len(options), options, margin + 106 + 6, offset + i * mcq_height + int(mcq_height / 2),
                               image, "mcq", 20, 20)
    if (in_limit):
        cv2.rectangle(image, (margin + 16, offset - 20), (margin + 385 + 8, offset + col1 * mcq_height + 20), black, 2)
    else:
        cv2.rectangle(image, (margin + 16, offset - 20), (margin + 8 + 385, height - margin), black, 2)

    delta = 23

    for i in range(col2):
        # cv2.rectangle(image,(half+40,margin+ i*mcq_height),(width - margin,margin+ i*mcq_height+mcq_height),red,2)
        # cv2.rectangle(image,(530+margin,-8+offset + i*mcq_height+int(mcq_height/2)),(530+margin+16,8+offset + i*mcq_height+int(mcq_height/2)),black,-5)
        cv2.putText(image, str(total_count + i + 1 + col1),
                    (margin + 410 + delta + dist, inc + 4 + offset + i * mcq_height + int(mcq_height / 2)), font, 1,
                    black, 3)
        image = create_circles(len(options), options, delta + margin + 510,
                               offset + i * mcq_height + int(mcq_height / 2, ), image, "mcq", 20, 20)
    if in_limit:
        cv2.rectangle(image, (margin + 404 + delta, offset - 20),
                      (margin + 789 + delta, offset + col2 * mcq_height + 20), black, 2)
    else:
        cv2.rectangle(image, (margin + delta + 404, offset - 20), (margin + delta + 789, height - margin), black, 2)

    for i in range(col3):
        # cv2.rectangle(image,(half+40,margin+ i*mcq_height),(width - margin,margin+ i*mcq_height+mcq_height),red,2)
        # cv2.rectangle(image,(1162+margin,-8+offset + i*mcq_height+int(mcq_height/2)),(1162+margin+16,8+offset + i*mcq_height+int(mcq_height/2)),black,-5)
        cv2.putText(image, str(total_count + i + 1 + col2 + col1),
                    (margin + 820 + delta + dist, inc + 4 + offset + i * mcq_height + int(mcq_height / 2)), font, 1,
                    black, 3)
        image = create_circles(len(options), options, margin + delta + 920,
                               offset + i * mcq_height + int(mcq_height / 2, ), image, "mcq", 20, 20)

    if in_limit:
        cv2.rectangle(image, (margin + 814 + delta, offset - 20),
                      (margin + 1199 + delta, offset + col3 * mcq_height + 20), black, 2)
    else:
        cv2.rectangle(image, (margin + 814 + delta, offset - 20), (margin + delta + 1199, height - margin), black, 2)

    for i in range(col4):
        # cv2.rectangle(image,(margin,margin+ i*mcq_height),(margin+mcq_width,margin+ i*mcq_height+mcq_height),red,2)
        # cv2.rectangle(image,(margin-100,-8+offset + i*mcq_height+int(mcq_height/2)),(margin-84,8+offset + i*mcq_height+int(mcq_height/2)),black,-5)
        cv2.putText(image, str(total_count + i + 1 + col1 + col2 + col3),
                    (margin + delta + 1230 + dist, inc + 4 + offset + i * mcq_height + int(mcq_height / 2)), font, 1,
                    black, 3)
        # cv2.line(image,(margin-100,+margin + i*mcq_height+int(mcq_height/2)),(width-margin,margin+ i*mcq_height+int(mcq_height/2)),black,2)
        image = create_circles(len(options), options, margin + 1330 + delta,
                               offset + i * mcq_height + int(mcq_height / 2), image, "mcq", 20, 20)

    if in_limit:
        cv2.rectangle(image, (margin + 1224 + delta, offset - 20),
                      (margin + 1609 + delta, offset + col4 * mcq_height + 20), black, 2)
    else:
        cv2.rectangle(image, (margin + delta + 1224, offset - 20), (margin + delta + 1609, height - margin), black, 2)
    for i in range(col5):
        # cv2.rectangle(image,(margin,margin+ i*mcq_height),(margin+mcq_width,margin+ i*mcq_height+mcq_height),red,2)
        # cv2.rectangle(image,(margin-100,-8+offset + i*mcq_height+int(mcq_height/2)),(margin-84,8+offset + i*mcq_height+int(mcq_height/2)),black,-5)
        cv2.putText(image, str(total_count + i + 1 + col1 + col2 + col3 + col4),
                    (margin + delta + dist + 1640, inc + 4 + offset + i * mcq_height + int(mcq_height / 2)), font, 1,
                    black, 3)
        # cv2.line(image,(margin-100,+margin + i*mcq_height+int(mcq_height/2)),(width-margin,margin+ i*mcq_height+int(mcq_height/2)),black,2)
        image = create_circles(len(options), options, margin + delta + 1740,
                               offset + i * mcq_height + int(mcq_height / 2), image, "mcq", 20, 20)

    if in_limit:
        cv2.rectangle(image, (margin + 1634 + delta, offset - 20),
                      (margin + 2019 + delta, offset + col5 * mcq_height + 20), black, 2)
    else:
        cv2.rectangle(image, (margin + 1634 + delta, offset - 20), (margin + 2019 + delta, height - margin), black, 2)

    # now writing options for DECIMALS

    offset += mcq_height * col1
    total_count += total

    return image, total_count, offset


def add_decimals(image, total, options, answers, total_count, offset, mcq_height, height, width, margin):
    col1 = 0
    col2 = 0
    col3 = 0

    temp = total
    while (temp != 0):

        if (temp >= 3):
            temp -= 3

            col1 += 1
            col2 += 1
            col3 += 1
        else:

            if (temp == 2):
                col2 += 1
                col1 += 1
                temp -= 2
            if temp == 1:
                col1 += 1
                temp -= 1

    offset += 50

    dist = 5
    print("Col1: ", col1, "   Col2:  ", col2, "  col3:  ", col3)
    image = write_options(options, image, margin + 136, offset - 20)
    image = write_options(options, image, margin + 810, offset - 20)
    image = write_options(options, image, margin + 1500, offset - 20)
    offset += 60

    in_limit = 0

    if (offset + col1 * mcq_height + 20 <= height - margin):
        in_limit = 1
    # print("Col1: ",col1,"   Col2:  ",col2)
    for i in range(col1):
        # cv2.rectangle(image,(margin,margin+ i*mcq_height),(margin+mcq_width,margin+ i*mcq_height+mcq_height),red,2)
        cv2.rectangle(image, (margin - 100, -10 + offset + i * mcq_height + int(mcq_height / 2)),
                      (margin - 60, 10 + offset + i * mcq_height + int(mcq_height / 2)), black, -5)
        cv2.putText(image, str(total_count + i + 1),
                    (margin + 6 + dist, offset + 4 + i * mcq_height + int(mcq_height / 2)), font, 1, black, 3)
        # cv2.rectangle(image,(margin+3,offset+4 + i*mcq_height),(margin+80,offset+4 + i*mcq_height))
        image = create_circles(answers, options, margin + 100, offset + i * mcq_height + int(mcq_height / 2), image,
                               "deci", 20, 20)
    if (in_limit):
        cv2.rectangle(image, (margin, offset - 20), (margin + 670, offset + col1 * mcq_height + 20), black, 2)
    else:
        cv2.rectangle(image, (margin, offset - 20), (margin + 670, height - margin), black, 2)
    cv2.rectangle(image, (margin, offset - 70), (margin + 670, offset - 20), black, 2)
    for i in range(col2):
        # cv2.rectangle(image,(margin,margin+ i*mcq_height),(margin+mcq_width,margin+ i*mcq_height+mcq_height),red,2)
        # cv2.rectangle(image,(margin+1130-100,-8+offset + i*mcq_height+int(mcq_height/2)),(margin+1130-68,8+offset + i*mcq_height+int(mcq_height/2)),black,-5)
        cv2.putText(image, str(total_count + i + 1 + col1),
                    (margin + 675 + dist, offset + 4 + i * mcq_height + int(mcq_height / 2)), font, 1, black, 3)
        image = create_circles(answers, options, margin + 780, offset + i * mcq_height + int(mcq_height / 2), image,
                               "deci", 20, 20)

    if in_limit:
        cv2.rectangle(image, (margin + 670, offset - 20), (margin + 1350, offset + col1 * mcq_height + 20), black, 2)
    else:
        cv2.rectangle(image, (margin + 670, offset - 20), (margin + 1350, height - margin), black, 2)
    cv2.rectangle(image, (margin + 670, offset - 70), (margin + 1350, offset - 20), black, 2)

    for i in range(col3):
        # cv2.rectangle(image,(margin,margin+ i*mcq_height),(margin+mcq_width,margin+ i*mcq_height+mcq_height),red,2)
        # cv2.rectangle(image,(margin-100,-8+offset + i*mcq_height+int(mcq_height/2)),(margin-68,8+offset + i*mcq_height+int(mcq_height/2)),black,-5)
        cv2.putText(image, str(total_count + i + 1 + col1 + col2),
                    (margin + 1360 + dist, offset + 4 + i * mcq_height + int(mcq_height / 2)), font, 1, black, 3)
        image = create_circles(answers, options, margin + 1470, offset + i * mcq_height + int(mcq_height / 2), image,
                               "deci", 20, 20)
    if in_limit:
        cv2.rectangle(image, (margin + 1350, offset - 20), (width - margin, offset + col1 * mcq_height + 20), black, 2)
    else:
        cv2.rectangle(image, (margin + 1350, offset - 20), (width - margin, height - margin), black, 2)
    cv2.rectangle(image, (margin + 1350, offset - 70), (width - margin, offset - 20), black, 2)

    offset += mcq_height * col1
    total_count += total
    return image, total_count, offset


green = (0, 255, 0)
red = (0, 0, 255)
black = (0, 0, 0)


def create_OMR(path_of_json):
    total_count = 0
    canvas = np.zeros((3508, 2480, 3), dtype="uint8")
    canvas[:, :] = (255, 255, 255)
    image = canvas

    (height, width, channels) = image.shape

    '''margin was 220'''

    margin = 210
    half = int(width / 2)
    page_height = height - 2 * margin
    mcq_width = half - 40 - margin
    mcq_height = 60

    # here is the path of the JSON input file
    file = open(path_of_json)
    data = json.load(file)

    roll_height = data["rollNoLength"]

    offset = margin + 80

    col1 = roll_height
    # print("Col1: ",col1,"   Col2:  ",col2)
    digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    digit_height = 50

    roll_off = 150
    roll_off_y = 50
    image = write_roll(digits, image, roll_off + margin + 120 + 50 + 15 - 80 + 15, roll_off_y + margin + 68 + 4,
                       mcq_height)
    # cv2.putText(image,"Roll No.#", (margin + 6,margin + 50), font, 1,black,1)
    # cv2.line(image, (margin, margin+ 58), (300+margin, margin+58), black,2)
    for i in range(10):
        # cv2.rectangle(image,(margin,margin+ i*mcq_height),(margin+mcq_width,margin+ i*mcq_height+mcq_height),red,2)
        # cv2.rectangle(image,(margin-60,-2+offset + i*mcq_height+int(mcq_height/2)),(margin-20,2+offset + i*mcq_height+int(mcq_height/2)),black,-5)
        # cv2.putText(image,str(i+1)+".", (margin + 6,offset+4 + i*mcq_height+int(mcq_height/2)), font, 1,black,3)
        image = create_circles(roll_height, None, roll_off + margin + 75 + 15,
                               roll_off_y + -2 + offset + i * mcq_height + int(mcq_height / 2),
                               image, "roll", 20, 15)

    cv2.putText(image, "Roll No:", (margin + 6 + 15, 25 + margin + 48), font, 1, black, 3)
    cv2.rectangle(image, (roll_off + margin + 15, roll_off_y + margin + 60),
                  (roll_off + margin + 15 + mcq_height * roll_height + 25, roll_off_y + margin + 700), black,
                  2)

    cv2.rectangle(image, (roll_off + margin + 15, roll_off_y + margin + 60 - 75),
                  (roll_off + margin + 15 + mcq_height * roll_height + 25, roll_off_y + margin + 60 - 10), black, 2)
    # ROLL NUMBER READING BOXES...
    zero_mark = 40

    cv2.rectangle(image, (roll_off + margin + 15 , roll_off_y + margin + 60 - 75 ),
                  (roll_off + margin + 15 + zero_mark + 10 , roll_off_y + margin + 60 + 65 - 75 ),
                  black, -1)
    #cv2.rectangle(image, (roll_off + margin + 15-1  , roll_off_y + margin + 60 - 75-1),
     #             (roll_off + margin + 15 + zero_mark+10-1, roll_off_y + margin + 60 + 65 - 75+1 ), (255,255,255), -1)

    zero_mark = roll_off + margin + 15 + zero_mark + 10
    cv2.line(image, (zero_mark , roll_off_y + margin + 60 - 75),
             (zero_mark , roll_off_y + margin + 60 + 65 - 75), black, 2)
    distance = 54


    for lines in range(0,roll_height):

        cv2.line(image, (zero_mark+distance,roll_off_y + margin + 60 - 75),(zero_mark+distance , roll_off_y + margin + 60 + 65 - 75),black,2)
        zero_mark+=distance

    zero_mark-=distance
    cv2.rectangle(image, (zero_mark + distance , roll_off_y + margin + 60 - 75 ),
                  (zero_mark + distance + 24, roll_off_y + margin + 60 + 65 - 75 ), black, -1)
    #cv2.rectangle(image, ( zero_mark+distance+2,roll_off_y + margin + 60 - 75-1) ,(zero_mark+distance + 24,roll_off_y + margin + 60 + 65 - 75+1),(255,255,255),-1)


    box_offset = 40
    len_offset = 950
    cv2.putText(image, "Name:", (margin + 10 * digit_height + len_offset, margin + 55), font, 1, black, 3)


    cv2.rectangle(image, (margin + 10 * digit_height + len_offset, margin + 68),
                  (width - margin - 15, margin + 168 + box_offset), black,
                  3)  # name

    cv2.putText(image, "Date:", (margin + 10 * digit_height + len_offset, margin + 200 + 55), font, 1, black, 3)
    cv2.rectangle(image, (margin + 10 * digit_height + len_offset, margin + 68 + 200),
                  (width - margin - 15, margin + 168 + 200 + box_offset), black, 3)  # date
    sig_offset = 70
    cv2.putText(image, "Signature:", (margin + 10 * digit_height + len_offset, margin + 455), font, 1, black, 3)
    cv2.rectangle(image, (margin + 10 * digit_height + len_offset, margin + 68 + 400),
                  (width - margin - 15, margin + 580 + box_offset + sig_offset),
                  black, 3)  # signature
    offset = margin + digit_height * roll_height + 30 + 150

    cv2.rectangle(image, (margin, margin), (width - margin, height - margin), black, 6)

    cv2.rectangle(image, (int(margin / 2), margin), (int(margin / 2) + 16, margin + 180), black, -1)
    cv2.rectangle(image, (int(margin / 2) + width - margin, margin),
                  (int(margin / 2) + width - margin + 16, margin + 180), black, -1)
    # logo = cv2.imread("logo.png")
    # image = add_logo(image,logo)
    offset = margin + 750  # 700 = size of logo +
    cv2.line(image, (margin, offset + 15), (width - margin, offset + 15), black, 3)

    groups = (data["questions"])

    for i in range(len(groups)):
        type = groups[i]["type"]
        count = groups[i]["count"]
        options = groups[i]["options"]

        if (type == "M"):
            (image, total_count, offset) = add_mcqs(image, count, options, total_count, offset, mcq_height, height,
                                                    margin)

        if (type == "D"):
            (image, total_count, offset) = add_decimals(image, count, options, len(options), total_count, offset,
                                                        mcq_height, height, width, margin)

    logo_url = data["logoUrl"]

    low_logo = cv2.imread(logo_url)
    # print("LOGO size = ", low_logo.shape)
    # cv2.imshow("logo",low_logo)
    # cv2.waitKey(0)
    # low_logo = cv2.resize(low_logo,(500,106))
    # print("LOGO size = ",low_logo.shape)
    h_off = 75
    image[height - margin + h_off:height - margin + h_off + low_logo.shape[0], half - int(low_logo.shape[1] / 2):half + int(low_logo.shape[1] / 2)] = low_logo
    ##cv2.rectangle(image, (half - int(low_logo.shape[1] / 2) - 10, height - margin + 10),
    #             (half + int(low_logo.shape[1] / 2) + 10, height - margin + 30 + low_logo.shape[0]), black, 4)

    name = data["fileName"]

    cv2.imwrite(name + ".png", image)
    print("\n====================\n    OMR created & saved!\n")
    '''image = cv2.imread("results.png")
    cv2.imshow("OMR",image)
    cv2.waitKey(0)
    '''


create_OMR("test_json.json.txt")