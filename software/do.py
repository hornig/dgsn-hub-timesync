import numpy as np
import sys

import hashlib

def pairs(a,b):
    for index in range(min(len(a), len(b))):
        yield (a[index], b[index])

def convert_a(x):
    out = -127 + x
    out = (out[::2]**2 + out[1::2]**2)**0.5
    return out

def convert_b(x):
    out = np.zeros(int(len(x)/2.0))
    for i in range(0,len(x),2):
        out[int(i/2)] = ( (int(x[i]) -127)**2 + (int(x[i+1]) -127)**2 )**0.5
    return out

def do_sha224(x):
    hashed = hashlib.sha224(x)
    hashed = hashed.hexdigest()
    return hashed

def smooth_line(x,y, iter):
    x1 = []
    y1 = []
    for i in range(len(x)):
        x1.append(x[i])
        y1.append(y[i])

    b, a, rxy, Rsquared, y_hat, y_residuum = linear_regression(x1,y1)

    print("smoothing", i, len(x), len(y), len(x1), len(y1), b, a, rxy, Rsquared, np.argmax(np.abs(y_residuum)))
    print(np.argmax(np.abs(y_residuum)), np.max(np.abs(y_residuum)))
    print(np.sum(np.abs(y_residuum)))#, np.abs(y_residuum))
    #print(x1
    #print(y1
    for i in range(iter):
        del x1[np.argmax(np.abs(y_residuum))]
        del y1[np.argmax(np.abs(y_residuum))]
        del y_residuum[np.argmax(np.abs(y_residuum))]
        #print(x1
        #print(y1
    print("")
    b, a, rxy, Rsquared, y_hat, y_residuum = linear_regression(x1,y1)
    # conditions:
    # no negative slope for time!
    # slope must be near to 1!

    y_hat1 = []
    for i in range(len(x)):
        y_hat1.append(a + b * x[i])

    return x, y_hat1


def linear_regression(x, y):
    x_sum = np.sum(x)
    y_sum = np.sum(y)
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    Sx = x - x_mean
    Sy = y - y_mean

    Sxy = Sx * Sy
    Sxx = Sx * Sx
    Syy = Sy * Sy

    Sxy_sum = np.sum(Sxy)
    Sxx_sum = np.sum(Sxx)
    Syy_sum = np.sum(Syy)

    b = Sxy_sum / Sxx_sum # slope
    a = y_sum / len(y) - (b * x_sum / len(x)) # achsenabschnitt der regr. geraden

    rxy = Sxy_sum / (Sxx_sum * Syy_sum)**0.5 # empirische korelation
    Rsquared = rxy**2 # bestimtheitsmass

    y_hat = []
    y_residuum = []
    for i in range(len(x)):
        y_hat.append(a + b * x[i])
        y_residuum.append(y[i] - y_hat[i])

    return b, a, rxy, Rsquared, y_hat, y_residuum

def find_line_properties(points):
    m = float(points[1][1] - points[0][1]) / (points[1][0] - points[0][0] + sys.float_info.epsilon)  # slope (gradient) of the line
    c = points[1][1] - m * points[1][0]                                     # y-intercept of the line
    return m, c

def find_intercept_point(m, c, x0, y0):
    # intersection point with the model
    x = (x0 + m*y0 - m*c)/(1 + m**2)
    y = (m*x0 + (m**2)*y0 - (m**2)*c)/(1 + m**2) + c

    return x, y

def bad_sphere_center(long, lat, alt):
    import math

    dots = np.zeros((len(long),4))
    for i in range(len(long)):
        dots[i][0] = long[i]
        dots[i][1] = lat[i]
        dots[i][2] = alt[i][1]
        dots[i][3] = i
    print(dots)

    iterations = 40000
    distance_allowed = 250

    counterbest = 0
    Rearth = 6371000.0

    found_dots = 0

    for l in range(2000):
        np.random.shuffle(dots)
        dots_sum = 0

        x0 = (Rearth + dots[0][2]) * math.cos(dots[0][1]) * math.cos(dots[0][0])
        y0 = (Rearth + dots[0][2]) * math.cos(dots[0][1]) * math.sin(dots[0][0])
        z0 = (Rearth + dots[0][2]) * math.sin(dots[0][1])

        index = []
        index.append(dots[0][3])

        for i in range(1, len(long)):
            x1 = (Rearth + dots[i][2]) * math.cos(dots[i][1]) * math.cos(dots[i][0])
            y1 = (Rearth + dots[i][2]) * math.cos(dots[i][1]) * math.sin(dots[i][0])
            z1 = (Rearth + dots[i][2]) * math.sin(dots[i][1])

            distance = ((x0-x1)**2 + (y0-y1)**2 + (z0-z1)**2)**0.5

            if distance < distance_allowed:
                dots_sum += 1
                index.append(dots[i][3])

        if found_dots < dots_sum:
            out4 = []
            found_dots = dots_sum
            dots_center = [dots[0][0], dots[0][1], dots[0][2], dots[0][3]]
            for i in range(len(dots)):
                yes = 0
                for j in range(len(index)):
                    if dots[i][3] == index[j]:
                        yes = 1

                if yes == 1:
                    out4.append(dots[i][3])

    print("center", dots_center, found_dots)
    return out4

def bad_ransac(x, y):

    dots = np.zeros((len(x),2), dtype=np.float64)
    for i in range(len(x)):
        dots[i][0] = float(x[i])
        dots[i][1] = float(y[i])
        #print(dots[i][0], dots[i][1])
    print("dots")
    #print(dots)

    iterations1 = 70000
    iterations = iterations1
    distance_allowed = 1.0
    ppp = 12

    counterbest = 0

    while iterations >= 0:
        np.random.shuffle(dots)

        m, c = find_line_properties(dots[0:2])
        '''
        xx = [dots[0][0], dots[1][0]]
        yy = [dots[0][1], dots[1][1]]

        if xx[1] - xx[0] == 0:
            print(xx[1] - xx[0])
        tmp = linregress(xx,yy)
        #print(tmp)
        m = tmp.slope
        c = tmp.intercept
        '''


        counter = 0
        for i in range(2,len(dots)):
            x1, y1 = find_intercept_point(m, c, dots[i][0], dots[i][1])
            distance = ((x1-dots[i][0])**2 + (y1-dots[i][1])**2)**0.5
            #print(dots[i][0], dots[i][1], distance)

            if distance < distance_allowed:
            # didn't work that good with noisy measurements
            #if m*2048000.0 < 1.5 and m*2048000.0 > 0.5:
               counter += 1

        #print("counter", counter, len(x)-2

        #if float(counter)/float(len(x)-2) > 0.2 and counterbest < counter:
        if counterbest < counter and np.abs(m*2048000.0 - 1.0) < 1.0:
            #print("hgggg"
            dotsbest = dots[0:2]
            mbest = m
            cbest = c
            counterbest = counter

        iterations -= 1
        if iterations <= 1 and counter <= 1 and ppp > 0:
            iterations = iterations1
            ppp -= 1

    y_new = []

    passed = 0
    if counter > 0:
        print("counter", counter, "ppp", ppp, mbest, cbest, dotsbest[0][0], dotsbest[0][1],dotsbest[1][0], dotsbest[1][1])
        for i in range(len(x)):
            y_new.append(np.round(cbest + mbest*x[i]))

        #print("final linear properties", cbest, mbest, mbest*2048000.0, float(counterbest)/float(len(x)-2), counterbest)
        passed = 1
    else:
        for i in range(len(x)):
            #print("counter", counter
            y_new.append(float(y[i]))

        #print("final normal properties", counter)
        passed = 0

    return x, y_new, passed

def bad_ransac_lin(x, y):
    distance_allowed = 1.0
    counterbest = 0

    for i in range(len(x)-1):
        for j in range(i+1, len(x)):
            #print(i,j)
            dots = [[x[i],y[i]],[x[j],y[j]]]
            m, c = find_line_properties(dots)


            counter = 0
            for k in range(len(x)):
                x1, y1 = find_intercept_point(m, c, x[k], y[k])
                distance = ((x1-x[k])**2 + (y1-y[k])**2)**0.5
                #print(dots[i][0], dots[i][1], distance)

                if distance < distance_allowed:
                # didn't work that good with noisy measurements
                #if m*2048000.0 < 1.5 and m*2048000.0 > 0.5:
                    counter += 1

             #if float(counter)/float(len(x)-2) > 0.2 and counterbest < counter:
            if counterbest < counter and np.abs(m*2048000.0 - 1.0) < 1.0:
                #print("hgggg"
                dotsbest = dots
                mbest = m
                cbest = c
                counterbest = counter

                y_new = []

    passed = 0
    if counter > 0:
        print("counter", counter, "ppp", mbest, cbest, dotsbest[0][0], dotsbest[0][1],dotsbest[1][0], dotsbest[1][1])
        for i in range(len(x)):
            y_new.append(np.round(cbest + mbest*x[i]))

        #print("final linear properties", cbest, mbest, mbest*2048000.0, float(counterbest)/float(len(x)-2), counterbest)
        passed = 1
    else:
        for i in range(len(x)):
            #print("counter", counter
            y_new.append(float(y[i]))

        #print("final normal properties", counter)
        passed = 0
    print(passed)
    return x, y_new, passed