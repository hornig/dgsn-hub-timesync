import numpy as np
#import matplotlib.pylab as plt
#import tiqs
import time
#import do
import os
import importlib

def stream_seq(stream, seq_start, frame, extra):
    seq_len = frame + extra
    return np.abs(stream[seq_start : seq_start + seq_len : 2]+ 1j * stream[seq_start + 1 : seq_start + seq_len : 2])


def find_file_in_folder(folder, needle):
    # traverse root directory, and list directories as dirs and files as files
    list = []
    for root, dirs, files in os.walk(folder):
        path = root.split(os.sep)
        #print((len(path) - 1) * '---', os.path.basename(root))
        for file in files:
            if file.find(needle) > -1:
                list.append(root + os.path.sep + file)
                #print(root + os.path.sep + file)
    return list

def run(path_storing, input_filename):
    load = str.join(".", (find_file_in_folder(".", "do.py")[0][1:].split(".")[0].split(os.path.sep)[1:]))
    do = importlib.import_module(load)

    load = str.join(".", (find_file_in_folder(".", "tiqs.py")[0][1:].split(".")[0].split(os.path.sep)[1:]))
    tiqs = importlib.import_module(load)

    stream = np.memmap(input_filename)
    stream = -127 + stream

    gaps_gps_code = []


    time1 = time.time()
    samples_per_second = 4000000
    print(int(len(stream)/samples_per_second))

    gps_tiqs = []

    for i in range(int(len(stream)/samples_per_second)):
        print(i)
        bin = 10 # must be possible to keep the shape of the original array
        extra_samples = 0
        binned = np.mean(np.array(stream_seq(stream, samples_per_second*i, samples_per_second, extra_samples)).reshape((-1, bin)), axis=1)
        #print(binned)
        #print("binning1", time.time() - time1)

        binned_min = np.min(binned)
        threshold = binned_min + (tiqs.statistics(binned, 2)-binned_min)/3.0
        #print("threshold", time.time() - time1)

        gap_state= tiqs.find_gaps1(binned, threshold)
        #print("state", time.time() - time1)

        gap_gps_1pps = tiqs.gap_resetter1(gap_state, 6, 15) # gps tiqs
        gap_dab_synch = tiqs.gap_resetter1(gap_state, 240, 280) # gps tiqs
        #print("gps 1pps", time.time() - time1)
        gap_gps_code = tiqs.gap_grower1(gap_state, 2)
        gap_gps_code = tiqs.gap_resetter1(gap_gps_code, 0, 5)
        #plt.plot(binned)
        #plt.plot(gap_gps_code)
        #plt.plot(gap_gps_1pps)
        #plt.plot(gap_dab_synch)
        #plt.show()
        print("gps code", time.time() - time1)

        gaps_gps_code.append(gap_gps_code)


        stream_part = stream_seq(stream, samples_per_second*i, samples_per_second, extra_samples)
        for sample in range(1,len(gap_gps_1pps)):
            if gap_gps_1pps[sample-1] == 0 and gap_gps_1pps[sample] == 1:
                gaps = stream_part[sample*bin-200: sample*bin+300]
                gaps = tiqs.find_gaps1(gaps, threshold)
                gaps = tiqs.gap_grower1(gaps, 2)
                gaps = tiqs.gap_resetter1(gaps, 59, 129)
                for end in range(10): # artefact delete
                    gaps[-end] = 0

                for g in range(len(gaps)):
                    if gaps[g-1] == 0 and gaps[g] == 1:
                        gps_tiqs.append((samples_per_second*i)/2 + (sample*bin - 200 + g))


    '''
        plt.plot(gap_gps_1pps)
        plt.plot(gap_gps_code)
        plt.show()
    '''

    gaps_gps_code = np.array(gaps_gps_code).reshape((-1))

    gaps_period = []
    print(len(gps_tiqs), gps_tiqs)
    for i in range(1, len(gps_tiqs)):
        gaps_period.append(gps_tiqs[i] - gps_tiqs[i-1])

    gaps_period1, gap_start1 = tiqs.gap_period_counter_betweener(gaps_period, gps_tiqs, 1900000, 2200000)
    best_period1, period_length1 = tiqs.gap_period_stacker(gaps_period1, 0.0)
    print(gaps_period1)
    print(len(gap_start1), gap_start1)
    print(best_period1)
    print(period_length1)

    print("end", time.time() - time1)

    gps_timestamp = []
    gps_long = []
    gps_lat = []
    gps_alt = []
    gps_hdop = []
    gps_fixquality = []
    gps_satellites = []

    for i in range(len(gap_start1)):
        tmp = gaps_gps_code[int(gap_start1[i]/bin) + 1000: int(gap_start1[i]/bin)+7000]
        #plt.plot(tmp)
        #plt.show()

        synchgap = []
        for j in range(len(tmp)-1):
            if j > 0: #int(np.round(5000.0 / bin_size)):
                # ignoring the first n samples because code will start later
                if tmp[j] == 0 and tmp[j+1] == 1:
                    synchgap.append(j)
        del tmp

        bit_distance = 18 # is fixed now, but should be altered in the next if
        bit_duration = 50
        bit_str = ""

        for j in range(len(synchgap)-1):
            if j == 0:
                bit_distance = synchgap[j+1]-synchgap[j]
                print("bitdistance", bit_distance)
            bit_str = bit_str + "1"
            for k in range(int(round((synchgap[j+1]-synchgap[j])/float(bit_distance)))-1):
                bit_str = bit_str + "0"
            if j == len(synchgap)-2:
                bit_str = bit_str + "1"

        bits = 4+ 3*8+ 32+ 1+14+17+ 1+13+17+ 4+19 +11+4+7
        if len(bit_str) < bits:
            for j in range(bits-len(bit_str)):
                bit_str = bit_str + "0"

        print(len(bit_str), bit_str[0:bits-1])

        print("time", int(bit_str[4+3*8:4+3*8+32], base=2))
        gps_timestamp.append(int(bit_str[4+3*8:4+3*8+32], base=2))
        factor = 1
        if int(bit_str[4+3*8+32:4+3*8+32+1], base=2) == 1:
            factor = -1
        gps_long.append(factor *(float(int(bit_str[4+3*8+32+1:4+3*8+32+1+14], base=2))
                                     + float(int(bit_str[4+3*8+32+1+14 : 4+3*8+32+1+14+17], base=2)) / 100000.0) / 60.0)
        factor = 1
        if int(bit_str[4+3*8+32+1+14+17 : 4+3*8+32+1+14+17+1], base=2) == 1:
            factor = -1
        gps_lat.append(factor * (float(int(bit_str[4+3*8+32+1+14+17+1 : 4+3*8+32+1+14+17+1+13], base=2))
                                         + float(int(bit_str[4+3*8+32+1+14+17+1+13 : 4+3*8+32+1+14+17+1+13+17], base=2)) / 100000.0) / 60.0)
        gps_alt.append([int(bit_str[4+3*8+32+1+14+17+1+13+17 : 4+3*8+32+1+14+17+1+13+17+4], base=2), int(bit_str[4+3*8+32+1+14+17+1+13+17+4 : 4+3*8+32+1+14+17+1+13+17+4+19], base=2)/10.0 - 1000.0])
        gps_hdop.append(int(bit_str[4 + 3 * 8 + 32 + 1 + 14 + 17 + 1 + 13 + 17 + 4 + 19 : 4 + 3 * 8 + 32 + 1 + 14 + 17 + 1 + 13 + 17 + 4 + 19 + 11], base=2) / 100.0)
        gps_fixquality.append(int(bit_str[4 + 3 * 8 + 32 + 1 + 14 + 17 + 1 + 13 + 17 + 4 + 19 + 11 : 4 + 3 * 8 + 32 + 1 + 14 + 17 + 1 + 13 + 17 + 4 + 19 + 11 + 4], base=2))
        gps_satellites.append(int(bit_str[4 + 3 * 8 + 32 + 1 + 14 + 17 + 1 + 13 + 17 + 4 + 19 + 11 + 4 : 4 + 3 * 8 + 32 + 1 + 14 + 17 + 1 + 13 + 17 + 4 + 19 + 11 + 4 + 7], base=2))
        print(gps_lat[-1], gps_long[-1], gps_alt[-1], gps_hdop[-1], gps_fixquality[-1], gps_satellites[-1])

    print("end", time.time() - time1)


    print("huhubb", len(gap_start1))

    if len(gap_start1)> 0:
        geo_filter = do.bad_sphere_center(gps_long, gps_lat, gps_alt)
        print(geo_filter)
        r1 = []
        r2 = []
        r3 = []
        for i in range(len(gps_long)):
            r1.append(360.0)
            r2.append(360.0)
            r3.append(-12345.0)

        for j in range(len(geo_filter)):
            r1[int(geo_filter[j])] = gps_long[int(geo_filter[j])]
            r2[int(geo_filter[j])] = gps_lat[int(geo_filter[j])]
            r3[int(geo_filter[j])] = gps_alt[int(geo_filter[j])][1]
            yes = 1

        for i in range(len(gps_long)):
            if r1[i] >= 360.0 and i < len(gps_long)-1:
                j = 1
                while r1[i+j] >=360.0 and i+j < len(r1)-1:
                    j+=1
                r1[i] = r1[i+j]
                r2[i] = r2[i+j]
                r3[i] = r3[i+j]
                if i+j == len(r1)-1:
                    r1[i] = r1[i-1]
                    r2[i] = r2[i-1]
                    r3[i] = r3[i-1]
            if r1[i] >= 360.0 and i == len(gps_long)-1:
                r1[i] = r1[i-1]
                r2[i] = r2[i-1]
                r3[i] = r3[i-1]

        print(np.mean(r2), np.mean(r1), np.min(r2), np.max(r2), np.min(r1), np.max(r1), len(r1))

        print(gps_long)
        print(gps_lat)
        print(gps_alt)
        #plt.plot(gps_pps_start, gps_pps_timestamp, 'o')

        #splitme = file.split("_")
        #np.save(a+"/"+splitme[0]+"_"+splitme[1]+"_"+splitme[2]+"_gaps.npy", [gps_pps_start, gps_pps_timestamp, smoothed_x, gps_pps_timestamp_smoothed])
        smoothed_x, gps_pps_timestamp_smoothed, passed_ransac = do.bad_ransac_lin(gap_start1, gps_timestamp)
        #plt.plot(gps_pps_start, gps_pps_timestamp, 'o')
        #plt.plot(smoothed_x, gps_pps_timestamp_smoothed, 'x')
        #plt.show()

        print(len(smoothed_x), len(gps_pps_timestamp_smoothed), len(r1), len(r2))
        gapsxxx = []
        for i in range(0,len(gps_pps_timestamp_smoothed)-1):
            gapsxxx.append(smoothed_x[i+1] - smoothed_x[i])

        best_period1, period_length1 = tiqs.gap_period_stacker(gapsxxx, 0.0)
        #print(best_period1
        #print(period_length1
        print("these are the droids you are looking for", np.max(best_period1), period_length1[np.argmax(best_period1)])

        # selecting the good pulses by period length
        good_gaps = []
        good_gaps_start = []
        for i in range(0,len(gps_pps_timestamp_smoothed)):
            # created all before it can be set in the next loop
            good_gaps.append(0)

        for i in range(0,len(gps_pps_timestamp_smoothed)):
            if i < len(gps_pps_timestamp_smoothed)-1:
                if gapsxxx[i] == period_length1[np.argmax(best_period1)]:
                    good_gaps[i] = 1
                    good_gaps[i+1] = 1

            print(gap_start1[i], smoothed_x[i], good_gaps[i])


        backwards = 0
        counter = 1
        smoothed_x_all = []
        gps_pps_timestamp_smoothed_all = []
        good_all = []
        gps_long_all = []

        ##############
        # gps flattening
        gps_long_all = []
        gps_lat_all = []
        gps_alt_all = []
        gps_hdop_all = []
        gps_fixquality_all = []
        gps_satellites_all = []

        if passed_ransac == 1:
            for i in range(len(good_gaps)):
                if good_gaps[i] == 1:
                    if backwards == 0:
                        #print("cc", smoothed_x[i] / float(period_length1[np.argmax(best_period1)])
                        j = int(float(smoothed_x[i]) / float(period_length1[np.argmax(best_period1)]))
                        for k1 in range(j):
                            smoothed_x_all.append(float(smoothed_x[i]) - float(j-k1)*float(period_length1[np.argmax(best_period1)]))
                            gps_pps_timestamp_smoothed_all.append(float(gps_pps_timestamp_smoothed[i]) - float(j-k1))
                            good_all.append(0)
                            #print("zzz", k1, (gps_pps_timestamp_smoothed[i] - (j-k1)), float(smoothed_x[i]) - (j-k1)*float(period_length1[np.argmax(best_period1)])
                        backwards = 1


                    print(counter, len(good_gaps), np.sum(good_gaps), i, len(smoothed_x)-1)

                    if i < len(good_gaps)-1:
                        j = 1
                        while good_gaps[i+j] == 0 and i+j < len(good_gaps)-1:
                            j+=1

                        k = int(np.round((smoothed_x[i+j] - smoothed_x[i]) /  float(period_length1[np.argmax(best_period1)])))

                        for k1 in range(k):
                            good = 1
                            if k1 > 0:
                                good = 0

                            smoothed_x_all.append(float(smoothed_x[i]) + float(k1) * (smoothed_x[i+j] - smoothed_x[i]) / np.round((smoothed_x[i+j] - smoothed_x[i]) /  float(period_length1[np.argmax(best_period1)])))
                            gps_pps_timestamp_smoothed_all.append(np.round( float(gps_pps_timestamp_smoothed[i]) + float(k1)) )
                            good_all.append(good)

                            #print("yyy", i, (gps_pps_timestamp_smoothed[i] + k1), smoothed_x[i] + k1 * (smoothed_x[i+j] - smoothed_x[i]) / np.round((smoothed_x[i+j] - smoothed_x[i]) /  float(period_length1[np.argmax(best_period1)])), (smoothed_x[i+j] - smoothed_x[i]) / np.round((smoothed_x[i+j] - smoothed_x[i]) /  float(period_length1[np.argmax(best_period1)])), good, k, k1

                    if counter >= np.sum(good_gaps) and i <len(smoothed_x)-1:
                        #print(len(smoothed_x), i, j
                        last_i = int((len(stream)/2 - smoothed_x[i+j]) / float(period_length1[np.argmax(best_period1)]))
                        #print("hier", i, last_i, input_length/2, smoothed_x[i+j]
                        for k1 in range(last_i+1):
                            smoothed_x_all.append(float(smoothed_x[i+j]) + k1*float(period_length1[np.argmax(best_period1)]))
                            gps_pps_timestamp_smoothed_all.append(np.round( float(gps_pps_timestamp_smoothed[i+j]) + float(k1)) )
                            good_all.append(good)
                            #print("xxx", i+j+k, gps_pps_timestamp_smoothed[i+j] + k1, float(smoothed_x[i+j]) + k1*float(period_length1[np.argmax(best_period1)]), float(period_length1[np.argmax(best_period1)])

                    if counter == np.sum(good_gaps) and i == len(smoothed_x)-1:
                        loops = int((len(stream)/2 - smoothed_x[i]) / float(period_length1[np.argmax(best_period1)]))
                        for k1 in range(loops+1):
                            smoothed_x_all.append(float(smoothed_x[i]) + k1*float(period_length1[np.argmax(best_period1)]))
                            gps_pps_timestamp_smoothed_all.append(np.round( float(gps_pps_timestamp_smoothed[i]) + float(k1)))
                            good_all.append(good)
                            #print("xxx", i+k, gps_pps_timestamp_smoothed[i] + k1, float(smoothed_x[i]) + k1*float(period_length1[np.argmax(best_period1)]), float(period_length1[np.argmax(best_period1)])

                    counter += 1


            for i in range(0, len(smoothed_x)):
                if i==0:
                    loops = int(smoothed_x[0] / float(period_length1[np.argmax(best_period1)]))
                    for j in range(loops):
                        gps_long_all.append(r1[i])
                        gps_lat_all.append(r2[i])
                        gps_alt_all.append(r3[i])
                        #print(i, loops, gps_pps_timestamp_smoothed[i], "sss"
                        gps_hdop_all.append(gps_hdop[i])
                        gps_fixquality_all.append(gps_fixquality[i])
                        gps_satellites_all.append(gps_satellites[i])

                if i < len(smoothed_x)-1:
                    loops = int(np.round(gps_pps_timestamp_smoothed[i+1]-gps_pps_timestamp_smoothed[i]))
                    for j in range(loops):
                        gps_long_all.append(r1[i])
                        gps_lat_all.append(r2[i])
                        gps_alt_all.append(r3[i])
                        #print(i, loops, gps_pps_timestamp_smoothed[i], "ttt"
                        gps_hdop_all.append(gps_hdop[i])
                        gps_fixquality_all.append(gps_fixquality[i])
                        gps_satellites_all.append(gps_satellites[i])

                if i == len(smoothed_x)-1:
                    loops = int((len(stream)/2 - smoothed_x[i]) / float(period_length1[np.argmax(best_period1)]))
                    for j in range(loops): #+1?
                        gps_long_all.append(r1[i])
                        gps_lat_all.append(r2[i])
                        gps_alt_all.append(r3[i])
                        #print(i, loops, gps_pps_timestamp_smoothed[i], "uuu"
                        gps_hdop_all.append(gps_hdop[i])
                        gps_fixquality_all.append(gps_fixquality[i])
                        gps_satellites_all.append(gps_satellites[i])
                    for _ in range(len(smoothed_x_all)-len(gps_long_all)): # xxx nicht toll, aber bugfix
                        gps_long_all.append(r1[i])
                        gps_lat_all.append(r2[i])
                        gps_alt_all.append(r3[i])
                        gps_hdop_all.append(gps_hdop[i])
                        gps_fixquality_all.append(gps_fixquality[i])
                        gps_satellites_all.append(gps_satellites[i])

        else:
            print("not passed")
            smoothed_x_all = smoothed_x
            gps_pps_timestamp_smoothed_all = np.round( gps_pps_timestamp_smoothed )
            gps_long_all = r1
            gps_lat_all = r2
            gps_alt_all = r3
            gps_hdop_all = gps_hdop
            gps_fixquality_all = gps_fixquality
            gps_satellites_all = gps_satellites

        print("ttttttttt")
        print(gps_timestamp)
        print(gps_pps_timestamp_smoothed)
        print(gps_pps_timestamp_smoothed_all)
        #plt.plot(smoothed_x_all, gps_pps_timestamp_smoothed_all, 'x-')
        #plt.show()

        print(len(gps_long_all), len(smoothed_x), len(smoothed_x_all))
        for l in range(0,len(smoothed_x_all)):
            print(l, gps_pps_timestamp_smoothed_all[l], smoothed_x_all[l], gps_long_all[l], gps_lat_all[l], gps_alt_all[l], gps_hdop_all[l], gps_fixquality_all[l], gps_satellites_all[l])

        if input_filename.find(os.path.sep) > -1:
            filename = input_filename.split(".")[0].split(os.path.sep)[-1]
        elif input_filename.find("/") > -1:
            filename = input_filename.split(".")[0].split("/")[-1]
        #print(filename)

        np.save(path_storing + filename, [gps_pps_timestamp_smoothed_all, smoothed_x_all, gps_long_all, gps_lat_all,
                                          gps_alt_all, passed_ransac, gap_start1, gps_timestamp,
                                          gps_pps_timestamp_smoothed, smoothed_x, gps_hdop_all, gps_fixquality_all,
                                          gps_satellites_all, period_length1[np.argmax(best_period1)]])

    else:
        if input_filename.find(os.path.sep) > -1:
            filename = input_filename.split(".")[0].split(os.path.sep)[-1]
        elif input_filename.find("/") > -1:
            filename = input_filename.split(".")[0].split("/")[-1]

        np.save(path_storing + filename, path_storing + filename)

    print("end", time.time() - time1)

if __name__ == '__main__':
    #input_filename = "C:/Users/Andreas/PycharmProjects/sdr_gainlevel/rec/iq/a70030daac692295e3f2f7cab2a66d78f302897e45c7a8e1341cce03_869525000_1484855202.dat"
    #input_filename = "C:/Users/Andreas/PycharmProjects/sdr_gainlevel/rec/iq/eda8b2407095640fddc5a770a591d570f18699e3d9e84e2e8c49d46b_869525000_1484855194.dat"
    input_filename = "C:/Users/Andreas/dgsn-node-data/rec/iq/eda8b2407095640fddc5a770a591d570f18699e3d9e84e2e8c49d46b_178000000_1459719017.npy"

    #input_filename = ""
    path_storing = ""
    run(path_storing, input_filename)