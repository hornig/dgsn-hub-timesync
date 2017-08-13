import numpy as np

import time

import pickle

def binning1(stream, window):
    out = []
    for i in range(0, len(stream), window):
        mean = np.mean(stream[i:i+window])
        out.append(mean)
    return out

def rolling_window(stream, window):
    #window = 25
    out = []
    for i in range(len(stream)-window):
        out.append(np.mean(stream[i:i+window]))
    return out

def find_gaps(x):
    binned_min = np.min(x)
    binned_mean = np.mean(x)

    bin_gaps = []
    for bin in x:
        if bin <= (binned_min+binned_mean)/2.1: #2.2
            bin_gaps.append(1)
        else:
            bin_gaps.append(0)

    return bin_gaps

def find_gaps1(x, number):

    bin_gaps = []
    for bin in x:
        if bin <= number: #(binned_min+binned_mean)/2.2 2.2
            bin_gaps.append(1)
        else:
            bin_gaps.append(0)

    return bin_gaps

def gap_grower(x, iterating):
    y = x[:]
    for i in range(iterating):
        for i in range(1,len(y)):
            if y[i] == 1:
                y[i-1] = 1
    return y

def gap_grower1(x, iterating):
    y = x[:]
    for i in range(iterating):
        for j in range(len(x)-2, 1, -1):
            if y[j] == 1:
                y[j+1] = 1
    return y

def gap_resetter(x, gap_length_min):
    y = x[:]
    counter = 0
    for i in range(1,len(x)):
        if x[i] == 1 and x[i-1] == 1:
            counter += 1
        else:
            if counter < gap_length_min:
                for j in range(counter+1):
                    y[i-j-2] = 0

            counter = 0
    return y

def gap_resetter1(x, gap_length_min, gap_length_max):
    y = x[:]
    counter = 0
    for i in range(1,len(x)):
        if x[i] == 1 and x[i-1] == 1:
            counter += 1
        else:
            if counter < gap_length_min or counter > gap_length_max:
                for j in range(counter+1):
                    y[i-j-2] = 0

            counter = 0
    return y

def correlate(haystack, needle):
    # https://www.youtube.com/watch?v=ngEC3sXeUb4
    needle2 = np.sum(needle * needle)

    window = len(needle)

    norm = []

    for i in range(0, len(haystack)-len(needle)):
        haystack_part = np.array(haystack[i:i+window])
        normed_cross_correlation = np.sum(haystack_part * needle)
        normed_cross_correlation = normed_cross_correlation / (np.sum(haystack_part * haystack_part) * needle2)**0.5
        norm.append(normed_cross_correlation)

    return norm

def gap_compare_and(x, y):
    out = np.zeros((len(x),), dtype=np.int)
    for i in range(len(x)):
        if x[i] == 0 and y[i] == 1:
            out[i] = 1
    return out

def gap_period_counter(x):
    gap_start = []
    gap_period = []
    for i in range(1,len(x)):
        if x[i-1] < x[i]:
            gap_start.append(i)

    for i in range(1,len(gap_start)):
        gap_period.append(gap_start[i] - gap_start[i-1])
    return gap_period, gap_start

def gap_period_counter_betweener(x, y, min, max):
    gap_start = []
    gap_period = []
    for i in range(len(x)):
        if x[i] >= min and x[i] <= max:
             gap_period.append(x[i])
             gap_start.append(y[i])
    return gap_period, gap_start

def gap_period_stacker(x, spreading):
    gap_period_count = []
    gap_period_stack = []

    for i in range(len(x)):
        if i == 0:
            gap_period_stack.append(x[i])
            gap_period_count.append(1)
        else:
            isstacked = -1
            for j in range(len(gap_period_stack)):
                if x[i] >= gap_period_stack[j] - x[i]*spreading and x[i] <= gap_period_stack[j] + x[i]*spreading:
                    isstacked = j

            if isstacked == -1:
                gap_period_stack.append(x[i])
                gap_period_count.append(1)
            else:
                gap_period_count[isstacked] = gap_period_count[isstacked] + 1

    return gap_period_count, gap_period_stack

def detect_gap(samplerate, input_stream):
    time1 = time.time()
    window = 100
    input_stream2 = binning1(input_stream,window)
    print("windowing end2", time.time()-time1)

    #print("gap number", np.min(input_stream2), np.mean(input_stream2), (np.min(input_stream2)+np.mean(input_stream2))/2.1)
    number = np.min(input_stream2) + (np.mean(input_stream)-np.min(input_stream2))/3.0
    print(number, np.min(input_stream2),(np.mean(input_stream)))
    gaps_x = find_gaps1(input_stream2, number)
    print("windowing end3", time.time()-time1)
    gaps1_x = gap_grower1(gaps_x,1)
    #plt.plot(input_stream2)
    #plt.plot(gaps1_x)
    gaps1_x = gap_resetter1(gaps1_x, 12, 37)

    #plt.show()

    print("windowing end5", time.time()-time1)
    gaps1_start = []
    for i in range(1,len(gaps1_x)):
        if gaps1_x[i]-gaps1_x[i-1] == 1:
            gaps1_start.append(i*window)
    print(gaps1_start)

    symbol_length = 0.001297
    symbol_samples = int(symbol_length * samplerate)
    print(symbol_samples)
    gaps_period = []
    phase_symbols = []
    phase_start_relative = []

    print("hier", time.time()-time1)

    for i in range(0,len(gaps1_start)):
        #print(i, gaps1_start[i+1]-gaps1_start[i]
        if i == 0:
            phase_symbols = np.array(input_stream[gaps1_start[i]+symbol_samples*1-40:gaps1_start[i]+symbol_samples*2-60])
            #phase_symbols = pickle.load( open( "phase1.p", "rb") )
            with open("phase1.p", 'rb') as f:
                phase_symbols = pickle.load(f, encoding='latin1')
                #print(phase_symbols)
            #pickle.dump( phase_symbols, open( "phase_3072000.p", "wb" ) )
            print(phase_symbols)

        #gaps_period.append(gaps1_start[i+1]-gaps1_start[i])
        sequence = input_stream[gaps1_start[i]:gaps1_start[i]+7000]
        #plt.plot(sequence)
        norm = correlate(sequence, phase_symbols)
        if len(norm) > 0:
            phase_start_relative.append(np.argmax(norm))
        else:
            phase_start_relative.append(0)
        #plt.plot(norm)
    #plt.show()

    symbol_periods = []
    dab_symbol_start = []
    for i in range(0,len(gaps1_start)-1):
        period = (gaps1_start[i+1]+phase_start_relative[i+1]) - (gaps1_start[i]+phase_start_relative[i])
        if period > 0:
            symbol_periods.append(period)
            dab_symbol_start.append(gaps1_start[i]+phase_start_relative[i])
            print(i, gaps1_start[i], symbol_periods[-1], np.mean(phase_start_relative)-phase_start_relative[i])
            if i == len(gaps1_start)-2:
                dab_symbol_start.append(gaps1_start[i+1]+phase_start_relative[i+1])

    if len(gaps1_start) > 1:
        print(np.mean(symbol_periods), np.min(symbol_periods), np.max(symbol_periods))
        #print((gaps1_start)
        print((dab_symbol_start))
        #print((phase_start_relative)

    print("time", time.time()-time1)
    return dab_symbol_start

def detect_gps(input_stream, other_gaps_start):
    # here, we will look for two kinds of gps injected tiqs
    # 1. tiqs: gps 1 pulse per second. longer duration and always before 2# tiqs.
    # 2. tiqs: gps neam data coded as tiqs. shorter duration


    # in case there were further gaps analysed, they can be ruled out in this analysis at this point
    gaps_dab = np.zeros((len(input_stream),), dtype=np.int)

    filler = 5600
    for i in range(len(other_gaps_start)):
        for j in range(filler):
            if other_gaps_start[i]-filler/2+j < len(gaps_dab)-1:
                gaps_dab[other_gaps_start[i] - int(filler / 2) + j] = 1


    ############
    # starting the preparation
    # by creating a state array (0,1) for the gaps in general
    # by binning the input stream by a given bin-size
    # by using a threshold "number" on the stream and filling the state array according to values higher or lower

    gps_gap = np.zeros((len(input_stream),), dtype=np.int)

    time1 = time.time()
    bin_size = 12
    binned = binning1(input_stream, bin_size)


    # starting the quick finding of gps gaps

    #number = np.min(binned) + (np.mean(input_stream)-np.min(binned))/3.0
    number = np.min(binned) + (statistics(binned, 4)-np.min(binned))/3.0
    print("number", number, np.min(input_stream), np.mean(input_stream), np.min(binned))
    gap_state= find_gaps1(binned, number)
    gap_gps_1pps = gap_resetter1(gap_state, 6, 13) # gps tiqs
    gap_gps_code = gap_grower1(gap_state, 1)
    gap_gps_code = gap_resetter1(gap_gps_code, 0, 5) # gps code

    '''
    import matplotlib.pylab as plt
    limit = []
    for i in range(len(binned)):
        limit.append(number)
    plt.plot(binned)
    plt.plot(limit)
    #plt.plot(gap_gps_code)
    plt.plot(gap_gps_1pps)
    plt.show()
    del limit
    '''


    tmp = 0
    for sample in range(1,len(gap_gps_1pps)):
        if gap_gps_1pps[sample-1] == 0 and gap_gps_1pps[sample] == 1:

            goppy = find_gaps1(input_stream[sample*bin_size-1100: sample*bin_size+300], number)

            print(len(goppy), sample*bin_size-1100, sample*bin_size-1100-tmp)
            tmp=sample*bin_size-1100
            for k in range(len(goppy)):
                gps_gap[sample*bin_size-1100 + k] = goppy[k]

    #print(gps_gap)
    print("oooold", time.time()-time1)


    gap_gps_1pps_2 = gap_grower1(gps_gap, 2)
    gap_gps_1pps_2 = gap_resetter1(gap_gps_1pps_2, 59, 129)
    #####################
    if len(gaps_dab) > 0:
        gaps4 = gap_compare_and(gaps_dab, gap_gps_1pps_2)
    else:
        gaps4 = gap_gps_1pps_2

        del gap_gps_1pps_2

    gaps_code = []
    until = -1


    gaps_period, gap_start = gap_period_counter(gaps4)
    del gaps4

    gaps_period1, gap_start1 = gap_period_counter_betweener(gaps_period, gap_start, 1900000, 2200000)
    best_period1, period_length1 = gap_period_stacker(gaps_period1, 0.0)

    gps_timestamp = []
    gps_long = []
    gps_lat = []
    gps_alt = []
    hdop = []
    fixquality = []
    satellites = []

    if len(period_length1) > 0:
        print("gaps period stacked", period_length1)
        print("gaps period stacked counts", best_period1)

        print("gaps_period", gaps_period1)
        print("gaps period start", gap_start1)
        for i in range(len(gap_start1)):

            tmp = gap_gps_code[int(np.round(gap_start1[i]/bin_size)) : int(np.round((gap_start1[i]/bin_size + 5600.0)))]

            synchgap = []
            for j in range(len(tmp)-1):
                if j > 2000: #int(np.round(5000.0 / bin_size)):
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
            hdop.append(int(bit_str[4+3*8+32+1+14+17+1+13+17+4+19 : 4+3*8+32+1+14+17+1+13+17+4+19+11], base=2) / 100.0)
            fixquality.append(int(bit_str[4+3*8+32+1+14+17+1+13+17+4+19+11 : 4+3*8+32+1+14+17+1+13+17+4+19+11+4], base=2))
            satellites.append(int(bit_str[4+3*8+32+1+14+17+1+13+17+4+19+11+4 : 4+3*8+32+1+14+17+1+13+17+4+19+11+4+7], base=2))
            print(gps_lat[-1], gps_long[-1], gps_alt[-1], hdop[-1], fixquality[-1], satellites[-1])

        #plt.show()
    return gap_start1, gps_timestamp, gps_long, gps_lat, gps_alt, hdop, fixquality, satellites

def statistics(samples, loops):
    mean = np.max(samples)
    s = 0

    for i in range(loops):
        fresh_samples = []
        for j in range(len(samples)):
            if samples[j] <= mean + s:
                fresh_samples.append(samples[j])
        x_mean = np.mean(fresh_samples)
        s = np.std(fresh_samples)
        mean = x_mean
        print("standard_deviation", i, s, x_mean)
    return x_mean
