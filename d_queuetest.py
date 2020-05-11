from multiprocessing import Process, Queue, Pipe, Array
import numpy as np
from time import time
import logging
import matplotlib.pyplot as plt

class QueueTest(Process):
    def __init__(self, inq, outq,datasize):
        Process.__init__(self)
        self.data = np.random.rand(datasize,8)
        self.inq = inq
        self.outq = outq
        print('\tinitiation complete...')

    def run(self):
        # print(self.inq, self.outq)
        var = self.inq.get()
        # print(var)
        if var == 1:
            self.outq.put(self.data)

class PipeTest(Process):
    def __init__(self, pipe,datasize):
        Process.__init__(self)
        self.data = np.random.rand(datasize,8)
        self.pipe = pipe
        print('\tinitiation complete...')

    def run(self):
        var = self.pipe.recv()
        # print(var)
        if var == 1:
            self.pipe.send(self.data)

class SM(Process):
    '''
    Shared Memeory Test
    '''
    def __init__(self, arr1, arr2,datasize):
        Process.__init__(self)
        self.data = np.random.rand(datasize,8)
        self.inarr = arr1
        self.outarr = arr2
        print('\tinitiation complete...')

    def run(self):
        inarr = np.zeros(1)
        # print(len(self.inarr))
        # print(len(self.outarr))
        while True:
            if inarr[0]!=self.inarr[0]:
                # print(self.outarr[0], self.data[0,0])
                for i in range(len(inarr)):
                    inarr[i] = self.inarr[i]
                data = self.data.reshape(self.data.shape[0]*self.data.shape[1],)
                for i in range(len(self.outarr)):
                    self.outarr[i] = data[i]
                self.inarr[0] = 0.
                break         

        
if __name__=='__main__':
    ds = []
    qtime = []
    ptime = []
    smtime = []

    for dsize in range(30):
        datasize = 5000 * dsize+1
        ds.append(datasize*8)
        # Queue Test
        print('Queue test')
        inq = Queue()
        outq = Queue()
        qtest = QueueTest(inq, outq,datasize)
        qtest.start()
        inq.put(1)
        s = time()
        _ = outq.get()
        print('\tdata shape:', _.shape)
        _ = float(time()-s)
        print(datasize, '\tQueue share time:', _)
        qtime.append(_)
        print('\n')
        qtest.join()
        ##########################

        # Pipe test
        print('Pipe test')
        (con1, con2) = Pipe()
        ptest = PipeTest(con1, datasize)
        ptest.start()
        con2.send(1)
        s = time()
        _ = con2.recv()
        print('\tdata shape:', _.shape)
        _ = float(time()-s)
        print(datasize, '\tPipe share time:', _)
        ptime.append(_)
        print('\n')
        ptest.join()
        ##########################

        # Shared Memory test
        print('Shared Memory test')
        arr1 = Array('d', np.zeros(1))
        arr2 = Array('d', np.random.rand(datasize,8).reshape(datasize*8,))
        smtest = SM(arr1, arr2, datasize)
        smtest.start()
        print('\tinit res 0:',arr2[0])
        arr1[0] = 1.
        s = time()
        while True:
            if arr1[0] == 0.:
                _ = np.zeros(datasize*8)
                for i in range(len(_)):
                    _[i] = arr2[i]
                print('\tnow res 0:', _[0])
                _ = float(time()-s)
                print(datasize, '\tShared Memory share time:', _)
                smtime.append(_)
                break
        smtest.join()
        print('\n\n')

    # plt.plot(ds, qtime, label='Queue')
    # plt.plot(ds, ptime, label='Pipe')
    # # plt.plot(ds, smtime, label='Shared Memory')
    # plt.xlabel('Sharing Data Size')
    # plt.ylabel('Sharing time')
    # plt.legend()
    # plt.grid()
    # plt.savefig('share_time2.png')

    
