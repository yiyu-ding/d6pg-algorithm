import numpy as np

Net_Topo_Loop_4u = np.array([[2, 1, 0, 1],
                       [1, 2, 1, 0],
                       [0, 1, 2, 1],
                       [1, 0, 1, 2]])

Net_Topo_Loop_4d = np.array([[2, 1, 0, 0],
                       [0, 2, 1, 0],
                       [0, 0, 2, 1],
                       [1, 0, 0, 2]])

Net_Topo_Loop_5u = np.array([[2, 1, 0, 0, 1],
                       [1, 2, 1, 0, 0],
                       [0, 1, 2, 1, 0],
                       [0, 0, 1, 2, 1],
                       [1, 0, 0, 1, 2]])

Net_Topo_Loop_5d = np.array([[2, 1, 0, 0, 0],
                       [0, 2, 1, 0, 0],
                       [0, 0, 2, 1, 0],
                       [0, 0, 0, 2, 1],
                       [1, 0, 0, 0, 2]])

Net_Topo_Loop_6u = np.array([[2, 1, 0, 0, 0, 1],
                       [1, 2, 1, 0, 0, 0],
                       [0, 1, 2, 1, 0, 0],
                       [0, 0, 1, 2, 1, 0],
                       [0, 0, 0, 1, 2, 1],
                       [1, 0, 0, 0, 1, 2]])

Net_Topo_Loop_6d = np.array([[2, 1, 0, 0, 0, 0],
                       [0, 2, 1, 0, 0, 0],
                       [0, 0, 2, 1, 0, 0],
                       [0, 0, 0, 2, 1, 0],
                       [0, 0, 0, 0, 2, 1],
                       [1, 0, 0, 0, 0, 2]])

Net_Topo_Loop_Star = np.array([[2, 0, 1, 1, 0],
                       [0, 2, 0, 1, 1],
                       [1, 0, 2, 0, 1],
                       [1, 1, 0, 2, 0],
                       [0, 1, 1, 0, 2]])

Net_Topo_HalfNet = np.array([[2, 1, 0, 1, 0, 1],
                       [1, 2, 1, 0, 1, 0],
                       [0, 1, 2, 1, 0, 1],
                       [1, 0, 1, 2, 1, 0],
                       [0, 1, 0, 1, 2, 1],
                       [1, 0, 1, 0, 1, 2]])

Net_Topo_DualFlat = np.array([[2, 1, 1, 0, 1, 0, 0, 0],
                            [1, 2, 1, 0, 1, 0, 0, 0],
                            [1, 0, 2, 1, 0, 0, 1, 0],
                            [0, 1, 1, 2, 0, 0, 0, 1],
                            [1, 0, 0, 0, 2, 1, 1, 0],
                            [1, 0, 0, 0, 1, 2, 0, 1],
                            [0, 0, 1, 0, 1, 0, 2, 1],
                            [0, 0, 0, 1, 0, 1, 1, 2]])

Net_Topo_NotConnected = np.array([[2, 1, 0, 0, 0, 0],
                       [1, 2, 0, 0, 0, 0],
                       [0, 0, 2, 1, 0, 0],
                       [0, 0, 1, 2, 0, 0],
                       [0, 0, 0, 0, 2, 1],
                       [0, 0, 0, 0, 1, 2]])


def createundirectedloop(num):
    Net_Topo_Loop_num = np.ones([num, num], dtype=np)
    for i in range(num):
        for j in range(num):
            Net_Topo_Loop_num[i][j] = 0
    for i in range(num):
        if i == 0:
            Net_Topo_Loop_num[i][0] = 2
            Net_Topo_Loop_num[i][1] = 1
            Net_Topo_Loop_num[i][num - 1] = 1
        if i == num - 1:
            Net_Topo_Loop_num[i][num - 1] = 2
            Net_Topo_Loop_num[i][num - 2] = 1
            Net_Topo_Loop_num[i][0] = 1
        if 1 <= i <= num - 2:
            Net_Topo_Loop_num[i][i - 1] = 1
            Net_Topo_Loop_num[i][i] = 2
            Net_Topo_Loop_num[i][i + 1] = 1
    return Net_Topo_Loop_num


Net_Topo_Loop_10 = createundirectedloop(10)
Net_Topo_Loop_15 = createundirectedloop(15)
Net_Topo_Loop_20 = createundirectedloop(20)
Net_Topo_Loop_25 = createundirectedloop(25)
Net_Topo_Loop_30 = createundirectedloop(30)
Net_Topo_Loop_40 = createundirectedloop(40)
Net_Topo_Loop_50 = createundirectedloop(50)
Net_Topo_Loop_60 = createundirectedloop(60)

Net_Topo_Loop_4 = createundirectedloop(4)
Net_Topo_Loop_6 = createundirectedloop(6)
Net_Topo_Loop_8 = createundirectedloop(8)