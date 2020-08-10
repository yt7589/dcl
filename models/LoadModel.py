import sys
import numpy as np
from torch import nn
import torch
from torchvision import models, transforms, datasets
import torch.nn.functional as F
import pretrainedmodels

from config import pretrained_model

import pdb
#torch.backends.cudnn.benchmark = False

class MainModel(nn.Module):
    RUN_MODE_NORMAL = 100
    RUN_MODE_FEATURE_EXTRACT = 101

    def __init__(self, config):
        super(MainModel, self).__init__()
        self.use_dcl = config.use_dcl
        self.num_brands = config.num_brands
        self.num_bmys = config.num_bmys
        self.backbone_arch = config.backbone
        self.use_Asoftmax = config.use_Asoftmax
        self.run_mode = MainModel.RUN_MODE_NORMAL # 1-正常运行；2-输出最后一层的特征；
        self.train_batch = config.train_batch
        self.val_batch = config.val_batch
        print(self.backbone_arch)
        self.fc_size = {'resnet50': 2048, 'resnet18': 512}

        if self.backbone_arch in dir(models):
            self.model = getattr(models, self.backbone_arch)()
            if self.backbone_arch in pretrained_model:
                self.model.load_state_dict(torch.load(pretrained_model[self.backbone_arch]))
        else:
            if self.backbone_arch in pretrained_model:
                # export TORCH_HOME="/media/zjkj/35196947-b671-441e-9631-6245942d671b/yantao/fgvc/dcl/models/pretrained/"
                # export TORCH_MODEL_ZOO="/media/zjkj/35196947-b671-441e-9631-6245942d671b/yantao/fgvc/dcl/models/pretrained/"
                #self.model = pretrainedmodels.__dict__[self.backbone_arch](num_classes=1000, pretrained='imagenet')
                self.model = pretrainedmodels.__dict__[self.backbone_arch](num_classes=1000, pretrained=None)
                self.model.load_state_dict(torch.load(pretrained_model[self.backbone_arch]))
            else:
                self.model = pretrainedmodels.__dict__[self.backbone_arch](num_classes=478)

        if self.backbone_arch == 'resnet18':
            self.model = nn.Sequential(*list(self.model.children())[:-2])
        if self.backbone_arch == 'resnet50' or self.backbone_arch == 'se_resnet50':
            self.model = nn.Sequential(*list(self.model.children())[:-2])
        if self.backbone_arch == 'senet154':
            self.model = nn.Sequential(*list(self.model.children())[:-3])
        if self.backbone_arch == 'se_resnext101_32x4d':
            self.model = nn.Sequential(*list(self.model.children())[:-2])
        if self.backbone_arch == 'se_resnet101':
            self.model = nn.Sequential(*list(self.model.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        # 品牌分类器
        self.classifier = nn.Linear(self.fc_size[self.backbone_arch], self.num_brands, bias=False)
        # 年款分类器
        self.brand_clfr = nn.Linear(self.fc_size[self.backbone_arch], self.num_bmys, bias=False)

        if self.use_dcl:
            if config.cls_2:
                self.classifier_swap = nn.Linear(self.fc_size[self.backbone_arch], 2, bias=False)
            if config.cls_2xmul:
                self.classifier_swap = nn.Linear(self.fc_size[self.backbone_arch], 2*self.num_classes, bias=False)
            self.Convmask = nn.Conv2d(self.fc_size[self.backbone_arch], 1, 1, stride=1, padding=0, bias=True)
            self.avgpool2 = nn.AvgPool2d(2, stride=2)

        if self.use_Asoftmax:
            self.Aclassifier = AngleLinear(self.fc_size[self.backbone_arch], self.num_classes, bias=False)

        self.initialize_bmy_masks()

    def forward(self, x, last_cont=None, run_mode=RUN_MODE_NORMAL):
        x = self.model(x)
        if self.use_dcl:
            mask = self.Convmask(x)
            mask = self.avgpool2(mask)
            mask = torch.tanh(mask)
            mask = mask.view(mask.size(0), -1)
        x = self.avgpool(x)
        if MainModel.RUN_MODE_FEATURE_EXTRACT == run_mode:
            return x
        #x = x.view(x.size(0), -1)
        #x = x.view(x.size(0), x.size(1))
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        out = []
        out.append(self.classifier(x))
        y_brand = self.brand_clfr(x)

        if self.use_dcl:
            out.append(self.classifier_swap(x))
            out.append(mask)

        if self.use_Asoftmax:
            if last_cont is None:
                x_size = x.size(0)
                out.append(self.Aclassifier(x[0:x_size:2]))
            else:
                last_x = self.model(last_cont)
                last_x = self.avgpool(last_x)
                last_x = last_x.view(last_x.size(0), -1)
                out.append(self.Aclassifier(last_x))
        out.append(y_brand)
        '''
        if not self.training:
            # 由品牌决定年款输出（仅在实际运行中开启）
            print('Use brand to controll bmy...')
            brand_out = out[0]
            brand_result = torch.argmax(brand_out, dim=1)
            bmy_out = out[-1]
            for idx1 in range(brand_out.shape[0]):
                brand_idx = int(brand_result[idx1].cpu().item())
                bmy_mask = self.bmy_masks[brand_idx]
                bmy_out[idx1] = bmy_out[idx1] * bmy_mask
        '''
        return out

    def initialize_bmy_masks(self):
        self.bmy_masks = np.zeros((self.num_brands, self.num_bmys), dtype=np.float32)
        for bi in range(self.num_brands):
            bmy_idxs = MainModel.BRAND_BMYS_DICT[bi]
            for bmy_idx in bmy_idxs:
                self.bmy_masks[bi][bmy_idx] = 1.0
        self.bmy_masks = torch.from_numpy(self.bmy_masks).cuda()

    BRAND_BMYS_DICT = {
        0:[0, 34, 1159, 1163, 1171],
        1:[1, 2, 3, 4, 5, 6, 7, 24, 148, 149, 150, 151, 154, 155, 1148, 1149, 1150, 1151, 1152, 1153, 1154, 1158],
        2:[8, 9, 10, 11, 30, 31, 32, 75, 81, 87, 94, 99, 274, 316, 320, 357, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 510, 511, 512, 515, 518, 522, 1541, 1615, 1616],
        3:[12, 13, 14, 15, 22, 2079, 2085, 2148, 2150, 2151, 2154, 2158, 2159, 2162, 2167],
        4:[16, 17, 18, 19, 20, 21],
        5:[25, 35, 36, 39, 51, 52, 53, 54],
        6:[26, 27, 46, 47, 48, 49, 50, 71],
        7:[28, 29, 258, 265, 951],
        8:[37, 38, 952],
        9:[42, 43, 44, 77, 78, 82, 98, 139, 173, 187, 188, 196, 197, 198, 199, 200, 210, 229, 230, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 254, 255, 256, 257, 259, 260, 261, 262, 263, 264, 266, 267, 268, 269, 272, 358, 359, 737, 933, 982, 1024, 1025, 1026, 1027, 1107, 1146, 1393, 1427, 1428, 1438, 2571, 2572, 2573, 2574, 2575, 2576, 2577, 2578, 2579, 2580, 2581, 2582, 2583, 2584, 2585, 2586, 2587, 2588, 2589, 2590, 2591, 2592, 2593, 2594, 2595, 2596, 2597, 2598, 2599, 2600, 2601, 2602, 2603, 2604, 2605, 2606, 2607, 2608, 2609, 2610, 2611, 2612, 2613, 2614, 2615, 2616, 2617, 2618, 2619, 2620, 2621, 2622, 2623, 2624, 2625, 2626, 2627, 2757, 2758, 2785, 2786, 2818, 2819, 2820, 2821],
        10:[55, 56, 57, 821, 822, 863, 864, 865, 872, 873, 1533, 1534, 1535, 1536, 1537, 2172],
        11:[58, 59, 86, 93, 379, 388, 389, 390, 391, 392, 393, 394, 417, 422, 428, 429, 1142, 1147, 1190, 1192, 1193, 1194, 1195, 1196, 1197, 1198, 1199, 1200, 1201, 1202, 1203, 1204, 1205, 1206, 1207, 1208, 1209, 1210, 1211, 1212, 1213, 1214, 1215, 1216, 1429, 1430, 1437, 1545, 1546, 1547, 1548, 1549, 1550, 1551, 1552, 1553, 1554, 1555, 1556, 1557, 1558, 1591, 1592, 1593, 1594, 1595, 1798, 1826, 2059, 2060, 2455, 2456, 2457, 2458, 2459, 2460, 2461, 2462, 2463, 2464, 2465, 2466, 2467, 2468, 2469, 2470, 2471, 2472, 2473, 2474, 2475, 2476, 2477, 2478, 2479, 2480, 2495],
        12:[66, 67, 68, 69],
        13:[40, 72, 73, 74, 1041, 1042, 1053, 1059, 1061, 1062, 1063, 1071, 1072, 1073, 1074, 1081, 1082, 1083, 1084, 1085, 1089, 1090, 1091, 1092, 1093, 1094, 1095, 1096, 1097, 1098, 1101, 2454, 2498, 2499, 2500, 2501, 2502, 2503, 2504, 2505, 2506, 2507, 2508, 2509, 2510, 2511, 2512, 2513, 2514, 2515, 2516, 2517, 2518, 2519, 2520, 2521, 2522, 2523, 2524],
        14:[88, 89, 91, 309, 698, 1230, 1231, 1232, 1233, 1234, 1235, 1236, 1238, 1239, 1240, 1241, 1242, 1243, 1244, 1245, 1246, 1247, 1248, 1249, 1250, 1251, 1252, 1253, 1254, 1255, 1256, 1257, 1258, 1259, 1260, 1261, 1262, 1263, 1264, 1265, 1266, 1267, 1268, 1269, 1270, 1271, 1272, 1273, 1797, 2441, 2790, 2791],
        15:[90],
        16:[92, 97],
        17:[101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 1646, 1647, 1648, 1649, 1650, 1651, 1853, 2068],
        18:[141, 360, 361, 362, 363, 365, 366, 369, 370, 372, 373, 374, 376, 377, 382, 383, 384, 395, 396, 397, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 413, 414, 415, 416, 418, 419, 420, 421, 423, 425, 432, 434, 435, 436, 2431, 2445, 2446, 2447, 2448],
        19:[140],
        20:[142, 143, 144, 145, 146, 147, 152, 153, 156, 157, 158, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 176, 178, 179, 180, 181, 182, 184, 185, 186, 190, 191, 192, 193, 194, 195, 201, 203, 204, 205, 206, 207, 209, 211, 212, 215, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 231, 232, 233, 234, 235, 236, 237, 251, 252, 253, 662, 2173],
        21:[177],
        22:[159, 170, 189, 938, 949, 965, 966, 967, 968, 969, 970, 1182, 1183, 1185, 1186, 1187, 1189, 1412, 1413, 1414, 1415, 1416, 1417, 1418, 1419],
        23:[171, 172, 174, 213, 216, 217, 1864, 2788],
        24:[183, 208, 646, 647, 650, 651, 656, 657, 658, 659, 660, 661, 663, 664, 667, 678, 681, 682, 683],
        25:[41, 80, 270, 271, 297, 298, 371, 427, 695, 1040, 1043, 1044, 1045, 1046, 1047, 1048, 1049, 1050, 1051, 1052, 1054, 1055, 1056, 1057, 1058, 1060, 1064, 1065, 1066, 1067, 1068, 1069, 1070, 1075, 1076, 1077, 1078, 1079, 1080, 1086, 1087, 1088, 1099, 1100, 1102, 1103, 1104, 1408, 1439, 2294, 2297, 2298, 2299, 2302, 2303, 2304, 2305, 2306, 2307, 2309, 2310, 2312, 2315, 2316, 2317, 2318, 2319, 2320, 2321, 2322, 2323, 2324, 2325, 2326, 2327, 2332, 2333, 2334, 2335, 2336, 2339, 2340, 2341, 2342, 2343, 2345, 2346, 2347, 2348, 2352, 2353, 2354, 2355, 2356, 2357, 2358, 2360, 2363, 2364, 2365, 2366, 2367, 2368, 2369, 2370, 2372, 2658, 2659, 2660, 2661, 2662, 2663, 2664, 2665, 2666, 2667, 2668, 2669, 2787],
        26:[276, 277, 278],
        27:[45, 60, 61, 62, 63, 64, 65, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 1799, 1800, 1801, 2072, 2525, 2526, 2527, 2528, 2529, 2530, 2531, 2532, 2533, 2534, 2535, 2536, 2537, 2538, 2539, 2540, 2541, 2542, 2543, 2544, 2545, 2546, 2547, 2548, 2549, 2550, 2551, 2552, 2553, 2554, 2555, 2556, 2557, 2558, 2559, 2560, 2561, 2562, 2563, 2564, 2565, 2566, 2567, 2568, 2569, 2570, 2628, 2629, 2630, 2631, 2632, 2633, 2634, 2635, 2636, 2637, 2638, 2639, 2640, 2641, 2642],
        28:[299, 300, 301, 302, 303, 304, 305, 306, 307, 308],
        29:[202, 312, 313, 314],
        30:[321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 1856, 1857, 1858, 1859, 1860, 1861, 1862],
        31:[367, 368],
        32:[378],
        33:[375, 385, 387, 437, 439, 440, 441, 442, 443, 444, 445, 447, 448, 449, 450, 451, 452, 453, 473, 474, 487, 488, 489, 490, 491, 509, 521, 524, 525, 526, 527, 528, 1364, 1365, 1367, 1484, 1485, 1486],
        34:[380],
        35:[411, 412],
        36:[364],
        37:[398, 424, 426, 430, 431, 433, 438, 446, 454, 455, 456, 457, 458],
        38:[513, 514, 516, 517, 519, 520, 2481, 2482, 2483, 2484, 2485, 2486, 2487, 2488, 2489, 2490, 2729, 2730, 2731, 2732, 2733, 2734, 2735, 2736, 2737, 2738, 2739, 2740, 2741, 2742],
        39:[529, 530, 531, 533],
        40:[532, 739, 740, 741, 743, 745, 746, 747, 752, 753, 756, 757, 758, 759, 760, 763, 764, 765, 774, 775, 776, 777, 779, 780, 781, 783, 785, 786],
        41:[534, 535, 536, 537, 538, 541, 542, 543, 551, 552, 556, 557, 567, 568, 582, 588, 597, 598, 600, 601, 602, 605, 606, 607, 609, 610, 611, 612, 613, 616, 618, 619, 622, 623, 626, 627, 628, 1746],
        42:[544, 545, 546, 547, 548, 553, 554, 555, 558, 559, 560, 561, 562, 563, 564, 565, 566, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 583, 584, 585, 586, 587, 589, 590, 591, 594, 595, 596, 599, 608, 614, 615, 617, 620, 621, 624, 625, 629, 630, 631],
        43:[549, 550, 592, 593],
        44:[603, 604],
        45:[632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 1670, 1671, 1672, 2748, 2749],
        46:[648, 649, 652, 653, 654, 655, 665, 666, 669, 670, 671, 672, 673, 674, 675, 676, 677, 679, 680, 1521, 1522, 1543, 1978, 1979, 1980, 1982, 1984, 1985, 1990, 1991, 1992, 1995, 1996, 1997, 2003, 2004, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031],
        47:[684, 690, 691, 692, 1872, 1873, 1874, 1875],
        48:[685, 686, 687, 688, 1876, 1877, 1878, 1879, 1880, 1881, 1882, 1883, 1884, 1885, 1886, 1887, 1888, 1889, 1890, 1891, 1892, 1893, 1894, 1895, 1896, 1897, 1898, 1899, 1900],
        49:[701, 702, 703, 705, 706, 707, 708, 709, 710, 711, 714, 718, 720, 721, 722, 723, 724, 730, 731, 732, 733],
        50:[704, 712, 713, 715, 716, 717, 719, 725, 726, 727, 728, 729, 734, 735, 736, 1842, 1843, 1850, 1851, 1901],
        51:[738, 742, 744, 748, 749, 750, 751, 754, 755, 761, 762, 767, 768, 769, 770, 771, 772, 773, 778, 782, 784, 2494],
        52:[95, 697, 700, 766, 799, 800, 802, 803, 804, 805, 806, 876, 877, 878, 879, 880, 881, 882, 885, 886, 888, 889, 890, 891, 892, 893, 934, 971, 972, 973, 974, 976, 977, 978, 979, 980, 981, 996, 997, 998, 999, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1009, 1010, 1011, 1012, 1013, 1014, 1406, 1710, 1711, 1712, 1713, 1714, 1715, 1716, 1717, 1718, 1719, 1720, 1721, 1722, 1723, 1724, 1725, 1726, 1727, 1728, 1729, 1730, 1731, 1732, 1733, 1734, 1735, 1736, 1737, 1738, 1739, 1740, 1741, 1742, 1743, 1744, 1745, 2792, 2798, 2800, 2801, 2802, 2803, 2804, 2806, 2807, 2808, 2809, 2810],
        53:[787, 788, 789, 791, 792, 793, 794, 795, 796, 797],
        54:[801, 1829, 1830, 1831, 1832, 1833],
        55:[807, 808, 809, 815, 816, 818, 819, 820, 823, 824, 825, 827, 829, 830, 831, 832, 833, 834, 837, 838, 839, 840, 841, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851, 852, 853, 860, 861, 862, 867, 868, 869, 870, 871, 874, 875, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1490, 1491, 1492, 2789, 2793, 2794, 2795, 2796, 2797, 2799, 2805, 2811],
        56:[810, 811, 812, 813, 814, 817, 826, 828, 835, 836, 854, 855, 856, 857, 858, 859, 866],
        57:[883, 884, 887, 1008],
        58:[315, 894, 895, 896, 2491, 2492, 2493],
        59:[798, 897],
        60:[23, 33, 898, 899, 900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912, 913, 914, 915, 916, 917, 918, 921, 922, 923, 924, 925, 926, 927, 928, 929, 930, 931, 932, 1160, 1161, 1162, 1164, 1165, 1166, 1167, 1168, 1169, 1170, 1296, 1297, 1298, 1299, 1300, 1304, 1305, 1306, 1308, 1309, 1310, 1311, 1312, 1313, 1314, 1315, 1316, 1317, 1318, 1319, 1320, 1321, 1322, 1323, 1324, 1325, 1326],
        61:[919, 920],
        62:[935, 936, 937, 939, 940, 941, 942, 943, 944, 945, 946, 947, 948, 950, 953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964],
        63:[983, 984, 985, 986, 987, 988, 989, 990, 991, 992, 993, 994],
        64:[975],
        65:[1028, 1029, 1030, 1031, 1032, 1033],
        66:[1034, 1035, 1036],
        67:[1037, 1038],
        68:[381, 386, 1108, 1109, 1110, 1111, 1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119],
        69:[1120, 1121, 1122, 1123, 1124, 1125, 1126, 1127, 1128, 1129, 1130, 1131, 1132, 1133, 1134, 1135, 1136, 1137, 1138, 1139, 1140, 1141, 1184, 1188, 1191],
        70:[1143, 1144, 1145],
        71:[1155, 1156, 1157, 1838, 1839, 1840, 1841, 1845, 1846, 1847, 1848, 1849],
        72:[1172, 1173, 1174],
        73:[1175, 1176, 1177, 1178],
        74:[1179, 1180, 1181, 2058],
        75:[1217, 1218, 1219, 1220],
        76:[1221, 1222, 1223, 1224, 1225, 1394],
        77:[1274, 1275, 1276, 1277, 1278, 1279, 1280, 1281, 1282, 1283, 1284, 1285, 1286, 1287, 1288, 1289, 1290, 1291, 1292, 1293, 1294, 1295],
        78:[1301, 1303],
        79:[1302, 1307],
        80:[1327, 1328, 1329, 1330, 1331, 1332, 1333, 1334],
        81:[1336, 1337, 1338, 1339, 1340, 1341, 1342, 1343, 1344, 1345, 1346, 1347, 1348, 1349, 1350, 1351, 1352, 1353, 1354, 1355, 1356, 1357, 1358, 1359, 1360, 1361, 1362, 1363, 1366, 1368, 1369, 1370, 1371, 1372, 1373, 2814, 2815, 2816],
        82:[1375, 1376, 1377, 1378, 1379, 1380, 1381, 1382, 1383, 1385, 1386, 1387, 1388, 1389, 1390, 1391, 1392, 1440, 1441, 1442, 1443, 1444, 1445, 1446, 1447, 1450, 1452, 1453, 1456, 1457, 1458, 1459, 1460, 1461, 1462, 1463, 1464, 1465, 1466, 1467, 1468, 1469, 1470, 1471, 1472, 1473, 1474, 1475, 1476, 1477, 1478, 1479, 1480, 1481, 1482, 1802, 1803, 1804, 1805, 1806, 1807, 1808, 1809, 1810, 1811, 1812, 1813, 1814, 1815, 1816, 1817, 1819, 1821, 1823, 1824, 1825, 1869, 1870, 2177, 2178, 2179, 2180, 2181, 2182, 2183, 2184, 2185, 2186, 2187, 2188, 2189],
        83:[1384],
        84:[1395, 1396, 1397],
        85:[1398, 1400, 1401, 1402, 1403, 1404],
        86:[1399],
        87:[1407],
        88:[693, 1335, 1374, 1410, 1596, 1597, 1598, 1599, 1600, 1601, 1602, 1603, 1604, 1605, 1606, 1607, 1608, 1609, 1610, 1611, 1612, 1613, 1614, 1617, 1618, 1619, 1620, 1621, 1622, 1623, 1624, 1625, 1626, 1627, 1628, 1629, 1630, 1631, 1632, 1633, 1634, 2746, 2747, 2750, 2751],
        89:[1420, 1421, 1422, 1423, 1424, 1425, 1426],
        90:[1431, 1433, 1435, 1436],
        91:[1432, 1434],
        92:[1448, 1449, 1818, 1820, 1822],
        93:[1487, 1488, 1489],
        94:[319, 2192, 2193],
        95:[668, 1493, 1494, 1495, 1496, 1497, 1498, 1499, 1500, 1501, 1503, 1504, 1505, 1506, 1507, 1508, 1509, 1510, 1511, 1515, 1516, 1517, 1518, 1519, 1520, 1525, 1526, 1527, 1528, 1529, 1530, 1531, 1532],
        96:[1502, 1512, 1513],
        97:[1514, 1523, 1524],
        98:[1538],
        99:[1540],
        100:[1539],
        101:[85, 1409, 1411, 1635, 1636, 1865, 1866],
        102:[84, 175, 275, 310, 523, 539, 540, 696, 1039, 1229, 1405, 1483, 1542, 1836, 1837],
        103:[1544],
        104:[1559, 1560, 1561, 1562, 1563, 1564, 1565, 1566, 1567, 1568, 1569, 1570, 1571, 1572, 1573, 1574, 1575, 1576, 1577, 1578, 1579, 1580, 1581, 1582, 1583, 1584, 1585, 1586, 1587, 1588, 1589, 1590],
        105:[1638, 1639, 1787, 1788, 1789, 1790, 2083, 2084, 2092, 2093, 2094, 2095, 2099, 2100, 2101, 2102, 2103, 2105, 2106, 2107, 2112, 2113, 2114, 2115, 2116, 2117, 2119, 2120, 2123, 2125, 2129, 2130, 2136, 2140, 2142, 2143, 2145, 2146, 2153, 2155, 2156, 2157, 2394],
        106:[1640, 1641, 1642, 1643, 1644, 1645],
        107:[995, 1652, 1653, 1654, 1655, 1656, 1657, 1658, 1659, 1660, 1661, 2685, 2686, 2687, 2688, 2689, 2690, 2691, 2692, 2693, 2694, 2695, 2696, 2697, 2698, 2699, 2700, 2701, 2702, 2703, 2704, 2705, 2706, 2707, 2708, 2709, 2710, 2711, 2712, 2713, 2714, 2715, 2716, 2717, 2718, 2719, 2720, 2721, 2722, 2723, 2724, 2725, 2726, 2727, 2728],
        108:[1662],
        109:[1664, 1665, 1666, 1667, 1668, 1669],
        110:[1673, 1674, 1675, 1676, 1677, 1678, 1679, 1680, 1681, 1682, 1683, 1684, 1685, 1686, 1687, 1688, 1689, 1690, 1691, 1692, 1693, 1694, 1695, 1696, 1697, 1698, 1699, 1700, 1701, 1702],
        111:[1703, 1704, 1705, 1706, 1707, 1708, 1709, 1747, 1748, 1749, 1750, 1751, 1752, 1753, 1754, 1755, 1756, 1757, 1760, 1761, 1762, 1763, 1764, 1765, 1766, 1767, 1768, 1769, 1770, 1782, 1783, 1785],
        112:[1758, 1759, 1771, 1772, 1773, 1774, 1775, 1776, 1777, 1778, 1779, 1780, 1781, 1784, 1786, 1791, 1792, 1793, 1794, 1795, 1796],
        113:[1835],
        114:[1834, 1844],
        115:[214, 1852],
        116:[1854, 1855],
        117:[70, 1105, 1228, 1863, 2080, 2081, 2082, 2086, 2087, 2088, 2089, 2090, 2091, 2096, 2097, 2098, 2104, 2108, 2109, 2110, 2111, 2118, 2121, 2122, 2124, 2126, 2127, 2128, 2131, 2132, 2133, 2134, 2135, 2137, 2138, 2139, 2141, 2144, 2147, 2149, 2152, 2160, 2161, 2163, 2164, 2165, 2166, 2168],
        118:[1867],
        119:[1868],
        120:[1871],
        121:[1902],
        122:[76, 1903, 1904, 1905, 1906, 1907, 1908, 1909, 1910, 1911, 1912, 1913, 1914, 1915, 1916, 1917, 1918, 1919, 1920, 1921, 1922, 1923, 1924, 1925, 1926, 1927, 1928, 1929, 1930, 1931, 1932, 1933, 1934, 1935, 1936, 1937, 1938, 1939, 1940, 1941, 1942, 1943, 1944, 1945, 1946, 1947, 1948, 1949, 1950, 1951, 1952, 1953, 1954, 1955, 1956, 1957, 1958, 1959, 1960, 1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1981, 1983, 1986, 1987, 1988, 1989, 1993, 1994, 1998, 1999, 2000, 2001, 2002, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039, 2040, 2057],
        123:[2041, 2042, 2043, 2044, 2045, 2046, 2047, 2048],
        124:[2049],
        125:[2050, 2051, 2052, 2053, 2054, 2055, 2056],
        126:[2061, 2062, 2063, 2064, 2065, 2066, 2067, 2069, 2070, 2071, 2073, 2074, 2075, 2076, 2077],
        127:[83, 96, 100, 273, 311, 694, 790, 1106, 2169, 2170, 2759],
        128:[2171],
        129:[317],
        130:[2174],
        131:[1451, 2175],
        132:[1454, 1455, 2176],
        133:[2190, 2191, 2194, 2195, 2196, 2197, 2198, 2199],
        134:[2201, 2203, 2204, 2205, 2206, 2208, 2209, 2210, 2213, 2214, 2215, 2221, 2671],
        135:[689, 1827, 1828, 2200, 2202, 2212, 2216, 2217, 2222, 2223, 2224, 2225, 2230, 2231, 2234, 2236, 2238, 2239, 2240, 2243, 2244, 2246, 2247, 2248, 2249, 2250, 2251, 2252, 2253, 2254, 2255, 2256, 2257, 2258, 2260, 2261, 2262, 2263, 2264, 2267, 2268, 2269, 2270, 2271, 2272, 2273, 2274, 2275, 2276, 2277, 2278, 2280, 2281, 2282, 2283, 2284, 2285, 2286, 2287, 2289, 2290, 2291],
        136:[2218, 2219, 2220, 2237, 2265, 2266, 2279],
        137:[2207, 2228, 2229],
        138:[2226, 2232, 2233],
        139:[2227],
        140:[2211, 2235, 2241, 2242, 2245, 2259, 2288],
        141:[2292],
        142:[2293],
        143:[2295, 2296, 2300, 2301, 2308, 2311, 2313, 2314, 2328, 2329, 2330, 2331, 2337, 2338, 2344, 2349, 2350, 2351, 2359, 2361, 2362, 2371, 2373],
        144:[2374],
        145:[79, 318, 1226, 1227, 1237, 1663, 2375, 2376, 2378, 2379, 2382, 2383, 2384, 2385, 2386, 2387, 2388, 2389, 2390, 2391, 2393, 2395, 2396, 2397, 2399, 2400, 2402, 2428],
        146:[2377, 2392, 2398, 2403, 2404, 2405, 2406, 2407, 2408, 2409, 2410, 2411, 2412, 2413, 2414, 2415, 2416, 2417, 2418, 2419, 2420, 2421, 2422, 2423, 2424],
        147:[2380, 2381, 2401],
        148:[2427],
        149:[2426],
        150:[2429],
        151:[2430],
        152:[2078],
        153:[2432, 2433, 2434, 2435, 2436, 2437, 2438, 2439, 2440, 2442, 2443, 2444, 2449, 2450, 2452, 2453],
        154:[2451],
        155:[2496, 2497],
        156:[2643, 2644, 2645, 2646, 2647, 2648, 2649, 2650, 2651, 2652, 2653, 2654, 2655, 2656, 2657],
        157:[2670],
        158:[2672],
        159:[699, 2425, 2673, 2674],
        160:[2675, 2676, 2677, 2678, 2679, 2680],
        161:[2681],
        162:[2682, 2683, 2684],
        163:[2743, 2744, 2745],
        164:[2752, 2753, 2754, 2812],
        165:[2755, 2756],
        166:[2760],
        167:[2761, 2762, 2763, 2764, 2765, 2766, 2767, 2768, 2769, 2770, 2771, 2772, 2773, 2774, 2775],
        168:[2776, 2777, 2778, 2779, 2780, 2781, 2782, 2783, 2784, 2813],
        169:[1637],
        170:[2817],
    }

