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
        #y_bmy = F.softmax(self.classifier(x), dim=1)
        y_bmy = self.classifier(x)
        out.append(y_bmy)
        y_brand = self.brand_clfr(x)
        #y_brand = F.softmax(self.brand_clfr(x), dim=1)

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
    # 20200922
    BRAND_BMYS_DICT = {
		0:[1723, 1724, 1725, 1726, 1727, 1728, 1729, 1730, 1731, 1732, 1734, 1735, 1736, 1737, 1738, 1739, 1740, 1741, 1811, 1812, 1813, 1814, 1815, 1816, 1817, 1818, 1819, 1822, 1823, 1825, 1826, 1827, 1831, 1833, 1834, 1835, 1836, 1837, 1838, 1839, 1840, 1841, 1842, 1843, 1844, 1845, 1846, 1847, 1848, 1849, 1850, 1851, 1852, 1853, 1854, 1855, 1856, 1857, 1858, 1859, 1860, 1861, 2281, 2282, 2283, 2284, 2285, 2286, 2287, 2288, 2289, 2290, 2291, 2292, 2293, 2294, 2295, 2296, 2297, 2298, 2300, 2302, 2304, 2305, 2306, 2373, 2374, 2723, 2724, 2725, 2726, 2727, 2728, 2729, 2730, 2731, 2732, 2733, 2734, 2735, 2736, 2737, 2738, 2739],
		1:[140, 450, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 1441],
		2:[906, 907, 908, 909, 2385, 2386, 2387, 2388, 2389, 2390, 2391, 2392, 2393, 2394, 2395, 2396, 2397, 2398, 2399, 2400, 2401, 2402, 2403, 2404, 2405, 2406, 2407, 2408, 2409, 2410, 2411],
		3:[3432, 3434, 3435, 3436],
		4:[114, 478, 479, 480, 1548, 1549, 1562, 2110, 2277, 2955, 2956, 2959, 2961, 2962, 2963, 2964, 2965, 2966, 2967, 2970, 2971, 2972, 2973, 2974, 2975, 2976, 2977, 2978, 2979, 2980, 2981, 2982, 2984, 2987, 2988, 2989, 2990, 2992, 2993, 2994, 2995, 2996, 2997, 2998, 3000, 3001, 3002, 3033, 3034],
		5:[194, 533, 535, 536, 537, 538, 539, 540, 542, 544, 548, 549, 550, 552, 553, 554, 556, 557, 561, 564, 565, 566, 580, 581, 582, 583, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 600, 601, 602, 603, 605, 606, 607, 608, 610, 612, 619, 621, 622, 623, 646, 715, 1526, 3040, 3055, 3056, 3057, 3058, 3531],
		6:[68, 69, 112, 115, 410, 411, 447, 448, 551, 614, 918, 1325, 1328, 1329, 1330, 1331, 1332, 1333, 1334, 1335, 1336, 1337, 1338, 1340, 1341, 1342, 1343, 1344, 1346, 1350, 1351, 1352, 1353, 1354, 1355, 1356, 1361, 1362, 1363, 1364, 1365, 1366, 1367, 1368, 1374, 1375, 1376, 1377, 1388, 1389, 1391, 1392, 1393, 1770, 1810, 2872, 2875, 2876, 2877, 2880, 2881, 2882, 2883, 2884, 2885, 2887, 2888, 2890, 2894, 2895, 2896, 2897, 2898, 2899, 2900, 2901, 2902, 2903, 2904, 2905, 2906, 2911, 2912, 2913, 2914, 2915, 2918, 2919, 2920, 2921, 2922, 2924, 2925, 2926, 2927, 2931, 2932, 2933, 2934, 2935, 2936, 2937, 2939, 2942, 2943, 2944, 2945, 2946, 2947, 2948, 2949, 2951, 3324, 3325, 3326, 3327, 3328, 3329, 3330, 3331, 3332, 3333, 3334, 3335, 3336, 3337, 3338, 3339, 3487],
		7:[969, 973, 975, 979, 980, 981, 982, 985, 986, 992, 993, 998, 999, 1000, 1001, 1002, 1003, 1004, 1009, 1013, 1015, 3130],
		8:[905, 911, 912, 913, 2378, 2379, 2380, 2381, 2382, 2383, 2384],
		9:[154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 2085, 2086, 2087, 2088, 2089, 2090, 2347, 2605, 3323],
		10:[1258, 2091, 2092, 2093, 2094, 2095, 2096, 2097, 2098, 2099, 2100, 2101, 3367, 3368, 3369, 3370, 3371, 3372, 3373, 3374, 3375, 3376, 3377, 3378, 3379, 3380, 3381, 3382, 3383, 3384, 3385, 3386, 3387, 3388, 3389, 3390, 3391, 3392, 3393, 3394, 3395, 3396, 3397, 3398, 3399, 3400, 3401, 3402, 3403, 3404, 3405, 3406, 3407, 3408, 3409, 3410, 3411, 3412],
		11:[20, 21, 22, 23, 24, 49, 50, 51, 52, 108, 116, 122, 141, 145, 148, 414, 476, 484, 527, 528, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 699, 700, 701, 704, 707, 711, 712, 1775, 1943, 1945, 2022, 2026, 2033, 2037, 2038, 2042, 3060],
		12:[48, 1679, 1680, 1681, 1682, 1683, 1684, 1685, 1686, 1687, 1688, 1689, 1690, 1691, 1692, 1693, 1694, 1695, 1696, 1697, 1698, 1699, 1700, 1701, 1702, 1703, 1704, 1705, 1706, 1707, 1708, 1709, 1712, 1714, 1715, 1716, 1717, 1718, 1719, 3522, 3523, 3524],
		13:[1492, 2162, 2163, 2164, 2165, 2166, 2167, 2168, 2169, 2214, 2215, 2216, 2217, 2218, 2219, 2220, 2221, 2222, 2223, 2224, 2225, 2226, 2227, 2228, 2229, 2232, 2233, 2234, 2235, 2236, 2237, 2238, 2239, 2240, 2241, 2242, 2243, 2257, 2258, 2260],
		14:[91, 92, 121, 139, 559, 573, 574, 575, 576, 577, 578, 579, 604, 609, 615, 616, 1394, 1442, 1454, 1502, 1504, 1505, 1506, 1507, 1508, 1509, 1510, 1511, 1512, 1513, 1514, 1515, 1516, 1517, 1518, 1519, 1520, 1521, 1522, 1523, 1524, 1525, 1527, 1528, 1529, 1530, 1531, 1751, 1798, 1799, 1806, 1807, 1950, 1951, 1953, 1954, 1955, 1956, 1957, 1958, 1959, 1960, 1961, 1962, 1963, 1964, 1965, 2002, 2003, 2004, 2005, 2006, 2007, 2276, 2308, 2593, 2594, 2595, 2755, 3038, 3072, 3073, 3074, 3075, 3076, 3077, 3078, 3079, 3080, 3081, 3082, 3083, 3084, 3085, 3086, 3087, 3088, 3089, 3090, 3091, 3092, 3093, 3094, 3095, 3096, 3098, 3099, 3100, 3101, 3102, 3103, 3104, 3105, 3106, 3107, 3108, 3109, 3110, 3111, 3112, 3133, 3134, 3437, 3438],
		15:[1886, 1896, 1897],
		16:[888, 1876, 1877, 1878, 1879, 1880, 1881, 1882, 1883, 1884, 1885, 1887, 1888, 1889, 1890, 1891, 1892, 1893, 1894, 1895, 1900, 1901, 1902, 1903, 1904, 1905, 1906, 1907, 1908, 1909, 1916, 1917, 1918, 1919, 1920, 1921, 1922, 1923, 1924, 1925],
		17:[725, 970, 971, 972, 974, 976, 977, 978, 983, 984, 987, 988, 989, 990, 991, 994, 995, 996, 1005, 1006, 1007, 1008, 1010, 1011, 1012, 1014, 1016, 1017, 3131, 3132],
		18:[2873, 2874, 2878, 2879, 2886, 2889, 2892, 2893, 2907, 2908, 2909, 2910, 2916, 2917, 2923, 2928, 2929, 2930, 2938, 2940, 2941, 2950, 2952, 3069],
		19:[1606, 1607, 1608, 1610, 1611, 1612, 1613, 1614, 1615, 1616, 1617, 1618, 1619, 1620, 1621, 1622, 1623, 1624, 1625, 1626, 1627, 1628, 1629, 1630],
		20:[2230, 2231, 2244, 2245, 2246, 2247, 2248, 2249, 2250, 2251, 2252, 2253, 2254, 2255, 2256, 2259, 2261, 2262, 2267, 2268, 2269, 2270, 2271, 2272],
		21:[1051, 1052, 1053, 1059, 1060, 1062, 1063, 1064, 1067, 1068, 1069, 1071, 1073, 1074, 1075, 1076, 1077, 1078, 1079, 1082, 1083, 1084, 1085, 1086, 1087, 1088, 1089, 1090, 1091, 1092, 1093, 1094, 1095, 1096, 1097, 1098, 1105, 1106, 1107, 1112, 1113, 1114, 1115, 1116, 1119, 1120, 1288, 1289, 1290, 1291, 1292, 1293, 1294, 1295, 1296, 1297, 1873, 1874, 1875, 3490, 3498, 3499, 3500, 3502, 3503, 3504, 3506, 3507, 3513, 3519, 3526],
		22:[71, 72, 73, 74, 111, 113, 117, 147, 152, 192, 244, 281, 282, 292, 293, 294, 295, 301, 314, 363, 364, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 389, 390, 391, 392, 393, 394, 395, 398, 399, 400, 401, 402, 403, 404, 406, 407, 408, 409, 412, 530, 531, 926, 968, 1184, 1245, 1298, 1299, 1300, 1301, 1302, 1399, 1452, 1743, 1795, 1796, 1808, 2103, 2348, 3225, 3226, 3227, 3228, 3229, 3230, 3231, 3232, 3233, 3234, 3235, 3236, 3237, 3238, 3239, 3240, 3241, 3242, 3243, 3244, 3245, 3246, 3247, 3248, 3249, 3250, 3251, 3252, 3253, 3254, 3255, 3256, 3257, 3258, 3259, 3260, 3261, 3262, 3263, 3264, 3265, 3266, 3267, 3268, 3269, 3270, 3271, 3272, 3273, 3274, 3275, 3276, 3277, 3278, 3279, 3280, 3283, 3284, 3285, 3457, 3458, 3484, 3485, 3486, 3528, 3529, 3536, 3537],
		23:[1054, 1055, 1056, 1057, 1058, 1061, 1070, 1072, 1080, 1081, 1099, 1100, 1101, 1102, 1103, 1104, 1111],
		24:[863, 864, 868, 869, 870, 871, 885, 886, 889, 890, 891, 892, 893, 894, 895, 896, 897, 899, 900, 1910, 1911, 1946, 1947, 2370, 2506, 2507, 2508, 2510, 2512, 2513, 2518, 2519, 2520, 2523, 2524, 2525, 2532, 2533, 2553, 2554, 2555, 2556, 2557, 2558, 2559, 2560, 2561],
		25:[910, 2310, 2311, 2758, 2760, 2774, 2778, 2779, 2780, 2786, 2787, 2788, 2789, 2795, 2796, 2800, 2803, 2806, 2807, 2808, 2811, 2812, 2813, 2815, 2816, 2817, 2818, 2819, 2820, 2821, 2822, 2823, 2824, 2826, 2827, 2828, 2830, 2831, 2832, 2833, 2834, 2838, 2839, 2840, 2841, 2842, 2843, 2844, 2845, 2846, 2847, 2848, 2849, 2852, 2853, 2854, 2855, 2856, 2857, 2858, 2859, 2860, 2863, 2864, 2865, 2867, 2868],
		26:[2596, 2597, 2598, 2599, 2600, 2601, 2602, 2603, 2604, 2606, 2607, 2608, 2609, 2611, 2612, 2613, 2614, 2615, 2616],
		27:[87, 88, 89, 1065, 1066, 1108, 1109, 1110, 1117, 1118, 1926, 1927, 1928, 1929, 1930, 1931, 1932, 1933, 1940, 1941, 2715],
		28:[125, 129, 130, 134, 135, 136, 137, 469, 482, 570, 924, 1323, 1554, 1555, 1556, 1557, 1558, 1559, 1560, 1561, 1563, 1564, 1565, 1566, 1567, 1568, 1569, 1570, 1572, 1573, 1574, 1575, 1576, 1577, 1578, 1579, 1580, 1581, 1582, 1583, 1584, 1585, 1586, 1587, 1588, 1589, 1590, 1591, 1592, 1593, 1594, 1595, 1596, 1597, 1598, 1599, 1600, 1601, 1602, 1603, 1604, 1605, 2274, 3050, 3493, 3494],
		29:[39, 53, 56, 1148, 1149, 1150, 1151, 1152, 1153, 1154, 1155, 1156, 1157, 1158, 1159, 1160, 1161, 1162, 1163, 1164, 1165, 1166, 1167, 1168, 1169, 1172, 1173, 1174, 1175, 1176, 1177, 1178, 1179, 1180, 1181, 1182, 1183, 1468, 1469, 1470, 1472, 1473, 1474, 1475, 1476, 1477, 1478, 1632, 1633, 1634, 1635, 1636, 1638, 1641, 1642, 1643, 1644, 1645, 1647, 1649, 1650, 1651, 1652, 1653, 1654, 1655, 1656, 1657, 1658, 1659, 1660, 1661, 1662, 1663, 1664, 1665, 1666, 1667],
		30:[914, 1314, 1398, 1678, 1721, 2008, 2009, 2010, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2023, 2024, 2025, 2027, 2028, 2029, 2030, 2031, 2032, 2034, 2035, 2036, 2039, 2041, 2043, 2044, 2045, 2046, 2047, 2048, 2049, 2050, 2051, 2052, 2053, 2054, 2055, 2056, 2057, 2058, 2059, 2060, 2061, 2062, 3439, 3440, 3441, 3442, 3445, 3447, 3449],
		31:[744, 745, 746, 747, 748, 754, 755, 756, 759, 760, 761, 762, 763, 764, 765, 766, 767, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 785, 786, 787, 788, 789, 792, 793, 794, 798, 799, 800, 804, 813, 819, 820, 822, 825, 826, 830, 831, 835, 837, 838],
		32:[103, 1395, 1551, 1757, 2362, 2620, 2621, 2622, 2626, 2627, 2628, 2629, 2630, 2631, 2636, 2637, 2638, 2644, 2648, 2649, 2650, 2651, 2658, 2661, 2662, 2664, 2666, 2667, 2668, 2671, 2672, 2673, 2674, 2675, 2677, 2678, 2679, 2681, 2684, 2687, 2689, 2692, 2700, 2701, 2703, 2704, 2705, 2706, 2708],
		33:[1246, 1247, 1248, 1249, 1250, 1251, 1252, 1253, 1254, 1255, 1256, 1257],
		34:[196, 197, 198, 199, 200, 201, 206, 207, 210, 211, 212, 213, 216, 217, 218, 219, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 262, 268, 270, 271, 272, 273, 276, 278, 279, 280, 284, 285, 286, 287, 288, 289, 290, 291, 300, 302, 303, 305, 306, 307, 308, 309, 310, 312, 317, 318, 319, 330, 332, 339, 340, 342, 343, 344, 345, 346, 348, 349, 350, 351, 352, 353, 354, 355, 357, 359, 360, 361, 362, 365, 366, 367, 368, 369, 370, 371, 385, 386, 387, 388, 396, 881],
		35:[109, 680, 721, 858, 872, 1942, 2414, 2415, 2416, 2417, 2418, 2419, 2420, 2421, 2422, 2423, 2424, 2425, 2426, 2427, 2428, 2429, 2430, 2431, 2432, 2433, 2434, 2435, 2436, 2437, 2438, 2439, 2440, 2441, 2442, 2443, 2444, 2445, 2446, 2447, 2448, 2449, 2450, 2451, 2452, 2453, 2454, 2455, 2456, 2457, 2458, 2459, 2460, 2461, 2462, 2463, 2464, 2465, 2466, 2467, 2468, 2469, 2470, 2471, 2472, 2473, 2474, 2475, 2476, 2477, 2478, 2479, 2480, 2481, 2482, 2483, 2484, 2485, 2486, 2487, 2488, 2489, 2490, 2491, 2492, 2493, 2494, 2495, 2496, 2497, 2498, 2500, 2501, 2502, 2503, 2504, 2505, 2509, 2511, 2514, 2515, 2516, 2517, 2521, 2522, 2526, 2527, 2528, 2529, 2530, 2531, 2534, 2535, 2536, 2537, 2538, 2539, 2540, 2541, 2542, 2543, 2544, 2545, 2546, 2547, 2548, 2549, 2550, 2551, 2552, 2562, 2563, 2564, 2565, 2566, 2567, 2568, 2569, 2570, 2591],
		36:[143, 483, 920, 922, 927, 928, 929, 997, 1038, 1039, 1040, 1041, 1042, 1044, 1045, 1046, 1047, 1048, 1049, 1050, 1121, 1122, 1123, 1124, 1125, 1126, 1127, 1128, 1132, 1133, 1135, 1136, 1137, 1138, 1139, 1140, 1185, 1232, 1233, 1234, 1235, 1236, 1238, 1239, 1240, 1241, 1242, 1243, 1244, 1259, 1260, 1261, 1262, 1263, 1264, 1265, 1266, 1267, 1268, 1269, 1270, 1271, 1272, 1273, 1274, 1275, 1276, 1278, 1279, 1280, 1281, 1282, 1283, 1284, 1285, 1286, 1287, 1396, 1550, 1764, 2170, 2171, 2172, 2173, 2174, 2176, 2177, 2178, 2179, 2180, 2181, 2182, 2183, 2184, 2185, 2186, 2187, 2188, 2189, 2190, 2191, 2192, 2193, 2194, 2195, 2196, 2197, 2198, 2199, 2200, 2201, 2202, 2203, 2204, 2205, 2206, 2207, 2208, 2209, 2210, 2211, 2344, 3496, 3497, 3505, 3508, 3509, 3510, 3511, 3512, 3514, 3515, 3516, 3517, 3518],
		37:[485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 2352, 2354, 2355, 2356, 2357, 2358, 2359, 2360, 2361],
		38:[3041, 3042, 3043, 3044, 3045, 3046, 3047, 3048, 3049, 3051, 3052, 3053, 3054, 3059, 3061, 3062, 3065, 3067, 3068],
		39:[110, 119, 260, 415, 470, 713, 739, 740, 921, 1324, 1552, 1763, 1863, 1944, 2111, 2324, 2325, 2617],
		40:[555, 567, 572, 624, 626, 627, 628, 629, 630, 631, 632, 634, 635, 636, 637, 638, 639, 640, 661, 662, 675, 676, 677, 678, 679, 698, 710, 716, 717, 718, 719, 720, 1710, 1711, 1713, 1864, 1865, 1866, 1867],
		41:[2074, 2084, 3351, 3352, 3353, 3354, 3355, 3356, 3357, 3358, 3359],
		42:[67, 105, 106, 107, 1326, 1327, 1339, 1345, 1347, 1348, 1349, 1357, 1358, 1359, 1360, 1369, 1370, 1371, 1372, 1373, 1378, 1379, 1380, 1381, 1382, 1383, 1384, 1385, 1386, 1387, 1390, 3070, 3143, 3144, 3145, 3146, 3147, 3148, 3149, 3150, 3151, 3152, 3153, 3154, 3155, 3156, 3157, 3158, 3159, 3160, 3161, 3162, 3163, 3164, 3165, 3166, 3167, 3168, 3169, 3170, 3171, 3172, 3173, 3174, 3175, 3176],
		43:[930, 931, 932, 934, 935, 936, 937, 938, 939, 940, 943, 947, 949, 950, 951, 952, 953, 959, 960, 961, 962, 963, 967],
		44:[475, 1141, 1142, 1143, 3125, 3126, 3127, 3128, 3129],
		45:[1749, 2128, 2129, 2130, 2131, 2132, 2133, 2134, 2135, 2136, 2137, 2138, 2139, 2140, 2141, 2142, 2143, 2144, 2145, 2146, 2147, 2148, 2149, 2150, 2151, 2152, 2153, 2154, 2155, 2156, 2157, 2158, 2159],
		46:[1488, 1489, 1490, 1491, 2592],
		47:[277, 311, 860, 861, 862, 866, 867, 873, 874, 876, 877, 878, 879, 880, 883, 884, 887, 898, 901, 902, 903],
		48:[3306, 3307, 3308, 3309, 3310, 3311, 3312, 3313, 3314, 3315, 3316, 3317, 3318, 3319, 3320, 3321, 3322],
		49:[1668, 1669, 1670, 1671, 1672, 1673, 1674, 1675, 1676, 1677, 2069],
		50:[532, 534, 543, 545, 546, 547, 568, 571],
		51:[75, 93, 94, 95, 96, 97, 98, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 445, 446, 2278, 2279, 2280, 2610, 3177, 3178, 3179, 3180, 3181, 3182, 3183, 3184, 3185, 3186, 3187, 3188, 3189, 3190, 3191, 3192, 3193, 3194, 3195, 3196, 3197, 3198, 3199, 3200, 3201, 3202, 3203, 3204, 3205, 3206, 3207, 3208, 3209, 3210, 3211, 3212, 3213, 3214, 3215, 3216, 3217, 3218, 3219, 3220, 3221, 3222, 3223, 3224, 3286, 3287, 3288, 3289, 3290, 3291, 3292, 3293, 3294, 3295, 3296, 3297, 3298, 3299, 3300, 3301, 3302, 3303, 3304, 3305],
		52:[1187, 1188, 1189, 1190, 1192, 1193, 1194, 1195, 1197, 1198, 1199, 1200, 1201, 1202, 1203, 1204, 1206, 1207, 1210, 1211, 1213, 1214, 1215, 1216, 1217, 1218, 1219, 1220, 1221, 1222],
		53:[70, 214, 215, 239, 283, 841, 1191, 1196, 1205, 1223, 1224, 1225, 1226, 1227, 1228, 1229, 1230, 1446, 1493, 1494, 1496, 1497, 1498, 1500, 1773, 1777, 1778, 1779, 1780, 1781, 1782, 1783, 1784, 1785],
		54:[120, 923, 1231, 1547, 1772, 1774, 1862, 2063, 2064, 2065, 2365, 2367, 3532],
		55:[2957, 2958, 2960, 2983, 2985, 2991, 3003, 3004, 3005, 3006, 3007, 3008, 3009, 3010, 3011, 3012, 3013, 3014, 3015, 3016, 3017, 3018, 3019, 3020, 3021, 3022, 3023, 3024, 3025, 3026],
		56:[2070, 2071, 2263, 2264, 2265, 2266, 2273, 2623, 2624, 2632, 2633, 2634, 2635, 2639, 2640, 2641, 2642, 2643, 2645, 2646, 2647, 2652, 2653, 2654, 2655, 2656, 2657, 2659, 2660, 2663, 2665, 2669, 2670, 2676, 2680, 2682, 2683, 2685, 2686, 2693, 2695, 2696, 2697, 2986],
		57:[2759, 2761, 2762, 2763, 2764, 2765, 2767, 2768, 2769, 2770, 2771, 2773, 2775, 2776, 2777, 2784, 2785, 3281, 3342, 3344],
		58:[2371, 2372, 2375],
		59:[1418, 1419, 1420, 1421, 1422, 1423, 1424, 1425, 1426, 1427, 1428, 1429, 1430, 1431, 1432, 1433, 1434, 1435, 1436, 1437, 1438, 1439, 1440, 1495, 1499, 1503],
		60:[2869, 2871],
		61:[118, 144, 150, 413, 471, 916, 1026, 2161, 2710, 2711, 2712, 3459],
		62:[2781, 2782, 2783, 2804, 2805, 2835, 2836, 2837, 2850],
		63:[933, 941, 942, 944, 945, 946, 948, 954, 955, 956, 957, 958, 964, 965, 966, 2332, 2333, 2341, 2342, 2412],
		64:[3476, 3477, 3478, 3479, 3480, 3481, 3482, 3483, 3521],
		65:[702, 703, 705, 706, 708, 709, 3071, 3113, 3114, 3115, 3116, 3117, 3118, 3119, 3120, 3121, 3122, 3123, 3124, 3413, 3414, 3415, 3416, 3417, 3418, 3419, 3420, 3421, 3422, 3423, 3424, 3425, 3426, 3427, 3428, 3429, 3430, 3431],
		66:[2741, 2742, 2743, 2744, 2747, 2749, 2750, 2751, 2752, 2753, 2754, 2756, 2757],
		67:[562, 1018, 1019, 1020, 1022, 1023, 1024, 1025, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035, 1036, 1147],
		68:[749, 750, 795, 797],
		69:[584, 611, 613, 617, 618, 620, 625, 633, 641, 642, 643, 644, 645],
		70:[2067],
		71:[1949, 1952, 1966, 1967, 1968, 1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001],
		72:[563, 569, 1402, 1403, 1404, 1405, 1406, 1407, 1408, 1409, 1410, 1411, 1412, 1413, 1414, 1415, 1416, 1417],
		73:[30, 31, 32, 33, 34, 35],
		74:[727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 741, 742, 743, 751, 752, 753, 757, 758, 768, 769, 770, 784, 790, 791, 796, 801, 802, 803, 805, 806, 807, 810, 811, 812, 814, 815, 816, 817, 818, 821, 823, 824, 827, 828, 829, 832, 833, 834, 836, 2212],
		75:[1539, 1540, 1541, 1542, 1543, 1544, 1545, 1745],
		76:[1786, 1787, 1788, 1789, 1790, 1791, 1792, 1793, 1794],
		77:[1533, 1534, 1535, 1536, 1537, 1538],
		78:[25, 26, 27, 28, 29, 36, 37, 38, 2619, 2625, 2688, 2690, 2691, 2694, 2698, 2699, 2702, 2707],
		79:[11, 12, 13, 14, 15, 16, 17, 18, 40, 41, 202, 203, 204, 205, 208, 209, 1456, 1457, 1458, 1459, 1460, 1461, 1462, 1466],
		80:[0, 1, 2, 3, 4, 5, 6, 54, 55, 1467, 1471, 1479],
		81:[839, 840, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856, 1444, 2123, 2124, 2125, 2126, 3443, 3444, 3446],
		82:[1483, 1484, 1485, 1486, 1487, 1934],
		83:[42, 57, 58, 59, 60, 65, 66, 81, 82, 83, 84, 85, 86],
		84:[2323],
		85:[236, 925, 1401, 3027, 3345, 3346, 3347, 3348, 3349, 3350],
		86:[1637, 1640],
		87:[3361, 3362, 3363, 3364, 3365, 3366],
		88:[1937],
		89:[99, 100, 101, 102],
		90:[2075, 2076, 2077, 2078, 2079, 2080, 2081, 2082, 2083],
		91:[7, 8, 10, 19, 45, 46, 47, 397, 405, 1208],
		92:[2791, 2797, 2798],
		93:[1639, 1646, 1648],
		94:[1868, 1869, 1870, 1871, 1872],
		95:[1820, 1821, 2299, 2301, 2303],
		96:[1463, 1464, 1465, 2328, 2329, 2330, 2331, 2336, 2337, 2338, 2339, 2340],
		97:[1319, 1320, 1321],
		98:[1315, 1316, 1317],
		99:[2583, 2584, 2585, 2586, 2587, 2588, 2589, 2590],
		100:[2772, 2801, 2802, 2809, 2810, 2814, 2825, 2829, 2851, 2861, 2862, 2866],
		101:[2766, 2790, 2793, 2794, 2799],
		102:[1898, 1899, 1912, 1913, 1914, 1915],
		103:[1733],
		104:[1752, 1755, 1756, 1758, 1759, 1760, 1761],
		105:[304, 472, 473, 474],
		106:[3455, 3456],
		107:[722, 723, 724, 726],
		108:[1828, 1829, 1830, 1832, 2499, 2722],
		109:[3029, 3030, 3032],
		110:[2349, 2350],
		111:[2413],
		112:[127, 1043, 1571, 2313, 2314, 2315, 2316, 2317, 2318, 3527],
		113:[43, 44, 76, 77, 78, 79, 80, 104],
		114:[1037, 1145, 1146],
		115:[3136, 3137, 3138, 3139, 3140, 3141, 3142],
		116:[9, 61, 62, 63, 64, 1209],
		117:[3461, 3462, 3463, 3464, 3465, 3466, 3467, 3468, 3469, 3470, 3471, 3472, 3473, 3474, 3475],
		118:[2968, 2969, 2999],
		119:[1443, 1445, 1447, 1448, 1449, 1450],
		120:[142, 220, 233, 234, 235, 237, 238, 240, 241, 242, 243, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 261, 263, 264, 265, 266, 274, 275, 296, 297, 298, 299, 313, 315, 316, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 333, 334, 335, 336, 337, 338, 865, 1021, 1397, 1400, 2363, 3488],
		121:[2112, 2113, 2114, 2115, 2116, 2117, 2118, 2119, 2120, 2121, 2122],
		122:[598, 599],
		123:[3031],
		124:[1938, 1939],
		125:[1753],
		126:[2579, 2580, 2581, 2582],
		127:[3451, 3452, 3453, 3454, 3520],
		128:[1130, 1131, 1134, 1277],
		129:[267, 875],
		130:[2571, 2572, 2573, 2574, 2575, 2576, 2577, 2578],
		131:[1800, 1802, 1804, 1805],
		132:[1801, 1803],
		133:[2102, 2104, 2105, 2106, 2107, 2108, 2109],
		134:[416, 417, 418],
		135:[3360],
		136:[138, 146, 149],
		137:[558],
		138:[1948],
		139:[1170, 1171, 2353],
		140:[2870],
		141:[859, 1303, 1305, 1306, 1307, 1308, 1309, 1310, 1311, 1312, 1313],
		142:[3340, 3341],
		143:[3039],
		144:[3525],
		145:[123, 124, 126, 128, 131, 132, 133],
		146:[2792],
		147:[3343],
		148:[331, 1754, 2345, 2346],
		149:[481, 2745, 2746, 2748],
		150:[2319, 2320, 2321, 2322, 2326, 2327, 2334, 2335],
		151:[1765, 1766, 1767, 1768, 1769],
		152:[560],
		153:[193],
		154:[477],
		155:[1824, 2721],
		156:[1480, 1481, 1482],
		157:[541],
		158:[2953],
		159:[3063, 3066],
		160:[1237],
		161:[2713, 2714],
		162:[3036, 3037],
		163:[2717, 2718, 2719],
		164:[808, 809],
		165:[1935],
		166:[1744, 1746, 1747, 1748, 1750],
		167:[2618],
		168:[2376, 2377],
		169:[2072, 2073],
		170:[2312],
		171:[90],
		172:[2716],
		173:[3282],
		174:[3135],
		175:[3491, 3492],
		176:[2127],
		177:[2720],
		178:[1318],
		179:[444],
		180:[2011, 2040],
		181:[449, 451, 468],
		182:[3495],
		183:[2175],
		184:[1451],
		185:[529],
		186:[1129],
		187:[1455],
		188:[2351],
		189:[2891],
		190:[915],
		191:[917, 919],
		192:[1553],
		193:[3028],
		194:[1936],
		195:[3064],
		196:[1722],
		197:[151, 153],
		198:[1809],
		199:[195, 269],
		200:[3433],
		201:[341, 347, 356],
		202:[2343],
		203:[3448],
		204:[2740],
		205:[358],
		206:[1742],
		207:[2364, 2366],
		208:[1546],
		209:[882],
		210:[1212],
		211:[3501],
		212:[1501],
		213:[714],
		214:[3535],
		215:[1609],
		216:[1720],
		217:[2309],
		218:[2068],
		219:[1797],
		220:[2954],
		221:[1322],
		222:[1771],
		223:[2275],
		224:[904],
		225:[2213],
		226:[3530],
		227:[1631],
		228:[2066],
		229:[1762],
		230:[2160],
		231:[1186],
		232:[1453],
		233:[3460],
		234:[2307],
		235:[2709],
		236:[2368],
		237:[3489],
		238:[1532],
		239:[3533, 3534],
		240:[3538],
		241:[3035],
		242:[3450],
		243:[1776],
		244:[1304],
		245:[1144],
		246:[3097],
		247:[857],
		248:[2369]
	}