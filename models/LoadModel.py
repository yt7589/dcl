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
		0:[1894, 1908, 1909, 1910, 1911, 1912, 1913, 1914, 1915, 1916, 1917, 1918, 1919, 1920, 1921, 1922, 1923, 1924, 1925, 1926, 1927, 1928, 1929, 1930, 1931, 1932, 1933, 1934, 1935, 1936, 1937, 1938, 1939, 1940, 1941, 1942, 1943],
		1:[1227, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 3226, 3227, 3228, 3229, 3230, 3231, 3232, 3233, 3234, 3235, 3236, 3237, 3238, 3239, 3240, 3241, 3242, 3243, 3244, 3245, 3246, 3247, 3248, 3249, 3250, 3251, 3252, 3253, 3254, 3255, 3256, 3257, 3258, 3259, 3260, 3261, 3262, 3263, 3264, 3265, 3266, 3267, 3268, 3269],
		2:[134, 476, 871, 873, 877, 878, 946, 982, 983, 984, 986, 987, 988, 989, 990, 991, 1072, 1073, 1074, 1075, 1076, 1077, 1078, 1079, 1082, 1083, 1085, 1086, 1087, 1088, 1089, 1090, 1140, 1202, 1203, 1204, 1205, 1206, 1208, 1209, 1210, 1211, 1212, 1213, 1228, 1229, 1230, 1231, 1232, 1233, 1234, 1235, 1236, 1237, 1238, 1239, 1241, 1242, 1243, 1244, 1245, 1246, 1709, 2129, 2130, 2131, 2132, 2133, 2134, 2135, 2136, 2137, 2138, 2139, 2140, 2141, 2142, 2143, 2144, 2145, 2146, 2147, 2148, 2149, 2150, 2151, 2152, 2153, 2154, 2155, 2156, 2157, 2158, 2159, 2160, 2161, 2162, 2163, 2164, 2165, 2291, 3383, 3384, 3391, 3393, 3394, 3395, 3396, 3397, 3399, 3400, 3401, 3402, 3403],
		3:[73, 91, 92, 93, 94, 95, 96, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 422, 423, 424, 425, 426, 2231, 2232, 2233, 2545, 3045, 3046, 3047, 3048, 3049, 3050, 3051, 3052, 3053, 3054, 3055, 3056, 3057, 3058, 3059, 3060, 3061, 3062, 3063, 3064, 3065, 3066, 3067, 3068, 3069, 3070, 3071, 3072, 3073, 3074, 3075, 3076, 3077, 3078, 3079, 3080, 3081, 3082, 3083, 3084, 3085, 3086, 3087, 3088, 3089, 3090, 3091, 3092, 3154, 3155, 3156, 3157, 3158, 3159, 3160, 3161, 3162, 3163, 3164, 3165, 3166, 3167, 3168, 3169, 3170, 3171, 3172, 3173],
		4:[2859, 2860, 2878, 2884, 2891, 2892, 2893, 2894, 2895, 2896, 2897, 2898, 2899, 2900, 2901, 2902, 2903, 2904, 2905, 2906, 2907, 2908, 2909, 2910, 2911, 2912, 2913],
		5:[3289, 3290, 3291],
		6:[115, 135, 139, 371, 464, 869, 972, 1348, 2117, 2642, 2643, 3310],
		7:[11, 12, 13, 14, 15, 16, 17, 18, 38, 39, 197, 198, 199, 200, 203, 204, 1395, 1396, 1397, 1398, 1399, 1400, 1401, 1405],
		8:[882, 890, 891, 893, 894, 895, 897, 903, 904, 905, 906, 907, 913, 914, 915, 2281, 2282, 2289, 2290, 2365],
		9:[3302, 3303, 3304, 3305, 3405],
		10:[147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 2010, 2011, 2012, 2013, 2014, 2015, 2298, 2541, 3191],
		11:[843, 1824, 1825, 1826, 1827, 1828, 1829, 1830, 1831, 1832, 1834, 1835, 1836, 1837, 1838, 1839, 1840, 1841, 1842, 1846, 1847, 1848, 1849, 1850, 1851, 1856, 1857, 1858, 1859, 1860, 1861, 1862, 1863, 1864],
		12:[2004, 2005, 2006, 2007, 2008, 2009],
		13:[108, 116, 241, 373, 463, 689, 708, 709, 872, 1276, 1484, 1708, 1814, 1888, 2273, 2274],
		14:[129, 136],
		15:[37, 52, 55, 1096, 1100, 1101, 1102, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1124, 1127, 1128, 1129, 1130, 1131, 1132, 1133, 1134, 1135, 1136, 1137, 1138, 1407, 1408, 1409, 1411, 1412, 1413, 1414, 1415, 1416, 1417, 1572, 1573, 1574, 1575, 1576, 1578, 1581, 1582, 1583, 1584, 1585, 1587, 1589, 1591, 1592, 1593, 1594, 1595, 1596, 1597, 1598, 1599, 1600, 1601, 1602, 1603, 1604, 1605, 1606, 1607, 1608, 1609, 1611, 1612, 1613],
		16:[3174, 3175, 3176, 3177, 3178, 3179, 3180, 3181, 3182, 3183, 3184, 3185, 3186, 3187, 3188, 3189, 3190],
		17:[20, 21, 22, 23, 24, 48, 49, 50, 51, 106, 113, 119, 132, 138, 372, 469, 477, 516, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 676, 677, 678, 681, 684, 688, 1727, 1972, 1973, 1975],
		18:[701, 702, 703, 704, 705, 706, 707, 710, 711, 712, 720, 721, 722, 726, 727, 737, 738, 752, 758, 767, 768, 770, 771, 772, 775, 776, 777, 779, 780, 781, 782, 783, 786, 788, 789, 792, 793, 796, 797, 798, 1503, 2171],
		19:[2690, 2716, 2722, 2723, 2726, 2740, 2769],
		20:[470],
		21:[107, 817, 1887, 2367, 2368, 2369, 2370, 2371, 2372, 2373, 2374, 2375, 2376, 2377, 2378, 2379, 2380, 2381, 2382, 2383, 2384, 2385, 2386, 2387, 2388, 2389, 2390, 2391, 2392, 2393, 2394, 2395, 2396, 2397, 2398, 2399, 2400, 2401, 2402, 2403, 2404, 2405, 2406, 2407, 2408, 2409, 2410, 2411, 2412, 2413, 2414, 2415, 2416, 2417, 2418, 2419, 2420, 2421, 2422, 2423, 2424, 2425, 2426, 2427, 2428, 2429, 2430, 2431, 2432, 2433, 2434, 2435, 2436, 2437, 2438, 2439, 2440, 2441, 2442, 2443, 2444, 2448, 2450, 2453, 2454, 2455, 2456, 2460, 2461, 2465, 2466, 2467, 2468, 2469, 2470, 2473, 2474, 2475, 2476, 2477, 2478, 2479, 2480, 2481, 2482, 2483, 2484, 2485, 2486, 2487, 2488, 2489, 2490, 2500, 2501, 2502, 2503, 2504, 2505, 2506, 2507, 2508, 2529],
		22:[131, 443, 444, 450, 453, 454, 455, 456, 457, 458, 459, 460, 1386],
		23:[2327],
		24:[818, 1262, 1263, 1264, 1265, 1266, 1267, 1268],
		25:[543, 802, 967, 968, 969, 970, 971, 973, 974, 975, 976, 977, 978, 979, 980, 1095],
		26:[1999, 2000, 2218, 2219, 2220, 2221, 2556, 2557, 2565, 2566, 2567, 2568, 2572, 2573, 2574, 2575, 2576, 2578, 2579, 2580, 2585, 2586, 2587, 2588, 2589, 2590, 2592, 2593, 2596, 2598, 2602, 2603, 2609, 2613, 2615, 2616, 2618, 2619, 2626, 2628, 2629, 2630, 2880],
		27:[868, 1269, 1347, 1624, 1676, 1720, 1949, 1950, 1951, 1952, 1953, 1954, 1955, 1956, 1957, 1958, 1959, 1960, 1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969, 1970, 1971, 1974, 1976, 1977, 1978, 1979, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 3294, 3295, 3296, 3300, 3301],
		28:[67, 112, 368, 369, 427, 428, 533, 593, 870, 1277, 1280, 1281, 1282, 1283, 1284, 1285, 1286, 1287, 1288, 1289, 1290, 1292, 1293, 1294, 1295, 1296, 1298, 1302, 1303, 1304, 1305, 1306, 1307, 1308, 1313, 1314, 1315, 1316, 1317, 1318, 1319, 1325, 1326, 1327, 1328, 1339, 1340, 1342, 1343, 1344, 1712, 1768, 2776, 2779, 2780, 2781, 2784, 2785, 2786, 2787, 2788, 2789, 2791, 2792, 2794, 2797, 2798, 2799, 2800, 2801, 2802, 2803, 2804, 2805, 2806, 2807, 2808, 2809, 2814, 2815, 2816, 2817, 2818, 2821, 2822, 2823, 2824, 2825, 2827, 2828, 2829, 2830, 2834, 2835, 2836, 2837, 2838, 2839, 2840, 2842, 2845, 2846, 2847, 2848, 2849, 2850, 2851, 2852, 2854, 3192, 3193, 3194, 3195, 3196, 3197, 3198, 3199, 3200, 3201, 3202, 3203, 3204, 3205, 3206, 3207, 3337],
		29:[1616, 1617, 1618, 1619, 1620, 1621, 1622, 1623, 1998],
		30:[3326, 3327, 3328, 3329, 3330, 3331, 3332, 3333, 3334, 3406],
		31:[0, 1, 2, 3, 4, 5, 6, 53, 54, 1406, 1410, 1418],
		32:[191, 192, 193, 194, 195, 196, 201, 202, 205, 206, 207, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 246, 251, 252, 253, 254, 255, 257, 259, 260, 261, 265, 266, 267, 268, 269, 270, 280, 282, 283, 284, 285, 286, 287, 289, 292, 293, 302, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 325, 326, 327, 328, 329, 330, 331, 345, 346, 347, 837, 2647],
		33:[1550, 1551, 1552, 1553, 1554, 1555, 1556, 1557, 1558, 1559, 1560, 1561, 1562, 1563, 1564, 1565, 1566, 1567, 1568, 1569, 1570, 1571],
		34:[1699, 2059, 2060, 2061, 2062, 2063, 2064, 2065, 2066, 2067, 2068, 2069, 2070, 2071, 2072, 2073, 2074, 2075, 2076, 2077, 2078, 2079, 2080, 2081, 2082, 2083, 2084, 2085, 2086, 2087, 2088, 2089],
		35:[859, 865, 866, 867, 2331, 2332, 2333, 2334, 2335, 2336, 2337],
		36:[1710, 1711],
		37:[1207],
		38:[478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 2304, 2305, 2306, 2307, 2308, 2309, 2310, 2311],
		39:[258, 288, 819, 820, 824, 825, 830, 832, 833, 834, 835, 836, 838, 839, 842, 853, 856, 857, 858],
		40:[2509, 2510, 2511, 2512, 2513, 2514, 2515, 2516],
		41:[2680, 2682, 2683, 2684, 2685, 2687, 2688, 2689, 2691, 2693, 2694, 2695, 2701, 3149, 3209],
		42:[1003, 1004, 1005, 1011, 1012, 1014, 1015, 1016, 1019, 1020, 1021, 1023, 1025, 1026, 1027, 1028, 1029, 1030, 1033, 1034, 1035, 1036, 1037, 1038, 1039, 1040, 1041, 1042, 1043, 1044, 1045, 1046, 1047, 1048, 1049, 1056, 1057, 1058, 1063, 1064, 1065, 1066, 1067, 1070, 1071, 1247, 1248, 1249, 1250, 1251, 1252, 1253, 1254, 1255, 1256, 1821, 1822, 1823, 3339, 3381, 3382, 3385, 3386, 3387, 3388, 3389, 3390, 3392, 3398, 3404],
		43:[66, 103, 104, 105, 1278, 1279, 1291, 1297, 1299, 1300, 1301, 1309, 1310, 1311, 1312, 1320, 1321, 1322, 1323, 1324, 1329, 1330, 1331, 1332, 1333, 1334, 1335, 1336, 1337, 1338, 1341, 2946, 3011, 3012, 3013, 3014, 3015, 3016, 3017, 3018, 3019, 3020, 3021, 3022, 3023, 3024, 3025, 3026, 3027, 3028, 3029, 3030, 3031, 3032, 3033, 3034, 3035, 3036, 3037, 3038, 3039, 3040, 3041, 3042, 3043, 3044],
		44:[1427, 1428, 1429, 2530],
		45:[2272],
		46:[1419, 1420, 1421],
		47:[117, 874, 1201, 1480, 1714, 1726, 1813, 1995, 1996, 2324, 2325, 3417],
		48:[1164, 1165, 1166, 1167, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1176, 1177, 1178, 1180, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190, 1191, 1192, 1193, 1194],
		49:[2774],
		50:[2923, 2924, 2925, 2926, 2927, 2928, 2929, 2930, 2931, 2933, 2934, 2935, 2940, 2941, 2943, 2944],
		51:[187, 519, 520, 521, 522, 523, 524, 526, 527, 530, 531, 532, 534, 535, 536, 538, 539, 545, 546, 547, 559, 560, 561, 562, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 579, 580, 581, 582, 584, 585, 586, 587, 589, 591, 598, 600, 601, 602, 690, 2922, 2936, 2937, 2938, 2939, 3416],
		52:[537, 548, 551, 603, 605, 606, 607, 608, 609, 610, 611, 613, 614, 615, 616, 617, 618, 619, 639, 640, 653, 654, 655, 656, 657, 675, 687, 691, 692, 693, 694, 695, 1666, 1667, 1669, 1815, 1816, 1817],
		53:[2003, 3215, 3216, 3217, 3218, 3219, 3220, 3221],
		54:[121, 123, 124, 126, 127, 128, 462, 475, 550, 875, 1275, 1485, 1486, 1487, 1488, 1489, 1490, 1491, 1493, 1494, 1495, 1496, 1497, 1498, 1499, 1501, 1502, 1504, 1505, 1506, 1507, 1508, 1509, 1510, 1511, 1512, 1513, 1514, 1515, 1516, 1517, 1518, 1519, 1520, 1521, 1522, 1523, 1524, 1525, 1526, 1527, 1528, 1529, 1530, 1531, 1532, 2228, 2932, 3379, 3380],
		55:[97, 98, 99, 100],
		56:[525],
		57:[528, 529],
		58:[821, 822, 826, 827, 828, 829, 840, 841, 844, 845, 846, 847, 848, 849, 850, 851, 852, 854, 855, 1852, 1853, 1889, 1890, 2445, 2446, 2447, 2449, 2451, 2452, 2457, 2458, 2459, 2462, 2463, 2464, 2471, 2472, 2491, 2492, 2493, 2494, 2495, 2496, 2497, 2498, 2499],
		59:[9, 60, 61, 62, 63, 1182],
		60:[2533, 2534, 2535, 2536, 2537, 2538, 2539, 2540, 2542, 2543, 2544, 2546, 2547, 2548, 2549, 2550],
		61:[111, 471, 472, 473, 1481, 1482, 1492, 2034, 2230, 2857, 2858, 2861, 2862, 2863, 2864, 2867, 2868, 2869, 2870, 2871, 2872, 2873, 2874, 2875, 2876, 2877, 2879, 2881, 2882, 2883, 2885, 2886, 2887, 2888, 2890, 2917, 2918],
		62:[1430, 2120, 2121, 2122, 2123, 2124, 2125, 2126, 2172, 2173, 2174, 2175, 2176, 2177, 2178, 2179, 2180, 2181, 2182, 2183, 2184, 2185, 2186, 2187, 2190, 2191, 2192, 2193, 2194, 2195, 2196, 2197, 2198, 2199, 2200, 2201, 2213, 2214, 2216],
		63:[1737, 1738, 1739, 1740, 1741, 1742, 1743, 1744, 1745],
		64:[43, 69, 70, 71, 72, 109, 110, 114, 137, 140, 185, 230, 262, 263, 271, 272, 273, 274, 279, 290, 323, 324, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 348, 349, 350, 351, 352, 353, 355, 356, 357, 358, 359, 360, 361, 363, 364, 365, 366, 367, 370, 517, 518, 917, 1139, 1214, 1257, 1258, 1259, 1260, 1261, 1349, 1393, 1695, 1748, 1750, 1767, 2027, 3093, 3094, 3095, 3096, 3097, 3098, 3099, 3100, 3101, 3102, 3103, 3104, 3105, 3106, 3107, 3108, 3109, 3110, 3111, 3112, 3113, 3114, 3115, 3116, 3117, 3118, 3119, 3120, 3121, 3122, 3123, 3124, 3125, 3126, 3127, 3128, 3129, 3130, 3131, 3132, 3133, 3134, 3135, 3136, 3137, 3138, 3139, 3140, 3141, 3142, 3143, 3144, 3145, 3146, 3147, 3148, 3151, 3152, 3153, 3308, 3309, 3335, 3336, 3414, 3415, 3419, 3420],
		65:[41, 42, 74, 75, 76, 77, 78, 102],
		66:[713, 714, 715, 716, 717, 723, 724, 725, 728, 729, 730, 731, 732, 733, 734, 735, 736, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 753, 754, 755, 756, 757, 759, 760, 761, 764, 765, 766, 769, 778, 784, 785, 787, 790, 791, 794, 795, 799, 800, 801],
		67:[699, 919, 920, 921, 923, 925, 926, 927, 932, 933, 936, 937, 938, 939, 940, 943, 944, 945, 954, 955, 956, 957, 959, 960, 961, 963, 965, 966, 3000, 3001],
		68:[860, 861, 862, 863, 2338, 2339, 2340, 2341, 2342, 2343, 2344, 2345, 2346, 2347, 2348, 2349, 2350, 2351, 2352, 2353, 2354, 2355, 2356, 2357, 2358, 2359, 2360, 2361, 2362, 2363, 2364],
		69:[68, 208, 224, 264, 1168, 1179, 1195, 1196, 1197, 1198, 1199, 1200, 1389, 1431, 1432, 1434, 1435, 1436, 1438, 1725, 1728, 1729, 1730, 1731, 1732, 1733, 1734, 1735, 1736],
		70:[1845, 1854, 1855],
		71:[679, 680, 682, 683, 685, 686, 2947, 2984, 2985, 2986, 2987, 2988, 2989, 2990, 2991, 2992, 2993, 3270, 3271, 3272, 3273, 3274, 3275, 3276, 3277, 3278, 3279, 3280, 3281, 3282, 3283, 3284, 3285, 3286, 3287, 3288],
		72:[85, 86, 87, 1017, 1018, 1059, 1060, 1061, 1068, 1069, 1865, 1866, 1867, 1868, 1869, 1870, 1871, 1872, 1886, 2645],
		73:[1364, 1365, 1366, 1367, 1368, 1369, 1370, 1371, 1372, 1373, 1374, 1375, 1376, 1377, 1378, 1379, 1380, 1381, 1382, 1383, 1384, 1385, 1433, 1437, 1440],
		74:[1270, 1271, 1272],
		75:[133, 220, 221, 222, 223, 226, 227, 228, 232, 233, 234, 235, 236, 237, 238, 239, 240, 244, 247, 248, 249, 256, 275, 276, 277, 278, 291, 294, 295, 296, 297, 298, 299, 300, 303, 304, 305, 823, 1346, 2313, 3338],
		76:[2521, 2522, 2523, 2524, 2525, 2526, 2527, 2528],
		77:[1997],
		78:[89, 90, 118, 130, 541, 552, 553, 554, 555, 556, 557, 558, 583, 588, 594, 595, 1387, 1394, 1439, 1441, 1442, 1443, 1444, 1445, 1446, 1447, 1448, 1449, 1450, 1451, 1452, 1453, 1454, 1455, 1456, 1457, 1458, 1459, 1460, 1461, 1462, 1463, 1464, 1465, 1466, 1758, 1759, 1766, 1892, 1893, 1895, 1896, 1897, 1898, 1899, 1900, 1901, 1902, 1903, 1904, 1905, 1906, 1907, 1944, 1945, 1946, 1947, 1948, 2229, 2258, 2531, 2532, 2920, 2948, 2949, 2950, 2951, 2952, 2953, 2954, 2955, 2956, 2957, 2958, 2959, 2960, 2961, 2962, 2963, 2964, 2965, 2966, 2967, 2968, 2969, 2970, 2971, 2972, 2973, 2974, 2975, 2976, 2977, 2978, 2979, 2980, 2981, 2982, 2983, 3002, 3292, 3293],
		79:[468, 1091, 1092, 1093, 2994, 2995, 2996, 2997, 2998],
		80:[40, 56, 57, 58, 59, 64, 65, 79, 80, 81, 82, 83, 84],
		81:[2777, 2778, 2782, 2783, 2790, 2793, 2795, 2796, 2810, 2811, 2812, 2813, 2819, 2820, 2826, 2831, 2832, 2833, 2841, 2843, 2844, 2853, 2855, 2945],
		82:[918, 922, 924, 928, 929, 930, 931, 934, 935, 941, 942, 947, 948, 949, 950, 951, 952, 953, 958, 962, 964, 2999],
		83:[47, 1638, 1639, 1640, 1641, 1642, 1643, 1644, 1645, 1646, 1647, 1648, 1649, 1650, 1651, 1652, 1653, 1654, 1655, 1656, 1657, 1658, 1659, 1660, 1661, 1662, 1663, 1664, 1665, 1668, 1670, 1671, 1672, 1673, 1674, 1675, 3407, 3408, 3409],
		84:[577, 578],
		85:[1677, 1678, 1679, 1680, 1681, 1682, 1683, 1684, 1685, 1687, 1688, 1689, 1690, 1691, 1692, 1693, 1694, 1769, 1770, 1771, 1772, 1773, 1774, 1775, 1776, 1779, 1781, 1782, 1785, 1786, 1787, 1788, 1789, 1790, 1791, 1792, 1793, 1794, 1795, 1796, 1797, 1798, 1799, 1800, 1801, 1802, 1803, 1804, 1805, 1806, 1807, 1808, 1809, 1810, 1811, 1812, 2234, 2235, 2236, 2237, 2238, 2239, 2240, 2241, 2242, 2243, 2244, 2245, 2246, 2247, 2248, 2249, 2251, 2253, 2255, 2256, 2257, 2328, 2329, 2652, 2653, 2654, 2655, 2656, 2657, 2658, 2659, 2660, 2661, 2662, 2663, 2664, 2665],
		86:[474, 2669, 2670],
		87:[2686, 2706, 2709, 2710],
		88:[1891],
		89:[301, 2292],
		90:[2698, 2699, 2700, 2718, 2746, 2747, 2760],
		91:[250, 831],
		92:[718, 719, 762, 763],
		93:[1783, 1784, 2651],
		94:[864, 2259, 2260, 2679, 2681, 2692, 2696, 2697, 2702, 2703, 2704, 2705, 2711, 2712, 2715, 2717, 2719, 2720, 2721, 2724, 2725, 2727, 2728, 2729, 2730, 2731, 2732, 2733, 2734, 2735, 2736, 2737, 2738, 2739, 2741, 2742, 2743, 2744, 2745, 2748, 2749, 2750, 2751, 2752, 2753, 2754, 2755, 2756, 2757, 2758, 2759, 2761, 2762, 2763, 2764, 2765, 2766, 2767, 2768, 2770, 2771, 2772, 2773],
		95:[3004, 3005, 3006, 3007, 3008, 3009, 3010],
		96:[25, 26, 27, 28, 35, 36, 2552, 2558, 2621, 2623, 2624, 2627, 2631, 2632, 2635, 2640],
		97:[2648, 2649],
		98:[2856],
		99:[981, 1094],
		100:[122, 985, 1500, 2264, 2265, 2266, 2267, 2268, 3413],
		101:[101, 1345, 1483, 2312, 2553, 2554, 2555, 2559, 2560, 2561, 2562, 2563, 2564, 2569, 2570, 2571, 2577, 2581, 2582, 2583, 2584, 2591, 2594, 2595, 2597, 2599, 2600, 2601, 2604, 2605, 2606, 2607, 2608, 2610, 2611, 2612, 2614, 2617, 2620, 2622, 2625, 2633, 2634, 2636, 2637, 2638, 2639, 2641],
		102:[2551],
		103:[1006, 1007, 1008, 1009, 1010, 1013, 1022, 1024, 1031, 1032, 1050, 1051, 1052, 1053, 1054, 1055, 1062],
		104:[1080, 1081, 1084, 1240],
		105:[696, 697, 698, 700],
		106:[3306, 3307],
		107:[1780, 2650],
		108:[2644],
		109:[2026, 2028, 2029, 2030, 2031, 2032, 2033],
		110:[2517, 2518, 2519, 2520],
		111:[2865, 2866, 2889],
		112:[1467, 1468, 1469, 1470],
		113:[7, 8, 10, 19, 44, 45, 46, 354, 362, 1181],
		114:[419],
		115:[563, 590, 592, 596, 597, 599, 604, 612, 620, 621, 622, 623, 624],
		116:[1701, 1703, 1704, 1705, 1706, 1707],
		117:[1702],
		118:[540],
		119:[544, 549, 1351, 1352, 1353, 1354, 1355, 1356, 1357, 1358, 1359, 1360, 1361, 1362, 1363],
		120:[542],
		121:[2366],
		122:[2915],
		123:[803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 1388, 2041, 2042, 2043, 3297, 3298, 3299],
		124:[2666, 2667, 2668, 2671, 2672, 2673, 2674, 2675, 2676, 2677, 2678],
		125:[876, 1350, 2914, 3211, 3212, 3213, 3214],
		126:[3311, 3312, 3313, 3314, 3315, 3316, 3317, 3318, 3319, 3320, 3321, 3322, 3323, 3324, 3325],
		127:[1474, 1475, 1476, 1477, 1478, 1479, 1696],
		128:[2188, 2189, 2202, 2203, 2204, 2205, 2206, 2207, 2208, 2209, 2210, 2211, 2212, 2215, 2217, 2222, 2223, 2224, 2225, 2226, 2227],
		129:[1215, 1216, 1217, 1218, 1219, 1220, 1221, 1222, 1223, 1224, 1225, 1226],
		130:[1697, 1698, 1700],
		131:[29, 30, 31, 32, 33, 34],
		132:[374, 375, 376],
		133:[2035, 2036, 2037, 2038, 2039, 2040],
		134:[1883],
		135:[1402, 1403, 1404, 2277, 2278, 2279, 2280, 2284, 2285, 2286, 2287, 2288],
		136:[3223, 3224, 3225],
		137:[281, 465, 466, 467],
		138:[879, 880, 881, 883, 884, 885, 886, 887, 888, 889, 892, 896, 898, 899, 900, 901, 902, 908, 909, 910, 911, 912, 916],
		139:[1625, 1626, 1627, 1628, 1629, 1630, 1631, 1632, 1633, 1634, 1635, 1636, 1637],
		140:[1533, 1534, 1535, 1536, 1537, 1538, 1539, 1540, 1541, 1542, 1543, 1544, 1545, 1546, 1547, 1548, 1549],
		141:[3340, 3341, 3342, 3343, 3344, 3345, 3346, 3347, 3348, 3349, 3350, 3351, 3352, 3353, 3354, 3355, 3356, 3357, 3358, 3359, 3360, 3361, 3362, 3363, 3364, 3365, 3366, 3367, 3368, 3369, 3370, 3371, 3372, 3373, 3374, 3375, 3376, 3377, 3378],
		142:[3410, 3411],
		143:[3210],
		144:[1141, 1142, 1143, 1144, 1145, 1146, 1147, 1148, 1149, 1150, 1151, 1152, 1153, 1154, 1155, 1156, 1157, 1158, 1159, 1160, 1161, 1162, 1163],
		145:[992, 993, 994, 995, 996, 997, 998, 999, 1000, 1001, 1002],
		146:[1614, 1615],
		147:[1273, 1274],
		148:[429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 445, 446, 447, 448, 449, 451, 452],
		149:[2044, 2045, 2046, 2047, 2048, 2049, 2050, 2051, 2052, 2053, 2054, 2055, 2056, 2057, 2058],
		150:[1713, 1715, 1716, 1717, 1718, 1719, 1721, 1722, 1723, 1724],
		151:[1746, 1747, 1749, 1751, 1752, 1753, 1754, 1755, 1756, 1757],
		152:[1873, 1874, 1875, 1876, 1877, 1878, 1879, 1880, 1881],
		153:[2166, 2167, 2168, 2169, 2170],
		154:[2261, 2262, 2263],
		155:[1980, 2293, 2294, 2295, 2296, 2297, 2314, 2315, 2316, 2317, 2318, 2319, 2320, 2321, 2322, 2323],
		156:[2090, 2091, 2092, 2093, 2094, 2095, 2096, 2097, 2098, 2099, 2100, 2101, 2102, 2103, 2104, 2105, 2106, 2107, 2108, 2109, 2110, 2111, 2112, 2113, 2114, 2115, 2116],
		157:[189],
		158:[2299, 2300, 2301],
		159:[141, 142, 143, 144, 145, 146],
		160:[2118, 2119],
		161:[188, 190, 225, 229, 231, 242, 243, 245],
		162:[1777, 1778, 2250, 2252, 2254],
		163:[3222],
		164:[2707, 2713, 2714],
		165:[3412],
		166:[1471, 1472, 1473],
		167:[3208],
		168:[2127, 2128],
		169:[3418],
		170:[2330],
		171:[2269, 2270, 2271, 2275, 2276, 2283],
		172:[1833, 1843, 1844],
		173:[1610],
		174:[420, 421],
		175:[2708],
		176:[773, 774],
		177:[1097, 1098, 1099],
		178:[1818, 1819, 1820],
		179:[1579, 1588, 1590],
		180:[1577, 1580],
		181:[1125, 1126],
		182:[2942],
		183:[2302, 2303],
		184:[2919],
		185:[2326],
		186:[1390, 1391, 1392],
		187:[120, 125],
		188:[186],
		189:[1686],
		190:[1422, 1423, 1424, 1425, 1426, 1882],
		191:[1884],
		192:[1885],
		193:[1761, 1763],
		194:[1586],
		195:[1760, 1762, 1764, 1765],
		196:[2921],
		197:[2916],
		198:[2775],
		199:[1103],
		200:[461],
		201:[2001, 2002],
		202:[88],
		203:[2646],
		204:[3150],
		205:[3003]
	}