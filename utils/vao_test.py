#
# 车管所测试工具类
#
from pathlib import Path

class VaoTest(object):
    MODE_PREPARE_DATASET = 1001
    MODE_GET_VEHICLE_CODES = 1002
    # 从yt_train.txt中统计出所里附件的180个品牌中未覆盖的品牌
    MODE_GET_UNCOVERED_VCS = 1003
    # 找出未处理过的车辆编号
    MODE_GET_UNKNOWN_VCS = 1004
    # 生成92类已有类型数据集
    MODE_KNOWN_VCS_DS = 1005

    def __init__(self):
        self.name = 'util.VaoTest'

    @staticmethod
    def create_v_bn_no():
        for k, v in VaoTest.vehicle_brands.items():
            VaoTest.v_bn_no[v] = k

    @staticmethod
    def startup():
        mode = VaoTest.MODE_KNOWN_VCS_DS
        if VaoTest.MODE_PREPARE_DATASET == mode:
            VaoTest.create_v_bn_no()
            VaoTest.process_imported_vehicles_main()
            VaoTest.process_domestic_vehicles_main()
            #VaoTest.process_test_data_main()
        elif VaoTest.MODE_GET_VEHICLE_CODES == mode:
            VaoTest.get_all_vehicle_codes()
        elif VaoTest.MODE_GET_UNCOVERED_VCS == mode:
            VaoTest.get_uncovered_vcs()
        elif VaoTest.MODE_GET_UNKNOWN_VCS == mode:
            VaoTest.get_unknown_vcs()
        elif VaoTest.MODE_KNOWN_VCS_DS == mode:
            VaoTest.known_vcs_ds_main()

    @staticmethod
    def get_all_vehicle_codes():
        '''
        从国产车guochanche_2目录中找出所有图片，取出车辆编号，
        形成不重复的车辆编号列表
        '''
        print('获取车辆型号列表...')
        base_dir = Path('/home/up/guochanche_2/')
        rst_file = './vehicle_codes.txt'
        vehicle_code_set = set()
        VaoTest.get_vehicle_codes_in_folder(base_dir, vehicle_code_set)
        print('编号总数量：{0};'.format(len(vehicle_code_set)))
        with open(rst_file, 'a+', encoding='utf-8') as fd:
            for vc in iter(vehicle_code_set):
                fd.write('{0}\r\n'.format(vc))

    @staticmethod
    def get_vehicle_codes_in_folder(folder_name, vehicle_code_set):
        path_obj = Path(folder_name)
        for file_obj in path_obj.iterdir():
            full_name = str(file_obj)
            if not file_obj.is_dir() and full_name.endswith(('jpg','png','jpeg','bmp')):
                arrs = full_name.split('/')
                arrs2 = arrs[-1].split('_')
                vehicle_code = arrs2[0]
                #rst_fd.write('{0}\r\n'.format(vehicle_code))
                vehicle_code_set.add(vehicle_code)
                print('vehicle_code: {0}; size={1};'.format(vehicle_code, len(vehicle_code_set)))
            elif file_obj.is_dir():
                VaoTest.get_vehicle_codes_in_folder(full_name, vehicle_code_set)
            else:
                print('ignore other file: {0};'.format(full_name))
        return vehicle_code_set

    @staticmethod
    def process_test_data_main():
        ds_file = './yt_test.txt'
        base_dir = Path('/media/zjkj/35196947-b671-441e-9631-6245942d671b/品牌')
        with open(ds_file, 'a+', encoding='utf-8') as fd:
            VaoTest.create_test_vehicle_dataset(fd, base_dir)

    @staticmethod
    def create_test_vehicle_dataset(ds_fd, base_dir):
        path_obj = Path(base_dir)
        for file_obj in path_obj.iterdir():
            imgs_dir = str(file_obj)
            arrs = imgs_dir.split('/')
            last_seg = arrs[-1]
            class_id = int(last_seg[0:3]) - 1
            if class_id < 179:
                brand_name = VaoTest.vehicle_brands[last_seg[0:3]]
                VaoTest.list_img_files(ds_fd, imgs_dir, class_id)


    @staticmethod
    def process_domestic_vehicles_main():
        ds_file = './yt_train.txt'
        base_dir = Path('/media/zjkj/My Passport/guochanche_all') #
        with open(ds_file, 'a+', encoding='utf-8') as fd:
            VaoTest.create_domestic_vehicle_dataset(fd, base_dir)
    
    @staticmethod
    def create_domestic_vehicle_dataset(ds_fd, base_dir):
        vc_dict = {} # 车型编号和品牌字典
        with open('./datasets/raw_domestic_brands.txt', 'r', encoding='utf-8') as fd:
            line = fd.readline()
            while line:
                arrs = line.split('*')
                if len(arrs) > 1:
                    arrs2 = arrs[1].split('_')
                    brand_name = arrs2[0]
                    vc_dict[arrs[0]] = brand_name
                line = fd.readline()
        domestic_brands_items = [x for x in base_dir.iterdir() if base_dir.is_dir()]
        for item in domestic_brands_items:
            if not item.is_dir():
                continue
            item_str = str(item)
            arrs = item_str.split('/')
            brand_name = vc_dict[arrs[-1]] if arrs[-1] in vc_dict else '*'
            if '*' == brand_name:
                arrs1 = arrs[-1].split('_')
                if len(arrs1) > 1:
                    brand_name = arrs1[0]
            if brand_name in VaoTest.v_bn_no:
                class_id = int(VaoTest.v_bn_no[brand_name]) - 1
                VaoTest.list_img_files(ds_fd, item_str, class_id)

    @staticmethod
    def process_imported_vehicles_main():
        ds_file = './yt_train.txt'
        base_dir = '/media/zjkj/35196947-b671-441e-9631-6245942d671b/vehicle_type_v2d/vehicle_type_v2d' #
        with open(ds_file, 'a+', encoding='utf-8') as fd:
            VaoTest.create_imported_vehicle_dataset(fd, base_dir)

    @staticmethod
    def create_imported_vehicle_dataset(ds_fd, folder_name):
        path_obj = Path(folder_name)
        for file_obj in path_obj.iterdir():
            if not file_obj.is_dir():
                continue
            imgs_dir = str(file_obj)
            arrs0 = imgs_dir.split('/')
            arrs = arrs0[-1].split('_')
            class_id = int(arrs[0]) - 1
            brand_name = arrs[1]
            VaoTest.list_img_files(ds_fd, imgs_dir, class_id)

    @staticmethod
    def list_img_files(ds_fd, folder_name, class_id):
        '''
        列出该目录以及其子目录下所有图片文件（以jpg为扩展名）列表
        '''
        path_obj = Path(folder_name)
        for file_obj in path_obj.iterdir():
            full_name = str(file_obj)
            if not file_obj.is_dir() and full_name.endswith(('jpg','png','jpeg','bmp')):
                print('{0}*{1}'.format(file_obj, class_id))
                ds_fd.write('{0}*{1}\r\n'.format(file_obj, class_id))
            elif file_obj.is_dir():
                VaoTest.list_img_files(ds_fd, str(file_obj), class_id)
            else:
                print('ignore other file: {0};'.format(full_name))

    @staticmethod
    def process_vehicles():
        _, uncovered_brand_names = VaoTest.process_imported_vehicles()
        #uncovered_brand_names = []
        #for bn in VaoTest.vehicle_brands.values():
        #    uncovered_brand_names.append(bn)
        VaoTest.process_domestic_vehicles(uncovered_brand_names)

    @staticmethod
    def process_domestic_vehicles(uncovered_brand_names):
        print('处理国产车')
        vehicle_codes = []
        vc_dict = {} # 车型编号和品牌字典
        with open('./datasets/raw_domestic_brands.txt', 'r', encoding='utf-8') as fd:
            line = fd.readline()
            while line:
                print('# {0};'.format(line))
                arrs = line.split('*')
                print('len={0};'.format(len(arrs)))
                if len(arrs) > 1:
                    arrs2 = arrs[1].split('_')
                    brand_name = arrs2[0]
                    vc_dict[arrs[0]] = brand_name
                line = fd.readline()
        srcPath = Path('/media/zjkj/My Passport/guochanche_all')
        domestic_brands_items = [x for x in srcPath.iterdir() if srcPath.is_dir()]
        for item in domestic_brands_items:
            item_str = str(item)
            arrs = item_str.split('/')
            brand_name = vc_dict[arrs[-1]] if arrs[-1] in vc_dict else '*'
            if '*' == brand_name:
                arrs1 = arrs[-1].split('_')
                if len(arrs1) > 1:
                    brand_name = arrs1[0]
            print('domestic: {0} => {1};'.format(arrs[-1], brand_name))
            if brand_name in uncovered_brand_names:
                uncovered_brand_names.remove(brand_name)
        print('uncoder brand number: {0};'.format(len(uncovered_brand_names)))
        for bn in uncovered_brand_names:
            print(bn)

    @staticmethod
    def process_imported_vehicles():
        # 列出所有
        srcPath = Path('/media/zjkj/35196947-b671-441e-9631-6245942d671b/vehicle_type_v2d/vehicle_type_v2d')
        imported_brands_items = [x for x in srcPath.iterdir() if srcPath.is_dir()]
        imported_brands_nos = []
        for item in imported_brands_items:
            item_str = str(item)
            arrs = item_str.split('_')
            arrs1 = arrs[-2].split('/')
            brand_no = arrs1[1]
            imported_brands_nos.append(brand_no)
            brand_name = arrs[-1]
            print('{0}: {1};'.format(brand_no, brand_name))
        # 从车型中去除进口车数据集中的品牌
        uncovered_brand_nos = []
        uncovered_brand_names = []
        for k in VaoTest.vehicle_brands.keys():
            if not (k in imported_brands_nos):
                uncovered_brand_nos.append(k)
                uncovered_brand_names.append(VaoTest.vehicle_brands[k])
        return uncovered_brand_nos, uncovered_brand_names
        # 从车型库中去除国产车数据集中的品牌
        # 打印未覆盖的车型


    @staticmethod
    def get_uncovered_vcs():
        our_vehicle_code_set = set()
        uncovered_vcs = []
        # 统计出已经处理完成的品牌
        train_ds = './yt_train.txt'
        with open(train_ds, 'r', encoding='utf-8') as fd:
            for line in fd:
                #print('正在处理：{0};'.format(line))
                arrs = line.split('*')
                class_id = arrs[1]
                our_vehicle_code_set.add(class_id[:-1])
        print('已经处理品牌数：{0}/180;'.format(len(our_vehicle_code_set)))
        for vc in our_vehicle_code_set:
            print(vc)
        # 找出未处理的品牌
        for k in VaoTest.vehicle_brands.keys():
            item = str(int(k)-1)
            rst = not (item in our_vehicle_code_set)
            print('{0} [{2}] => {1};'.format(k, rst, item))
            if rst:
                uncovered_vcs.append(k)
        print('未处理品牌数：{0};'.format(len(uncovered_vcs)))
        for vc in uncovered_vcs:
            print('##### {0};'.format(vc))

    @staticmethod
    def get_unknown_vcs():
        '''
        从./vehicle_codes.txt中挑出在./datasets/raw_domestic_brands.txt没有的编号
        '''
        # 获取所有的车辆编号
        all_vcs = set()
        with open('./vehicle_codes.txt', 'r', encoding='utf-8') as all_fd:
            for line in all_fd:
                all_vcs.add(line)
        print('所有车辆编号数量：{0};'.format(len(all_vcs)))
        # 获取已处理车辆编号
        known_vcs = set()
        with open('./datasets/raw_domestic_brands.txt', 'r', encoding='utf-8') as known_fd:
            for line in known_fd:
                arrs = line.split('*')
                known_vcs.add(arrs[0])
        print('已处理车辆编号数量：{0};'.format(len(known_vcs)))
        # 找出未处理车辆编号
        sum = 1
        for vc in known_vcs:
            if vc in all_vcs:
                all_vcs.remove(vc)
                print('从总体中删除：{0}; 共删除：{1};'.format(vc, sum))
                sum += 1
        with open('./unknown_vcs.txt', 'a+', encoding='utf-8') as unknown_fd:
            for vc in all_vcs:
                unknown_fd.write('{0}'.format(vc))

    @staticmethod
    def known_vcs_ds_main():
        #VaoTest.create_known_vcs_ds('./yt_train.txt', './datasets/CUB_200_2011/anno/yt_train_92.txt') # 生成训练数据集
        #VaoTest.create_known_vcs_ds('./datasets/CUB_200_2011/anno/yt_test.txt', './datasets/CUB_200_2011/anno/yt_test_92.txt')
        # 从0~91重复：找出yt_train_92.txt中所有符合条件的记录，
        # 从中随机取出1000张图片，形成yt_train_92_1000.txt作
        # 为训练数据集
        VaoTest.create_92_random_1000_ds()

    @staticmethod
    def create_known_vcs_ds(raw_file, ds_file):
        # 生成92类现有类别到0~91的对应表
        class_id = 0
        ob_nb_dict = {}
        nd_dict = {}
        with open('./work/known_brands.txt', 'r', encoding='utf-8') as kb_fd:
            for line in kb_fd:
                print('{0}=>{1};'.format(line[:-1], class_id))
                #on_fd.write('{0}={1}\n'.format(line[:-1], class_id))
                ob_nb_dict[line[:-1]] = class_id
                key = '{0:03d}'.format(int(line[:-1])+1)
                brand_name = VaoTest.vehicle_brands[key]
                #nd_fd.write('{0}={1}\n'.format('{0}'.format(class_id), brand_name))
                nd_dict['{0}'.format(class_id)] = brand_name
                class_id += 1
        sum = 0
        with open(raw_file, 'r', encoding='utf-8') as raw_fd:
            with open(ds_file, 'w+', encoding='utf-8') as ds92_fd:
                for line in raw_fd:
                    arrs = line.split('*')
                    img_file = arrs[0]
                    old_class_id = arrs[1][:-1]
                    if old_class_id in ob_nb_dict:
                        new_class_id = ob_nb_dict[old_class_id]
                        ds92_fd.write('{0}*{1}\n'.format(img_file, '{0}'.format(new_class_id)))
                        sum += 1
        print('共有{0}个样本！'.format(sum))

    @staticmethod
    def create_92_random_1000_ds():
        '''
        从0~91重复：找出yt_train_92.txt中所有符合条件的记录，
        从中随机取出1000张图片，形成yt_train_92_1000.txt
        作为训练数据集
        '''
        c_nums = {}
        fcid_set = set()
        for cid in range(92):
            c_nums['{0}'.format(cid)] = 0
        for cid in range(1):
            print('process {0} brand...'.format(cid))
            with open('./datasets/CUB_200_2011/anno/yt_train_92.txt', 'r', encoding='utf-8') as raw_fd:
                for line in raw_fd:
                    arrs = line.split('*')
                    fcid = int(arrs[1])
                    fcid_set.add(fcid)
                    img_file = arrs[0]
                    #print('fcid: {0};'.format(fcid))
                    if cid == fcid:
                        c_nums['{0}'.format(cid)] += 1
                        #print('get correct data')
        for k, v in c_nums:
            print('{0}: {1};'.format(k, v))
        print('fcid_set: {0};'.format(fcid_set))
        

    
    v_no_bn = {}
    v_bn_no = {}
    v_no_set = set()
    v_bn_set = set()
    vehicle_brands = {
        '001':'奥迪',
        '002':'阿尔法罗密欧',
        '003':'阿斯顿马丁',
        '004':'奔驰',
        '005':'宝马',
        '006':'宾利',
        '007':'布嘉迪',
        '008':'保时捷',
        '009':'别克',
        '010':'本田',
        '011':'标致',
        '012':'比亚迪',
        '013':'北京汽车',
        '014':'宝骏',
        '015':'宝腾',
        '016':'长城',
        '017':'长安',
        '018':'长丰',
        '019':'昌河',
        '020':'川汽野马',
        '021':'东风',
        '022':'风神',
        '023':'大发',
        '024':'帝豪',
        '025':'东南',
        '026':'道奇',
        '027':'大众',
        '028':'大宇',
        '029':'大迪',
        '030':'法拉利',
        '031':'丰田',
        '032':'皇冠',
        '033':'福特',
        '034':'野马',
        '035':'菲亚特',
        '036':'福田',
        '037':'福迪',
        '038':'富奇',
        '039':'广汽',
        '040':'GMC',
        '041':'光冈',
        '042':'海马',
        '043':'哈飞',
        '044':'悍马',
        '045':'霍顿',
        '046':'华普',
        '047':'华泰',
        '048':'红旗',
        '049':'黄海',
        '050':'汇众',
        '051':'捷豹',
        '052':'吉普',
        '053':'金杯',
        '054':'江淮',
        '055':'吉利',
        '056':'江铃',
        '057':'江南',
        '058':'吉奥',
        '059':'解放',
        '060':'九龙',
        '061':'金龙',
        '062':'凯迪拉克',
        '063':'克莱斯勒',
        '064':'柯尼塞格',
        '065':'开瑞',
        '066':'KTM',
        '067':'克尔维特',
        '068':'兰博基尼',
        '069':'劳斯莱斯',
        '070':'路虎',
        '071':'莲花',
        '072':'林肯',
        '073':'雷克萨斯',
        '074':'铃木',
        '075':'雷诺',
        '076':'力帆',
        '077':'陆风',
        '078':'理念',
        '079':'迈巴赫',
        '080':'名爵',
        '081':'迷你',
        '082':'玛莎拉蒂',
        '083':'马自达',
        '084':'纳智捷',
        '085':'南汽',
        '086':'欧宝',
        '087':'讴歌',
        '088':'奥兹莫比尔',
        '089':'帕加尼',
        '090':'庞蒂克',
        '091':'奇瑞',
        '092':'起亚',
        '093':'全球鹰',
        '094':'庆铃',
        '095':'启辰',
        '096':'尼桑',
        '097':'瑞麒',
        '098':'荣威',
        '099':'罗森',
        '100':'罗孚',
        '101':'萨博',
        '102':'斯巴鲁',
        '103':'双环',
        '104':'世爵',
        '105':'斯派朗',
        '106':'三菱',
        '107':'双龙',
        '108':'smart',
        '109':'斯柯达',
        '110':'塔塔',
        '111':'土星',
        '112':'沃尔沃',
        '113':'威麟',
        '114':'五菱',
        '115':'威兹曼',
        '116':'沃克斯豪尔',
        '117':'五十铃',
        '118':'现代',
        '119':'雪佛兰',
        '120':'夏利',
        '121':'雪铁龙',
        '122':'西亚特',
        '123':'英菲尼迪',
        '124':'英伦',
        '125':'一汽',
        '126':'奔腾',
        '127':'跃进',
        '128':'依维柯',
        '129':'永源',
        '130':'中华',
        '131':'众泰',
        '132':'中兴',
        '133':'中顺',
        '134':'华阳',
        '135':'飞虎',
        '136':'北汽幻速',
        '137':'北汽威望',
        '138':'北汽制造',
        '139':'安凯',
        '140':'北奔重汽',
        '141':'大运',
        '142':'东沃（优迪）',
        '143':'福达',
        '144':'日野',
        '145':'海格',
        '146':'红岩',
        '147':'华菱星马（CAMC）',
        '148':'金旅',
        '149':'凯马',
        '150':'东风柳汽',
        '151':'青年',
        '152':'三环',
        '153':'三一重工',
        '154':'陕汽重卡',
        '155':'少林',
        '156':'申龙',
        '157':'时代（福田）',
        '158':'时风',
        '159':'斯堪尼亚（SCANIA）',
        '160':'唐骏',
        '161':'五征',
        '162':'徐工汽车',
        '163':'亚星',
        '164':'英田',
        '165':'宇通',
        '166':'中国重汽',
        '167':'重汽王牌',
        '168':'中通',
        '169':'观致',
        '170':'特斯拉',
        '171':'福汽启腾',
        '172':'哈弗',
        '173':'江铃轻汽',
        '174':'江铃驭胜',
        '175':'上汽大通',
        '176':'思铭',
        '177':'腾势',
        '178':'英致',
        '179':'凯翼',
        '180':'北方客车'
        }













