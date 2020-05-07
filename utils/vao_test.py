#
# 车管所测试工具类
#
from pathlib import Path

class VaoTest(object):
    def __init__(self):
        self.name = 'util.VaoTest'

    @staticmethod
    def process_vehicles():
        _, uncovered_brand_names = VaoTest.process_imported_vehicles()
        #uncovered_brand_names = []
        for bn in VaoTest.vehicle_brands.values():
            uncovered_brand_names.append(bn)
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













