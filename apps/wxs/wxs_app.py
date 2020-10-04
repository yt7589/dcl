#
import sys
from apps.wxs.wxs_dsm import WxsDsm
from apps.wxs.controller.c_brand import CBrand
from apps.wxs.controller.c_model import CModel
from apps.wxs.wxs_utils import WxsUtils

class WxsApp(object):
    RM_EXP = 1000
    RM_GENERATE_SAMPLES = 1001
    RM_GENERATE_DATASET = 1002
    RM_GET_SIMPLIFIED_BMYS = 1003
    RM_CONVERT_TO_BRAND_DS_MAIN = 1004
    RM_BIND_BRAND_HEAD_BMY_HEAD = 1005
    RM_GENERATE_ZJKJ_CAMBRICON_LABELS = 1006
    RM_CHECK_WXS0901_MISSING_VINS = 1007
    RM_NORM_FILES_FOLDER = 1008
    RM_GENERATE_ERROR_SAMPLES_HTML = 1009
    RM_REFINE_PREV_DATASET = 1010
    RM_PROCESS_VD_JSONS = 1011
    RM_CHECK_GCC2N_VIN_CODES = 1012
    RM_LMDB_DEMO = 1013
    RM_SAVE_DS_IMGS_TO_LMDB = 1015
    RM_GET_FILES_IN_SUBFOLDERS_DICT = 1016
    RM_RUN_VD_CUT_SAVE = 1017
    RM_DELETE_ERROR_SAMPLES = 1018
    RM_DELETE_WXS_ERROR_20200919 = 1019
    RM_ADD_OLD_WXS_BRAND_DS = 1020
    RM_GENERATE_FDS_SAMPLES = 1021
    RM_RUN_VD_CUT_SAVE_ON_ES20200923 = 1022
    RM_GENERATE_RAW_DS_ESC20200923 = 1023
    RM_GET_VD_FAILED_IMAGES = 1024
    RM_COPY_FINE_RANDOM_IMAGES = 1025
    RM_ADD_ERR_IMGS_TO_RDS = 1026
    RM_PURIFY_FULL_DS_MAIN = 1027
    RM_COMPARE_NEW_WXS_BID_EXCEL = 1028
    RM_VD_CUT_BY_FOLDER = 1029
    RM_REFINE_FDS_TEST_DS = 1030
    RM_WXS_BID_DS_MAIN = 1031
    RM_RUN_VD_CUT_SAVE_ON_WXS_DS = 1032
    RM_FORM_WXS_BID_TEST_RST = 1033
    RM_GET_WXS_BID_DS_SCORES = 1034
    RM_GENERATE_ZJKJ_FDS_LABELS = 1035
    RM_GENERATE_SAMPLES_WXS_BID_DS = 1036

    def __init__(self):
        self.name = 'apps.wxs.WxsApp'

    def startup(self, args):
        print('2020年7月无锡所招标应用')
        mode = WxsApp.RM_FORM_WXS_BID_TEST_RST
        if WxsApp.RM_GENERATE_SAMPLES == mode:
            ''' 
            从fgvc_dataset/raw和guochanchezuowan_all目录生成样本列表
            '''
            print('generate samples')
            #WxsDsm.generate_samples()
            WxsDsm.generate_samples_wxs0901()
            #WxsDsm.correct_vin_bmy_codes_error()
            #WxsDsm.process_es0901_jsons()
        elif WxsApp.RM_GENERATE_DATASET == mode:
            '''
            生成原始数据集，采用稀疏品牌车型年款编号
            '''
            WxsDsm.generate_dataset()
        elif WxsApp.RM_GET_SIMPLIFIED_BMYS == mode:
            '''
            将品牌车型年款变为0开始递增的序号
            '''
            WxsDsm.get_simplified_bmys()
        elif WxsApp.RM_CONVERT_TO_BRAND_DS_MAIN == mode:
            '''
            向数据集中加入品牌信息
            '''
            WxsDsm.convert_to_brand_ds_main()
        elif WxsApp.RM_BIND_BRAND_HEAD_BMY_HEAD == mode:
            '''
            实现先预测出品牌类别，然后从年款头中除该品牌对应的年款索引外的其他
            类别全部清零，将年款头的内容输出作为输出
            '''
            WxsDsm.bind_brand_head_bmy_head()
        elif WxsApp.RM_GENERATE_ZJKJ_CAMBRICON_LABELS == mode:
            '''
            生成Cnstream要求的车辆品牌车型年款标签文件，格式为：
            {"品牌编号", "车型编号", "年款编号", "品牌_车型_年款"},{...},
            {...}
            '''
            WxsDsm.generate_zjkj_cambricon_labels()
        elif WxsApp.RM_CHECK_WXS0901_MISSING_VINS == mode:
            '''
            根据fgvc_dataset/raw和guochanchezuowan_all目录，生成车辆识别码图片数量字典；
            根据bid_brand_train_ds_090501.txt生成车辆识别码图片数量字典；
            统计出在文件系统中车辆识别码图片数量为0但是在数据集中该车辆识别码图片数量
            不为零的车辆识别码、品牌、品牌车型年款列表
            '''
            WxsDsm.check_wxs0901_missing_vins()
        elif WxsApp.RM_NORM_FILES_FOLDER == mode:
            '''
            将大量图片文件统一存储为每个子目录存储100个文件，目录层次为：
            d00/d00/d00/d00/d00/**.jpg
            其中最后一个目录存文件
            '''
            WxsDsm.norm_files_folder()
        elif WxsApp.RM_GENERATE_ERROR_SAMPLES_HTML == mode:
            '''
            根据错误分类样本列表，形成便于人工浏览的网页，保存于../../w1/es目录下
            '''
            WxsDsm.generate_error_samples_html()
        elif WxsApp.RM_REFINE_PREV_DATASET == mode:
            '''
            生成原来的训练集和只包括所里出错的3153张图的训练集
            '''
            WxsDsm.refine_prev_dataset()
        elif WxsApp.RM_PROCESS_VD_JSONS == mode:
            '''
            处理车辆检测Json文件
            '''
            print('?????')
            WxsDsm.process_vd_jsons()
        elif WxsApp.RM_CHECK_GCC2N_VIN_CODES == mode:
            '''
            确认guochanche_2n目录下的车辆识别码都不在所里列表中
            '''
            WxsDsm.check_gcc2n_vin_codes()
        elif WxsApp.RM_LMDB_DEMO == mode:
            '''
            学习LMDB的使用方法
            '''
            WxsDsm.run_lmdb_demo()
        elif WxsApp.RM_SAVE_DS_IMGS_TO_LMDB == mode:
            '''
            将所有数据集文件（通过遍历特定格式文件夹）保存到LMDB中
            '''
            WxsDsm.save_ds_imgs_to_lmdb()
        elif WxsApp.RM_GET_FILES_IN_SUBFOLDERS_DICT == mode:
            '''
            获取指定目录及其下所有子目录下文件名与全路径文件名字典
            '''
            WxsDsm.get_files_in_subfolders_dict()
        elif WxsApp.RM_RUN_VD_CUT_SAVE == mode:
            '''
            从图像目录中取出所有图片文件列表，将其分为20份，存为20个文本文件，
            启动20个线程，每个线程处理一个文本文件：
            1. 向服务器端发送请求，获取车辆检测结果；
            2. 解析Json文件，得到车辆检测框；
            3. 对原图进行切图，并缩放为224*224；
            4. 求出图片树形目录位置；
            5. 将文件保存到该目录下；
            '''
            WxsDsm.run_vd_cut_save()
        elif WxsApp.RM_DELETE_ERROR_SAMPLES == mode:
            '''
            从数据集中删除品牌分类错误的样本
            '''
            WxsDsm.delete_error_samples_main()
        elif WxsApp.RM_DELETE_WXS_ERROR_20200919 == mode:
            '''
            从9月1日错误集中删除品牌错误和车尾的数据
            '''
            WxsDsm.delete_wxs_error_20200919()
        elif WxsApp.RM_ADD_OLD_WXS_BRAND_DS == mode:
            '''
            '''
            WxsDsm.add_old_wxs_brand_ds()
        elif WxsApp.RM_GENERATE_FDS_SAMPLES == mode:
            '''
            生成全量数据集样本集，作为训练样本集
            '''
            WxsDsm.generate_fds_samples()
        elif WxsApp.RM_RUN_VD_CUT_SAVE_ON_ES20200923 == mode:
            '''
            将所里9月23日错误的品牌图片，调用切图程序进行切图
            '''
            WxsDsm.run_vd_cut_save_on_es20200923()
        elif WxsApp.RM_GENERATE_RAW_DS_ESC20200923 == mode:
            '''
            生成2020-09-23无锡所测试错误图片切图后图片的原始数据集
            '''
            WxsDsm.generate_raw_ds_esc20200923()
        elif WxsApp.RM_GET_VD_FAILED_IMAGES == mode:
            '''
            找出无锡所9月23日品牌错误图片中未检测出图片列表
            '''
            WxsDsm.get_vd_failed_images()
        elif WxsApp.RM_COPY_FINE_RANDOM_IMAGES == mode:
            '''
            将随机抽取的切图图片替换为质量更高的切图图片，即更新train_ds目录
            下的图片内容
            '''
            WxsDsm.copy_fine_random_images()
        elif WxsApp.RM_ADD_ERR_IMGS_TO_RDS == mode:
            '''
            将文件名与图片内容不符的样本加入到训练集中
            '''
            WxsDsm.add_err_imgs_to_rds()
        elif WxsApp.RM_PURIFY_FULL_DS_MAIN == mode:
            '''
            清理全量数据集
            '''
            WxsDsm.purify_full_ds_main()
        elif WxsApp.RM_COMPARE_NEW_WXS_BID_EXCEL == mode:
            '''
            检查所里9月18日发布的文件中，仅有2200个品牌车型的代码，与之前版本的
            品牌车型年款代码进行比较，看看是否一致
            '''
            WxsDsm.compare_new_wxs_bid_excel()
        elif WxsApp.RM_VD_CUT_BY_FOLDER == mode:
            '''
            对指定目录文件进行切图，以树形目录方式存到目录中，将切图失败的图片
            保存到另外的目录
            '''
            WxsDsm.vd_cut_by_folder()
        elif WxsApp.RM_REFINE_FDS_TEST_DS == mode:
            '''
            将全量数据集测试集图片换为切过的图片
            '''
            WxsDsm.refine_fds_test_ds()
        elif WxsApp.RM_WXS_BID_DS_MAIN == mode:
            '''
            无锡所招标品牌测试集
            '''
            WxsDsm.wxs_bid_ds_main()
        elif WxsApp.RM_RUN_VD_CUT_SAVE_ON_WXS_DS == mode:
            '''
            将所里9月23日错误的品牌图片，调用切图程序进行切图
            '''
            WxsDsm.run_vd_cut_save_on_wxs_ds()
        elif WxsApp.RM_FORM_WXS_BID_TEST_RST == mode:
            '''
            生成无锡所数据集测试正确结果，格式为：文件名*品牌代码*车型代码，用于
            计算在Pipeline中的品牌精度和车型精度
            '''
            WxsDsm.form_wxs_bid_test_rst()
        elif WxsApp.RM_GET_WXS_BID_DS_SCORES == mode:
            '''
            获取无锡所数据集品牌精度和车型精度
            '''
            WxsDsm.get_wxs_bid_ds_scores()
        elif WxsApp.RM_GENERATE_ZJKJ_FDS_LABELS == mode:
            '''
            生成全量数据集标签
            '''
            WxsDsm.generate_zjkj_fds_labels()
        elif WxsApp.RM_GENERATE_SAMPLES_WXS_BID_DS == mode:
            '''
            生成无锡所招标测试数据集
            '''
            WxsDsm.generate_samples_wxs_bid_ds()
        else:
            WxsDsm.exp001()
            
            
            
        '''
        利用所里最新Excel表格内容为主，work/ggh2_to_bmy_dict.txt内容为辅，
        生成数据库中t_brand、t_model、t_bmy、t_vin表格中内容
        '''
        #WxsDsm.initialize_db()
        '''
        向数据集中加入品牌信息
        '''
        #WxsDsm.convert_to_brand_ds_main()
        '''
        找出损坏的图片文件
        '''
        #WxsDsm.find_bad_images()
        #WxsDsm.report_current_status()
        #WxsDsm.exp001()
        #WxsDsm.get_fine_wxs_dataset()
        #WxsDsm.generate_wxs_bmy_csv()
        ''' 根据正确的测试集图片文件名，查出当前的品牌车型年款编号，没有的用-1表示，形成CSV文件 '''
        #WxsDsm.generate_test_ds_bmy_csv()
        ''' 生成Pipeline测试评价数据，将测试集中的图片文件拷贝到指定目录下 '''
        #WxsDsm.copy_test_ds_images_for_cnstream()
        '''
        集成无锡所测试集数据
        '''
        #WxsDsm.integrate_wxs_test_ds()
        '''
        生成车辆识别码和品牌车型年款对应关系表，用于修正所里品牌车型年款不合理的地方。2020.07.25
        '''
        #WxsDsm.generate_vin_bmy_csv()
        '''
        处理所里测试集中5664张正确图片中新车型和新年款记录，添加到数据集中
        '''
        #WxsDsm.process_unknown_wxs_tds()
        '''
        将子品牌合并为主品牌
        '''
        #WxsDsm.fix_test_ds_brand_errors()
        '''
        获取所里最新Excel表格中的品牌车型年款
        '''
        #WxsDsm.get_wxs_bmys()
        '''
        从t_vin表中获取当前不在所里5731个品牌车型年款中的车辆识别码
        '''
        #WxsDsm.get_non_wxs_vins()
        '''
        将无锡所测试文件放到Excel表格中
        '''
        #WxsDsm.generate_wxs_tds_table()
        '''
        求出无锡所Excel表格中每个车辆识别码的图片数，并列出图片数为零的
        车辆识别码编号
        '''
        #WxsDsm.get_wxs_vin_id_img_num()
        '''
        获取无锡所品牌列表
        '''
        #WxsDsm.get_wxs_brands()
        '''
        从samples.txt文件中统计出品牌数、车型数、年款数
        '''
        #WxsDsm.get_brand_bm_bmy_of_samples()
        '''
        求出无锡所测试集品牌与当前涉及的171个品牌的不同
        '''
        #WxsDsm.get_diff_wxs_tds_brands()
        '''
        标出无锡所测试集中可能出错的图片
        '''
        #WxsDsm.mark_error_img_in_wxs_tds()
        '''
        根据Csv文件生成测试数据集，其中年款值为一个不正确的值，因此年款精度为0，
        只测品牌精度
        '''
        #WxsDsm.generate_wxs_test_dataset()
        '''
        绘制在测试集上的精度变化曲线
        '''
        #WxsUtils.draw_tds_acc_curve()
        '''
        对所里测试集图片经过检测后的JSON文件进行解析，对原始图片进行切图，放
        到指定的目录下，子目录按00/00组织
        '''
        #WxsDsm.process_detect_jsons()
        #WxsDsm.merge_zhangcan_csv()
        '''
        将所里测试集中文件替换为切图后文件
        '''
        #WxsDsm.generate_cut_img_test_ds()
        '''
        将训练集或随机抽取测试集图片拷贝到单独目录下，便于调用切图软件
        '''
        #WxsDsm.cut_dataset_imgs()
        '''
        获取193万张原始图片和183万张检测图片之间，没有处理的图片列表，
        供后续查找原因
        '''
        #WxsDsm.find_diff_of_193_183()
        '''
        将经过检测的训练集图片对应的JSON文件进行解析，对原始图片进行切图，放
        到指定的目录下，按照车辆识别码作为目录进行组织
        '''
        #WxsDsm.process_training_ds_detect_jsons()
        '''
        将检测失败的图片文件拷贝到指定目录下，供耀辉改进其检测算法
        '''
        #WxsDsm.copy_detect_bad_image_files()
        '''
        生成由切过的图组成的数据集
        '''
        #WxsDsm.generate_cutted_dataset()
        '''
        生成由所里品牌测试切过的图组成的数据集
        '''
        #WxsDsm.generate_wxs_tds_cutted_dataset()
        '''
        通过随机采样生成int8量化需要的图片，以品牌为控制单位：
        当品牌下文件小于4个时，全部取；
        当大于4个时，随机从中取其中4张
        '''
        #WxsDsm.generate_int8_quant_imgs_by_brand()
        '''
        获取指定品牌在训练数据集和所里测试数据集中的图片，便于对比品牌预测
        错误图片的原因
        '''
        #WxsDsm.get_brand_images_main()
        '''
        获取bmy_id（年款头输出）与车型值对象的字典
        '''
        #WxsDsm.get_bmy_id_bm_vo_dict()
        '''
        将无锡所测试集年款信息添加到训练集中进行训练
        '''
        #WxsDsm.integrate_wxs_tds_bmy()
        '''
        生成由切图过图像和所里切图过测试集合并在一起的原始数据（bmyId为原始值）
        '''
        #WxsDsm.generate_cut_wxs_tds_merged_raw_ds()
        '''
        找出无锡所测试集中需要标注年款的记录
        '''
        #WxsDsm.get_to_anno_wxs_tds()
        '''
        获取无锡所品牌车型年款Excel中每个年款5张示例图片，并以品牌车型年款
        目录结构进行组织
        '''
        #WxsDsm.get_bmy_example_images()
        '''
        修改数据集中人工标注错误，主要是无锡所测试集中有405条记录需要修改，
        其他记录原样复制
        '''
        #WxsDsm.rectify_raw_bid_train_ds()
        '''
        将模型预测错误的样本整理为图片文件名, 标注年款, 预测年款 格式，发
        给张灿，由其进行处理。如果模型预测错误，则标注模型错误，如果标注
        错误则写上新年款编号 2020-08-16
        '''
        #WxsDsm.process_error_sample_for_zhangcan()
        '''
        处理20200817经过修正后的分类错误的样本数据
        '''
        #WxsDsm.process_error_sample_20200817()
        '''
        计算两张图片的相似度，通过获取两张图片ReID特征向量，然后计算二者的余弦
        距离作为相似度
        '''
        #WxsDsm.calculate_img_similarity_main()
        '''
        定期从车辆检测结果目录下读取检测的Json文件，将其存储到每100个文件一个目录的格式，
        并且保存原有的文件名。
        '''
        #WxsDsm.move_detect_json_to_folder()
        '''
        将1400万样本分割为10万一个单位，并保存为独立的文件: 
        samples_001.txt ~ sample_140.txt
        '''
        #WxsDsm.divide_samples()
        '''
        将10万个样本文件中的图片拷贝到client1.8/work/sample_files目录下，并按每100个
        文件一个目录组织
        '''
        #WxsDsm.copy_sample_image_files_main()
        '''
        根据车辆检测结果Json文件，对原始图像进行切图，并保存到指定目录下
        '''
        #WxsDsm.process_seg_ds_detect_jsons()
        '''
        将切图错误的图片原图和切过的图拷贝到同一目录下
        '''
        #WxsDsm.copy_cut_bad_images_pair()
        '''
        将无锡所测试集中标注出年款的图片文件名和年款编号写进文件文件中
        '''
        #WxsDsm.generate_txt_by_wxs_tds_ok_images()
        '''
        修改原始数据集中标注错误的样本，增加出错样本个数，生成新的数据集
        '''
        #WxsDsm.correct_augment_raw_ds()
        '''
        复制指定份数必错的样本，希望能够不再出错
        '''
        #WxsDsm.duplicate_miss_samples()
        '''
        求出无锡所测试集图片与品牌名称对应关系
        '''
        #WxsDsm.wxs_tds_to_image_brand()
        #WxsDsm.exp001()


        

    def exp(self):
        ds_file = './datasets/CUB_200_2011/anno/train_ds_v4.txt'
        train_id = self._t1(ds_file)
        ds_file = './datasets/CUB_200_2011/anno/test_ds_v4.txt'
        test_id = self._t1(ds_file)
        print('{0} vs {1};'.format(train_id, test_id))

    def _t1(self, ds_file):
        max_fgvc_id = 0
        with open(ds_file, 'r', encoding='utf-8') as nfd:
            for line in nfd:
                row = line.strip()
                arrs0 = row.split('*')
                fgvc_id = int(arrs0[1])
                if fgvc_id > max_fgvc_id:
                    max_fgvc_id = fgvc_id
        return max_fgvc_id
