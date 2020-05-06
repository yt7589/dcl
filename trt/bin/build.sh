OPENCV_HOME=/home/zjkj/opencv347/opencv/opencv-3.4.7
TENSORRT_HOME=/home/zjkj/working_zjw/onnx--prog/TensorRT-6.0.1.5
CUDA_HOME=/usr/local/cuda-10.1

# 编译文件
# 需要在CarType.cpp中修改标定数据集位置
/usr/local/cuda-10.1/bin/nvcc -c ../src/nv_crop_and_resize_and_norm.cu ../src/sample_options.cpp ../src/logger.cpp ../src/sample_engines.cpp ../src/entropy_calibrator.cpp ../src/onnx_trt.cpp ../src/fgvc.cpp -I ./include -I $CUDA_HOME/include -I $TENSORRT_HOME/include -I $OPENCV_HOME/modules/core/include -std=c++14 -g --compiler-options '-fPIC'
# 连接所有文件并输出so
export LD_LIBRARY_PATH=/home/zjkj/working_zjw/novio/
gcc --shared -fPIC sample_options.o logger.o sample_engines.o entropy_calibrator.o nv_crop_and_resize_and_norm.o onnx_trt.o fgvc.o -o libfgvc.so -L$TENSORRT_HOME/lib  -L$CUDA_HOME/lib -L$CUDA_HOME/lib64 -Wl,-rpath,$TENSORRT_HOME/lib:$CUDA_HOME/lib:$CUDA_HOME/lib64:$OPENCV_HOME/build/lib $OPENCV_HOME/build/lib/libopencv_features2d.so $OPENCV_HOME/build/lib/libopencv_highgui.so  $OPENCV_HOME/build/lib/libopencv_imgcodecs.so $OPENCV_HOME/build/lib/libopencv_imgproc.so $OPENCV_HOME/build/lib/libopencv_core.so -ldl -lnvinfer -lnvinfer_plugin -lnvonnxparser -lnvonnxparser_runtime -lnvparsers -lcudart -lm -lstdc++ -lpthread -lnppidei
# 编译测试用主函数
gcc ../test_main.cpp -o test_main -I./include -I $CUDA_HOME/include -I $OPENCV_HOME/modules/core/include  -I $OPENCV_HOME/modules/imgproc/include -I $OPENCV_HOME/modules/imgcodecs/include -I $TENSORRT_HOME/include -L./ -lfgvc -lpthread -L$OPENCV_HOME/build/lib -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -L$CUDA_HOME/lib -L$CUDA_HOME/lib64 -lcudart -lstdc++ -std=c++14 -lm

