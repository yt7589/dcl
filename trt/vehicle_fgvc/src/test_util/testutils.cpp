//
// Created by cao on 19-10-26.
//

#include "testutils.hpp"
#include <string>
#include <iostream>
#include <memory>
#include <vector>
#include <dirent.h>

#include "opencv2/opencv.hpp"

#include "predictor_api.hpp"



using namespace std;
void SaveEngine(const std::string& fileName, std::vector<char> &data) {
    if(fileName.empty() || data.empty()) {
        return;
    }
	std::ofstream file;
	file.open(fileName,std::ios::binary | std::ios::out);
	if(!file.is_open()) {
		return;
	}
	file.write((const char*)data.data(), data.size());
	file.close();

}

#include <sys/time.h>

double getTime() {
    struct timeval t;
    gettimeofday(&t, nullptr);
    double r;
    r = 1000.0 * t.tv_sec + t.tv_usec / 1000.0;
    return r;
}
std::vector<cv::Mat> getinputimg()
{
	std::string pathstr = "test_pic/fight_fi033/";
	std::vector<std::string> paths;
	for (int i = 1; i <17; ++i)
	{
		if (i<10)
			paths.push_back(pathstr+"image_0000"+std::to_string(i)+".jpg");
		
		else
			paths.push_back(pathstr+"image_000"+std::to_string(i)+".jpg");
	}

	std::vector<cv::Mat> inputs;
	for (int i = 0; i < 1; ++i)
	{
		
		auto img = cv::imread(paths[i]);
		cv::Mat resized;
		cv::resize(img, resized, cv::Size(224,224),0,0);
	
		inputs.push_back(resized);
	}
	
	return inputs;
}



std::string loadfile(const string& file){

		ifstream in(file, ios::in | ios::binary);
		if (!in.is_open())
			return "";

		in.seekg(0, ios::end);
		size_t length = in.tellg();

		string data;
		if (length > 0){
			in.seekg(0, ios::beg);
			data.resize(length);

			in.read(&data[0], length);
		}
		in.close();
		return data;
	}

    vector<string> loadList(const string& listfile){

		std::vector<string> lines;
		std::string data = loadfile(listfile);

		if (data.empty())
			return lines;

		char* ptr = (char*)&data[0];
		char* prev = ptr;
		string line;

		while (true){
			if (*ptr == '\n' || *ptr == 0){
				int length = ptr - prev;

				if (length > 0){
					if (length == 1){
						if (*prev == '\r')
							length = 0;
					}
					else {
						if (prev[length - 1] == '\r')
							length--;
					}
				}

				if (length > 0){
					line.assign(prev, length);
					lines.push_back(line);
				}

				if (*ptr == 0)
					break;

				prev = ptr + 1;
			}
			ptr++;
		}
		return lines;
	}



void getFileList(std::string dirPath, std::vector<std::string>& pathList)
{
    DIR *dp;
    struct dirent *dirp;

    if((dp = opendir(dirPath.c_str())) == NULL)
    {
        std::cout << "Can't open " << dirPath << std::endl;
    }

    while((dirp = readdir(dp)) != NULL)
    {
        if(dirp->d_name[0] == '.')
            continue;
        pathList.push_back(dirp->d_name);
        //std::cout << dirp->d_name << std::endl;
    }
    closedir(dp);
}