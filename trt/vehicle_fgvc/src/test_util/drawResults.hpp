#ifndef ALG_DRAWRESULTS_HPP_
#define ALG_DRAWRESULTS_HPP_


// c++
#include <string>
#include <vector>

namespace cv
{
    class Mat;
}

// alg
#include <pedestrianinfo.hpp>

class resultsDrawer{
  public:
    resultsDrawer(){};
    ~resultsDrawer(){};
    
    void showResults(cv::Mat& img, const std::vector<RectAction>& headInfoList, std::vector<PersonInfo>& personInfoList);

  private:
    void drawLine(cv::Mat& img, PersonInfo sinManInfo, int indexOne, int indexTwo, cv::Scalar color);
    void drawSinKeyPoint(cv::Mat& img,const  std::vector<PersonInfo> sinManInfoList);
    void showHeadResult(cv::Mat& img, const std::vector<RectAction>& headInfoList);
    void showBodyRect(cv::Mat& img, std::vector<PersonInfo>& personInfoList);
    void showKpHandRect(cv::Mat& img, const PersonInfo& sinManInfo);
    void showHandRect(cv::Mat& img, const PersonInfo& sinManInfo);
    void showAssembleResults(cv::Mat& img, const std::vector<PersonInfo>& personInfoList);
    void showFightResult(cv::Mat& img, const std::vector<PersonInfo>& personInfoList);

};

#endif  // ALG_DRAWRESULTS_HPP_
