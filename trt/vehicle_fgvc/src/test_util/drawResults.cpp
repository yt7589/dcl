// alg
//#include <common.hpp>
// opencv
//#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
//#include <opencv2/dnn.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <drawResults.hpp>
//#include <algorithm>
#include <pedestrianinfo.hpp>

 void resultsDrawer::drawLine(cv::Mat& img, PersonInfo sinManInfo, int indexOne, int indexTwo, cv::Scalar color = CV_RGB(255,255,255))
{
    if(sinManInfo.keyPoints.state[indexOne]&&sinManInfo.keyPoints.state[indexTwo])
    {

        cv::line(img, cv::Point(sinManInfo.keyPoints.x[indexOne], sinManInfo.keyPoints.y[indexOne]),
			cv::Point(sinManInfo.keyPoints.x[indexTwo], sinManInfo.keyPoints.y[indexTwo]), color, 2);
    }
}

void resultsDrawer::drawSinKeyPoint(cv::Mat& img,  std::vector<PersonInfo> sinManInfoList)
{
    for(int j=0; j<sinManInfoList.size(); j++)
    {
        bool flag = true;
		int id = sinManInfoList[j].personID;
		cv::Scalar color = cv::Scalar((0+id*156435 )%255, (125+id*54)%255, (255+id*944654)%255);
		
        drawLine(img,sinManInfoList[j],0,1, color); drawLine(img,sinManInfoList[j],1,3, color); drawLine(img,sinManInfoList[j],0,2, color); 
		drawLine(img,sinManInfoList[j],2,4, color); 

        drawLine(img,sinManInfoList[j],5,6, color); drawLine(img,sinManInfoList[j],5,7, color); 
        drawLine(img,sinManInfoList[j],7,9, color); drawLine(img,sinManInfoList[j],6,8, color); drawLine(img,sinManInfoList[j],8,10, color);
        drawLine(img,sinManInfoList[j],5,11, color); drawLine(img,sinManInfoList[j],6,12, color); drawLine(img,sinManInfoList[j],11,12, color); 
        drawLine(img,sinManInfoList[j],11,13, color); 
		drawLine(img,sinManInfoList[j],12,14, color); 
        drawLine(img,sinManInfoList[j],13,15, color);
        drawLine(img,sinManInfoList[j],14,16, color);
		
		 cv::circle(img, cv::Point(sinManInfoList[j].keyPoints.x[13], sinManInfoList[j].keyPoints.y[13])\
						,2, CV_RGB(111,111,255),3); 
		 cv::circle(img, cv::Point(sinManInfoList[j].keyPoints.x[14], sinManInfoList[j].keyPoints.y[14])\
						,2, CV_RGB(111,111,255),3); 				
        /*
        if(flag)
        {
            drawLine(img,sinManInfoList[j],19,5); drawLine(img,sinManInfoList[j],19,6); 
        }
        else
        {
            drawLine(img,sinManInfoList[j],17,18);
            drawLine(img,sinManInfoList[j],18,5);
            drawLine(img,sinManInfoList[j],18,6);
        }
        */
        

        //drawAssemblePoint
        /*if(sinManInfoList[j].assembleCoorFlag)
        {
            cv::circle(img,cv::Point(sinManInfoList[j].assembleCoor[0], sinManInfoList[j].assembleCoor[1]),8, CV_RGB(255,255,255),1); 
            cv::circle(img,cv::Point(sinManInfoList[j].assembleCoor[2], sinManInfoList[j].assembleCoor[3]),8, CV_RGB(255,255,255),1); 
            cv::circle(img,cv::Point(sinManInfoList[j].assembleCoor[4], sinManInfoList[j].assembleCoor[5]),8, CV_RGB(255,255,255),1); 
        }*/
    }

}


void resultsDrawer::showKpHandRect(cv::Mat& img, const PersonInfo& sinManInfo)
{
    for(int i=0;i<2;i++)
    {
        if(sinManInfo.handAction[i].score)
        {
            float x1 = sinManInfo.kpHandBbox[i].x1;
            float y1 = sinManInfo.kpHandBbox[i].y1;
            float x2 = sinManInfo.kpHandBbox[i].x2;
            float y2 = sinManInfo.kpHandBbox[i].y2;
            int id = sinManInfo.personID;

            int font_face = cv::FONT_HERSHEY_COMPLEX; 
            double font_scale = 2; int thickness = 2;
            cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar((0+id*156435)%255, (125+id*54)%255, (255+id*944654)%255), 3, 4);
        }
    }
}

void resultsDrawer::showResults(cv::Mat& img, const std::vector<RectAction>& headInfoList, std::vector<PersonInfo>& personInfoList)
{
    showHeadResult(img, headInfoList);
    showBodyRect(img, personInfoList);
    drawSinKeyPoint(img, personInfoList);
    //showAssembleResults(img, personInfoList);
    //showFightResult(img, personInfoList);
 
}

void resultsDrawer::showHeadResult(cv::Mat& img, const std::vector<RectAction>& headInfoList)
{
    float phoneThr = 0.6;
    float smokeThr = 0.6;
    for(int i=0; i<headInfoList.size(); i++)
    {
        float x1 = headInfoList[i].bboxInFrame.x1;
        float y1 = headInfoList[i].bboxInFrame.y1;
        float x2 = headInfoList[i].bboxInFrame.x2;
        float y2 = headInfoList[i].bboxInFrame.y2;
        if(headInfoList[i].action[0].score>phoneThr && headInfoList[i].action[1].score<=smokeThr)
            cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 255), 2);
        else if(headInfoList[i].action[0].score<=phoneThr && headInfoList[i].action[1].score>smokeThr)
            cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
        else if(headInfoList[i].action[0].score>phoneThr && headInfoList[i].action[1].score>smokeThr)
            cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0), 2);
        else
            cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(200, 200, 200), 2);

        /*
        */
        //画手机框
        if(headInfoList[i].phoneFlag)
        {
            x1 = headInfoList[i].phoneBbox.x1;
            y1 = headInfoList[i].phoneBbox.y1;
            x2 = headInfoList[i].phoneBbox.x2;
            y2 = headInfoList[i].phoneBbox.y2;
            cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 255, 255), 2);
        }

        //画香烟框
        if(headInfoList[i].smokeFlag)
        {
            x1 = headInfoList[i].smokeBbox.x1;
            y1 = headInfoList[i].smokeBbox.y1;
            x2 = headInfoList[i].smokeBbox.x2;
            y2 = headInfoList[i].smokeBbox.y2;
            cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 255, 255), 1);
        }
    }

}

void resultsDrawer::showBodyRect(cv::Mat& img, std::vector<PersonInfo>& personInfoList)
{
    for(int i=0; i<personInfoList.size(); i++)
    {
        float x1 = personInfoList[i].bodyBox.x1;
        float y1 = personInfoList[i].bodyBox.y1;
        float x2 = personInfoList[i].bodyBox.x2;
        float y2 = personInfoList[i].bodyBox.y2;
		
        int id = personInfoList[i].personID;
        std::string text = std::to_string(id);
        int font_face = cv::FONT_HERSHEY_COMPLEX; 
        double font_scale = 2; int thickness = 2;
        //cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 1);
        cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar((0+id*156435)%255, (125+id*54)%255, (255+id*944654)%255), 2, 4);
        cv::putText(img, text, cv::Point(x1, y1), font_face, font_scale, cv::Scalar(0, 255, 255), thickness, 2, 0);
		
		//head
		if (personInfoList[i].headBboxNum>= 1)
		{
			cv::rectangle(img, cv::Point(personInfoList[i].headBbox[0].x1, personInfoList[i].headBbox[0].y1), 
					cv::Point(personInfoList[i].headBbox[0].x2, personInfoList[i].headBbox[0].y2), 
					cv::Scalar((0+id*156435)%255, (125+id*54)%255, (255+id*944654)%255), 2, 4);

		}
        //没有显示手的检测框，只有通过手部检测框分类到抽烟时才显示框
        //showHandRect(img, personInfoList[i]);
        if(personInfoList[i].assembleCoorFlag)
            showKpHandRect(img, personInfoList[i]);

        //显示是否摔倒
        text = "Fall";
		auto iter_fall =  personInfoList[i].action.find("fall");
        //if(personInfoList[i].fallFlag)
		if(iter_fall != personInfoList[i].action.end() && iter_fall->second.score>= 0.80)	
		{
			text = "fall"+std::to_string((iter_fall->second.score));  
			text.resize(7);
            cv::putText(img, text, cv::Point(x2, y1), font_face, font_scale, cv::Scalar(0, 255, 255), thickness, 2, 0);
		}

        text = "speed"+std::to_string(int(personInfoList[i].action["speed"].score));    
        cv::putText(img, text, cv::Point(x2, (y2+y1)/2), font_face, font_scale, cv::Scalar(255, 0, 255), thickness, 2, 0);   
		
		auto iter_c = personInfoList[i].action.find("knees");
		if (iter_c != personInfoList[i].action.end())
		{
			text = std::to_string((iter_c->second.score));  
			text.resize(4);		
			cv::putText(img, text, cv::Point(x2, y2), font_face, font_scale, cv::Scalar(125, 0, 0), thickness-1, 2, 0);

			if ((iter_c->second.score) != (iter_c->second.score))
			{
				std::cout << "error " << std::endl;
				exit(-1);
			}				
		}
		iter_c = personInfoList[i].action.find("climb");
		if (iter_c != personInfoList[i].action.end())
		{
			text = std::to_string((iter_c->second.score));   
			text.resize(3);
			cv::putText(img, text, cv::Point((x2+x1)/2.0, y1), font_face, font_scale, cv::Scalar(255, 128, 255), thickness, 2, 0);
		}
        
    }

}

void resultsDrawer::showHandRect(cv::Mat& img, const PersonInfo& sinManInfo)
{
    for(int i=0;i<sinManInfo.handBboxNum;i++)
    {
        float x1 = sinManInfo.handBbox[i].x1;
        float y1 = sinManInfo.handBbox[i].y1;
        float x2 = sinManInfo.handBbox[i].x2;
        float y2 = sinManInfo.handBbox[i].y2;
        int id = sinManInfo.personID;

        int font_face = cv::FONT_HERSHEY_COMPLEX; 
        double font_scale = 2; int thickness = 2;
        cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar((0+id*156435)%255, (125+id*54)%255, (255+id*944654)%255), 3, 4);
    }

}

void resultsDrawer::showAssembleResults(cv::Mat& img, const std::vector<PersonInfo>& personInfoList)
{
    for(int i=0; i<personInfoList.size(); i++)
    {
        if(personInfoList[i].matchFlag)
        {
            int id = personInfoList[i].personID;
            std::string text = "smoke";
            int font_face = cv::FONT_HERSHEY_COMPLEX; 
            double font_scale = 1; int thickness = 1;
            float x1,y1,x2,y2;
            x1 = personInfoList[i].personHeadBox.x1;
            y1 = personInfoList[i].personHeadBox.y1;
            text = "head";
            //cv::putText(img, text, cv::Point(x1, y1), font_face, font_scale, cv::Scalar((0+id*156435)%255, (125+id*54)%255, (255+id*944654)%255), thickness, 1, 0);

            if(personInfoList[i].assembleAction[0].score<0.6) 
            {
                text = "smoke";
                x1 = personInfoList[i].bodyBox.x1;
                y1 = personInfoList[i].bodyBox.y1;
                x2 = personInfoList[i].bodyBox.x2;
                y2 = personInfoList[i].bodyBox.y2;
                cv::putText(img, text, cv::Point(x2, y1), font_face, font_scale, cv::Scalar(0, 255, 255), thickness, 1, 0);
            }
        }
    }

}

void resultsDrawer::showFightResult(cv::Mat& img, const std::vector<PersonInfo>& personInfoList)
{
    if(personInfoList.size())
    {
        if(personInfoList[0].wholeFightFlag)
        {
            int font_face = cv::FONT_HERSHEY_COMPLEX; 
            double font_scale = 2; int thickness = 2;
            std::string text = "Fighting !!";
            
            cv::putText(img, text, cv::Point(img.cols/2, img.rows/3), font_face, font_scale, cv::Scalar(0, 255, 255), thickness, 2, 0);
        }
    
    }
}

