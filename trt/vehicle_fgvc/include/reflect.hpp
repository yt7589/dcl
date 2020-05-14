#ifndef  __REFLECT_HPP__
#define __REFLECT_HPP__
#include <iostream>
#include <map>
#include <memory>

#define FUNC function<void*(void)>

using namespace std;


//Meyers Singleton
//Scott Meyers在《Effective C++》（Item 04）
class Reflector
{
private:
    map<std::string, FUNC>objectMap;
    Reflector(const Reflector&);
    Reflector& operator=(const Reflector&);
    Reflector(){
        //std::cout <<"Reflector"<< std::endl;
    };

public:
    void* CreateObject(const string &str)
    {
        for (auto & x : objectMap)
        {
            if(x.first == str)
                return x.second();
        }
        return nullptr;
    }

    void Register(const string &class_name, FUNC && generator)
    {
        objectMap[class_name] = generator;
    }

    static Reflector& Instance()
    {
        static Reflector re;
        return re;

    }



};


class RegisterAction
{
public:
    RegisterAction(const string &class_name, FUNC && generator)
    {
        Reflector::Instance().Register(class_name, forward<FUNC>(generator));
    }
};

#define REGISTER(CLASS_NAME) \
RegisterAction g_register_action_##CLASS_NAME(#CLASS_NAME, []()\
{\
    return new CLASS_NAME(); \
});


//class Base
//{
//public:
//    explicit Base() = default;
//    virtual void Print()
//    {
//        cout << "Base" << endl;
//    }
//};
//REGISTER(Base);
//
//class DeriveA : public Base
//{
//public:
//    void Print() override
//    {
//        cout << "DeriveA" << endl;
//    }
//};
//REGISTER(DeriveA);
//
//class DeriveB : public Base
//{
//public:
//    void Print() override
//    {
//        cout << "DeriveB" << endl;
//    }
//};
//REGISTER(DeriveB);

//int main()
//{
//    shared_ptr<Base> p1((Base*)Reflector::Instance()->CreateObject("Base"));
//    p1->Print();
//
//    shared_ptr<Base> p2((Base*)Reflector::Instance()->CreateObject("DeriveA"));
//    p2->Print();
//
//    shared_ptr<Base> p3((Base*)Reflector::Instance()->CreateObject("DeriveB"));
//    p3->Print();
//}

#endif