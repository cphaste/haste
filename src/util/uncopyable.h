#ifndef UTIL_UNCOPYABLE_H_
#define UTIL_UNCOPYABLE_H_

class Uncopyable {
protected:
    Uncopyable() {}
    ~Uncopyable() {}

private:
    Uncopyable(const Uncopyable&);
    Uncopyable& operator =(const Uncopyable&);
};

#endif // UTIL_UNCOPYABLE_H_
