#ifndef RAND_H
#define RAND_H

#include <math.h>
#include <stdlib.h>

// 返回值是0~1之间的double类型
inline double RandDouble()
{
    double r = static_cast<double>(rand());
    return r / RAND_MAX;
}

// 返回值是负无穷到正无穷之间的double类型
inline double RandNormal()
{
    double x1, x2, w;
    do{
        x1 = 2.0 * RandDouble() - 1.0;
        x2 = 2.0 * RandDouble() - 1.0;
        w = x1 * x1 + x2 * x2;
    }while( w >= 1.0 || w == 0.0);

    w = sqrt((-2.0 * log(w))/w);
    return x1 * w;
}

#endif // random.h