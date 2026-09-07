#ifndef QNN_TEST_UTIL_HPP
#define QNN_TEST_UTIL_HPP

#include <cstdio>

static int g_failures = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);        \
            ++g_failures;                                                      \
        }                                                                      \
    } while (0)

#define DONE()                                                                 \
    do {                                                                       \
        if (g_failures == 0) std::printf("PASS %s\n", __FILE__);               \
        return g_failures == 0 ? 0 : 1;                                        \
    } while (0)

#endif  // QNN_TEST_UTIL_HPP