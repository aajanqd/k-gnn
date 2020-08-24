//preprocessor directive
#pragma once

//this is a macro that replaces ITERATE_NODES with code inside for loop
//this is also a variadic function - a function that takes variable arguments
//required arguments are START, NAME, END. Elipses (...) indicate variable args
#define ITERATE_NODES(START, NAME, END, ...)                                   \
  {                                                                            \
    for (int64_t NAME = START; NAME < END; NAME++) {                           \
      __VA_ARGS__;                                                             \
    }                                                                          \
  }

//this is a macro that replaces ITERATE_NEIGHBORS with code inside for loop
#define ITERATE_NEIGHBORS(NODE, NAME, ROW, COL, ...)                           \
  {                                                                            \
    for (int64_t NAME##_i = ROW[NODE]; NAME##_i < ROW[NODE + 1]; NAME##_i++) { \
      auto NAME = COL[NAME##_i];                                               \
      __VA_ARGS__;                                                             \
    }                                                                          \
  }
