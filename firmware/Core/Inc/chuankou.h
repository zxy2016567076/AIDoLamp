#ifndef INC_CHUANKOU_H_
#define INC_CHUANKOU_H_

#include "main.h"

typedef struct {
    char command[64];     // 存储命令词（如"set_speed"）
    double params[6];      // 存储最多6个参数，索引0~5
    int param_count;      // 实际解析到的参数个数（0~6）
} CommandResult;


void chuankou_init();
CommandResult parse_command(const char *input);

#endif /* INC_CHUANKOU_H_ */
