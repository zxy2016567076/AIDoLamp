#include "chuankou.h"
//打印函数
int _write(int file, char *data, int len) {
    int i = 0;
    for(i = 0; i < len; i++) {
        // 发送一个字符到USART1
        HAL_UART_Transmit(&huart6, (unsigned char*)&data[i], 1, HAL_MAX_DELAY);
    }
    return len;
}

char receiveData[50];
extern DMA_HandleTypeDef hdma_usart6_rx;
extern DMA_HandleTypeDef hdma_usart6_tx;

// 工具函数（支持去除换行符）
// ------------------------------
static void trim_whitespace(char *str) {
    // 去除头尾所有空白字符（含空格、制表符、换行、回车）
    if (!str) return;
    int len = strlen(str);

    // 去除尾部空白
    while (len > 0 && isspace((unsigned char)str[len-1])) {
        str[--len] = '\0';
    }

    // 去除头部空白
    char *start = str;
    while (isspace((unsigned char)*start)) start++;
    if (start != str) {
        memmove(str, start, len - (start - str) + 1);
    }
}

static int is_numeric(const char *str) {
    if (*str == '-' || *str == '+') str++; // 允许符号

    int has_dot = 0;
    while (*str) {
        if (*str == '.' && !has_dot) {
            has_dot = 1;
        } else if (!isdigit(*str)) {
            return 0; // 非数字字符
        }
        str++;
    }
    return 1;
}

// ------------------------------
// 主解析函数（支持6参数）
// ------------------------------
CommandResult parse_command(const char *input) {
    CommandResult res = {0};
    res.param_count = 0;

    // 1. 安全拷贝输入并清理（关键：防止溢出）
    char buffer[256];
    strncpy(buffer, input, sizeof(buffer) - 1);
    buffer[sizeof(buffer) - 1] = '\0'; // 强制终止符
    trim_whitespace(buffer);  // 清理首尾空白

    // 2. 分割命令词和参数（支持空格、TAB、换行作为分隔符）
    const char *delim = " \t\r\n";
    char *token = strtok(buffer, delim);

    if (token == NULL) return res; // 处理空输入

    // 3. 保存命令词（直接使用首个token）
    strncpy(res.command, token, sizeof(res.command) - 1);
    res.command[sizeof(res.command) - 1] = '\0'; // 确保终止

    // 4. 循环提取参数（最多6个）
    for (int arg = 0; arg < 6; arg++) {
        token = strtok(NULL, delim);  // 继续分割后续内容

        if (token == NULL) break;     // 无更多参数时退出

        // 清理单个参数内的残留空白（如" 123 "→"123"）
        trim_whitespace(token);

        // 分析参数合法性并存储
        if (is_numeric(token)) {
            res.params[arg] = atof(token);
        } else {
            res.params[arg] = 0.0f; // 非法参数默认0
        }

        res.param_count++; // 增加参数计数
    }

    return res;
}

int flag = 0;
void chuankou_init()
{
	  HAL_UARTEx_ReceiveToIdle_DMA(&huart6,(uint8_t *)receiveData,sizeof(receiveData));
	  __HAL_DMA_DISABLE_IT(&hdma_usart6_rx,DMA_IT_HT);
}


void HAL_UARTEx_RxEventCallback(UART_HandleTypeDef *huart, uint16_t Size){
	if(huart == &huart6){
		printf("ok\n");
		flag = 1;
	  HAL_UARTEx_ReceiveToIdle_DMA(&huart6,(uint8_t *)receiveData,sizeof(receiveData));
	  __HAL_DMA_DISABLE_IT(&hdma_usart6_rx,DMA_IT_HT);
	}
}
