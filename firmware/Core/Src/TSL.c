/*
 * TSL.c
 *
 *  Created on: May 2, 2025
 *      Author: stay
 */
#include "TSL.h"
#include "i2c.h"
#include "tim.h"

#define TSL2591_ADDR 0x29 << 1  // HAL库要求左移1位
#define COMMAND_BIT  0xA0  // 寄存器命令位

int pulse_duration=0;
float distance=0;

void TriggerUltrasonic(void) {
    HAL_GPIO_WritePin(GPIOE, GPIO_PIN_12, GPIO_PIN_SET);//触发超声波模块
    HAL_Delay(1);  // 保持10us
    HAL_GPIO_WritePin(GPIOE, GPIO_PIN_12, GPIO_PIN_RESET);
}

void CalculateDistance(void) {
    while (HAL_GPIO_ReadPin(GPIOE, GPIO_PIN_13) == GPIO_PIN_RESET);//测量距离
    __HAL_TIM_SET_COUNTER(&htim1, 0);
    TIM2->CR1 |= TIM_CR1_CEN;
    while (HAL_GPIO_ReadPin(GPIOE, GPIO_PIN_13) == GPIO_PIN_SET);
    TIM2->CR1 &= ~TIM_CR1_CEN;
    pulse_duration = __HAL_TIM_GET_COUNTER(&htim1);
    distance = (pulse_duration * 343) / 20000;
}
void TSL2591_Init(I2C_HandleTypeDef *hi2c) {
    uint8_t config[2];

    // 启动传感器（ENABLE寄存器）
    uint8_t enable_cmd[2] = {0xA0 | 0x00, 0x03}; // 启动传感器
    HAL_I2C_Master_Transmit(hi2c, TSL2591_ADDR, enable_cmd, 2, 100);

    // 配置增益和积分时间（CONTROL寄存器）
    uint8_t gain_cmd[2] = {0xA0 | 0x01, 0x03};        // 设置增益和积分时间
    HAL_I2C_Master_Transmit(hi2c, TSL2591_ADDR, gain_cmd, 2, 100);
}

float TSL2591_ReadLux(I2C_HandleTypeDef *hi2c) {
    uint8_t data[4];
    HAL_I2C_Mem_Read(hi2c, TSL2591_ADDR, 0x14 | 0xA0, I2C_MEMADD_SIZE_8BIT, data, 4, 100);
    uint16_t ch0 = (data[1] << 8) | data[0];
    uint16_t ch1 = (data[3] << 8) | data[2];
	if (ch0 == 0) return 0.0; // 防止除零错误
    float ratio = (float)ch1 / ch0;
    float lux;

    if (ratio <= 0.5) {
        lux = (ch0 * 1.0 - ch1 * 1.7)/25;
    } else {
        lux = (ch0 * 0.8 - ch1 * 0.6)/25;
    }
    return (lux > 0) ? lux : 0; // 避免负值
    return lux;
}
void PWM_VAL(int PWM_Val)
{
	__HAL_TIM_SET_COMPARE(&htim2,TIM_CHANNEL_1,PWM_Val);
}
void PWM_Adjust(uint16_t lux) {
    static uint16_t pwm_val;
    // 将光照值映射到PWM范围（0-1000）

    if(lux*100 > 1000)
		lux = 1000;

	pwm_val = 1000 - lux*100;
    __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_1, pwm_val);
}




