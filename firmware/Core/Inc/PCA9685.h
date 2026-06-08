#ifndef __STM32PCA9685_H
#define __STM32PCA9685_H

#include "stm32f4xx_hal.h"

#define pca_adrr 0x80

#define pca_mode1 0x0
#define pca_pre 0xFE

#define LED0_ON_L 0x6
#define LED0_ON_H 0x7
#define LED0_OFF_L 0x8
#define LED0_OFF_H 0x9

extern float now_angle_1,now_angle_2,now_angle_3,now_angle_0;

//各个舵机的零度位置
#define SERVO_ZERO_0 135.0f
#define SERVO_ZERO_1 90.0f
#define SERVO_ZERO_2 0.0f
#define SERVO_ZERO_3 90.0f

void pca_write(uint8_t adrr, uint8_t data);
uint8_t pca_read(uint8_t adrr);
void PCA_Servo_Init1(void);
void PCA_Servo_Init2(void);
//void PCA_Servo_Init3(void);
//void PCA_Servo_Init4(void);
void pca_setfreq(float freq);
void pca_setpwm(uint8_t num, uint32_t on, uint32_t off);
void PCA_Servo_180(uint8_t num, float end_angle);
void PCA_Servo_270(uint8_t num, float end_angle);
void PCA_Servo_180_Speed(uint8_t num, float start_angle, float end_angle, float speed);
void PCA_Servo_270_Speed(uint8_t num, float start_angle, float end_angle, float speed);
void pca_set_duty_cycle(uint8_t channel, float duty_cycle);
//void PCA_Paw_Status(int status);

#endif
