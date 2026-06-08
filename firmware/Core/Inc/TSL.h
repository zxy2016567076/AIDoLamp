/*
 * TSL.h
 *
 *  Created on: May 2, 2025
 *      Author: stay
 */

#ifndef INC_TSL_H_
#define INC_TSL_H_
#include "stm32f4xx_hal.h"
extern float distance;
void TriggerUltrasonic(void);
void CalculateDistance(void);
void TSL2591_Init(I2C_HandleTypeDef *hi2c);
float TSL2591_ReadLux(I2C_HandleTypeDef *hi2c);
void PWM_VAL(int PWM_Val);
void PWM_Adjust(uint16_t lux);
#endif /* INC_TSL_H_ */
