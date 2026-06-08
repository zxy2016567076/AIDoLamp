#include "PCA9685.h"
#include "math.h"
#include "i2c.h"
float now_angle_1,now_angle_2,now_angle_3,now_angle_0;
uint8_t pca_read(uint8_t startAddress)
{
	uint8_t tx[1];
	uint8_t buffer[1];
	tx[0] = startAddress;

	HAL_I2C_Master_Transmit(&hi2c2, pca_adrr, tx, 1, 10000);
	HAL_I2C_Master_Receive(&hi2c2, pca_adrr, buffer, 1, 10000);
	return buffer[0];
}
void pca_write(uint8_t startAddress, uint8_t buffer)
{
	uint8_t tx[2];
	tx[0] = startAddress;
	tx[1] = buffer;

	HAL_I2C_Master_Transmit(&hi2c2, pca_adrr, tx, 2, 10000);
}
void pca_setfreq(float freq)
{
	uint8_t prescale, oldmode, newmode;
	double prescaleval;
	freq *= 0.937;
	prescaleval = 25000000;
	prescaleval /= 4096;
	prescaleval /= freq;
	prescaleval -= 1;
	prescale = floor(prescaleval + 0.5f);
	oldmode = pca_read(pca_mode1);
	newmode = (oldmode & 0x7F) | 0x10;

	pca_write(pca_mode1, newmode);
	pca_write(pca_pre, prescale);
	pca_write(pca_mode1, oldmode);

	HAL_Delay(2);

	pca_write(pca_mode1, oldmode | 0xa1);
}
void pca_setpwm(uint8_t num, uint32_t on, uint32_t off)
{
	pca_write(LED0_ON_L + 4 * num, on);
	pca_write(LED0_ON_H + 4 * num, on >> 8);
	pca_write(LED0_OFF_L + 4 * num, off);
	pca_write(LED0_OFF_H + 4 * num, off >> 8);
}
/*
	设置pwm占空比
	num：控制的舵机口
	duty_cycle：占空比 2.5~12.5
*/
void set_pwm_duty_cycle(uint8_t num, float duty_cycle)
{
	if (duty_cycle < 0.0f)
		duty_cycle = 0.0f; // 限制占空比范围
	if (duty_cycle > 100.0f)
		duty_cycle = 100.0f;

	uint32_t on = 0;											// 默认从 0 开始计数
	uint32_t off = (uint32_t)((duty_cycle / 100.0f) * 4096.0f); // 根据占空比计算 off 值

	pca_setpwm(num, on, off);
}
/*
	初始化舵机驱动模块
*/
void PCA_Servo_Init1(void)
{
	pca_write(pca_mode1, 0x0);
	pca_setfreq(50);
	HAL_Delay(500);
	PCA_Servo_270(0, 0);
	PCA_Servo_180(1, -20);
	PCA_Servo_180(2, 80);
	PCA_Servo_180(3, 0);
}
void PCA_Servo_Init2(void)
{
	pca_write(pca_mode1, 0x0);
	pca_setfreq(50);
	HAL_Delay(500);
	PCA_Servo_270(0, 0);
	PCA_Servo_180(1, -20);
	PCA_Servo_180(2, 80);
	PCA_Servo_180(3, -20);
}
//void PCA_Servo_Init3(void)
//{
//	pca_write(pca_mode1, 0x0);
//	pca_setfreq(50);
//	HAL_Delay(500);
//	PCA_Servo_270(0, 0);
//	PCA_Servo_180(1, 0);
//	PCA_Servo_180(2, 0);
//	PCA_Servo_180(3, 0);
//}
//void PCA_Servo_Init4(void)
//{
//	pca_write(pca_mode1, 0x0);
//	pca_setfreq(50);
//	HAL_Delay(500);
//	PCA_Servo_270(0, 0);
//	PCA_Servo_180(1, 0);
//	PCA_Servo_180(2, 0);
//	PCA_Servo_180(3, 0);
//}


/*
	获取真实旋转角度
*/
float GET_Servo_Real(uint8_t num, float end_angle)
{
	float angle = end_angle;
	switch (num)
	{
	case 0:
		angle = SERVO_ZERO_0 + end_angle;
		break;
	case 1:
		angle = SERVO_ZERO_1 + end_angle;
		break;
	case 2:
		angle = SERVO_ZERO_2 + end_angle;
		break;
	case 3:
		angle = SERVO_ZERO_3 + end_angle;
		break;
	default:
		angle = end_angle;
		break;
	}
	return angle;
}

/*
	控制180度舵机的旋转
	180度舵机实际最大可旋转200度
*/
void PCA_Servo_180(uint8_t num, float end_angle)
{
	// 角度输入合法性检查
	/*
	if(end_angle < -90 || end_angle > 200) {
	    Error_Handler();
	}
	*/
	float duty_cycle = 2.5f + (GET_Servo_Real(num, end_angle) / 180.0f) * 10.0f;
	set_pwm_duty_cycle(num, duty_cycle);

	if(num == 1)now_angle_1 = end_angle;
	if(num == 2)now_angle_2 = end_angle;
	if(num == 3)now_angle_3 = end_angle;

}

/*
	控制270度舵机的旋转
*/
void PCA_Servo_270(uint8_t num, float end_angle)//end_angle为逻辑角度
{
	float duty_cycle = 2.5f + (GET_Servo_Real(num, end_angle) / 270.0f) * 10.0f;
	set_pwm_duty_cycle(num, duty_cycle);
	now_angle_0 = end_angle;
}

