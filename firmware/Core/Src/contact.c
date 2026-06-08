#include "contact.h"





#define STRING_STOP "STOP"
unsigned char m=0;
char OLED_printf[30]="";
char str[20]="";
uint8_t dis[20];
int PWM_Val=500;

extern char receiveData[50];

void contact()
{


//	printf("命令词: %s\n", res.command);  // 输出：set_speed
//	printf("参数个数: %d\n", res.param_count); // 输出：3
//
//	for(int i=0; i<res.param_count; i++) {
//	    printf("参数%d: %f\n", i+1, res.params[i]);
//	}

	CommandResult res = parse_command(receiveData);
	printf("%s,%f,%f\n",res.command, res.params[0], res.params[1]);
	//模式
	if(strcmp(res.command, STRING_MODE) == 0)
	{
		if(res.params[0] == 1)
		{
			PCA_Servo_Init2();
			TriggerUltrasonic();//开启超声波模块
			CalculateDistance();
			//sprintf(str,"%.2f",distance);
			//HAL_UART_Transmit_DMA(&huart3,&str , 50);
			dis[0] = distance;
			HAL_UART_Transmit_DMA(&huart6,dis , 20);
			OLED_NewFrame();//OLED显示
			sprintf(OLED_printf,"AIDoLamp");
			OLED_PrintString(0,0,OLED_printf,&font16x16,OLED_COLOR_NORMAL);
			sprintf(OLED_printf,"状态:待机");
			OLED_PrintString(0,16,OLED_printf,&font16x16,OLED_COLOR_NORMAL);
			OLED_ShowFrame();

		}
		else if(res.params[0] == 2)
		{
			PCA_Servo_Init1();
			OLED_NewFrame();//OLED显示
			sprintf(OLED_printf,"AIDoLamp");
			OLED_PrintString(0,0,OLED_printf,&font16x16,OLED_COLOR_NORMAL);
			sprintf(OLED_printf,"状态:陪伴");
			OLED_PrintString(0,16,OLED_printf,&font16x16,OLED_COLOR_NORMAL);
			OLED_ShowFrame();
			__HAL_TIM_SET_COMPARE(&htim2,TIM_CHANNEL_1,PWM_Val);			//开启灯光



		}
		else if(res.params[0] == 3)
		{
			PCA_Servo_Init2();
			OLED_NewFrame();//OLED显示
			sprintf(OLED_printf,"AIDoLamp");
			OLED_PrintString(0,0,OLED_printf,&font16x16,OLED_COLOR_NORMAL);
			sprintf(OLED_printf,"状态:互动");
			OLED_PrintString(0,16,OLED_printf,&font16x16,OLED_COLOR_NORMAL);
			OLED_ShowFrame();

		}
		else
		{
			PCA_Servo_Init1();
			OLED_NewFrame();//OLED显示
			sprintf(OLED_printf,"AIDoLamp");
			OLED_PrintString(0,0,OLED_printf,&font16x16,OLED_COLOR_NORMAL);
			sprintf(OLED_printf,"状态:写作");
			OLED_PrintString(0,16,OLED_printf,&font16x16,OLED_COLOR_NORMAL);
			OLED_ShowFrame();
			TSL2591_Init(&hi2c3);
			float lux=TSL2591_ReadLux(&hi2c3);
			PWM_Adjust(lux);

							//自动调节LED光强
		}
	}

	//舵机
	if(strcmp(res.command,STRING_Adroup_1) == 0)//上
	{
		PCA_Servo_180(1,now_angle_1-10);
	}
	if(strcmp(res.command,STRING_Adroup_2) == 0)//上
	{
		PCA_Servo_180(2,now_angle_2-10);
	}
	if(strcmp(res.command,STRING_Adroup_3) == 0)//上
	{
		PCA_Servo_180(3,now_angle_3-10);
	}


	if(strcmp(res.command,STRING_Adrodown_1) == 0)// 下
	{
		PCA_Servo_180(1,now_angle_1+10);
	}
	if(strcmp(res.command,STRING_Adrodown_2) == 0)// 下
	{
		PCA_Servo_180(2,now_angle_2+10);
	}
	if(strcmp(res.command,STRING_Adrodown_3) == 0)// 下
	{
		PCA_Servo_180(3,now_angle_3+10);
	}


	if(strcmp(res.command,STRING_Adroleft) == 0)// 左
	{
		PCA_Servo_270(0,now_angle_0+10);
	}

	if(strcmp(res.command,STRING_Adroright) == 0)// 右
	{
		PCA_Servo_270(0,now_angle_0-10);
	}

	if(strcmp(res.command,STRING_Alldro) == 0)// 机械臂移动
	{

		PCA_Servo_270(0,res.params[0]);
		PCA_Servo_180(1,res.params[1]);
		PCA_Servo_180(2,res.params[2]);
		PCA_Servo_180(3,res.params[3]);

	}

	//灯强
	if(strcmp(res.command, STRING_LIGHT_ZJ) == 0)//加强
	{
		PWM_Val+=300;
//		if(PWM_Val>1000)
//			PWM_Val=1000;
		PWM_VAL(PWM_Val);
//		__HAL_TIM_SET_COMPARE(&htim2,TIM_CHANNEL_1,PWM_Val);
		OLED_NewFrame();//OLED显示
		sprintf(OLED_printf,"AIDoLamp");
		OLED_PrintString(0,0,OLED_printf,&font16x16,OLED_COLOR_NORMAL);
		OLED_ShowFrame();

	}
	if(strcmp(res.command, STRING_LIGHT_JX) == 0)//减弱
	{
		PWM_Val-=300;
//		if(PWM_Val<0)
//			PWM_Val=0;
		PWM_VAL(PWM_Val);
//		__HAL_TIM_SET_COMPARE(&htim2,TIM_CHANNEL_1,PWM_Val);
		OLED_NewFrame();//OLED显示
		sprintf(OLED_printf,"AIDoLamp");
		OLED_PrintString(0,0,OLED_printf,&font16x16,OLED_COLOR_NORMAL);
		OLED_ShowFrame();
	}


	//色温
	if(strcmp(res.command, STRING_light_zj) == 0)//加强
	{

	}
	if(strcmp(res.command, STRING_light_jx) == 0)//减弱
	{

	}
	if(strcmp(res.command, OLEDONE)==0)
	{
		OLED_NewFrame();//OLED显示
		sprintf(OLED_printf,"AIDoLamp");
		OLED_PrintString(0,0,OLED_printf,&font16x16,OLED_COLOR_NORMAL);
		sprintf(OLED_printf,"手势已激活");
		OLED_PrintString(0,16,OLED_printf,&font16x16,OLED_COLOR_NORMAL);
		OLED_ShowFrame();

	}
	if(strcmp(res.command, OLEDTWO)==0)
	{
		OLED_NewFrame();//OLED显示
		sprintf(OLED_printf,"AIDoLamp");
		OLED_PrintString(0,0,OLED_printf,&font16x16,OLED_COLOR_NORMAL);
		sprintf(OLED_printf,"已进入手势控制");
		OLED_PrintString(0,16,OLED_printf,&font16x16,OLED_COLOR_NORMAL);
		OLED_ShowFrame();
	}
	memset(receiveData,0,sizeof(receiveData));
}



