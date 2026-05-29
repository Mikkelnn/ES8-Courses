#include <Arduino.h>
#include <Arduino_LSM6DS3.h>
#include <BasicLinearAlgebra.h>

using namespace BLA;

class IMUExtended : public LSM6DS3Class
{
public:
    IMUExtended(TwoWire &wire, uint8_t addr) : LSM6DS3Class{wire, addr} {}

    void setRate13Hz()
    {
        writeRegister(0x10, 0b00011000); // CTRL1_XL
        writeRegister(0x11, 0b00011100); // CTRL2_G
    }
    void setRate26Hz()
    {
        writeRegister(0x10, 0b00101000);
        writeRegister(0x11, 0b00101100);
    }
    void setRate52Hz()
    {
        writeRegister(0x10, 0b00111000);
        writeRegister(0x11, 0b00111100);
    }
    void setRate104Hz()
    {
        writeRegister(0x10, 0b01001000);
        writeRegister(0x11, 0b01001100);
    }
};

IMUExtended imu{Wire, LSM6DS3_ADDRESS};

const float Ts = 1.0f / 13.0f;
const float omega_b = 2.0f * 3.14159f * 0.5f; // [rad/s]
const float phi = exp(-omega_b * Ts);         // AR(1) pole

const float sigma_a = 1.5f;                              // tune from stationary data
const float sigma_qa = sigma_a * sqrt(1.0f - phi * phi); // steady-state AR(1) noise

const float sigma_qb = 0.002f;
const float sigma_v = 0.05f; // tune from stationary data
const float R_var = sigma_v * sigma_v;

float bench_v = 0.0f;
float bench_p = 0.0f;

void benchmarkUpdate(float accel)
{
    bench_v += accel * Ts;
    bench_p += bench_v * Ts;
}

Matrix<3, 3> Phi_kf = {
    phi, 0, 0,
    Ts, 1, 0,
    0, Ts, 1};

Matrix<3, 3> Q_kf = {
    sigma_qa * sigma_qa, 0, 0,
    0, 0, 0,
    0, 0, 0};

Matrix<1, 3> H_kf = {1, 0, 0}; // measures acceleration only

Matrix<3, 1> x_kf; // [a, v, p]
Matrix<3, 3> P_kf; // error covariance

void kalmanUpdate(float accel_meas)
{
    // predict
    Matrix<3, 1> x_pred = Phi_kf * x_kf;
    Matrix<3, 3> P_pred = Phi_kf * P_kf * ~Phi_kf + Q_kf;

    float innov = accel_meas - (H_kf * x_pred)(0, 0); // residual
    float S = (H_kf * P_pred * ~H_kf)(0, 0) + R_var;  // innovation covariance
    Matrix<3, 1> K = P_pred * ~H_kf * (1.0f / S);     // Kalman gain

    // update
    x_kf = x_pred + K * innov;

    Matrix<3, 3> I_kf;
    I_kf.Fill(0);
    for (int i = 0; i < 3; i++)
        I_kf(i, i) = 1.0f;
    P_kf = (I_kf - K * H_kf) * P_pred;
}

Matrix<4, 4> Phi_bias = {
    phi, 0, 0, 0,
    Ts, 1, 0, 0,
    0, Ts, 1, 0,
    0, 0, 0, 1};

Matrix<4, 4> Q_bias = {
    sigma_qa * sigma_qa, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, sigma_qb *sigma_qb};

Matrix<1, 4> H_bias = {1, 0, 0, 1}; // measures acceleration + bias

Matrix<4, 1> x_bias; // [a, v, p, b]
Matrix<4, 4> P_bias; // error covariance

void kalmanBiasUpdate(float accel_meas)
{
    // predict
    Matrix<4, 1> x_pred = Phi_bias * x_bias;
    Matrix<4, 4> P_pred = Phi_bias * P_bias * ~Phi_bias + Q_bias;

    float innov = accel_meas - (H_bias * x_pred)(0, 0);  // residual
    float S = (H_bias * P_pred * ~H_bias)(0, 0) + R_var; // innovation covariance
    Matrix<4, 1> K = P_pred * ~H_bias * (1.0f / S);      // Kalman gain

    // update
    x_bias = x_pred + K * innov;

    Matrix<4, 4> I_bias;
    I_bias.Fill(0);
    for (int i = 0; i < 4; i++)
        I_bias(i, i) = 1.0f;
    P_bias = (I_bias - K * H_bias) * P_pred;
}

void setup()
{
    Serial.begin(115200);
    while (!Serial)
        ;

    if (!imu.begin())
    {
        Serial.println("IMU init failed");
        while (1)
            ;
    }
    imu.setRate13Hz();

    x_kf.Fill(0);
    x_bias.Fill(0);

    P_kf.Fill(0);
    P_kf(0, 0) = sigma_a * sigma_a;
    P_kf(1, 1) = 0.01f;
    P_kf(2, 2) = 0.001f;

    P_bias.Fill(0);
    P_bias(0, 0) = sigma_a * sigma_a;
    P_bias(1, 1) = 0.01f;
    P_bias(2, 2) = 0.001f;
    P_bias(3, 3) = sigma_a * sigma_a; // bias uncertainty prior

    // header
    Serial.println("# t_ms | accel | bench_v bench_p | kf_a kf_v kf_p kf_stdv kf_stdp | kfb_a kfb_v kfb_p kfb_b kfb_stdv kfb_stdp");
}

void loop()
{
    static uint32_t t0 = millis();
    float ax = 0, ay = 0, az = 0;

    if (!imu.accelerationAvailable())
    {
        return;
    }

    imu.readAcceleration(ax, ay, az);

    float accel = ax * 9.81f; // LSM6DS3 outputs g
    uint32_t t_ms = millis() - t0;

    benchmarkUpdate(accel);
    kalmanUpdate(accel);
    kalmanBiasUpdate(accel);

    float kf_std_v = sqrt(P_kf(1, 1)); // std dev from covariance diagonal
    float kf_std_p = sqrt(P_kf(2, 2));
    float kfb_std_v = sqrt(P_bias(1, 1));
    float kfb_std_p = sqrt(P_bias(2, 2));

    Serial.print("t_ms:");
    Serial.print(t_ms); // elapsed time [ms]
    Serial.print(" accel:");
    Serial.print(accel, 4); // raw accel measurement [m/s²]
    Serial.print(" bench_v:");
    Serial.print(bench_v, 4); // naive integrated velocity [m/s]
    Serial.print(" bench_p:");
    Serial.print(bench_p, 4); // naive integrated position [m]
    Serial.print(" kf_a:");
    Serial.print(x_kf(0, 0), 4); // kalman estimated acceleration [m/s²]
    Serial.print(" kf_v:");
    Serial.print(x_kf(1, 0), 4); // kalman estimated velocity [m/s]
    Serial.print(" kf_p:");
    Serial.print(x_kf(2, 0), 4); // kalman estimated position [m]
    Serial.print(" kf_stdv:");
    Serial.print(kf_std_v, 4); // velocity uncertainty [m/s]
    Serial.print(" kf_stdp:");
    Serial.print(kf_std_p, 4); // position uncertainty [m]
    Serial.print(" kfb_a:");
    Serial.print(x_bias(0, 0), 4); // bias kalman estimated acceleration [m/s²]
    Serial.print(" kfb_v:");
    Serial.print(x_bias(1, 0), 4); // bias kalman estimated velocity [m/s]
    Serial.print(" kfb_p:");
    Serial.print(x_bias(2, 0), 4); // bias kalman estimated position [m]
    Serial.print(" kfb_b:");
    Serial.print(x_bias(3, 0), 4); // estimated sensor bias [m/s²]
    Serial.print(" kfb_stdv:");
    Serial.print(kfb_std_v, 4); // velocity uncertainty [m/s]
    Serial.print(" kfb_stdp:");
    Serial.println(kfb_std_p, 4); // position uncertainty [m]
}
