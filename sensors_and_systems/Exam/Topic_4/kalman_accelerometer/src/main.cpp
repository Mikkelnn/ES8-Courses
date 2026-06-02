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
const float f = 104.0f;
const float Ts = 1.0f / f;
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

// 3-state EKF: [a, v, p]
Matrix<3, 1> x_ekf;
Matrix<3, 3> P_ekf;
Matrix<3, 3> Q_ekf;
Matrix<3, 3> Phi_ekf;
Matrix<1, 3> H_ekf;

// 4-state EKF: [a, v, p, b]
Matrix<4, 1> x_ekfb;
Matrix<4, 4> P_ekfb;
Matrix<4, 4> Q_ekfb;
Matrix<4, 4> Phi_ekfb;
Matrix<1, 4> H_ekfb;

void initializeEKF()
{
    // Jacobian F = Phi (linear system, constant)
    Phi_ekf.Fill(0);
    Phi_ekf(0, 0) = phi;
    Phi_ekf(1, 0) = Ts;
    Phi_ekf(1, 1) = 1.0f;
    Phi_ekf(2, 1) = Ts;
    Phi_ekf(2, 2) = 1.0f;

    Q_ekf.Fill(0);
    Q_ekf(0, 0) = sigma_qa * sigma_qa;

    // Jacobian H (linear measurement, constant)
    H_ekf.Fill(0);
    H_ekf(0, 0) = 1.0f;

    // Jacobian F = Phi (linear system, constant)
    Phi_ekfb.Fill(0);
    Phi_ekfb(0, 0) = phi;
    Phi_ekfb(1, 0) = Ts;
    Phi_ekfb(1, 1) = 1.0f;
    Phi_ekfb(2, 1) = Ts;
    Phi_ekfb(2, 2) = 1.0f;
    Phi_ekfb(3, 3) = 1.0f;

    Q_ekfb.Fill(0);
    Q_ekfb(0, 0) = sigma_qa * sigma_qa;
    Q_ekfb(3, 3) = sigma_qb * sigma_qb;

    // Jacobian H (linear measurement, constant)
    H_ekfb.Fill(0);
    H_ekfb(0, 0) = 1.0f;
    H_ekfb(0, 3) = 1.0f;
}

void ekfUpdate(float z_meas)
{
    // Linear system: F = Phi, H constant
    Matrix<3, 1> x_pred = Phi_ekf * x_ekf;
    Matrix<3, 3> P_pred = Phi_ekf * P_ekf * ~Phi_ekf + Q_ekf;

    float z_pred = (H_ekf * x_pred)(0, 0);
    float innov = z_meas - z_pred;

    float S = (H_ekf * P_pred * ~H_ekf)(0, 0) + R_var;
    Matrix<3, 1> K = P_pred * ~H_ekf * (1.0f / S);

    x_ekf = x_pred + K * innov;

    Matrix<3, 3> I;
    I.Fill(0);
    for (int i = 0; i < 3; i++)
        I(i, i) = 1.0f;
    P_ekf = (I - K * H_ekf) * P_pred;
}

void ekfUpdateBias(float z_meas)
{
    // Linear system: F = Phi, H constant
    Matrix<4, 1> x_pred = Phi_ekfb * x_ekfb;
    Matrix<4, 4> P_pred = Phi_ekfb * P_ekfb * ~Phi_ekfb + Q_ekfb;

    float z_pred = (H_ekfb * x_pred)(0, 0);
    float innov = z_meas - z_pred;

    float S = (H_ekfb * P_pred * ~H_ekfb)(0, 0) + R_var;
    Matrix<4, 1> K = P_pred * ~H_ekfb * (1.0f / S);

    x_ekfb = x_pred + K * innov;

    Matrix<4, 4> I;
    I.Fill(0);
    for (int i = 0; i < 4; i++)
        I(i, i) = 1.0f;
    P_ekfb = (I - K * H_ekfb) * P_pred;
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
    imu.setRate104Hz();

    x_ekf.Fill(0);
    x_ekfb.Fill(0);
    initializeEKF();

    P_ekf.Fill(0);
    P_ekf(0, 0) = sigma_a * sigma_a;
    P_ekf(1, 1) = 0.01f;
    P_ekf(2, 2) = 0.001f;

    P_ekfb.Fill(0);
    P_ekfb(0, 0) = sigma_a * sigma_a;
    P_ekfb(1, 1) = 0.01f;
    P_ekfb(2, 2) = 0.001f;
    P_ekfb(3, 3) = sigma_a * sigma_a;

    Serial.println("# Extended Kalman Filter (EKF)");
    Serial.println("# t_ms | accel | bench_v bench_p | ekf_a ekf_v ekf_p ekf_stdv ekf_stdp | ekfb_a ekfb_v ekfb_p ekfb_b ekfb_stdv ekfb_stdp");
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

    float accel = ax * 9.82f;
    uint32_t t_ms = millis() - t0;

    benchmarkUpdate(accel);
    ekfUpdate(accel);
    ekfUpdateBias(accel);

    float ekf_std_v = sqrt(P_ekf(1, 1));
    float ekf_std_p = sqrt(P_ekf(2, 2));
    float ekfb_std_v = sqrt(P_ekfb(1, 1));
    float ekfb_std_p = sqrt(P_ekfb(2, 2));

    Serial.print("t_ms:");
    Serial.print(t_ms);
    Serial.print(" accel:");
    Serial.print(accel, 4);
    Serial.print(" bench_v:");
    Serial.print(bench_v, 4);
    Serial.print(" bench_p:");
    Serial.print(bench_p, 4);
    Serial.print(" ekf_a:");
    Serial.print(x_ekf(0, 0), 4);
    Serial.print(" ekf_v:");
    Serial.print(x_ekf(1, 0), 4);
    Serial.print(" ekf_p:");
    Serial.print(x_ekf(2, 0), 4);
    Serial.print(" ekf_stdv:");
    Serial.print(ekf_std_v, 4);
    Serial.print(" ekf_stdp:");
    Serial.print(ekf_std_p, 4);
    Serial.print(" ekfb_a:");
    Serial.print(x_ekfb(0, 0), 4);
    Serial.print(" ekfb_v:");
    Serial.print(x_ekfb(1, 0), 4);
    Serial.print(" ekfb_p:");
    Serial.print(x_ekfb(2, 0), 4);
    Serial.print(" ekfb_b:");
    Serial.print(x_ekfb(3, 0), 4);
    Serial.print(" ekfb_stdv:");
    Serial.print(ekfb_std_v, 4);
    Serial.print(" ekfb_stdp:");
    Serial.println(ekfb_std_p, 4);
}