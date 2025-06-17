import asyncio
import platform
import pygame
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

#Stałe fizyczne i symulacyjne
g = 9.81  #Przyspieszenie grawitacyjne [m/s^2]
L = 4.0  #Długość wahadła [m]
b = 0.0  #Współczynnik tłumienia
FPS = 60  #Klatki na sekundę
dt = 1.0 / FPS  #Krok czasowy

#Parametry początkowe
theta0_values = np.array([np.pi / 6, np.pi / 3, np.pi / 2])  #Amplitudy: 30°, 60°, 90° (rad)
theta_harmonic = theta0_values[0]
theta_real = theta0_values[0]
omega_harmonic = 0.0
omega_real = 0.0
time = 0.0
omega_0 = np.sqrt(g / L)  #Częstość kątowa wahadła harmonicznego

#Listy do przechowywania przebiegów
harmonic_data = []
real_data = []
time_data = []

#Parametry Pygame
width, height = 1200, 600
pendulum_length = 200
origin_x1, origin_y = 300, 100
origin_x2 = 900
origin_y2 = 100

#Inicjalizacja Pygame
pygame.init()
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Symulator Wahadeł")
font = pygame.font.Font(None, 36)

def setup():
    global screen
    screen.fill((255, 255, 255))

def real_pendulum(state):
    """Równanie wahadła rzeczywistego: d^2theta/dt^2 + b*dtheta/dt + (g/L)sin(theta) = 0"""
    theta, omega = state
    dtheta = omega
    domega = -(g / L) * np.sin(theta) - b * omega
    return np.array([dtheta, domega])

def rk4_step(f, state, dt):
    """Metoda Runge-Kutta 4. rzędu"""
    k1 = f(state)
    k2 = f(state + 0.5 * dt * k1)
    k3 = f(state + 0.5 * dt * k2)
    k4 = f(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def update_loop():
    global theta_harmonic, omega_harmonic, theta_real, omega_real, time
    #Wahadło harmoniczne (rozwiązanie analityczne)
    theta_harmonic = theta0 * np.cos(omega_0 * time)
    omega_harmonic = -theta0 * omega_0 * np.sin(omega_0 * time)

    #Wahadło rzeczywiste (Runge-Kutta)
    state_r = np.array([theta_real, omega_real])
    state_r_new = rk4_step(real_pendulum, state_r, dt)
    theta_real, omega_real = state_r_new

    #Zapis danych
    time_data.append(time)
    harmonic_data.append(theta_harmonic)
    real_data.append(theta_real)
    time += dt

    #Rysowanie
    screen.fill((255, 255, 255))
    #Wahadło harmoniczne
    x_h = origin_x1 + pendulum_length * np.sin(theta_harmonic)
    y_h = origin_y + pendulum_length * np.cos(theta_harmonic)
    pygame.draw.line(screen, (0, 0, 255), (origin_x1, origin_y), (x_h, y_h), 2)
    pygame.draw.circle(screen, (0, 0, 255), (int(x_h), int(y_h)), 10)
    text_h = font.render("Harmoniczne", True, (0, 0, 255))
    screen.blit(text_h, (origin_x1 - 50, 50))
    #Wyświetlanie kąta theta dla wahadła harmonicznego
    theta_h_deg = np.degrees(theta_harmonic)
    text_theta_h = font.render(f"θ = {theta_h_deg:.1f}°", True, (0, 0, 255))
    screen.blit(text_theta_h, (origin_x1 - 50, 80))

    #Wahadło rzeczywiste
    x_r = origin_x2 + pendulum_length * np.sin(theta_real)
    y_r = origin_y2 + pendulum_length * np.cos(theta_real)
    pygame.draw.line(screen, (255, 0, 0), (origin_x2, origin_y2), (x_r, y_r), 2)
    pygame.draw.circle(screen, (255, 0, 0), (int(x_r), int(y_r)), 10)
    text_r = font.render("Rzeczywiste", True, (255, 0, 0))
    screen.blit(text_r, (origin_x2 - 50, 50))
    #Wyświetlanie kąta theta dla wahadła rzeczywistego
    theta_r_deg = np.degrees(theta_real)
    text_theta_r = font.render(f"θ = {theta_r_deg:.1f}°", True, (255, 0, 0))
    screen.blit(text_theta_r, (origin_x2 - 50, 80))

    pygame.display.flip()

def estimate_period(data, times):
    """Oszacowanie okresu na podstawie detekcji szczytów"""
    peaks, _ = find_peaks(data, distance=5)
    if len(peaks) < 2:
        return None
    periods = np.diff(times[peaks])
    return np.mean(periods) if len(periods) > 0 else None

async def main():
    global theta_harmonic, theta_real, omega_harmonic, omega_real, time, theta0
    setup()
    periods_harmonic = []
    periods_real = []

    for theta0 in theta0_values:
        #Resetowanie parametrów
        theta_harmonic = theta0
        theta_real = theta0
        omega_harmonic = 0.0
        omega_real = 0.0
        time = 0.0
        harmonic_data.clear()
        real_data.clear()
        time_data.clear()

        #Symulacja przez 15 sekund
        for _ in range(int(15.0 / dt)):
            update_loop()
            await asyncio.sleep(1.0 / FPS)

        #Oszacowanie okresów
        period_h = estimate_period(np.array(harmonic_data), np.array(time_data))
        period_r = estimate_period(np.array(real_data), np.array(time_data))
        if period_h and period_r:
            periods_harmonic.append(period_h)
            periods_real.append(period_r)

        #Wykres kąta w czasie
        plt.figure(figsize=(8, 6))
        plt.plot(time_data, harmonic_data, 'b-', label='Harmoniczne')
        plt.plot(time_data, real_data, 'r-', label='Rzeczywiste')
        plt.title(f'Kąt vs czas (θ₀ = {np.degrees(theta0):.0f}°)')
        plt.xlabel('Czas [s]')
        plt.ylabel('Kąt [rad]')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'pendulum_theta_{np.degrees(theta0):.0f}deg.png')
        plt.close()

    #Wykres stosunku okresów
    ratios = [pr / ph for pr, ph in zip(periods_real, periods_harmonic)]
    percent_diff = [((pr - ph) / ph) * 100 for pr, ph in zip(periods_real, periods_harmonic)]

    plt.figure(figsize=(8, 6))
    plt.plot(np.degrees(theta0_values), ratios, 'o-', label='T_rzeczywiste / T_harmoniczne')
    for i, (theta0, ratio, pd) in enumerate(zip(np.degrees(theta0_values), ratios, percent_diff)):
        plt.annotate(f'{pd:.1f}%', (theta0, ratio), textcoords="offset points", xytext=(0, 10), ha='center')
    plt.title('Stosunek okresów w funkcji amplitudy początkowej')
    plt.xlabel('Amplituda początkowa θ₀ [°]')
    plt.ylabel('Stosunek okresów T_r / T_h')
    plt.grid(True)
    plt.legend()
    plt.savefig('period_ratio.png')
    plt.close()

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())